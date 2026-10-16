"""Small CPU-only sklearn-compatible PyTorch classifier."""

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn


class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        epochs: int = 45,
        hidden: int = 48,
        lr: float = 0.002,
        random_state: int = 42,
    ) -> None:
        self.epochs = epochs
        self.hidden = hidden
        self.lr = lr
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        a, b, c, d = train_test_split(
            X, y, test_size=0.18, stratify=y, random_state=self.random_state
        )
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.network_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 1),
        )
        opt = torch.optim.AdamW(
            self.network_.parameters(), lr=self.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=3, factor=0.5
        )
        loss_fn = nn.BCEWithLogitsLoss()
        best = float("inf")
        stale = 0
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(a), torch.tensor(c)),
            batch_size=128,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )
        for epoch in range(self.epochs):
            self.network_.train()
            for xx, yy in loader:
                opt.zero_grad()
                loss = loss_fn(self.network_(xx).ravel(), yy)
                loss.backward()
                opt.step()
            self.network_.eval()
            with torch.no_grad():
                val = loss_fn(
                    self.network_(torch.tensor(b)).ravel(), torch.tensor(d)
                ).item()
            scheduler.step(val)
            if val < best - 1e-4:
                best = val
                weights = copy.deepcopy(self.network_.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 7:
                break
        self.network_.load_state_dict(weights)
        self.network_.eval()
        self.epochs_trained_ = epoch + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.network_.eval()
        with torch.no_grad():
            p = (
                torch.sigmoid(
                    self.network_(torch.tensor(np.asarray(X, dtype=np.float32)))
                )
                .numpy()
                .ravel()
            )
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
