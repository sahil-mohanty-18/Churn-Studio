"""Permutation SHAP in probability units for the exact deployed model."""

import numpy as np
import pandas as pd
import shap


class ExplanationService:
    def __init__(self, model, background: pd.DataFrame, columns: list[str]):
        self.model = model
        self.columns = columns
        self.categories = {
            c: list(background[c].dropna().unique())
            for c in columns
            if background[c].dtype == "object"
        }
        self.background = background

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        d = df[self.columns].copy()
        for c, cats in self.categories.items():
            for value in d[c].dropna().unique():
                if value not in cats:
                    cats.append(value)
            d[c] = d[c].map({v: i for i, v in enumerate(cats)})
        return d.to_numpy(dtype=float)

    def predict(self, array: np.ndarray) -> np.ndarray:
        d = pd.DataFrame(array, columns=self.columns)
        for c, cats in self.categories.items():
            d[c] = [cats[int(round(v))] if np.isfinite(v) else None for v in d[c]]
        return self.model.predict_proba(d)[:, 1]

    def values(self, df: pd.DataFrame) -> np.ndarray:
        encoded = self.encode(df)
        background = self.encode(self.background)
        explainer = shap.Explainer(
            self.predict,
            shap.maskers.Independent(background),
            algorithm="permutation",
            seed=42,
        )
        return explainer(
            encoded, max_evals=2 * len(self.columns) + 1, silent=True
        ).values

    def reasons(self, df: pd.DataFrame) -> list[dict]:
        values = self.values(df)[0]
        order = np.argsort(-np.abs(values))[:5]
        return [
            {
                "feature": self.columns[i],
                "value": None
                if pd.isna(df.iloc[0, self.columns.index(self.columns[i])])
                else str(df.iloc[0][self.columns[i]]),
                "contribution": float(values[i]),
                "direction": "increases" if values[i] > 0 else "decreases",
            }
            for i in order
        ]
