"""Stateless engineering followed by fold-local imputation and encoding."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICES = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().drop_duplicates(subset=["customerID"] if "customerID" in df else None)
    for c in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "SupportTickets",
    ]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy().drop(
            columns=["customerID", "Churn", "SupportTickets"], errors="ignore"
        )
        d["TenureBucket"] = pd.cut(
            d.tenure,
            [-1, 12, 36, 60, 1000],
            labels=["new", "growing", "established", "loyal"],
        ).astype(str)
        d["ChargesRatio"] = d.TotalCharges / (
            d.tenure.clip(lower=1) * d.MonthlyCharges.clip(lower=1)
        )
        d["ServiceCount"] = sum(d[c].eq("Yes").astype(int) for c in SERVICES)
        d["ContractRisk"] = d.Contract.eq("Month-to-month").astype(int)
        d["ChargeContract"] = d.MonthlyCharges * d.ContractRisk
        d["NewFiber"] = (d.tenure.lt(12) & d.InternetService.eq("Fiber optic")).astype(
            int
        )
        return d


def preprocess() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    category = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            (
                "columns",
                ColumnTransformer(
                    [
                        ("num", numeric, make_column_selector(dtype_include=np.number)),
                        (
                            "cat",
                            category,
                            make_column_selector(dtype_exclude=np.number),
                        ),
                    ],
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
