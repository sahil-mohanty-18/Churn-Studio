"""Reproducible Telco-like fallback; never presented as real IBM observations."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 3500, seed: int = 42) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:06d}" for i in range(n)],
            "gender": r.choice(["Male", "Female"], n),
            "SeniorCitizen": r.binomial(1, 0.16, n),
        }
    )
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[c] = r.choice(["Yes", "No"], n, p=[0.6, 0.4])
    d["tenure"] = r.integers(0, 73, n)
    d["Contract"] = r.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.22, 0.23]
    )
    d["InternetService"] = r.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    for c in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        d[c] = np.where(
            d.InternetService.eq("No"),
            "No internet service",
            r.choice(["Yes", "No"], n),
        )
    d["MultipleLines"] = np.where(
        d.PhoneService.eq("No"), "No phone service", r.choice(["Yes", "No"], n)
    )
    d["PaymentMethod"] = r.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
    )
    d["MonthlyCharges"] = (
        20
        + 25 * d.InternetService.eq("DSL")
        + 45 * d.InternetService.eq("Fiber optic")
        + r.uniform(0, 25, n)
    ).round(2)
    d["TotalCharges"] = (d.tenure * d.MonthlyCharges * r.uniform(0.85, 1.05, n)).round(
        2
    )
    d["SupportTickets"] = r.poisson(1.4, n)
    z = (
        1.6 * d.Contract.eq("Month-to-month")
        - 0.045 * d.tenure
        + 0.8 * d.InternetService.eq("Fiber optic")
        + 0.6 * d.PaymentMethod.eq("Electronic check")
        + 0.3 * d.SupportTickets
        - 0.6 * d.TechSupport.eq("Yes")
    )
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + mid)))) > 0.26:
            hi = mid
        else:
            lo = mid
    d["Churn"] = np.where(r.random(n) < 1 / (1 + np.exp(-(z + mid))), "Yes", "No")
    return d


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    generate().to_csv("data/telco.csv", index=False)
