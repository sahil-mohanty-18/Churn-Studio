import numpy as np
import pandas as pd

from data.generate_synthetic import generate
from src.ml.features import clean, preprocess


def test_synthetic_schema_and_rate():
    d = generate(5000)
    assert 0.23 < d.Churn.eq("Yes").mean() < 0.29
    assert d.customerID.is_unique
    assert {"TotalCharges", "SupportTickets", "TechSupport", "Contract"}.issubset(
        d.columns
    )


def test_clean_numeric_missing_duplicates():
    d = pd.DataFrame(
        {
            "customerID": ["a", "a", "b"],
            "TotalCharges": [" ", " ", "5"],
            "tenure": ["0", "0", "2"],
        }
    )
    out = clean(d)
    assert (
        len(out) == 2
        and pd.isna(out.iloc[0].TotalCharges)
        and out.iloc[1].TotalCharges == 5
    )


def test_pipeline_unknown_categories_and_no_target():
    d = generate(50)
    p = preprocess()
    a = p.fit_transform(d.drop(columns=["Churn"]))
    new = d.iloc[:1].copy()
    new["Contract"] = "Unseen"
    new["TotalCharges"] = np.nan
    b = p.transform(new.drop(columns=["Churn"]))
    assert a.shape[1] == b.shape[1] and np.isfinite(b).all()
    assert not any(
        "Churn" in x or "customerID" in x
        for x in p.named_steps["columns"].get_feature_names_out()
    )


def test_download_fallback(monkeypatch, tmp_path):
    import data.download_data as downloader

    def fail(*args, **kwargs):
        raise OSError("offline test")

    monkeypatch.setattr(downloader, "urlopen", fail)
    monkeypatch.chdir(tmp_path)
    result = downloader.download()
    assert (
        result["kind"] == "synthetic fallback"
        and (tmp_path / "data/telco.csv").exists()
    )
