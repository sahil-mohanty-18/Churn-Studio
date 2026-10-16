"""Generate reproducible EDA without changing training data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.ml.features import clean


def run() -> None:
    d = clean(pd.read_csv("data/telco.csv"))
    d["churn"] = d.Churn.eq("Yes")
    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    d.groupby("Contract").churn.mean().plot.bar(
        ax=ax[0], color="#735bff", title="Observed churn by contract"
    )
    d.groupby(
        pd.cut(d.tenure, [0, 12, 24, 48, 72], include_lowest=True), observed=True
    ).churn.mean().plot.bar(ax=ax[1], color="#25bca7", title="Observed churn by tenure")
    d.boxplot(column="MonthlyCharges", by="Churn", ax=ax[2])
    fig.suptitle("Telco data exploration")
    fig.tight_layout()
    fig.savefig("reports/eda.png", dpi=140)
    plt.close(fig)
    Path("reports/EDA_INSIGHTS.md").write_text(
        f"# Data insights\n\n{len(d):,} unique customers; observed churn {d.churn.mean():.1%}.\n\nMissing TotalCharges: {d.TotalCharges.isna().sum()} (imputed inside each training fold).\n\nContract churn rates:\n"
        + d.groupby("Contract").churn.mean().to_string()
        + "\n\nAssociations are descriptive, not evidence that changing a contract causes churn to fall.\n"
    )


if __name__ == "__main__":
    run()
