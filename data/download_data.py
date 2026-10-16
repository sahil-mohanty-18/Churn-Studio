"""Download IBM sample or transparently use synthetic offline fallback."""

import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from data.generate_synthetic import generate

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download() -> dict:
    Path("data").mkdir(exist_ok=True)
    try:
        with urlopen(URL, timeout=20) as response:
            raw = response.read()
        df = pd.read_csv(BytesIO(raw))
        assert {"customerID", "Churn", "TotalCharges"}.issubset(df.columns)
        source = {
            "kind": "IBM Telco sample",
            "url": URL,
            "support_tickets": "Not collected by IBM; omitted from real-data modeling",
        }
    except Exception as exc:
        df = generate()
        source = {
            "kind": "synthetic fallback",
            "reason": str(exc),
            "support_tickets": "Simulated",
        }
    df.to_csv("data/telco.csv", index=False)
    source.update(rows=len(df), churn_rate=float(df.Churn.eq("Yes").mean()))
    Path("data/source.json").write_text(json.dumps(source, indent=2))
    return source


if __name__ == "__main__":
    print(download())
