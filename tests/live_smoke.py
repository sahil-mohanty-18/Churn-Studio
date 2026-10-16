"""Exercise real HTTP responses from the documented production server command."""

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]
log = (root / "reports/live-server.log").open("w")
server = subprocess.Popen(
    [sys.executable, "run.py", "serve", "--port", "5057"],
    cwd=root,
    stdout=log,
    stderr=log,
)
base = "http://127.0.0.1:5057"
try:
    for _ in range(60):
        try:
            with urlopen(base + "/api/health", timeout=1) as r:
                assert json.load(r)["status"] == "ok"
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise RuntimeError("Server did not become healthy")
    pages = ["/", "/predict", "/batch", "/customers", "/insights", "/simulator"]
    for page in pages:
        with urlopen(base + page) as r:
            assert r.status == 200 and b"page-content" in r.read()
    paths = [
        "health",
        "schema",
        "customers",
        "analytics/kpis",
        "analytics/contract",
        "analytics/tenure",
        "analytics/payment",
        "analytics/risk",
        "analytics/feature-importance",
        "analytics/model-metrics",
    ]
    for path in paths:
        with urlopen(base + "/api/" + path) as r:
            assert r.status == 200
            json.load(r)
    profile = {
        "tenure": 12,
        "MonthlyCharges": 79,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    with urlopen(
        Request(
            base + "/api/predict",
            data=json.dumps(profile).encode(),
            headers={"Content-Type": "application/json"},
        )
    ) as r:
        result = json.load(r)
        assert len(result["reasons"]) == 5
    with urlopen(base + "/api/customers?page_size=1") as r:
        cid = json.load(r)["items"][0]["customer_id"]
    for path in ["customers/" + cid, "retention-advice/" + cid]:
        with urlopen(base + "/api/" + path) as r:
            assert r.status == 200
            json.load(r)
    print(
        "PASS: one-command server, 6 pages, 12 GET APIs, prediction and SHAP over real HTTP."
    )
finally:
    server.terminate()
    server.wait(timeout=10)
    log.close()
