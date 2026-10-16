import io

import pytest


@pytest.mark.parametrize(
    "path",
    ["/", "/dashboard", "/predict", "/batch", "/customers", "/insights", "/simulator"],
)
def test_pages(client, path):
    r = client.get(path)
    assert r.status_code == 200 and b"Churn Studio" in r.data


@pytest.mark.parametrize(
    "path",
    [
        "health",
        "schema",
        "customers",
        "analytics/kpis",
        "analytics/contract",
        "analytics/tenure",
        "analytics/payment",
        "analytics/risk",
        "analytics/model-metrics",
        "analytics/feature-importance",
    ],
)
def test_json_endpoints(client, path):
    r = client.get("/api/" + path)
    assert r.status_code == 200 and r.is_json


def test_predict_and_explanation(client, profile):
    r = client.post("/api/predict", json=profile)
    assert r.status_code == 200
    d = r.json
    assert 0 <= d["probability"] <= 1 and len(d["reasons"]) == 5
    assert (
        d["risk_level"] in ["Low", "Medium", "High"]
        and d["retention"]["mode"] == "offline"
    )
    assert all(isinstance(x["contribution"], float) for x in d["reasons"])


@pytest.mark.parametrize(
    "payload", [{}, [], {"tenure": -1}, {"tenure": "not a number"}, None]
)
def test_invalid_payload(client, payload):
    assert client.post("/api/predict", json=payload).status_code in [400, 415]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("MonthlyCharges", "NaN"),
        ("Contract", "<script>"),
        ("SeniorCitizen", 0.5),
        ("TotalCharges", float("inf")),
    ],
)
def test_validation(client, profile, field, value):
    profile[field] = value
    assert client.post("/api/predict", json=profile).status_code == 400


def test_customer_filters_and_detail(client):
    r = client.get(
        "/api/customers?risk=High&page_size=3&sort=tenure&direction=asc"
    ).json
    assert len(r["items"]) == 3 and all(x["risk"] == "High" for x in r["items"])
    cid = r["items"][0]["customer_id"]
    assert client.get("/api/customers/" + cid).json["profile"]["customerID"] == cid
    assert client.get("/api/retention-advice/" + cid).json["actions"]
    assert client.get("/api/customers?search=" + cid).json["total"] == 1
    for path in [
        "customers?sort=bad",
        "customers?page=0",
        "customers?page=abc",
        "customers?page_size=101",
    ]:
        assert client.get("/api/" + path).status_code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/api/customers/missing",
        "/api/retention-advice/missing",
        "/api/analytics/missing",
        "/api/downloads/missing",
        "/missing",
    ],
)
def test_missing(client, path):
    assert client.get(path).status_code == 404 and client.get(path).is_json


def test_batch_download(client):
    raw = "customerID,tenure,MonthlyCharges,Contract,InternetService,PaymentMethod\n=1+1,12,79,Month-to-month,Fiber optic,Electronic check\n"
    r = client.post(
        "/api/predict/batch", data={"file": (io.BytesIO(raw.encode()), "customers.csv")}
    )
    assert r.status_code == 200 and r.json["count"] == 1
    download = client.get(r.json["download_url"])
    assert download.status_code == 200 and b"'=1+1" in download.data
    assert "attachment" in download.headers["Content-Disposition"]


def test_invalid_batch(client):
    assert client.post("/api/predict/batch").status_code == 400
    for raw in ["bad\n1\n", "tenure\n"]:
        assert (
            client.post(
                "/api/predict/batch", data={"file": (io.BytesIO(raw.encode()), "x.csv")}
            ).status_code
            == 400
        )


def test_security_headers_and_cors(client):
    r = client.get("/api/health", headers={"Origin": "https://malicious.example"})
    assert (
        r.headers["X-Content-Type-Options"] == "nosniff"
        and "Access-Control-Allow-Origin" not in r.headers
    )
    r = client.get("/api/health", headers={"Origin": "http://localhost:5000"})
    assert r.headers["Access-Control-Allow-Origin"] == "http://localhost:5000"


def test_logging_prediction(client, app, profile):
    cid = client.get("/api/customers?page_size=1").json["items"][0]["customer_id"]
    profile["customerID"] = cid
    assert client.post("/api/predict", json=profile).status_code == 200
    import sqlite3

    with sqlite3.connect(app.config["DATABASE"]) as con:
        assert (
            con.execute(
                "SELECT COUNT(*) FROM predictions WHERE customer_id=?", (cid,)
            ).fetchone()[0]
            >= 2
        )


def test_online_failure_falls_back(monkeypatch, profile):
    import app.services as services

    monkeypatch.setenv("ENABLE_FREE_AI", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    def fail(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(services, "urlopen", fail)
    assert services.advice(profile, [])["mode"] == "offline"


def test_rate_limit(tmp_path):
    from app import create_app

    a = create_app({"TESTING": True, "RATELIMIT_ENABLED": True})
    c = a.test_client()
    responses = [c.get("/api/health") for _ in range(121)]
    assert responses[-1].status_code == 429 and responses[-1].is_json


def test_invalid_customer_id(client, profile):
    profile["customerID"] = {"bad": "type"}
    assert client.post("/api/predict", json=profile).status_code == 400


def test_batch_row_limit(client):
    row = "12,79,Month-to-month,Fiber optic,Electronic check\n"
    raw = "tenure,MonthlyCharges,Contract,InternetService,PaymentMethod\n" + row * 501
    response = client.post(
        "/api/predict/batch", data={"file": (io.BytesIO(raw.encode()), "too-many.csv")}
    )
    assert response.status_code == 400


def test_blank_total_charges(client, profile):
    profile["TotalCharges"] = ""
    assert client.post("/api/predict", json=profile).status_code == 200
