"""JSON endpoints; parameterized SQL and bounded, validated inputs."""

import csv
import io
import json
import time
import uuid

import pandas as pd
from flask import Blueprint, Response, current_app, jsonify, request, send_file

from app.db import get_db, query
from app.services import Predictor, advice

api = Blueprint("api", __name__, url_prefix="/api")


def predictor() -> Predictor:
    return current_app.extensions["predictor"]


def integer(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except ValueError:
        raise ValueError(f"{name} must be an integer.") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}.")
    return value


@api.get("/health")
def health() -> Response:
    get_db().execute("SELECT 1")
    return jsonify(
        status="ok", model_loaded=True, model=predictor().bundle["metrics"]["winner"]
    )


@api.get("/schema")
def schema() -> Response:
    from app.services import CHOICES, NUMERIC

    return jsonify(
        choices=CHOICES, numeric=NUMERIC, defaults=predictor().bundle["defaults"]
    )


@api.post("/predict")
def predict() -> Response:
    payload = request.get_json()
    result = predictor().predict(payload)
    customer_id = payload.get("customerID")
    if (
        customer_id
        and get_db()
        .execute("SELECT 1 FROM customers WHERE customer_id=?", (customer_id,))
        .fetchone()
    ):
        db = get_db()
        run_id = (
            predictor().bundle.get("model_run_id")
            or db.execute("SELECT MAX(id) FROM model_runs").fetchone()[0]
        )
        db.execute(
            "INSERT INTO predictions(customer_id,model_run_id,probability,risk) VALUES(?,?,?,?)",
            (customer_id, run_id, result["probability"], result["risk_level"]),
        )
        db.commit()
    return jsonify(result)


@api.post("/predict/batch")
def batch() -> Response:
    upload = request.files.get("file")
    if upload is None or not upload.filename.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file in the file field.")
    try:
        frame = pd.read_csv(upload, dtype=str, keep_default_na=False)
    except Exception:
        raise ValueError("CSV could not be parsed.") from None
    if frame.empty or len(frame) > 500:
        raise ValueError("CSV must have 1–500 customer rows.")
    rows = []
    for i, payload in enumerate(frame.to_dict("records")):
        try:
            result = predictor().predict(payload, explain=False)
        except ValueError as exc:
            raise ValueError(f"Row {i + 2}: {exc}") from None
        rows.append(
            {
                "customerID": str(payload.get("customerID", f"row-{i + 1}")),
                "probability": result["probability"],
                "risk_level": result["risk_level"],
            }
        )
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["customerID", "probability", "risk_level"]
    )
    writer.writeheader()
    for row in rows:
        safe = dict(row)
        if safe["customerID"].startswith(("=", "+", "-", "@", "\t", "\r")):
            safe["customerID"] = "'" + safe["customerID"]
        writer.writerow(safe)
    # Short-lived bounded cache, suitable for the documented single-process deployment.
    cache = current_app.extensions["downloads"]
    now = time.time()
    for token in list(cache):
        if now - cache[token][0] > 900:
            del cache[token]
    if len(cache) >= 20:
        del cache[min(cache, key=lambda k: cache[k][0])]
    token = uuid.uuid4().hex
    cache[token] = (now, output.getvalue())
    return jsonify(
        count=len(rows), predictions=rows, download_url="/api/downloads/" + token
    )


@api.get("/downloads/<token>")
def download(token: str) -> Response | tuple[Response, int]:
    item = current_app.extensions["downloads"].get(token)
    if item is None or time.time() - item[0] > 900:
        return jsonify(error="Download expired or not found."), 404
    return send_file(
        io.BytesIO(item[1].encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_predictions.csv",
    )


@api.get("/customers")
def customers() -> Response:
    page = integer("page", 1, 1, 100000)
    size = integer("page_size", 15, 1, 100)
    conditions = ["1=1"]
    args = []
    search = request.args.get("search", "")[:100]
    if search:
        conditions.append("c.customer_id LIKE ?")
        args.append("%" + search + "%")
    for param, col in [("contract", "c.contract"), ("risk", "p.risk")]:
        if request.args.get(param):
            conditions.append(col + "=?")
            args.append(request.args[param])
    sorts = {
        "customer_id": "c.customer_id",
        "tenure": "c.tenure",
        "monthly_charges": "c.monthly_charges",
        "probability": "p.probability",
    }
    sort = request.args.get("sort", "probability")
    direction = request.args.get("direction", "desc")
    if sort not in sorts or direction not in ["asc", "desc"]:
        raise ValueError("Invalid sort or direction.")
    where = " AND ".join(conditions)
    base = (
        " FROM customers c LEFT JOIN latest_predictions p USING(customer_id) WHERE "
        + where
    )
    db = get_db()
    total = db.execute("SELECT COUNT(*)" + base, args).fetchone()[0]
    rows = db.execute(
        "SELECT c.customer_id,c.tenure,c.contract,c.monthly_charges,c.payment_method,p.probability,p.risk"
        + base
        + " ORDER BY "
        + sorts[sort]
        + " "
        + direction
        + ",c.customer_id LIMIT ? OFFSET ?",
        args + [size, (page - 1) * size],
    ).fetchall()
    return jsonify(
        items=[dict(r) for r in rows], total=total, page=page, page_size=size
    )


@api.get("/customers/<customer_id>")
def customer(customer_id: str) -> Response | tuple[Response, int]:
    row = (
        get_db()
        .execute("SELECT profile FROM customers WHERE customer_id=?", (customer_id,))
        .fetchone()
    )
    if not row:
        return jsonify(error="Customer not found."), 404
    profile = json.loads(row["profile"])
    return jsonify(profile=profile, prediction=predictor().predict(profile))


@api.get("/analytics/<name>")
def analytics(name: str) -> Response | tuple[Response, int]:
    if name in ["kpis", "contract", "tenure", "payment", "risk"]:
        return jsonify(query(name)[0] if name == "kpis" else query(name))
    if name in ["model-metrics", "feature-importance"]:
        rows = query("metrics")
        if not rows:
            return jsonify(error="No model run available."), 503
        metrics = json.loads(rows[0]["metrics"])
        return jsonify(
            metrics if name == "model-metrics" else metrics["feature_importance"]
        )
    return jsonify(error="Unknown analytics query."), 404


@api.get("/retention-advice/<customer_id>")
def retention(customer_id: str) -> Response | tuple[Response, int]:
    row = (
        get_db()
        .execute("SELECT profile FROM customers WHERE customer_id=?", (customer_id,))
        .fetchone()
    )
    if not row:
        return jsonify(error="Customer not found."), 404
    profile = json.loads(row["profile"])
    result = predictor().predict(profile)
    return jsonify(customer_id=customer_id, **advice(profile, result["reasons"]))
