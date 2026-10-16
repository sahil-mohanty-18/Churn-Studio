"""Validation, model inference and scoped retention recommendations."""

import json
import math
import os
import threading
from urllib.request import Request, urlopen

import joblib
import pandas as pd

from src.ml.explain import ExplanationService


def risk(p: float, threshold: float) -> str:
    """Risk categories follow the validated contact policy."""
    return "High" if p >= threshold else ("Medium" if p >= threshold * 0.5 else "Low")


NUMERIC = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 100000),
    "SeniorCitizen": (0, 1),
}
CHOICES = {
    "gender": ["Male", "Female"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    CHOICES[c] = ["Yes", "No"]
CHOICES["MultipleLines"] = ["Yes", "No", "No phone service"]
for c in [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]:
    CHOICES[c] = ["Yes", "No", "No internet service"]


class Predictor:
    def __init__(self, path: str):
        self.bundle = joblib.load(path)
        self.model = self.bundle["model"]
        self.lock = threading.Lock()

    def validate(self, payload: dict) -> tuple[dict, list[str]]:
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        if "customerID" in payload and (
            not isinstance(payload["customerID"], str)
            or not 1 <= len(payload["customerID"]) <= 100
        ):
            raise ValueError(
                "customerID must be a non-empty string of at most 100 characters."
            )
        required = [
            "tenure",
            "MonthlyCharges",
            "Contract",
            "InternetService",
            "PaymentMethod",
        ]
        missing = [
            k
            for k in required
            if k not in payload or payload[k] is None or str(payload[k]).strip() == ""
        ]
        if missing:
            raise ValueError("Missing required fields: " + ", ".join(missing))
        unknown = (
            set(payload)
            - set(self.bundle["columns"])
            - {"customerID", "Churn", "SupportTickets"}
        )
        if unknown:
            raise ValueError("Unknown fields: " + ", ".join(sorted(unknown)))
        d = dict(self.bundle["defaults"])
        d.update({k: v for k, v in payload.items() if k in d})
        defaulted = [k for k in self.bundle["columns"] if k not in payload]
        for c, (lo, hi) in NUMERIC.items():
            if c == "TotalCharges" and (d[c] is None or str(d[c]).strip() == ""):
                d[c] = float("nan")
                continue
            try:
                value = float(d[c])
            except (TypeError, ValueError):
                raise ValueError(f"{c} must be numeric.") from None
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{c} must be between {lo} and {hi}.")
            if c in ["tenure", "SeniorCitizen"] and value != int(value):
                raise ValueError(f"{c} must be an integer.")
            d[c] = value
        for c, choices in CHOICES.items():
            if d[c] not in choices:
                raise ValueError(f"{c} must be one of: " + ", ".join(choices))
        if d["InternetService"] == "No":
            for c in [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]:
                d[c] = "No internet service"
        if d["PhoneService"] == "No":
            d["MultipleLines"] = "No phone service"
        return d, defaulted

    def predict(self, payload: dict, explain: bool = True) -> dict:
        d, defaulted = self.validate(payload)
        frame = pd.DataFrame([d], columns=self.bundle["columns"])
        p = float(self.model.predict_proba(frame)[0, 1])
        reasons = []
        if explain:
            with self.lock:
                explainer = ExplanationService(
                    self.model, self.bundle["background"], self.bundle["columns"]
                )
                reasons = explainer.reasons(frame)
        return {
            "probability": p,
            "risk_level": risk(p, self.bundle["threshold"]),
            "threshold": self.bundle["threshold"],
            "reasons": reasons,
            "defaulted_fields": defaulted,
            "retention": advice(d, reasons, online=False),
        }


def advice(profile: dict, reasons: list, online: bool = True) -> dict:
    actions = []
    drivers = {r["feature"] for r in reasons if r.get("contribution", 0) > 0}
    if profile.get("Contract") == "Month-to-month" or "Contract" in drivers:
        actions.append(
            "Offer an optional 12-month price guarantee; explain terms and avoid lock-in pressure."
        )
    if profile.get("TechSupport") != "Yes" or drivers & {
        "TechSupport",
        "OnlineSecurity",
    }:
        actions.append(
            "Schedule a complimentary technical support check and resolve outstanding service problems."
        )
    if float(profile.get("MonthlyCharges", 0)) > 75 or "MonthlyCharges" in drivers:
        actions.append(
            "Review plan usage and offer a lower-cost package or a time-limited 10% loyalty credit."
        )
    if profile.get("PaymentMethod") == "Electronic check":
        actions.append(
            "Offer optional automatic payments with clear consent and a billing walkthrough."
        )
    if float(profile.get("tenure", 24)) < 12:
        actions.append(
            "Arrange an onboarding check-in within seven days and confirm the service meets expectations."
        )
    if not actions:
        actions = [
            "Thank the customer for their loyalty and check whether their current plan still fits their needs."
        ]
    result = {
        "mode": "offline",
        "actions": actions,
        "note": "Recommendations are hypotheses to test; SHAP associations do not establish intervention effects.",
    }
    # Explicit free-tier opt-in prevents accidental use of a billed API account.
    if not online or os.getenv("ENABLE_FREE_AI", "false").lower() != "true":
        return result
    key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return result
    prompt = (
        "Write a brief retention plan using only these actions. Do not invent facts or request identifying data. "
        + json.dumps(actions)
    )
    try:
        if os.getenv("GROQ_API_KEY"):
            url = "https://api.groq.com/openai/v1/chat/completions"
            body = {
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 250,
            }
            headers = {
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            }
        else:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                + ":generateContent"
            )
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 350},
            }
        with urlopen(
            Request(url, data=json.dumps(body).encode(), headers=headers), timeout=5
        ) as response:
            data = json.load(response)
        text = (
            data["choices"][0]["message"]["content"]
            if os.getenv("GROQ_API_KEY")
            else data["candidates"][0]["content"]["parts"][0]["text"]
        )
        if text:
            result.update(mode="online", summary=text)
    except Exception:
        pass
    return result
