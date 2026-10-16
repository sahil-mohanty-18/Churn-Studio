"""Isolate all API writes from the shipped SQLite database."""

import pytest

from app import create_app
from app.config import ROOT


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    # SQLite backup includes all committed WAL data.
    import sqlite3

    with sqlite3.connect(ROOT / "data/churn.db") as src, sqlite3.connect(path) as dest:
        src.backup(dest)
    return create_app(
        {"TESTING": True, "DATABASE": str(path), "RATELIMIT_ENABLED": False}
    )


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def profile():
    return {
        "tenure": 12,
        "MonthlyCharges": 79.0,
        "TotalCharges": 948.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
