"""Request-local SQLite connections and named SQL queries."""

import sqlite3

from flask import current_app, g

from app.config import ROOT


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"], timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def query(name: str) -> list[dict]:
    return [
        dict(r)
        for r in get_db().execute((ROOT / "sql" / f"{name}.sql").read_text()).fetchall()
    ]
