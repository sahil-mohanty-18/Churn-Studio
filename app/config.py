"""Environment configuration, rooted independently of working directory."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class Config:
    DATABASE = str(ROOT / "data/churn.db")
    MODEL_PATH = str(ROOT / "models/churn.joblib")
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    RATELIMIT_STORAGE_URI = "memory://"
    RATELIMIT_HEADERS_ENABLED = True
    CORS_ORIGINS = [
        x
        for x in os.getenv(
            "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
        ).split(",")
        if x
    ]
    JSON_SORT_KEYS = False
