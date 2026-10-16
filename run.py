"""One-command workflows. Run from any directory: python run.py [command]."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(ROOT)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["setup", "download", "train", "serve", "test"],
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.command == "setup":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
    elif args.command == "download":
        from data.download_data import download

        print(download())
    elif args.command == "train":
        from src.ml.train import run

        run()
    elif args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    else:
        import sqlite3

        ready = False
        if (ROOT / "data/churn.db").exists():
            try:
                with sqlite3.connect(ROOT / "data/churn.db") as con:
                    ready = (
                        con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] > 0
                        and con.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
                        > 0
                    )
            except sqlite3.Error:
                pass
        if not (ROOT / "models/churn.joblib").exists() or not ready:
            from src.ml.train import run

            run()
        from waitress import serve

        from app import create_app

        print(f"Churn Studio: http://{args.host}:{args.port}", flush=True)
        serve(create_app(), host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
