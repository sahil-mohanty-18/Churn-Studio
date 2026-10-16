"""Leakage-safe comparison, bounded tuning, holdout evaluation and SQL publishing."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.ml.eda import run as eda
from src.ml.features import clean, preprocess
from src.ml.neural import TorchMLP

ROOT = Path(__file__).resolve().parents[2]


def pipeline(model):
    return Pipeline([("preprocess", preprocess()), ("model", model)])


def risk(p, threshold):
    return "High" if p >= threshold else ("Medium" if p >= threshold * 0.5 else "Low")


def run(trials: int = 6) -> dict:
    start = time.time()
    os.chdir(ROOT)
    Path("models").mkdir(exist_ok=True)
    if not Path("data/telco.csv").exists():
        from data.download_data import download

        download()
    source = (
        json.loads(Path("data/source.json").read_text())
        if Path("data/source.json").exists()
        else {"kind": "user-provided CSV", "rows": None}
    )
    df = clean(pd.read_csv("data/telco.csv"))
    eda()
    X = df.drop(columns=["Churn", "customerID", "SupportTickets"], errors="ignore")
    y = df.Churn.eq("Yes").astype(int)
    dev, xt, yd, yt = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    xtr, xv, ytr, yv = train_test_split(
        dev, yd, test_size=0.2, stratify=yd, random_state=43
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(C=1, max_iter=1200),
        "Logistic balanced": LogisticRegression(
            C=1, max_iter=1200, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=9,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=2,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=2,
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            num_leaves=12,
            learning_rate=0.035,
            min_child_samples=35,
            verbosity=-1,
            n_jobs=2,
            random_state=42,
        ),
        "PyTorch MLP": TorchMLP(),
    }
    results = []
    fitted = {}
    for name, model in candidates.items():
        p = pipeline(model)
        scores = cross_val_score(p, xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
        p.fit(xtr, ytr)
        prob = p.predict_proba(xv)[:, 1]
        results.append(
            dict(
                model=name,
                cv_auc=float(scores.mean()),
                cv_std=float(scores.std()),
                validation_auc=float(roc_auc_score(yv, prob)),
            )
        )
        fitted[name] = p
        print(name, results[-1], flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        m = LogisticRegression(
            C=trial.suggest_float("C", 0.005, 20, log=True), max_iter=1200
        )
        return cross_val_score(pipeline(m), xtr, ytr, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=trials)
    tuned = pipeline(LogisticRegression(**study.best_params, max_iter=1200))
    tuned.fit(xtr, ytr)
    fitted["Tuned Logistic"] = tuned
    results.append(
        dict(
            model="Tuned Logistic",
            cv_auc=float(study.best_value),
            cv_std=None,
            validation_auc=float(roc_auc_score(yv, tuned.predict_proba(xv)[:, 1])),
        )
    )
    # Rank ensemble members on training CV only, never on test outcomes.
    top = sorted(results, key=lambda x: x["cv_auc"], reverse=True)[:3]
    ensemble = VotingClassifier(
        [(f"m{i}", fitted[row["model"]]) for i, row in enumerate(top)],
        voting="soft",
        n_jobs=1,
    )
    scores = cross_val_score(ensemble, xtr, ytr, cv=cv, scoring="roc_auc")
    ensemble.fit(xtr, ytr)
    fitted["Soft Voting"] = ensemble
    results.append(
        dict(
            model="Soft Voting",
            cv_auc=float(scores.mean()),
            cv_std=float(scores.std()),
            validation_auc=float(roc_auc_score(yv, ensemble.predict_proba(xv)[:, 1])),
        )
    )
    winner = max(results, key=lambda x: x["validation_auc"])["model"]
    model = fitted[winner]
    pv = model.predict_proba(xv)[:, 1]
    # Assumptions: successful contact retains 30% of would-be churners, value $300, contact cost $10.
    thresholds = np.linspace(0.05, 0.8, 151)
    profits = [
        float(((pv >= t) & (yv == 1)).sum() * 90 - (pv >= t).sum() * 10)
        for t in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(profits))])
    pt = model.predict_proba(xt)[:, 1]
    labels = pt >= threshold
    fpr, tpr, _ = roc_curve(yt, pt)
    prec, rec, _ = precision_recall_curve(yt, pt)
    observed, predicted = calibration_curve(yt, pt, n_bins=10, strategy="quantile")
    metrics = dict(
        roc_auc=float(roc_auc_score(yt, pt)),
        pr_auc=float(average_precision_score(yt, pt)),
        precision=float(precision_score(yt, labels, zero_division=0)),
        recall=float(recall_score(yt, labels)),
        f1=float(f1_score(yt, labels)),
        brier=float(brier_score_loss(yt, pt)),
        confusion_matrix=confusion_matrix(yt, labels).tolist(),
        threshold=threshold,
        winner=winner,
        comparison=results,
        roc={"x": fpr.tolist(), "y": tpr.tolist()},
        pr={"x": rec.tolist(), "y": prec.tolist()},
        calibration={"x": predicted.tolist(), "y": observed.tolist()},
        source=source,
        split={"train": len(xtr), "validation": len(xv), "test": len(xt)},
        business={
            "contact_cost": 10,
            "retained_value": 300,
            "success_rate": 0.3,
            "validation_net_value": max(profits),
            "validation_contact_all_value": int(yv.sum()) * 90 - len(yv) * 10,
        },
        optuna={"trials": trials, "best_params": study.best_params},
        ensemble_members=[row["model"] for row in top],
    )
    # Model-agnostic permutation SHAP on actual deployed probability, including ensemble.
    # Encode only for SHAP's numerical masker; decoded values feed the original fitted pipeline.
    from src.ml.explain import ExplanationService

    explanation = ExplanationService(
        model, xtr.sample(32, random_state=42), list(X.columns)
    )
    sample = xv.sample(40, random_state=42)
    values = explanation.values(sample)
    metrics["feature_importance"] = [
        {"feature": c, "importance": float(v)}
        for c, v in sorted(
            zip(X.columns, np.abs(values).mean(axis=0)), key=lambda z: -z[1]
        )
    ]
    bundle = {
        "model": model,
        "threshold": threshold,
        "columns": list(X.columns),
        "background": xtr.sample(32, random_state=42),
        "defaults": xtr.mode().iloc[0].to_dict(),
        "metrics": metrics,
    }
    joblib.dump(model, "models/pipeline.joblib")
    joblib.dump(ensemble, "models/ensemble.joblib")
    if hasattr(model, "named_steps"):
        joblib.dump(model.named_steps["preprocess"], "models/preprocessing.joblib")
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    Path("models/threshold.json").write_text(
        json.dumps({"threshold": threshold, "business": metrics["business"]}, indent=2)
    )
    con = sqlite3.connect("data/churn.db")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(Path("sql/schema.sql").read_text())
    cur = con.execute(
        "INSERT INTO model_runs(model_name,metrics,source) VALUES(?,?,?)",
        (winner, json.dumps(metrics), source["kind"]),
    )
    run_id = cur.lastrowid
    probs = model.predict_proba(X)[:, 1]
    for (_, row), p in zip(df.iterrows(), probs):
        profile = row.drop(labels=["Churn"]).where(row.notna(), None).to_dict()
        con.execute(
            "INSERT INTO customers VALUES(?,?,?,?,?,?,?) ON CONFLICT(customer_id) DO UPDATE SET tenure=excluded.tenure,contract=excluded.contract,monthly_charges=excluded.monthly_charges,payment_method=excluded.payment_method,churn=excluded.churn,profile=excluded.profile",
            (
                row.customerID,
                int(row.tenure),
                row.Contract,
                float(row.MonthlyCharges),
                row.PaymentMethod,
                int(row.Churn == "Yes"),
                json.dumps(profile),
            ),
        )
        con.execute(
            "INSERT INTO predictions(customer_id,model_run_id,probability,risk) VALUES(?,?,?,?)",
            (row.customerID, run_id, float(p), risk(p, threshold)),
        )
    con.commit()
    con.close()
    bundle["model_run_id"] = run_id
    # Atomic replacement avoids partially written inference artifacts. Restart after training.
    joblib.dump(bundle, "models/churn.joblib.tmp")
    os.replace("models/churn.joblib.tmp", "models/churn.joblib")
    report = (
        "# Model report\n\nSource: "
        + source["kind"]
        + ". Split: 64% training / 16% validation / 20% untouched test. Seed 42 (validation seed 43).\n\n| Model | 5-fold training CV ROC-AUC | SD | Validation ROC-AUC |\n|---|---:|---:|---:|\n"
    )
    for r in results:
        report += f"| {r['model']} | {r['cv_auc']:.4f} | {r['cv_std'] if r['cv_std'] is not None else 'tuning objective'} | {r['validation_auc']:.4f} |\n"
    report += f"\nSelected: **{winner}**. Untouched test ROC-AUC **{metrics['roc_auc']:.4f}**, PR-AUC **{metrics['pr_auc']:.4f}**, precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, F1 {metrics['f1']:.4f}, Brier {metrics['brier']:.4f}. Threshold **{threshold:.3f}**.\n\n"
    report += "Threshold maximizes assumed validation net value: $300 retained value × 30% intervention success per true positive, minus $10 per contact. These are configurable business assumptions, not measured causal effects. No test data is used for selection, early stopping, preprocessing, or threshold tuning. MLP early stopping uses a training-fold subset; its preprocessing is fit on that outer training fold. Optuna tunes logistic regularization in six bounded five-fold trials. Tuned CV is an optimization score, not an unbiased generalization estimate.\n\nBalanced and unweighted logistic baselines compare imbalance handling. Selected model remains fit on training only to preserve validation threshold consistency. All candidates and their CV results are logged in each run JSON. Global SHAP uses 40 validation observations and a 32-row training background; local SHAP explains the deployed probability. Estimates are approximate, associational, and not causal.\n\nDatabase portfolio scores include training observations and must not be read as out-of-sample evaluation. Gender and senior status are present in the sample; fairness and drift reviews are required before real customer decisions. Support tickets are absent from IBM and excluded. Dataset has no true acquisition dates, so tenure cohorts are used. No resampling or SMOTE was necessary because class weighting was evaluated.\n"
    report += (
        f"\nTraining completed in {time.time() - start:.1f} seconds. Target 0.84: "
        + (
            "met."
            if metrics["roc_auc"] >= 0.84
            else "not met; engineered features and ensemble were evaluated without tuning on test data."
        )
        + "\n"
    )
    Path("MODEL_REPORT.md").write_text(report)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "winner",
                    "roc_auc",
                    "pr_auc",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                ]
            }
        ),
        flush=True,
    )
    return metrics


if __name__ == "__main__":
    run()
