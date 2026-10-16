import pandas as pd

from src.ml.explain import ExplanationService


def test_artifact_consistency(app, profile):
    p = app.extensions["predictor"]
    first = p.predict(profile, explain=False)
    second = p.predict(profile, explain=False)
    assert first["probability"] == second["probability"]
    m = p.bundle["metrics"]
    assert (
        0.5 < m["roc_auc"] <= 1
        and sum(map(sum, m["confusion_matrix"])) == m["split"]["test"]
    )
    assert len(m["comparison"]) == 8


def test_shap_local_accuracy(app, profile):
    p = app.extensions["predictor"]
    d, _ = p.validate(profile)
    frame = pd.DataFrame([d], columns=p.bundle["columns"])
    e = ExplanationService(p.model, p.bundle["background"], p.bundle["columns"])
    values = e.values(frame)[0]
    baseline = p.model.predict_proba(p.bundle["background"])[:, 1].mean()
    assert abs(baseline + values.sum() - p.model.predict_proba(frame)[0, 1]) < 1e-5
