# Model report

Source: IBM Telco sample. Split: 64% training / 16% validation / 20% untouched test. Seed 42 (validation seed 43).

| Model | 5-fold training CV ROC-AUC | SD | Validation ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8496 | 0.014705698440826915 | 0.8406 |
| Logistic balanced | 0.8492 | 0.014459166683600849 | 0.8404 |
| Random Forest | 0.8463 | 0.016009647125739473 | 0.8446 |
| XGBoost | 0.8479 | 0.016160061238730525 | 0.8452 |
| LightGBM | 0.8424 | 0.015720887020394 | 0.8439 |
| PyTorch MLP | 0.8449 | 0.015282254352924157 | 0.8361 |
| Tuned Logistic | 0.8497 | tuning objective | 0.8403 |
| Soft Voting | 0.8496 | 0.01449984928766218 | 0.8406 |

Selected: **XGBoost**. Untouched test ROC-AUC **0.8454**, PR-AUC **0.6591**, precision 0.3907, recall 0.9465, F1 0.5531, Brier 0.1353. Threshold **0.085**.

Threshold maximizes assumed validation net value: $300 retained value × 30% intervention success per true positive, minus $10 per contact. These are configurable business assumptions, not measured causal effects. No test data is used for selection, early stopping, preprocessing, or threshold tuning. MLP early stopping uses a training-fold subset; its preprocessing is fit on that outer training fold. Optuna tunes logistic regularization in six bounded five-fold trials. Tuned CV is an optimization score, not an unbiased generalization estimate.

Balanced and unweighted logistic baselines compare imbalance handling. Selected model remains fit on training only to preserve validation threshold consistency. All candidates and their CV results are logged in each run JSON. Global SHAP uses 40 validation observations and a 32-row training background; local SHAP explains the deployed probability. Estimates are approximate, associational, and not causal.

Database portfolio scores include training observations and must not be read as out-of-sample evaluation. Gender and senior status are present in the sample; fairness and drift reviews are required before real customer decisions. Support tickets are absent from IBM and excluded. Dataset has no true acquisition dates, so tenure cohorts are used. No resampling or SMOTE was necessary because class weighting was evaluated.

Training completed in 21.8 seconds. Target 0.84: met.
