"""Stage 10.3 — baseline scorecard: majority class / logistic regression / default
LightGBM, each run through all three CV tiers (evaluation.cv.evaluate_all_tiers) and
logged to MLflow. Low/high anchors for every later Stage 11-14 candidate."""

from __future__ import annotations

import lightgbm as lgb
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from climate_health.data.loaders import load_train_full
from climate_health.evaluation.cv import evaluate_all_tiers
from climate_health.features.pipeline import build_feature_matrix
from climate_health.tracking import configure_tracking

TARGET_COL = "is_climate_sensitive"
RANDOM_STATE = 42


def competition_score(y_true, y_pred_proba) -> float:
    """0.60*F1@0.5 + 0.40*AUC — the fixed competition metric (blueprint Stage 13)."""
    f1 = f1_score(y_true, (y_pred_proba >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_pred_proba)
    return 0.60 * f1 + 0.40 * auc


def numeric_feature_cols(X) -> list[str]:
    """Numeric columns only, target excluded — the dual categorical/encoded branch
    split (Stage 11.2) doesn't exist yet, so baselines get what every candidate can
    consume today without special-casing per model."""
    return [c for c in X.select_dtypes(include="number").columns if c != TARGET_COL]


BASELINE_MODELS = {
    "majority_baseline": lambda: DummyClassifier(strategy="most_frequent"),
    # LogisticRegression can't handle the NaN cells the ratio features leave where a
    # denominator is 0 (documented deferral in features/climate.py), and the raw
    # feature scales (e.g. day_of_year vs. month_sin) blow up lbfgs convergence —
    # impute + scale inside the pipeline so both fit per-fold, not once globally.
    "logistic_regression": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    ),
    "lightgbm_default": lambda: lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
}


def run_baselines(df=None) -> dict[str, dict]:
    """Runs every model in BASELINE_MODELS through evaluate_all_tiers, logs each to
    MLflow, and returns {model_name: as_summary_row()} — the first rows of the Stage
    12 model-comparison table."""
    configure_tracking()
    if df is None:
        df = load_train_full()
    X, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    feature_cols = numeric_feature_cols(X)

    scorecard = {}
    for name, make_estimator in BASELINE_MODELS.items():
        result = evaluate_all_tiers(
            make_estimator=make_estimator,
            df=X,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            score_fn=competition_score,
            spatial_cluster_col="spatial_cluster",
            random_state=RANDOM_STATE,
        )
        row = result.as_summary_row()
        scorecard[name] = row

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("feature_pipeline_version", "stage10.2")
            mlflow.log_param("n_features", len(feature_cols))
            for k, v in row.items():
                mlflow.log_metric(k, v)

    return scorecard


if __name__ == "__main__":
    for name, row in run_baselines().items():
        print(name, row)
