"""Stage 11.3 — full model zoo, wired through the two-branch feature pipeline
(Stage 11.2) and the shared estimator contract (Stage 11.1).

Why a separate module from baselines.py: Stage 10.3's three baselines stay as
committed, reproducible anchor numbers (see pipeline.py's ADR — numeric_feature_cols
is deliberately not touched). This module is the actual model-zoo comparison table,
each candidate wired to whichever branch its family actually needs.

evaluate_all_tiers needs the FULL feature matrix (location/zone/deathdate/target)
for its splitters, but a branch function narrows columns down to just what a model
should see. `_branch_frame` reconciles the two: start from the full matrix, overlay
the branch's columns (adding/replacing), and use the branch's column list as
feature_cols — so the splitters keep everything they need while the model only
gets what the branch selected.
"""

from __future__ import annotations

import catboost as cb
import lightgbm as lgb
import mlflow
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from climate_health.data.loaders import load_train_full
from climate_health.evaluation.cv import evaluate_all_tiers
from climate_health.features.pipeline import (
    branch_a_native_categorical,
    branch_b_encoded,
    build_feature_matrix,
    categorical_feature_positions,
)
from climate_health.models.baselines import RANDOM_STATE, TARGET_COL, competition_score
from climate_health.models.interface import assert_valid_estimator_factory
from climate_health.tracking import configure_tracking


def _branch_frame(X_full, branch_fn):
    branch_df = branch_fn(X_full, target_col=TARGET_COL)
    combined = X_full.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values
    return combined, list(branch_df.columns)


# name -> (branch, factory(cat_indices) -> fresh unfitted estimator)
# cat_indices is only used by the branch_a candidate; branch_b factories ignore it.
ZOO_MODELS = {
    "catboost_native": (
        "branch_a",
        lambda cat_idx: cb.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            cat_features=cat_idx,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    ),
    "xgboost_tuned": (
        "branch_b",
        lambda _: xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    ),
    "lightgbm_tuned": (
        "branch_b",
        lambda _: lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    ),
    "histgradientboosting": (
        "branch_b",
        lambda _: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    ),
    "extra_trees": (
        "branch_b",
        lambda _: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE),
        ),
    ),
    "logistic_regression_v2": (
        "branch_b",
        lambda _: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    ),
}


def run_zoo(df=None) -> dict[str, dict]:
    configure_tracking()
    if df is None:
        df = load_train_full()
    X_full, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)

    scorecard = {}
    for name, (branch, make_model) in ZOO_MODELS.items():
        branch_fn = branch_a_native_categorical if branch == "branch_a" else branch_b_encoded
        combined, feature_cols = _branch_frame(X_full, branch_fn)
        cat_idx = categorical_feature_positions(feature_cols) if branch == "branch_a" else None

        def make_estimator(mm=make_model, ci=cat_idx):
            return mm(ci)

        assert_valid_estimator_factory(make_estimator, name)

        result = evaluate_all_tiers(
            make_estimator=make_estimator,
            df=combined,
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
            mlflow.log_param("branch", branch)
            mlflow.log_param("feature_pipeline_version", "stage11.2")
            mlflow.log_param("n_features", len(feature_cols))
            for k, v in row.items():
                mlflow.log_metric(k, v)
    return scorecard


if __name__ == "__main__":
    for name, row in run_zoo().items():
        print(name, row)
