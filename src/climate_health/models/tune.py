"""Stage 13.2 — Optuna study tuning `catboost_native` against the literal competition
score, evaluated inside the Tier-2 geographic CV loop, honoring the locked internal
holdout decided in ADR-001
(docs/decisions/ADR-001-stage13-tuning-scope-and-holdout.md).

Per that ADR: only CatBoost is tuned here — LightGBM stays a Stage 11.3 diversity
candidate until Stage 14.1's prediction-correlation analysis shows it's worth a second
tuning pass. The objective is Tier 2's mean competition score, never Tier 1 (Stage 5's
whole point is that Tier 1 overstates geographic generalization), and one specific
(repeat, fold) combination is always skipped — never used as a per-trial signal — so
it stays available for the one-time Stage 13.4 check.
"""

from __future__ import annotations

from collections.abc import Iterator

import catboost as cb
import mlflow
import optuna
import pandas as pd

from climate_health.data.loaders import load_train_full
from climate_health.evaluation.cv import Tier2Result, summarize_tier2_scores, tier2_splits
from climate_health.features.pipeline import (
    branch_a_native_categorical,
    build_feature_matrix,
    categorical_feature_positions,
)
from climate_health.models.baselines import RANDOM_STATE, TARGET_COL, competition_score
from climate_health.tracking import REPO_ROOT, configure_tracking

# ADR-001: the first fold tier2_splits yields under its default seed — locked before
# any per-fold score for any candidate was inspected, specifically to avoid the choice
# of *which* fold to lock becoming itself a form of data snooping.
LOCKED_HOLDOUT: tuple[int, int] = (0, 0)  # (repeat, fold)

STUDY_STORAGE = f"sqlite:///{REPO_ROOT / 'optuna_studies.db'}"
STUDY_NAME = "catboost_tuned_tier2"
N_TRIALS = 40  # ADR-001: modest budget, fits the 2-day wall-clock limit


def non_locked_tier2_splits(
    df: pd.DataFrame, locked: tuple[int, int] = LOCKED_HOLDOUT, **tier2_kwargs
) -> Iterator[Tier2Result]:
    """Every Tier2Result from `tier2_splits` except `locked` (repeat, fold).

    A thin wrapper rather than a change to `evaluation/cv.py`: `evaluate_all_tiers`
    has no notion of an excluded fold, and Stage 13 is the only caller that needs
    one — adding that concept to the shared orchestrator would be scope creep for a
    single-caller need.
    """
    for split in tier2_splits(df, **tier2_kwargs):
        if (split.repeat, split.fold) == locked:
            continue
        yield split


def build_tuning_frame(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Reproduces Stage 11.2's Branch A (native-categorical) feature frame — the same
    representation `catboost_native` used in the Stage 11.3 zoo comparison, so tuning
    isn't silently evaluated against a different feature set than the one that won."""
    if df is None:
        df = load_train_full()
    X_full, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    branch_df = branch_a_native_categorical(X_full, target_col=TARGET_COL)
    combined = X_full.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values
    feature_cols = list(branch_df.columns)
    cat_idx = categorical_feature_positions(feature_cols)
    return combined, feature_cols, cat_idx


def suggest_catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def make_objective(df: pd.DataFrame, feature_cols: list[str], cat_idx: list[int]):
    """Returns an Optuna objective closed over the already-built feature frame, so
    `build_feature_matrix` and the Branch A transform run once per study, not once per
    trial."""
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    def objective(trial: optuna.Trial) -> float:
        params = suggest_catboost_params(trial)
        scores = [
            competition_score(
                y[split.val_idx],
                cb.CatBoostClassifier(
                    **params, cat_features=cat_idx, random_state=RANDOM_STATE, verbose=False
                )
                .fit(X[split.train_idx], y[split.train_idx])
                .predict_proba(X[split.val_idx])[:, 1],
            )
            for split in non_locked_tier2_splits(
                df, spatial_cluster_col="spatial_cluster", random_state=RANDOM_STATE
            )
        ]
        summary = summarize_tier2_scores(scores)
        trial.set_user_attr("tier2_std", summary["std"])
        trial.set_user_attr("tier2_min", summary["min"])
        trial.set_user_attr("tier2_max", summary["max"])
        with mlflow.start_run(run_name=f"{STUDY_NAME}_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("tier2_mean", summary["mean"])
            mlflow.log_metric("tier2_std", summary["std"])
        return summary["mean"]

    return objective


def run_study(df: pd.DataFrame | None = None, n_trials: int = N_TRIALS) -> optuna.Study:
    configure_tracking()
    combined, feature_cols, cat_idx = build_tuning_frame(df)
    objective = make_objective(combined, feature_cols, cat_idx)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        direction="maximize",
        load_if_exists=True,
    )
    with mlflow.start_run(run_name=STUDY_NAME):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_metric("best_tier2_mean", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    return study


if __name__ == "__main__":
    completed_study = run_study()
    print("best_trial:", completed_study.best_trial.number)
    print("best_value:", completed_study.best_value)
    print("best_params:", completed_study.best_params)
