"""Stage 15.3 — the single scripted entry point from trained artifacts to a
validated `submission.csv`. Never a manual notebook step: every later
interpretability/model-card artifact (Phase 8) needs to be pinned to the model
that was actually scored, which only holds if that model came from a
reproducible script, not a slightly-different notebook re-run.

Final chosen pipeline, per docs/experiment_registry.md (Stage 13.4, 14.3, 15.2):
  - catboost_tuned (Optuna trial #26, configs/model_best.yaml), retrained on the
    FULL training data — not a CV fold — since this is the deployed model, not
    a validation exercise.
  - No ensembling: Stage 14.3's repeated-CV check found the weighted blend's
    apparent gain was noise (paired diff -0.0011 across 24 folds).
  - Platt calibration, fit ONCE on catboost_tuned's Stage 14.1 OOF predictions
    (the full OOF set, not per-fold as in Stage 15.2's cross-validated
    comparison — Stage 15.2 needed leakage-free evaluation to pick a method;
    the winning method's single production calibrator is fit on everything OOF
    has, since OOF predictions were already never seen by the model that
    produced them).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import yaml

from climate_health.data.loaders import load_sample_submission, load_test_full, load_train_full
from climate_health.features.pipeline import (
    branch_a_native_categorical,
    build_feature_matrix,
    categorical_feature_positions,
)
from climate_health.models.baselines import TARGET_COL
from climate_health.models.calibrate import PlattCalibrator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_BEST_PATH = REPO_ROOT / "configs" / "model_best.yaml"
OOF_PATH = REPO_ROOT / "notebooks" / "artifacts" / "stage14_1_oof_predictions.npz"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"


def build_submission_frame(
    ids: list[str], calibrated_probs: list[float] | np.ndarray
) -> pd.DataFrame:
    """`ID` + the two competition targets: `TargetF1` (the fixed 0.5-threshold
    call) and `TargetRAUC` (the probability itself — AUC is rank-based, so
    raw vs. calibrated makes no difference to it, but the calibrated value is
    the better estimate to actually report)."""
    calibrated_probs = np.asarray(calibrated_probs)
    return pd.DataFrame(
        {
            "ID": ids,
            "TargetF1": (calibrated_probs >= 0.5).astype(int),
            "TargetRAUC": calibrated_probs,
        }
    )


def validate_against_sample(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Schema-exact check against SampleSubmission.csv before anything is written
    to disk: same columns, same row count, same ID set (order-independent — Zindi
    joins on ID, not row position), TargetF1 in {0, 1}, TargetRAUC in [0, 1]."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError(
            f"column mismatch: submission has {list(submission.columns)}, "
            f"expected {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        raise ValueError(
            f"row count mismatch: submission has {len(submission)} rows, expected {len(sample)}"
        )
    if set(submission["ID"]) != set(sample["ID"]):
        raise ValueError(
            "ID set mismatch: submission's IDs don't match SampleSubmission.csv's IDs exactly"
        )
    if not submission["TargetF1"].isin([0, 1]).all():
        raise ValueError("TargetF1 contains values outside {0, 1}")
    if not submission["TargetRAUC"].between(0, 1).all():
        raise ValueError("TargetRAUC contains values outside [0, 1]")


def retrain_champion_on_full_data() -> tuple[cb.CatBoostClassifier, dict, list[str], list[int]]:
    """Retrains catboost_tuned's exact Optuna-found config on ALL training rows —
    the tuning/CV folds already did their job (selecting this config); the
    deployed model uses every training row available, per Stage 15's blueprint
    instruction to retrain on full data before generating the submission."""
    with open(MODEL_BEST_PATH) as f:
        params = yaml.safe_load(f)["params"]

    df = load_train_full()
    X_full, fitted_state = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    branch_df = branch_a_native_categorical(X_full, target_col=TARGET_COL)
    combined = X_full.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values
    feature_cols = list(branch_df.columns)
    cat_idx = categorical_feature_positions(feature_cols)

    model = cb.CatBoostClassifier(**params, cat_features=cat_idx, verbose=False)
    model.fit(combined[feature_cols].values, combined[TARGET_COL].values)
    return model, fitted_state, feature_cols, cat_idx


def fit_production_calibrator() -> PlattCalibrator:
    """Platt scaling fit once on the full Stage 14.1 OOF prediction set — the
    winning method from Stage 15.2's cross-validated comparison, now fit on
    everything OOF has for the actual deployed calibrator (Stage 15.2's
    fold-wise fitting was only needed to evaluate the method without leakage;
    the production calibrator has no such constraint since OOF predictions were
    never seen by the model that produced them)."""
    data = np.load(OOF_PATH)
    return PlattCalibrator().fit(data["catboost_tuned"], data["y_true"])


def predict_test_set(
    model: cb.CatBoostClassifier,
    calibrator: PlattCalibrator,
    fitted_state,
    feature_cols: list[str],
) -> pd.DataFrame:
    test_df = load_test_full()
    X_test_full, _ = build_feature_matrix(
        test_df, fit=False, fitted_state=fitted_state, target_col=TARGET_COL
    )
    branch_df = branch_a_native_categorical(X_test_full, target_col=TARGET_COL)
    combined = X_test_full.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values

    # Guards categorical_feature_positions' assumption that column order at
    # inference matches training exactly — silent misalignment here would feed
    # CatBoost the wrong columns as "categorical" with no error, just bad scores.
    if list(combined[feature_cols].columns) != feature_cols:
        raise ValueError(
            "predict_test_set: test feature columns don't match training feature_cols order"
        )

    raw_probs = model.predict_proba(combined[feature_cols].values)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)
    return build_submission_frame(list(combined["ID"]), calibrated_probs)


def generate_submission() -> Path:
    """The single command Stage 15.3 asks for: retrain, calibrate, predict,
    validate, archive under submissions/ named by the current commit hash."""
    model, fitted_state, feature_cols, _ = retrain_champion_on_full_data()
    calibrator = fit_production_calibrator()
    submission = predict_test_set(model, calibrator, fitted_state, feature_cols)

    sample = load_sample_submission()
    validate_against_sample(submission, sample)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "generate_submission: could not read the git commit hash. This script's whole "
            "purpose is a reproducible, tag-able submission — it needs a git repo to name the "
            "output file after. Run it from inside the project's git checkout."
        ) from exc
    if dirty:
        commit_hash += "-dirty"
        print(
            "WARNING: working tree has uncommitted changes — the submission filename is "
            f"tagged '{commit_hash}' rather than a clean commit hash, since it would "
            "otherwise misrepresent uncommitted code as that commit's output. Commit your "
            "changes before the real submission run."
        )
    out_path = SUBMISSIONS_DIR / f"submission_{commit_hash}.csv"
    submission.to_csv(out_path, index=False)
    # Zindi's upload expects the file at this exact repo-root path.
    submission.to_csv(REPO_ROOT / "submission.csv", index=False)
    n, h = len(submission), commit_hash
    print(f"Wrote {out_path} and submission.csv ({n} rows, commit {h})")
    return out_path


if __name__ == "__main__":
    generate_submission()
