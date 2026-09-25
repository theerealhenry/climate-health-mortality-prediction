"""Stage 17, check 3 of 8 — re-run adversarial validation on the champion's own
predicted-probability output (not just raw input features).

Stage 4's original adversarial validation (notebooks/01_eda.ipynb) found a perfect
1.0000 AUC separating Train from Test rows, carried almost entirely by raw
latitude/elevation/longitude/slope acting as a location fingerprint -- not a leak,
but a known, already-documented train/test distributional gap. That check never
included the model's own output as a feature, because the model didn't exist yet.

This script asks the follow-up question the blueprint's Stage 17 gate names
explicitly: does catboost_tuned + Platt calibration's predicted probability itself
carry train/test-distinguishing signal *beyond* what's already known and accepted?
If the champion's calibrated probability on Test.csv looks distributionally
different from its calibrated OOF probability on Train.csv in a way that isn't just
"test has fewer high-target-rate rural rows" (already expected from Section 1's
13.5pp zone-composition shift), that would suggest the model is exploiting a
train-specific artifact rather than a genuine, transferable pattern.

Two independent methods, mirroring Stage 4's own triangulation style:
  1. A two-sample Kolmogorov-Smirnov test on the probability distributions alone.
  2. A single-feature adversarial classifier (probability as the only input) --
     if this single feature's AUC is anywhere near the 1.0000 the raw coordinates
     already produce, that's the red flag; if it's close to 0.5, the champion's
     output isn't adding a *new* train/test tell beyond what Stage 4 already found
     and Stage 3/16 already accepted as the dataset's known geographic split.

Run from the repo root:  python scripts/run_stage17_adversarial_residual_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from climate_health.models.calibrate import PlattCalibrator

OOF_PATH = "notebooks/artifacts/stage14_1_oof_predictions.npz"
SUBMISSION_PATH = "submission.csv"
RANDOM_STATE = 42


def main() -> None:
    oof = np.load(OOF_PATH)
    raw_train_oof = oof["catboost_tuned"]
    y_true = oof["y_true"]

    # Same production calibration as predict.py: Platt fit once on the full OOF set.
    calibrator = PlattCalibrator().fit(raw_train_oof, y_true)
    calibrated_train_oof = calibrator.predict(raw_train_oof)

    test_probs = pd.read_csv(SUBMISSION_PATH)["TargetRAUC"].to_numpy()

    print("=== Stage 17, check 3 of 8 — adversarial validation on champion's own output ===")
    print(
        f"Train OOF (calibrated) n={len(calibrated_train_oof)}, Test predicted n={len(test_probs)}"
    )
    print(
        f"Train OOF mean={calibrated_train_oof.mean():.4f} std={calibrated_train_oof.std():.4f} | "
        f"Test mean={test_probs.mean():.4f} std={test_probs.std():.4f}"
    )

    ks_stat, ks_p = ks_2samp(calibrated_train_oof, test_probs)
    print(
        f"\nKS test (train OOF prob vs. test predicted prob): statistic={ks_stat:.4f}, p={ks_p:.4g}"
    )

    X = np.concatenate([calibrated_train_oof, test_probs]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(calibrated_train_oof)), np.ones(len(test_probs))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(LogisticRegression(), X, y, cv=cv, scoring="roc_auc")
    print(
        f"Single-feature adversarial AUC (predicted probability only): "
        f"{auc_scores.mean():.4f} +/- {auc_scores.std():.4f}"
    )

    print(
        "\nDecision rule: AUC near 1.0 (matching Stage 4's raw-coordinate result) would mean "
        "the champion's own output is itself a near-perfect train/test fingerprint -- a red "
        "flag distinct from the already-accepted geographic split. AUC near 0.5, or only "
        "modestly above it and consistent with the already-known 13.5pp zone-composition "
        "shift, means the champion's predictions don't add a *new* train/test tell."
    )


if __name__ == "__main__":
    main()
