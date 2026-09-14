"""Stage 15.2 — compare raw / Platt / isotonic calibration of the champion
(`catboost_tuned`, standalone per Stage 14.3) against Stage 15.1's finding of
real miscalibration near p=0.5.

Reuses the same OOF predictions and fold assignment Stage 15.1 already loaded
(notebooks/artifacts/stage14_1_oof_predictions.npz) — no retraining needed,
since compare_calibration_methods only ever fits a calibrator on top of
already-computed OOF probabilities.

Run from the repo root:  python scripts/run_stage15_2_calibration_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from climate_health.models.calibrate import compare_calibration_methods

OOF_PATH = Path("notebooks/artifacts/stage14_1_oof_predictions.npz")


def main() -> None:
    data = np.load(OOF_PATH)
    y_true = data["y_true"]
    fold_id = data["fold_id"]
    raw = data["catboost_tuned"]

    results = compare_calibration_methods(y_true, raw, fold_id)

    print("\n=== Stage 15.2 — calibration comparison (catboost_tuned, OOF) ===")
    header = f"{'method':10s} {'f1@0.5':>8s} {'auc':>8s} {'comp_score':>11s} {'brier':>8s}"
    print(header)
    for name, row in results.items():
        f1, auc = row["f1_at_0.5"], row["auc"]
        comp, brier = row["competition_score"], row["brier"]
        print(f"{name:10s} {f1:8.4f} {auc:8.4f} {comp:11.4f} {brier:8.4f}")

    best = max(results, key=lambda k: results[k]["competition_score"])
    print(f"\nBest by competition score: {best} ({results[best]['competition_score']:.4f})")


if __name__ == "__main__":
    main()
