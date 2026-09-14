"""Stage 14.2 — run all three ensembling strategies over Stage 14.1's OOF
predictions and compare them against the best single model.

Run from the repo root:  python scripts/run_stage14_2_ensembling.py

Expects `notebooks/04_modeling_experiments.ipynb`'s Stage 14.1 cells to have
already saved the OOF predictions and shared fold assignments — if you saved
them differently, adjust OOF_PATH / the load below to match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from climate_health.models.baselines import competition_score
from climate_health.models.ensemble import (
    oof_stack,
    optimize_weights,
    rank_average,
    weighted_average,
)

OOF_PATH = Path("notebooks/artifacts/stage14_1_oof_predictions.npz")


def main() -> None:
    data = np.load(OOF_PATH)
    y_true = data["y_true"]
    fold_id = data["fold_id"]  # which of the 5 shared folds (random_state=99) each row is in
    candidate_names = (
        "catboost_tuned",
        "extra_trees",
        "histgradientboosting",
        "logistic_regression_v2",
    )
    oof_preds = {name: data[name] for name in candidate_names}

    splits = [
        (np.where(fold_id != f)[0], np.where(fold_id == f)[0])
        for f in sorted(set(fold_id.tolist()))
    ]

    results = {name: competition_score(y_true, preds) for name, preds in oof_preds.items()}

    weights = optimize_weights(oof_preds, y_true, competition_score)
    results["weighted_average"] = competition_score(y_true, weighted_average(oof_preds, weights))

    results["rank_average"] = competition_score(y_true, rank_average(oof_preds))

    stacked = oof_stack(oof_preds, y_true, splits)
    results["oof_stack"] = competition_score(y_true, stacked)

    print("\n=== Stage 14.2 comparison (OOF competition score) ===")
    for name, score in sorted(results.items(), key=lambda kv: -kv[1]):
        marker = ""
        if name == "weighted_average":
            rounded = {k: round(v, 4) for k, v in weights.items()}
            marker = f"  <- optimized weights: {json.dumps(rounded)}"
        print(f"{name:28s} {score:.4f}{marker}")

    best = max(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.4f})")


if __name__ == "__main__":
    main()
