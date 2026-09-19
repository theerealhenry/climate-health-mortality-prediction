"""Stage 8.5 Task 7 — does the shrinkage-smoothed cluster-level NDVI feature
(`ndvi_30d_cluster_smoothed` / `ndvi_90d_cluster_smoothed`, NDVIClusterFeaturizer in
src/climate_health/features/climate.py) actually move `catboost_tuned`'s CV score,
under all three tiers, or is it noise/nothing?

Same with/without comparison pattern as Task 3's
scripts/run_stage8_5_ndvi_trend_cv.py. Decision rule is identical: keep only if
Tier-2 mean improves AND std doesn't blow up (this project's primary
generalization signal, given the confirmed 0% train/test coordinate overlap).

Run from the repo root:  python scripts/run_stage8_5_ndvi_cluster_cv.py
"""

from __future__ import annotations

import catboost as cb
import yaml

from climate_health.data.loaders import load_train_full
from climate_health.evaluation.cv import evaluate_all_tiers
from climate_health.features.pipeline import (
    branch_a_native_categorical,
    build_feature_matrix,
    categorical_feature_positions,
)
from climate_health.models.baselines import TARGET_COL, competition_score

with open("configs/model_best.yaml") as f:
    CATBOOST_TUNED_PARAMS = yaml.safe_load(f)["params"]


def _make_estimator(cat_idx: list[int]):
    return lambda: cb.CatBoostClassifier(
        **CATBOOST_TUNED_PARAMS, cat_features=cat_idx, verbose=False
    )


def main() -> None:
    df = load_train_full()

    results = {}
    for label, extra_features in (
        ("without_ndvi_cluster", frozenset()),
        ("with_ndvi_cluster", frozenset({"ndvi_cluster_smoothed"})),
    ):
        combined, _ = build_feature_matrix(
            df, fit=True, target_col=TARGET_COL, extra_features=extra_features
        )
        branch_df = branch_a_native_categorical(combined, target_col=TARGET_COL)
        full_combined = combined.copy()
        for col in branch_df.columns:
            full_combined[col] = branch_df[col].values
        feature_cols = list(branch_df.columns)

        if label == "with_ndvi_cluster" and "ndvi_30d_cluster_smoothed" not in feature_cols:
            raise RuntimeError(
                "run_stage8_5_ndvi_cluster_cv: 'ndvi_30d_cluster_smoothed' not found in "
                "the Branch A feature columns — did Task 6's pipeline wiring not take "
                "effect? Re-check build_feature_matrix before trusting this comparison."
            )

        cat_idx = categorical_feature_positions(feature_cols)
        result = evaluate_all_tiers(
            make_estimator=_make_estimator(cat_idx),
            df=full_combined,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            score_fn=competition_score,
            spatial_cluster_col="spatial_cluster",
        )
        results[label] = result.as_summary_row()

    print("\n=== Stage 8.5 Task 7 — ndvi cluster-smoothed with/without comparison ===")
    header = f"{'':20s} {'tier1_mean':>11s} {'tier2_mean':>11s} {'tier2_std':>10s} {'tier3':>8s}"
    print(header)
    for label, row in results.items():
        print(
            f"{label:20s} {row['tier1_mean']:11.4f} {row['tier2_mean']:11.4f} "
            f"{row['tier2_std']:10.4f} {row['tier3_score']:8.4f}"
        )

    delta_t2_mean = (
        results["with_ndvi_cluster"]["tier2_mean"] - results["without_ndvi_cluster"]["tier2_mean"]
    )
    delta_t2_std = (
        results["with_ndvi_cluster"]["tier2_std"] - results["without_ndvi_cluster"]["tier2_std"]
    )
    print(f"\nTier-2 mean delta (with - without): {delta_t2_mean:+.4f}")
    print(f"Tier-2 std delta  (with - without): {delta_t2_std:+.4f}")
    print(
        "\nDecision rule: keep only if Tier-2 mean improves AND std doesn't blow up "
        "— read the two deltas above and record the keep/reject call in "
        "docs/experiment_registry.md as F-006."
    )


if __name__ == "__main__":
    main()
