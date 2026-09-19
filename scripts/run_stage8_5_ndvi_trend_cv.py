"""Stage 8.5 Task 3 — does wiring in `ndvi_trend_30_90` (Task 2) actually move
`catboost_tuned`'s CV score, under all three tiers, or is it noise/nothing?

Per this project's standing rule (Stage 5's deliverable, reaffirmed at every stage
since): no feature ships on a Tier-1-only improvement. This script runs
`evaluate_all_tiers` twice on the identical tuned CatBoost config — once on the
feature set as it stood before Task 2 (ndvi_trend_30_90 dropped), once on the
current default output of build_feature_matrix (ndvi_trend_30_90 included) — and
prints both full scorecards side by side plus the delta, so the keep/reject
decision is made on Tier-2 mean AND std, not Tier-1 alone.

Run from the repo root:  python scripts/run_stage8_5_ndvi_trend_cv.py
"""

from __future__ import annotations

import catboost as cb
import yaml

from climate_health.data.loaders import load_train_full
from climate_health.evaluation.cv import evaluate_all_tiers
from climate_health.features.pipeline import build_feature_matrix
from climate_health.models.baselines import TARGET_COL, competition_score

with open("configs/model_best.yaml") as f:
    CATBOOST_TUNED_PARAMS = yaml.safe_load(f)["params"]


def _make_estimator(cat_idx: list[int]):
    return lambda: cb.CatBoostClassifier(
        **CATBOOST_TUNED_PARAMS, cat_features=cat_idx, verbose=False
    )


def main() -> None:
    df = load_train_full()
    combined, state = build_feature_matrix(df, fit=True, target_col=TARGET_COL)

    from climate_health.features.pipeline import (
        branch_a_native_categorical,
        categorical_feature_positions,
    )

    branch_df = branch_a_native_categorical(combined, target_col=TARGET_COL)
    full_combined = combined.copy()
    for col in branch_df.columns:
        full_combined[col] = branch_df[col].values
    feature_cols_with = list(branch_df.columns)
    feature_cols_without = [c for c in feature_cols_with if c != "ndvi_trend_30_90"]
    if len(feature_cols_without) == len(feature_cols_with):
        raise RuntimeError(
            "run_stage8_5_ndvi_trend_cv: 'ndvi_trend_30_90' not found in the Branch A "
            "feature columns — did Task 2's pipeline wiring not take effect? Re-check "
            "build_feature_matrix before trusting this comparison."
        )

    results = {}
    for label, feature_cols in (
        ("without_ndvi_trend", feature_cols_without),
        ("with_ndvi_trend", feature_cols_with),
    ):
        # cat_idx is positional into feature_cols_with; recompute for the trimmed
        # column list so "without" doesn't silently mis-align CatBoost's cat_features.
        this_cat_idx = categorical_feature_positions(feature_cols)
        result = evaluate_all_tiers(
            make_estimator=_make_estimator(this_cat_idx),
            df=full_combined,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            score_fn=competition_score,
            spatial_cluster_col="spatial_cluster",
        )
        results[label] = result.as_summary_row()

    print("\n=== Stage 8.5 Task 3 — ndvi_trend_30_90 with/without comparison ===")
    header = f"{'':20s} {'tier1_mean':>11s} {'tier2_mean':>11s} {'tier2_std':>10s} {'tier3':>8s}"
    print(header)
    for label, row in results.items():
        print(
            f"{label:20s} {row['tier1_mean']:11.4f} {row['tier2_mean']:11.4f} "
            f"{row['tier2_std']:10.4f} {row['tier3_score']:8.4f}"
        )

    delta_t2_mean = (
        results["with_ndvi_trend"]["tier2_mean"] - results["without_ndvi_trend"]["tier2_mean"]
    )
    delta_t2_std = (
        results["with_ndvi_trend"]["tier2_std"] - results["without_ndvi_trend"]["tier2_std"]
    )
    print(f"\nTier-2 mean delta (with - without): {delta_t2_mean:+.4f}")
    print(f"Tier-2 std delta  (with - without): {delta_t2_std:+.4f}")
    print(
        "\nDecision rule: keep only if Tier-2 mean improves AND std doesn't blow up "
        "(same standard Stage 13.4's locked-holdout check used) — read the two deltas "
        "above and record the keep/reject call in docs/experiment_registry.md."
    )


if __name__ == "__main__":
    main()
