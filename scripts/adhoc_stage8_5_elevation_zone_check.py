"""Stage 8.5 Task 8 — cheap, throwaway check: does `elevation * (zone == "Peri_urban")`
move catboost_tuned's CV score at all?

Deliberately NOT a real transformer with tests (see the amended scope doc — this
task is "low priority, cheap check only"). The interaction column is added
directly to the already-built feature matrix, inline, in this script — nothing
here touches src/climate_health. If the result surprises us (helps), *then* it
gets promoted to a proper FeatureFitState-aware transformer with tests, same as
Tasks 4-6, before it goes anywhere near build_feature_matrix.

Run from the repo root:  python scripts/adhoc_stage8_5_elevation_zone_check.py
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
    combined, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    branch_df = branch_a_native_categorical(combined, target_col=TARGET_COL)

    full_combined = combined.copy()
    for col in branch_df.columns:
        full_combined[col] = branch_df[col].values
    full_combined["elevation_x_periurban"] = full_combined["elevation"] * (
        full_combined["zone"] == "Peri_urban"
    ).astype(float)

    feature_cols_without = list(branch_df.columns)
    feature_cols_with = feature_cols_without + ["elevation_x_periurban"]

    results = {}
    for label, feature_cols in (
        ("without_elevation_x_zone", feature_cols_without),
        ("with_elevation_x_zone", feature_cols_with),
    ):
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

    print("\n=== Stage 8.5 Task 8 — elevation x zone interaction with/without comparison ===")
    header = f"{'':25s} {'tier1_mean':>11s} {'tier2_mean':>11s} {'tier2_std':>10s} {'tier3':>8s}"
    print(header)
    for label, row in results.items():
        print(
            f"{label:25s} {row['tier1_mean']:11.4f} {row['tier2_mean']:11.4f} "
            f"{row['tier2_std']:10.4f} {row['tier3_score']:8.4f}"
        )

    delta_t2_mean = (
        results["with_elevation_x_zone"]["tier2_mean"]
        - results["without_elevation_x_zone"]["tier2_mean"]
    )
    delta_t2_std = (
        results["with_elevation_x_zone"]["tier2_std"]
        - results["without_elevation_x_zone"]["tier2_std"]
    )
    print(f"\nTier-2 mean delta (with - without): {delta_t2_mean:+.4f}")
    print(f"Tier-2 std delta  (with - without): {delta_t2_std:+.4f}")
    print(
        "\nDecision rule: keep only if Tier-2 mean improves AND std doesn't blow up. "
        "Expected: 'not worth building as a real feature' (elevation likely near-"
        "redundant with spatial_cluster, which was fit jointly with elevation in "
        "Stage 7). If the deltas surprise you and favor 'with', don't ship this "
        "script's column directly — promote it to a proper transformer with tests "
        "first, per the amended scope doc."
    )


if __name__ == "__main__":
    main()
