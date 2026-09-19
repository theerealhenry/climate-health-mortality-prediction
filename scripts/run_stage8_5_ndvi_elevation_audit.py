"""Stage 8.5, kill-or-proceed gate — audits the four downloaded-climate columns
that Stage 8's Track D (NDVI) and the elevation/slope direction would build on,
before any feature code is written.

Checks the failure mode that already killed `hot_days_30d` (Stage 3/8: confirmed
constant across all 4,176 rows) — missingness, near-constant/zero-variance, and
basic distribution shape — for `ndvi_30d`, `ndvi_90d`, `elevation`, `slope`.
Also reports per-spatial-cluster row counts for NDVI, since the recommended
direction (cluster-level NDVI aggregation) needs enough rows per cluster to be a
stable statistic rather than noise.

Run from the repo root:  python scripts/run_stage8_5_ndvi_elevation_audit.py
"""

from __future__ import annotations

import pandas as pd

from climate_health.data.loaders import load_train_full
from climate_health.features.pipeline import build_feature_matrix
from climate_health.models.baselines import TARGET_COL

CANDIDATE_COLS = ["ndvi_30d", "ndvi_90d", "elevation", "slope"]


def audit_column(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    n = len(series)
    missing_pct = series.isna().mean() * 100
    nunique = series.nunique(dropna=True)
    # Near-constant: one value covers almost every row (the hot_days_30d failure
    # mode was 100% one value; flag anything at or above 95% as suspect).
    top_value_share = (
        series.value_counts(normalize=True, dropna=True).iloc[0] * 100 if nunique else 100.0
    )
    return {
        "column": col,
        "n_rows": n,
        "missing_pct": round(missing_pct, 2),
        "n_unique": nunique,
        "top_value_share_pct": round(top_value_share, 2),
        "mean": round(series.mean(), 4) if series.notna().any() else float("nan"),
        "std": round(series.std(), 4) if series.notna().any() else float("nan"),
        "min": round(series.min(), 4) if series.notna().any() else float("nan"),
        "max": round(series.max(), 4) if series.notna().any() else float("nan"),
    }


def main() -> None:
    df = load_train_full()
    missing_cols = [c for c in CANDIDATE_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"run_stage8_5_ndvi_elevation_audit: expected columns not found after "
            f"load_train_full(): {missing_cols}. Check the merge with climate_features.csv."
        )

    print("\n=== Stage 8.5 gate — missingness / near-constant / distribution audit ===")
    rows = [audit_column(df, col) for col in CANDIDATE_COLS]
    report = pd.DataFrame(rows).set_index("column")
    print(report.to_string())

    print("\nVerdict per column (>20% missing OR >=95% one value => FAIL, don't build on it):")
    for row in rows:
        verdict = (
            "FAIL" if (row["missing_pct"] > 20 or row["top_value_share_pct"] >= 95) else "PASS"
        )
        print(f"  {row['column']:14s} {verdict}")

    # Cluster-level NDVI stability check: only meaningful if spatial_cluster
    # exists (added by build_feature_matrix's fit step) and NDVI passed above.
    X_full, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    if "spatial_cluster" in X_full.columns and "ndvi_30d" in df.columns:
        merged = X_full[["spatial_cluster"]].join(df[["ndvi_30d", "ndvi_90d"]])
        cluster_counts = merged.groupby("spatial_cluster").size()
        cluster_ndvi_std = merged.groupby("spatial_cluster")["ndvi_30d"].std()
        print("\n=== Cluster-level NDVI stability (rows per cluster, within-cluster std) ===")
        stability = pd.DataFrame({"n_rows": cluster_counts, "ndvi_30d_std": cluster_ndvi_std})
        print(stability.to_string())
        small_clusters = (cluster_counts < 15).sum()
        print(
            f"\nClusters with < 15 rows: {small_clusters}/{len(cluster_counts)} "
            "(cluster-level NDVI aggregation is noisy below this - treat as a"
            " caution, not a hard fail)"
        )
    else:
        print(
            "\nSkipping cluster-level NDVI check: 'spatial_cluster' not found in "
            "build_feature_matrix output, or ndvi_30d missing from the raw frame."
        )


if __name__ == "__main__":
    main()
