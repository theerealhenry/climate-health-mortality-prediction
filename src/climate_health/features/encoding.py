"""
Stage 9 — Out-of-fold, Bayesian-smoothed target encoding.

Encodes `spatial_cluster` and `zone` (categorical, moderate cardinality). Deliberately
does NOT encode raw `location`: at 39-43 distinct values against ~3,100 training rows,
many locations have only 3-5 records, so a per-location encoding would mostly memorize
those few labels rather than learn signal (see docs/PROJECT_BLUEPRINT.md Stage 9). A
`region` column was specified in the original stage brief but does not exist in this
dataset (see data/loaders.py's column list) — there is no coarser geographic grouping
between `location` and `spatial_cluster`/`zone` to encode, so it's omitted rather than
invented.

The one rule this module exists to enforce: an encoding used as a *training* feature
must never be computed using that same row's own target value, directly or through any
row sharing its fold. `fit_target_encoding_map`/`apply_target_encoding` are the
production fit-once/reuse-everywhere pair (fit on train, applied verbatim to
test/production, like Stage 6/7/8's other fitted transforms). `compute_oof_target_encoding`
is the *training-set-only* variant: for each row, the encoding is built from every fold
except the one that row is in, using exactly the fold boundaries `cv.py`'s
`tier1_splits`/`tier2_splits` already produce (twin-record-safe) rather than an
independent K-fold that could split a twin-record pair across the encoding computation
and its target — see `test_encoding.py`'s leakage tests for the check that actually
proves this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from climate_health.utils.validation import validate_no_nulls, validate_required_columns


def _smoothed_stats(
    train_target: pd.Series, cat: pd.Series, global_mean: float, smoothing: float
) -> pd.Series:
    """Bayesian/count-smoothed per-category mean: pulls small-count categories toward
    `global_mean`, large-count categories toward their own empirical mean. `smoothing`
    is the "pseudo-count" of prior observations at `global_mean` each category starts
    with — e.g. smoothing=10 means a category with only 2 real rows is still mostly
    the global mean, while a category with 500 rows is barely pulled at all."""
    stats = (
        pd.DataFrame({"cat": cat.values, "y": train_target.values})
        .groupby("cat")["y"]
        .agg(["mean", "count"])
    )
    return (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)


def fit_target_encoding_map(
    train_df: pd.DataFrame, target_col: str, cat_col: str, smoothing: float = 10.0
) -> tuple[dict, float]:
    """Fits a smoothed category->encoding map on `train_df` only. Returns
    `(mapping, global_mean)` — reuse both verbatim on test/production data via
    `apply_target_encoding`, the same fit-once/reuse-everywhere pattern as
    `compute_heat_threshold`/`compute_reference_year` elsewhere in this project."""
    validate_required_columns(train_df, [target_col, cat_col], "fit_target_encoding_map")
    validate_no_nulls(train_df, [target_col, cat_col], "fit_target_encoding_map")
    if smoothing < 0:
        raise ValueError(f"fit_target_encoding_map: smoothing must be >= 0, got {smoothing}")

    global_mean = float(train_df[target_col].mean())
    smoothed = _smoothed_stats(train_df[target_col], train_df[cat_col], global_mean, smoothing)
    return smoothed.to_dict(), global_mean


def apply_target_encoding(
    df: pd.DataFrame, cat_col: str, mapping: dict, global_mean: float
) -> pd.Series:
    """Maps `df[cat_col]` through a fitted `mapping` (from `fit_target_encoding_map`).
    A category unseen at fit time (e.g. a test-set-only location) falls back to
    `global_mean` rather than raising or becoming NaN."""
    validate_required_columns(df, [cat_col], "apply_target_encoding")
    validate_no_nulls(df, [cat_col], "apply_target_encoding")
    return df[cat_col].map(mapping).fillna(global_mean).rename(f"{cat_col}_te")


def compute_oof_target_encoding(
    df: pd.DataFrame,
    target_col: str,
    cat_col: str,
    splits,
    smoothing: float = 10.0,
) -> pd.Series:
    """Out-of-fold smoothed target encoding, safe to use as a *training* feature.

    `splits` must be an iterable of `(train_idx, val_idx)` positional-index pairs that
    together partition every row of `df` exactly once as a val_idx — e.g.
    `cv.tier1_splits(df, target_col)` directly, or
    `((r.train_idx, r.val_idx) for r in cv.tier2_splits(df, ...) if <one repeat only>)`.
    Passing a raw, non-twin-aware K-fold defeats the purpose of this function: use a
    splitter from `climate_health.evaluation.cv`, which already guards against
    twin-record leakage (see module docstring).

    For each fold, the encoding applied to that fold's rows is computed using only the
    *other* folds' rows — the row's own target value, and every other row in its fold,
    never contributes to its own encoding.

    Raises
    ------
    KeyError
        If `target_col` or `cat_col` is missing.
    ValueError
        If either column contains nulls, `smoothing` is negative, or `splits` does not
        partition every row exactly once (a row appearing in zero or more than one
        val_idx signals a bug in whatever produced `splits`, not something this
        function should silently tolerate).
    """
    validate_required_columns(df, [target_col, cat_col], "compute_oof_target_encoding")
    validate_no_nulls(df, [target_col, cat_col], "compute_oof_target_encoding")
    if smoothing < 0:
        raise ValueError(f"compute_oof_target_encoding: smoothing must be >= 0, got {smoothing}")

    n = len(df)
    oof = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    global_mean = float(df[target_col].mean())

    for train_idx, val_idx in splits:
        if covered[val_idx].any():
            raise ValueError(
                "compute_oof_target_encoding: a row appears in more than one val_idx — "
                "`splits` must partition the data, not overlap (e.g. don't pass "
                "multiple Tier 2 repeats together)."
            )
        if set(np.asarray(train_idx).tolist()) & set(np.asarray(val_idx).tolist()):
            raise ValueError(
                "compute_oof_target_encoding: train_idx and val_idx overlap within a "
                "fold — this fold's own rows would leak into their own encoding. "
                "`splits` must come from a real train/val splitter (e.g. "
                "cv.tier1_splits/tier2_splits), not a hand-built or buggy pair."
            )
        train_target = df[target_col].iloc[train_idx]
        train_cat = df[cat_col].iloc[train_idx]
        smoothed = _smoothed_stats(train_target, train_cat, global_mean, smoothing)
        oof[val_idx] = df[cat_col].iloc[val_idx].map(smoothed).fillna(global_mean).values
        covered[val_idx] = True

    if not covered.all():
        raise ValueError(
            f"compute_oof_target_encoding: {(~covered).sum()} row(s) were never covered "
            "by any val_idx — `splits` must partition every row exactly once."
        )
    return pd.Series(oof, index=df.index, name=f"{cat_col}_te_oof")
