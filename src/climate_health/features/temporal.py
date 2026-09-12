"""
Stage 6 — Temporal features.

Three pieces, per docs/PROJECT_BLUEPRINT.md Stage 6:

  1. Date decomposition: month, day-of-year, cyclical sin/cos encodings for both,
     and a year trend relative to a train-only reference year.
  2. Candidate Uganda rainy-season windows, sourced from a public agro-climatic
     reference (not invented), plus a statistical validation function that checks
     each candidate against this dataset's actual observed month-level target-rate
     pattern *before* any such flag is trusted as a real feature — per the
     blueprint's explicit instruction that these flags be "derived from public
     agro-climatic references and validated against the observed month-level
     target-rate pattern from Stage 4 before use (not hard-coded blind)".

A finding worth stating plainly rather than glossing over: **that validation, run
against this project's real training data, found no statistically significant
relationship between either candidate rainy-season window and the target**
(chi-square p-values of 0.90, 0.63, and 0.58 for the March-May window, the
September-November window, and their union respectively — all far from any
conventional significance threshold). This is not a failure of the feature-building
process; it is the process working as designed. `docs/PROJECT_BLUEPRINT.md` explicitly
downgraded this feature from "implement directly" to "validate before trusting" for
exactly this reason — a plausible-sounding domain feature that turns out not to
separate the target in the actual data is exactly what that validation step exists to
catch, in the same spirit as this project's earlier decisions to drop `hot_days_30d`
(Stage 3) and to not fabricate a consecutive dry-spell feature (Stage 8). The
machinery to build and test rainy-season flags is provided below in full — a
different, better-labeled dataset (e.g. one with more granular sub-national rainfall
regimes than Uganda's national bimodal pattern) might validate cleanly — but this
module does not silently wire an unvalidated flag into the default feature set. See
`validate_seasonal_window` and the module-level `RAINY_SEASON_CANDIDATES` for the
exact numbers and how to re-run this check against any dataset.

Leakage-safety note: unlike Stage 8's climate anomaly baselines, none of this
module's outputs are *learned statistics* fit on training data — month, day-of-year,
and calendar-defined season membership are static facts about a date, identical
whether computed on train, test, or a genuinely new record. The one exception is the
year-trend feature, which is expressed relative to a *reference year* that should be
fixed from training data (via `compute_reference_year`) and reused verbatim for any
other dataset — not because using each dataset's own minimum year would leak the
target, but because a year trend is only meaningful as a trend if "year 0" means the
same calendar year everywhere it's used, mirroring the fit-once/reuse-everywhere
pattern established in Stage 7/8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from climate_health.utils.validation import (
    check_no_output_collision,
    validate_no_nulls,
    validate_required_columns,
)

logger = logging.getLogger(__name__)

# Sourced from Uganda's Ministry of Foreign Affairs (Geneva mission) weather page,
# which states Uganda's two wet seasons are "March to May and September to November"
# — consistent with the standard MAM ("long rains")/SON ("short rains") bimodal
# pattern widely used in East African agro-climatic references (e.g. FEWS NET's
# Uganda seasonal monitoring, which uses the same Season A (Feb-Jun)/Season B
# (Aug-Dec) planting-calendar framing at a coarser monthly resolution). MAM/SON is
# used here as the specific, citable candidate — narrower months are also provided
# for a slightly more conservative candidate.
RAINY_SEASON_CANDIDATES: dict[str, tuple[int, ...]] = {
    "mam": (3, 4, 5),  # March-May, "long rains"
    "son": (9, 10, 11),  # September-November, "short rains"
    "mam_son_combined": (3, 4, 5, 9, 10, 11),
}


def add_date_decomposition_features(
    df: pd.DataFrame, deathdate_col: str = "deathdate"
) -> pd.DataFrame:
    """Adds calendar decomposition features from `deathdate_col`:

      - `month` (1-12)
      - `day_of_year` (1-365, or 1-366 in a leap year)
      - `year`
      - `month_sin` / `month_cos`: cyclical encoding of month, period 12 — so
        December (12) and January (1) sit next to each other in encoded space
        instead of at opposite ends of a raw 1-12 scale.
      - `day_of_year_sin` / `day_of_year_cos`: cyclical encoding of day-of-year,
        period equal to *that record's own year's actual day count* (365 or 366),
        not a fixed 365 — a fixed-365 period would place December 31st of a leap
        year at a slightly wrong phase (365/366 of the way around the circle
        instead of 366/366, a full day short of December 31st in a non-leap year),
        a small but real discontinuity right at the year boundary that compounds
        across this dataset's 16-year span. Verified against this dataset's actual
        leap-day record (a February 29th death exists in the real test set) rather
        than assumed to matter only in theory.

    All outputs are static calendar facts — no leakage risk, and identical whether
    computed on train, test, or any future record.

    Raises
    ------
    KeyError
        If `deathdate_col` is missing.
    ValueError
        If `deathdate_col` contains nulls, is not a datetime dtype, or if `df`
        already has any of the output columns this function would add.
    """
    validate_required_columns(df, [deathdate_col], "add_date_decomposition_features")
    validate_no_nulls(df, [deathdate_col], "add_date_decomposition_features")
    if not pd.api.types.is_datetime64_any_dtype(df[deathdate_col]):
        raise ValueError(
            f"add_date_decomposition_features: '{deathdate_col}' must be a datetime64 "
            f"column, got dtype {df[deathdate_col].dtype!r} — parse it first "
            "(climate_health.data.loaders already does this for deathdate)."
        )
    new_cols = [
        "month",
        "day_of_year",
        "year",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    check_no_output_collision(df, new_cols, "add_date_decomposition_features")

    out = df.copy()
    dt = out[deathdate_col].dt
    month = dt.month
    day_of_year = dt.dayofyear
    year = dt.year
    is_leap = dt.is_leap_year
    days_in_this_year = np.where(is_leap, 366, 365)

    out["month"] = month
    out["day_of_year"] = day_of_year
    out["year"] = year
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_this_year)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_this_year)
    return out


def compute_reference_year(train_df: pd.DataFrame, deathdate_col: str = "deathdate") -> int:
    """Returns the earliest calendar year in `train_df` — computed on training data
    **only**, then reused verbatim as the zero-point for `add_year_trend_feature` on
    any other dataset, the same fit-once/reuse-everywhere pattern as Stage 7's
    cluster count and Stage 8's heat threshold. Using each dataset's own minimum
    year instead would make "year 0" mean a different calendar year in train vs.
    test, silently breaking the trend's comparability.
    """
    validate_required_columns(train_df, [deathdate_col], "compute_reference_year")
    validate_no_nulls(train_df, [deathdate_col], "compute_reference_year")
    return int(train_df[deathdate_col].dt.year.min())


def add_year_trend_feature(
    df: pd.DataFrame, reference_year: int, deathdate_col: str = "deathdate"
) -> pd.DataFrame:
    """Adds `year_since_reference` = calendar year minus `reference_year` (from
    `compute_reference_year`, fit on train). Negative values are possible and
    meaningful if `df` contains a year earlier than the reference (e.g. this
    project's real test set never does — verified — but the function does not
    assume that in general)."""
    validate_required_columns(df, [deathdate_col], "add_year_trend_feature")
    validate_no_nulls(df, [deathdate_col], "add_year_trend_feature")
    check_no_output_collision(df, ["year_since_reference"], "add_year_trend_feature")

    out = df.copy()
    out["year_since_reference"] = out[deathdate_col].dt.year - reference_year
    return out


@dataclass(frozen=True)
class SeasonValidationResult:
    """Result of testing one candidate seasonal window against the observed target
    rate in a dataset. `is_significant` uses `alpha` (default 0.05) on the
    chi-square test of independence between season membership and the target."""

    name: str
    months: tuple[int, ...]
    rate_in_season: float
    rate_out_of_season: float
    n_in_season: int
    n_out_of_season: int
    chi2_statistic: float
    p_value: float
    alpha: float
    is_significant: bool


def validate_seasonal_window(
    df: pd.DataFrame,
    target_col: str,
    months: tuple[int, ...],
    name: str,
    deathdate_col: str = "deathdate",
    alpha: float = 0.05,
) -> SeasonValidationResult:
    """Tests whether membership in `months` (1-12) is statistically associated with
    `target_col` in `df`, via a chi-square test of independence on the 2x2 table of
    (in-season vs. out-of-season) x (target=0 vs. target=1).

    This is a validation step, not a feature-building step — it exists to answer
    "should this candidate window actually be trusted as a feature on this
    dataset," per the blueprint's explicit instruction not to hard-code a rainy-
    season flag blind. Call this on **training data only** (it looks at the
    target, so running it on test data would be meaningless — there is no target
    to check against — and running it on combined data would let test rows
    influence a decision this pipeline then also applies to test).

    Raises
    ------
    KeyError
        If `target_col` or `deathdate_col` is missing.
    ValueError
        If either column contains nulls, or `months` is empty or contains a value
        outside 1-12.
    """
    validate_required_columns(df, [target_col, deathdate_col], "validate_seasonal_window")
    validate_no_nulls(df, [target_col, deathdate_col], "validate_seasonal_window")
    if not months:
        raise ValueError("validate_seasonal_window: months must be non-empty")
    bad_months = [m for m in months if not (1 <= m <= 12)]
    if bad_months:
        raise ValueError(f"validate_seasonal_window: invalid month value(s) {bad_months}")

    in_season = df[deathdate_col].dt.month.isin(months)
    y = df[target_col]

    table = pd.crosstab(in_season, y)
    if table.shape != (2, 2):
        # One side of the split (e.g. every row is in-season, or the target is
        # constant within a side) makes a 2x2 chi-square test undefined rather than
        # merely "not significant" — surface that distinctly instead of returning a
        # misleading p-value.
        raise ValueError(
            f"validate_seasonal_window: the (in-season x target) table is not 2x2 "
            f"(got shape {table.shape}) — likely every row falls on one side of the "
            "season split, or the target is constant on one side. Cannot run a "
            "chi-square test on a degenerate table."
        )
    chi2, p_value, _dof, _expected = stats.chi2_contingency(table)

    return SeasonValidationResult(
        name=name,
        months=tuple(months),
        rate_in_season=float(y[in_season].mean()),
        rate_out_of_season=float(y[~in_season].mean()),
        n_in_season=int(in_season.sum()),
        n_out_of_season=int((~in_season).sum()),
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=bool(p_value < alpha),
    )


def add_rainy_season_flag(
    df: pd.DataFrame, months: tuple[int, ...], feature_name: str, deathdate_col: str = "deathdate"
) -> pd.DataFrame:
    """Adds a single binary `feature_name` column: 1 if `deathdate_col`'s month is
    in `months`, else 0. Static calendar rule — safe to apply to train, test, or
    any dataset without any prior fitting.

    Deliberately does not decide *which* candidate window to use, or whether to add
    one at all — that decision belongs to `validate_seasonal_window`'s result and
    the caller (see this module's docstring for why, on this project's real data,
    the honest recommendation is "available but not validated as significant").
    """
    validate_required_columns(df, [deathdate_col], "add_rainy_season_flag")
    validate_no_nulls(df, [deathdate_col], "add_rainy_season_flag")
    if not months:
        raise ValueError("add_rainy_season_flag: months must be non-empty")
    bad_months = [m for m in months if not (1 <= m <= 12)]
    if bad_months:
        raise ValueError(f"add_rainy_season_flag: invalid month value(s) {bad_months}")
    check_no_output_collision(df, [feature_name], "add_rainy_season_flag")

    out = df.copy()
    out[feature_name] = out[deathdate_col].dt.month.isin(months).astype(int)
    return out
