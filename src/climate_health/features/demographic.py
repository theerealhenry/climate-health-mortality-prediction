"""
Stage 6 — Demographic features.

Age transforms matched to public-health age bands, per docs/PROJECT_BLUEPRINT.md
Stage 6, motivated directly by Stage 3/4's forensics finding that `age` is by far the
strongest single predictor found so far (corr(age, target) = -0.44; a univariate
logistic regression on age alone reaches ~0.74 AUC) — younger age is associated with
a substantially higher climate-sensitive-mortality rate, consistent with established
public-health literature on climate-linked child mortality (diarrheal disease,
malaria, malnutrition disproportionately affecting children under 5).

Bands used: under-5 (0-4), 5-17, 18-59, 60+ — the standard WHO/UNICEF-style
age-banding convention for child/adult/older-adult mortality analysis, not an
arbitrary quantile split, per the blueprint's explicit instruction to match
"public-health age bands."

A data characteristic worth flagging up front rather than discovering it silently
downstream: **`age` is recorded in whole years, and age 0 alone accounts for 1,112 of
3,146 training rows (35.4%)** — almost certainly infant deaths (under 1 year) rounded
down to the nearest completed year, the standard convention for age-at-death data.
This means `age` cannot distinguish a death on day 3 of life from one at 11 months —
both are simply `age=0` — so no feature in this module (or elsewhere in this project)
should be read as having finer-than-yearly resolution within the under-1 population.
The `is_under5` flag and the `0-4` band both correctly capture this entire group
regardless of that resolution limit, which is why the blueprint calls for `is_under5`
specifically rather than a finer age-in-months feature this data cannot actually
support.

Leakage-safety note: every feature in this module is a deterministic function of
`age` alone (no fitting, no train-only statistic) — identical whether computed on
train, test, or any future record. There is nothing to fit here, unlike Stage 7/8.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from climate_health.utils.validation import (
    check_no_output_collision,
    validate_no_nulls,
    validate_required_columns,
)

logger = logging.getLogger(__name__)

# (label, inclusive lower bound, inclusive upper bound or None for open-ended)
AGE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("0-4", 0, 4),
    ("5-17", 5, 17),
    ("18-59", 18, 59),
    ("60+", 60, None),
)

# A generous but real sanity ceiling: the oldest verified human age on record is
# 122 years (Jeanne Calment). 130 gives headroom for any legitimate outlier without
# silently accepting obvious data-entry errors (e.g. a stray "999" placeholder).
MAX_PLAUSIBLE_AGE = 130


def validate_age_column(
    df: pd.DataFrame, age_col: str = "age", fn_name: str = "validate_age_column"
) -> None:
    """Raises `ValueError` if `age_col` contains a negative value, a non-integer
    value, or a value implausible for a human age (see `MAX_PLAUSIBLE_AGE`). Called
    internally by every function below before it trusts `age_col`; also usable
    standalone as an early data-quality gate.
    """
    validate_required_columns(df, [age_col], fn_name)
    validate_no_nulls(df, [age_col], fn_name)
    ages = df[age_col]
    if (ages < 0).any():
        n_bad = int((ages < 0).sum())
        raise ValueError(f"{fn_name}: {age_col} has {n_bad} negative value(s) — invalid age")
    non_integer = ages[(ages % 1) != 0]
    if len(non_integer) > 0:
        raise ValueError(
            f"{fn_name}: {age_col} has {len(non_integer)} non-integer value(s) (e.g. "
            f"{non_integer.iloc[0]}) — this project's data is recorded in whole years; "
            "a fractional age here likely indicates a different unit or a data error."
        )
    if (ages > MAX_PLAUSIBLE_AGE).any():
        n_bad = int((ages > MAX_PLAUSIBLE_AGE).sum())
        max_val = float(ages.max())
        raise ValueError(
            f"{fn_name}: {age_col} has {n_bad} value(s) exceeding the plausibility "
            f"ceiling of {MAX_PLAUSIBLE_AGE} (max observed: {max_val}) — investigate "
            "before treating these as genuine ages."
        )


def _assign_age_band(age: pd.Series) -> pd.Series:
    conditions = [age <= 4, age <= 17, age <= 59]
    choices = ["0-4", "5-17", "18-59"]
    return pd.Series(np.select(conditions, choices, default="60+"), index=age.index, dtype="object")


def add_age_band_features(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Adds:
      - `age_band`: categorical, one of `"0-4"`, `"5-17"`, `"18-59"`, `"60+"` —
        the standard public-health age-banding convention.
      - `is_under5`: 1 if `age_band == "0-4"`, else 0 — called out explicitly by
        the blueprint given age's dominance in the forensics pass and this
        band's outsized share of the data (35.4% of training rows are `age=0`
        alone, before even counting ages 1-4).

    Boundary ages are handled explicitly and tested directly (`age=4` -> `"0-4"`,
    `age=5` -> `"5-17"`, `age=17` -> `"5-17"`, `age=18` -> `"18-59"`, `age=59` ->
    `"18-59"`, `age=60` -> `"60+"`) rather than left to an off-by-one-prone `pd.cut`
    bin-edge choice.

    Raises
    ------
    KeyError
        If `age_col` is missing.
    ValueError
        If `age_col` fails `validate_age_column`, or `df` already has `age_band`/
        `is_under5`.
    """
    validate_age_column(df, age_col, "add_age_band_features")
    check_no_output_collision(df, ["age_band", "is_under5"], "add_age_band_features")

    out = df.copy()
    out["age_band"] = _assign_age_band(out[age_col])
    out["is_under5"] = (out["age_band"] == "0-4").astype(int)
    return out


def add_age_continuous_transforms(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Adds two continuous, monotonic transforms of `age`, useful for a linear
    model branch (Stage 11) where the strongly non-linear raw age -> target
    relationship (0-4: high rate, 60+: low rate, per the forensics pass) is better
    captured by a compressive transform than by raw age alone:

      - `age_log1p` = log(1 + age) — `log1p`, not `log`, specifically because
        `age=0` is 35.4% of this dataset's training rows; plain `log(0)` is
        undefined (-inf), which `log1p` avoids by construction.
      - `age_sqrt` = sqrt(age) — a gentler compression than log, well-defined at
        `age=0` (sqrt(0) = 0) without needing a `+1` adjustment.

    Both are deterministic, monotonic functions of `age` — tree-based models
    (LightGBM/XGBoost/CatBoost) gain nothing from them (a monotonic transform of a
    single feature cannot change a tree's split structure), but they matter for
    Stage 11's linear/logistic branch.
    """
    validate_age_column(df, age_col, "add_age_continuous_transforms")
    check_no_output_collision(df, ["age_log1p", "age_sqrt"], "add_age_continuous_transforms")

    out = df.copy()
    out["age_log1p"] = np.log1p(out[age_col])
    out["age_sqrt"] = np.sqrt(out[age_col])
    return out


# Categories confirmed present in both Train and Test (climate_health/data/loaders.py
# data, verified directly against the real files) — used only to give a clear,
# specific error on an unexpected category rather than silently passing one through.
EXPECTED_ZONE_CATEGORIES = ("Rural", "Peri_urban")
EXPECTED_GENDER_CATEGORIES = ("Male", "Female")


def validate_categorical_demographics(
    df: pd.DataFrame,
    zone_col: str = "zone",
    gender_col: str = "gender",
    expected_zones: tuple[str, ...] = EXPECTED_ZONE_CATEGORIES,
    expected_genders: tuple[str, ...] = EXPECTED_GENDER_CATEGORIES,
) -> None:
    """Raises `ValueError` if `zone_col`/`gender_col` contain any category outside
    the expected set. A defensive data-quality gate, not a feature — an unexpected
    category (a typo, a new category in a future data refresh, a schema drift) is
    exactly the kind of silent failure that should raise loudly during feature
    engineering rather than surface three stages later as a confusing model result.
    """
    validate_required_columns(df, [zone_col, gender_col], "validate_categorical_demographics")
    validate_no_nulls(df, [zone_col, gender_col], "validate_categorical_demographics")

    unexpected_zones = set(df[zone_col].unique()) - set(expected_zones)
    if unexpected_zones:
        raise ValueError(
            f"validate_categorical_demographics: unexpected {zone_col} categor"
            f"{'y' if len(unexpected_zones) == 1 else 'ies'} {sorted(unexpected_zones)} "
            f"— expected only {list(expected_zones)}"
        )
    unexpected_genders = set(df[gender_col].unique()) - set(expected_genders)
    if unexpected_genders:
        raise ValueError(
            f"validate_categorical_demographics: unexpected {gender_col} categor"
            f"{'y' if len(unexpected_genders) == 1 else 'ies'} {sorted(unexpected_genders)} "
            f"— expected only {list(expected_genders)}"
        )
