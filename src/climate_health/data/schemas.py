"""
Pandera data contracts for every raw input file this project depends on.

Design principle: a schema describes what's *structurally* valid about a raw
file, not today's incidental values. For example `hot_days_30d` in
climate_features.csv happens to be constant zero in the current data — that's
a feature-engineering finding (see docs/PROJECT_BLUEPRINT.md Phase 3, Track A),
not something this contract hard-codes. If a future refresh of the file had
non-zero values, this schema should still accept it.

Every schema is validated lazily (`lazy=True`) by the loaders in
`loaders.py`, so a bad file reports *every* violation in one pass instead of
stopping at the first — see docs/PROJECT_BLUEPRINT.md Stage 2.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# ID pattern shared by every file: "ID_" + 8 uppercase hex characters
_ID_PATTERN = r"^ID_[0-9A-F]{8}$"

# Uganda's rough geographic envelope (generous margins beyond the country's
# actual borders). Every record in this dataset is Ugandan — verified in
# Stage 1 forensics: every `location` string ends in "Uganda".
_LAT_RANGE = (-2.0, 5.0)
_LON_RANGE = (29.0, 36.0)

# Competition window bounds the sanity range for deathdate; actual observed
# span (Stage 1 forensics) is 2007-10-02 to 2022-12-13.
_DATE_MIN = pd.Timestamp("2000-01-01")
_DATE_MAX = pd.Timestamp("2026-12-31")


def _max_ge_avg_ge_min(df: pd.DataFrame) -> pd.Series:
    return (df["max_temperature"] >= df["avg_temperature"]) & (
        df["avg_temperature"] >= df["min_temperature"]
    )


def _date_in_range(s: pd.Series) -> pd.Series:
    return s.between(_DATE_MIN, _DATE_MAX)


# --- Core columns shared by Train.csv and Test.csv -------------------------

_CORE_COLUMNS = {
    "ID": Column(
        str,
        checks=Check.str_matches(_ID_PATTERN),
        unique=True,
        nullable=False,
    ),
    "zone": Column(str, checks=Check.isin(["Rural", "Peri_urban"]), nullable=False),
    "gender": Column(str, checks=Check.isin(["Male", "Female"]), nullable=False),
    "deathdate": Column("datetime64[ns]", checks=Check(_date_in_range), nullable=False),
    "age": Column(float, checks=Check.in_range(0, 120), nullable=False, coerce=True),
    "avg_temperature": Column(float, checks=Check.in_range(-10, 50), nullable=False, coerce=True),
    "max_temperature": Column(float, checks=Check.in_range(-10, 55), nullable=False, coerce=True),
    "min_temperature": Column(float, checks=Check.in_range(-10, 45), nullable=False, coerce=True),
    "precipitation": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
    "latitude": Column(float, checks=Check.in_range(*_LAT_RANGE), nullable=False, coerce=True),
    "longitude": Column(float, checks=Check.in_range(*_LON_RANGE), nullable=False, coerce=True),
    "location": Column(str, nullable=False),
}

_TEMPERATURE_CONSISTENCY_CHECK = Check(
    _max_ge_avg_ge_min,
    error="max_temperature must be >= avg_temperature >= min_temperature",
)

TEST_SCHEMA = DataFrameSchema(
    columns=dict(_CORE_COLUMNS),
    checks=[_TEMPERATURE_CONSISTENCY_CHECK],
    strict=True,
    coerce=False,
)

TRAIN_SCHEMA = DataFrameSchema(
    columns={
        **_CORE_COLUMNS,
        "is_climate_sensitive": Column(int, checks=Check.isin([0, 1]), nullable=False, coerce=True),
    },
    checks=[_TEMPERATURE_CONSISTENCY_CHECK],
    strict=True,
    coerce=False,
)


# --- climate_features.csv ---------------------------------------------------


def _rain_monotonic(df: pd.DataFrame) -> pd.Series:
    return (df["rain_sum_90d"] >= df["rain_sum_30d"]) & (df["rain_sum_30d"] >= df["rain_sum_7d"])


CLIMATE_FEATURES_SCHEMA = DataFrameSchema(
    columns={
        "ID": Column(str, checks=Check.str_matches(_ID_PATTERN), unique=True, nullable=False),
        "deathdate": Column("datetime64[ns]", checks=Check(_date_in_range), nullable=False),
        "rain_sum_7d": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "rain_sum_30d": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "rain_sum_90d": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "rain_days_30d": Column(int, checks=Check.in_range(0, 30), nullable=False, coerce=True),
        "max_daily_rain_30d": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "tavg_7d": Column(float, checks=Check.in_range(-10, 45), nullable=False, coerce=True),
        "tavg_30d": Column(float, checks=Check.in_range(-10, 45), nullable=False, coerce=True),
        "tavg_90d": Column(float, checks=Check.in_range(-10, 45), nullable=False, coerce=True),
        "tmax_30d": Column(float, checks=Check.in_range(-10, 55), nullable=False, coerce=True),
        "tmin_30d": Column(float, checks=Check.in_range(-10, 45), nullable=False, coerce=True),
        # Structurally an integer count in [0, 30] — currently constant zero
        # in this dataset (see docstring above), not asserted here.
        "hot_days_30d": Column(int, checks=Check.in_range(0, 30), nullable=False, coerce=True),
        "temp_range_mean_30d": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "ndvi_30d": Column(float, checks=Check.in_range(-1, 1), nullable=False, coerce=True),
        "ndvi_90d": Column(float, checks=Check.in_range(-1, 1), nullable=False, coerce=True),
        "elevation": Column(float, checks=Check.in_range(0, 5200), nullable=False, coerce=True),
        "slope": Column(float, checks=Check.in_range(0, 90), nullable=False, coerce=True),
    },
    checks=[Check(_rain_monotonic, error="rain_sum_90d >= rain_sum_30d >= rain_sum_7d must hold")],
    strict=True,
    coerce=False,
)


# --- SampleSubmission.csv / any generated submission.csv -------------------

SAMPLE_SUBMISSION_SCHEMA = DataFrameSchema(
    columns={
        "ID": Column(str, checks=Check.str_matches(_ID_PATTERN), unique=True, nullable=False),
        "TargetF1": Column(int, checks=Check.isin([0, 1]), nullable=False, coerce=True),
        "TargetRAUC": Column(float, checks=Check.in_range(0, 1), nullable=False, coerce=True),
    },
    strict=True,
    coerce=False,
)
