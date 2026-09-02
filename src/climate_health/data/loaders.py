"""
Typed, schema-validated loaders for every raw input file.

Every function here does exactly three things, in order:
  1. read the CSV off disk with the correct dtypes/date parsing,
  2. validate it against its pandera contract in `schemas.py` (lazily, so a
     bad file reports every violation in one pass),
  3. return a clean, guaranteed-valid DataFrame.

Nothing downstream of this module should call `pd.read_csv` on a raw file
directly — going through these functions is what makes "the file matched
its contract" a fact instead of an assumption. See
docs/PROJECT_BLUEPRINT.md Stage 2 for the rationale.
"""

from pathlib import Path

import pandas as pd

from climate_health.data.schemas import (
    CLIMATE_FEATURES_SCHEMA,
    SAMPLE_SUBMISSION_SCHEMA,
    TEST_SCHEMA,
    TRAIN_SCHEMA,
)

# src/climate_health/data/loaders.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class DataContractError(Exception):
    """Raised when a raw file fails its pandera schema validation.

    Wraps the underlying `pandera.errors.SchemaErrors` so callers get a
    project-specific exception type without losing the original, detailed
    failure_cases report (available via `__cause__`).
    """


def _load_and_validate(path: Path, schema, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected raw data file at {path}, but it does not exist. "
            "Raw files are gitignored — see docs/PROJECT_BLUEPRINT.md Stage 1 "
            "for how to obtain them, or confirm your data/raw/ folder is populated."
        )
    df = pd.read_csv(path, parse_dates=parse_dates)
    try:
        return schema.validate(df, lazy=True)
    except Exception as exc:  # pandera.errors.SchemaErrors
        raise DataContractError(
            f"{path.name} failed schema validation. "
            f"See the wrapped exception below for every violation found:\n{exc}"
        ) from exc


def load_train(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and validate Train.csv (3,146 rows, includes `is_climate_sensitive`)."""
    return _load_and_validate(raw_dir / "Train.csv", TRAIN_SCHEMA, parse_dates=["deathdate"])


def load_test(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and validate Test.csv (1,030 rows, no target column)."""
    return _load_and_validate(raw_dir / "Test.csv", TEST_SCHEMA, parse_dates=["deathdate"])


def load_climate_features(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and validate climate_features.csv (4,176 rows; covers train + test IDs)."""
    return _load_and_validate(
        raw_dir / "climate_features.csv",
        CLIMATE_FEATURES_SCHEMA,
        parse_dates=["deathdate"],
    )


def load_sample_submission(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and validate SampleSubmission.csv (1,030 rows)."""
    return _load_and_validate(raw_dir / "SampleSubmission.csv", SAMPLE_SUBMISSION_SCHEMA)


def load_all(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load and validate every raw file at once. Convenience for notebooks."""
    return {
        "train": load_train(raw_dir),
        "test": load_test(raw_dir),
        "climate_features": load_climate_features(raw_dir),
        "sample_submission": load_sample_submission(raw_dir),
    }
