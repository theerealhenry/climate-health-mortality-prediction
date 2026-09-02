"""
Stage 2 data-contract tests.

Two kinds of coverage, both required:
  - Positive: the real raw files validate cleanly through the loaders.
  - Negative: schemas actually reject corrupted data — a schema that
    silently accepts everything is worse than no schema, because it gives
    false confidence. Each negative test corrupts exactly one contract
    rule and asserts it's caught.

Run with: pytest tests/test_schemas.py -v
"""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from climate_health.data.loaders import (
    load_climate_features,
    load_sample_submission,
    load_test,
    load_train,
)
from climate_health.data.schemas import (
    CLIMATE_FEATURES_SCHEMA,
    TEST_SCHEMA,
    TRAIN_SCHEMA,
)

# ---------------------------------------------------------------------------
# Positive tests: the real files, as shipped, must validate cleanly.
# ---------------------------------------------------------------------------


def test_train_loads_and_validates():
    df = load_train()
    assert df.shape == (3146, 13)
    assert df["is_climate_sensitive"].isin([0, 1]).all()


def test_test_loads_and_validates():
    df = load_test()
    assert df.shape == (1030, 12)
    assert "is_climate_sensitive" not in df.columns


def test_climate_features_loads_and_validates():
    df = load_climate_features()
    assert df.shape == (4176, 18)


def test_sample_submission_loads_and_validates():
    df = load_sample_submission()
    assert df.shape == (1030, 3)


# ---------------------------------------------------------------------------
# Negative tests: deliberately corrupted data must be rejected.
#
# Each asserts pandera's own SchemaErrors specifically (not a bare
# Exception) — the whole point of a negative test is knowing exactly which
# failure mode fired, and pytest.raises(Exception) would also silently
# pass on an unrelated bug (a KeyError from a typo, say) rather than
# proving the contract itself caught the corruption.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_train() -> pd.DataFrame:
    from climate_health.data.loaders import RAW_DATA_DIR

    return pd.read_csv(RAW_DATA_DIR / "Train.csv", parse_dates=["deathdate"])


@pytest.fixture(scope="module")
def raw_climate_features() -> pd.DataFrame:
    from climate_health.data.loaders import RAW_DATA_DIR

    return pd.read_csv(RAW_DATA_DIR / "climate_features.csv", parse_dates=["deathdate"])


def test_duplicate_id_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[1, "ID"] = corrupted.loc[0, "ID"]
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_malformed_id_pattern_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "ID"] = "not-a-valid-id"
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_temperature_ordering_violation_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "min_temperature"] = corrupted.loc[0, "max_temperature"] + 5
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_unexpected_extra_column_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted["unexpected_column"] = 1
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_invalid_categorical_value_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "zone"] = "Urban"  # only Rural/Peri_urban are valid
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_invalid_target_value_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "is_climate_sensitive"] = 2  # only 0/1 are valid
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_null_in_required_column_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "age"] = None
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_out_of_range_coordinate_is_rejected(raw_train):
    corrupted = raw_train.copy()
    corrupted.loc[0, "latitude"] = 45.0  # far outside Uganda's envelope
    with pytest.raises(SchemaErrors):
        TRAIN_SCHEMA.validate(corrupted, lazy=True)


def test_rain_sum_monotonicity_violation_is_rejected(raw_climate_features):
    corrupted = raw_climate_features.copy()
    corrupted.loc[0, "rain_sum_7d"] = corrupted.loc[0, "rain_sum_90d"] + 100
    with pytest.raises(SchemaErrors):
        CLIMATE_FEATURES_SCHEMA.validate(corrupted, lazy=True)


def test_test_csv_correctly_rejects_leaked_target_column(raw_train):
    # If someone accidentally merged the target into a test-like frame,
    # strict=True on TEST_SCHEMA must catch it as an unexpected column.
    fake_test = raw_train.copy()
    with pytest.raises(SchemaErrors):
        TEST_SCHEMA.validate(fake_test, lazy=True)


def test_missing_raw_file_raises_data_contract_friendly_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train(raw_dir=tmp_path)
