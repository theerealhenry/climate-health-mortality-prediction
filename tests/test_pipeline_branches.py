"""Tests for Stage 11.2's two-branch column selection (pipeline.branch_a/branch_b)."""

import pandas as pd
import pytest

from climate_health.data.loaders import load_test_full, load_train_full
from climate_health.features.pipeline import (
    branch_a_native_categorical,
    branch_b_encoded,
    build_feature_matrix,
)


def _matrix():
    X, _ = build_feature_matrix(load_train_full(), fit=True)
    return X


def test_branch_a_keeps_raw_categoricals_as_category_dtype_and_drops_encoded():
    X = _matrix()
    out = branch_a_native_categorical(X)

    assert out["zone"].dtype == "category"
    assert out["spatial_cluster"].dtype == "category"
    assert "zone_te" not in out.columns
    assert "spatial_cluster_te" not in out.columns


def test_branch_b_drops_raw_ordinal_cluster_but_keeps_its_encoding():
    X = _matrix()
    out = branch_b_encoded(X)

    assert "spatial_cluster" not in out.columns
    assert "spatial_cluster_te" in out.columns
    assert "zone_te" in out.columns
    # gender/age_band one-hot expands into gender_<value> columns, not a raw column
    assert "gender" not in out.columns
    assert any(c.startswith("gender_") for c in out.columns)


def test_branch_b_output_is_fully_numeric():
    out = branch_b_encoded(_matrix())
    assert all(pd.api.types.is_numeric_dtype(out[c]) or out[c].dtype == bool for c in out.columns)


def test_fitted_state_freezes_branch_b_schema_matching_a_fresh_call():
    X_train, state = build_feature_matrix(load_train_full(), fit=True)
    assert state.branch_b_columns == tuple(branch_b_encoded(X_train).columns)


def test_branch_b_encoded_reindexes_test_data_to_the_fitted_schema():
    X_train, state = build_feature_matrix(load_train_full(), fit=True)
    X_test, _ = build_feature_matrix(load_test_full(), fit=False, fitted_state=state)

    out = branch_b_encoded(X_test, fitted_columns=state.branch_b_columns)
    assert tuple(out.columns) == state.branch_b_columns


def test_branch_b_encoded_fills_missing_category_columns_with_zero_and_warns():
    """A category present at fit time but entirely absent from a later call (e.g. a
    rare `spatial_cluster` value, or a gender value) must not silently reshape the
    matrix — it should reindex to the fitted schema, filling the missing one-hot
    column with 0, and warn so the gap is visible rather than silent."""
    X_train, state = build_feature_matrix(load_train_full(), fit=True)

    # Simulate data missing a category branch_b_encoded one-hots: drop every row
    # with the less common gender value.
    minority_gender = X_train["gender"].value_counts().idxmin()
    X_missing_category = X_train[X_train["gender"] != minority_gender]
    dropped_col = f"gender_{minority_gender}"
    assert dropped_col in state.branch_b_columns  # sanity: fit-time schema had it

    with pytest.warns(UserWarning, match="doesn't match the fitted schema"):
        out = branch_b_encoded(X_missing_category, fitted_columns=state.branch_b_columns)

    assert tuple(out.columns) == state.branch_b_columns
    assert (out[dropped_col] == 0).all()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
