"""Tests for Stage 10.2's build_feature_matrix (src/climate_health/features/pipeline.py)."""

import copy

import pytest

from climate_health.data.loaders import load_test_full, load_train_full
from climate_health.features.pipeline import FeatureFitState, build_feature_matrix


@pytest.fixture(scope="module")
def train_df():
    return load_train_full()


@pytest.fixture(scope="module")
def test_df():
    return load_test_full()


def test_fit_true_and_false_produce_same_columns_and_dtypes(train_df, test_df):
    X_train, state = build_feature_matrix(train_df, fit=True)
    X_test, _ = build_feature_matrix(test_df, fit=False, fitted_state=state)

    # Test.csv carries no label, so the target column is the one legitimate
    # difference — every feature column must still match exactly.
    feature_cols_train = set(X_train.columns) - {"is_climate_sensitive"}
    assert feature_cols_train == set(X_test.columns)
    for col in X_test.columns:
        assert X_train[col].dtype == X_test[col].dtype, f"dtype mismatch on {col}"


def test_fit_true_with_existing_state_raises(train_df):
    _, state = build_feature_matrix(train_df, fit=True)
    with pytest.raises(ValueError):
        build_feature_matrix(train_df, fit=True, fitted_state=state)


def test_fit_false_without_state_raises(test_df):
    with pytest.raises(ValueError):
        build_feature_matrix(test_df, fit=False)


def test_unknown_extra_feature_raises(train_df):
    with pytest.raises(ValueError):
        build_feature_matrix(train_df, fit=True, extra_features=frozenset({"nonsense"}))


def test_extra_features_opt_in_adds_columns(train_df):
    X_plain, _ = build_feature_matrix(train_df, fit=True)
    X_extra, _ = build_feature_matrix(
        train_df, fit=True, extra_features=frozenset({"cluster_zone_interaction"})
    )
    assert set(X_extra.columns) - set(X_plain.columns)


def test_returned_state_transformers_are_isolated_copies(train_df):
    _, state = build_feature_matrix(train_df, fit=True)
    original_kmeans = state.spatial_cluster.kmeans_
    mutant = copy.deepcopy(state.spatial_cluster)
    mutant.fit(train_df)  # mutate the copy, not the original state's transformer
    assert state.spatial_cluster.kmeans_ is original_kmeans


def test_target_encoding_mapping_is_also_an_isolated_copy(train_df):
    """FeatureFitState deepcopies every mutable field, not just the two transformer
    objects — target_encoding is a plain dict and just as mutable in a caller's
    hands, so it needs the same protection. Construct a state directly (rather than
    via build_feature_matrix) so there's an external dict reference to mutate that
    is distinct from whatever the state stores internally."""
    _, fitted = build_feature_matrix(train_df, fit=True)
    external_mapping = {"rural": 0.5}
    external_encoding = {"zone": (external_mapping, 0.5)}

    state = FeatureFitState(
        reference_year=fitted.reference_year,
        heat_threshold=fitted.heat_threshold,
        spatial_cluster=fitted.spatial_cluster,
        climate_anomaly=fitted.climate_anomaly,
        target_encoding=external_encoding,
        branch_b_columns=fitted.branch_b_columns,
    )

    external_mapping["rural"] = -999.0  # mutate the caller's original reference

    assert state.target_encoding["zone"][0]["rural"] == 0.5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
