"""
Stage 7 spatial-feature tests.

Coverage:
  - `build_location_profiles`: correctness of the per-coordinate aggregation, and its
    validation (missing columns, nulls).
  - `select_n_clusters`: sane behavior on a synthetic, well-separated dataset, and its
    guardrails (k too large for the data, empty k_range).
  - `SpatialClusterFeaturizer`: the core contract this module exists to deliver —
    every row sharing a coordinate gets the same cluster label; a genuinely unseen
    coordinate (present only at transform time) is assigned a sensible nearest cluster,
    not an error; the usual not-fitted/validation guardrails.
  - `summarize_clusters`, `add_coordinate_polynomial_features`, `location_fallback_tokens`:
    correctness on small, hand-built examples.
  - A real-data smoke test mirroring the empirical check run during development: fit
    on the real training data and confirm no coordinate is ever split across two
    cluster labels.

Run with: pytest tests/test_spatial.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from climate_health.data.loaders import load_test_full, load_train_full
from climate_health.features.spatial import (
    ClusterSelectionResult,
    SpatialClusterFeaturizer,
    add_coordinate_polynomial_features,
    build_location_profiles,
    location_fallback_tokens,
    select_n_clusters,
    summarize_clusters,
)

RANDOM_STATE = 42


@pytest.fixture(scope="module")
def train_full() -> pd.DataFrame:
    return load_train_full()


@pytest.fixture(scope="module")
def test_full() -> pd.DataFrame:
    return load_test_full()


def _make_synthetic_two_blob_df(n_per_blob: int = 30, seed: int = 0) -> pd.DataFrame:
    # Two well-separated coordinate/climate blobs so KMeans has an unambiguous answer
    # to check itself against, independent of the real dataset's messier structure.
    rng = np.random.RandomState(seed)
    blob_a_coords = rng.normal(loc=(0.0, 30.0), scale=0.01, size=(n_per_blob, 2))
    blob_b_coords = rng.normal(loc=(5.0, 40.0), scale=0.01, size=(n_per_blob, 2))
    coords = np.vstack([blob_a_coords, blob_b_coords])
    climate_a = rng.normal(loc=(20.0, 100.0, 1000.0), scale=0.1, size=(n_per_blob, 3))
    climate_b = rng.normal(loc=(28.0, 400.0, 500.0), scale=0.1, size=(n_per_blob, 3))
    climate = np.vstack([climate_a, climate_b])
    df = pd.DataFrame(
        {
            "latitude": coords[:, 0],
            "longitude": coords[:, 1],
            "tavg_30d": climate[:, 0],
            "rain_sum_90d": climate[:, 1],
            "elevation": climate[:, 2],
        }
    )
    return df


# ---------------------------------------------------------------------------
# build_location_profiles
# ---------------------------------------------------------------------------


def test_build_location_profiles_averages_repeated_coordinates():
    df = pd.DataFrame(
        {
            "latitude": [1.0, 1.0, 2.0],
            "longitude": [30.0, 30.0, 31.0],
            "tavg_30d": [20.0, 22.0, 25.0],
            "rain_sum_90d": [100.0, 200.0, 50.0],
            "elevation": [1000.0, 1000.0, 1200.0],
        }
    )
    profiles = build_location_profiles(df)
    assert len(profiles) == 2  # two unique coordinates
    row = profiles[(profiles["latitude"] == 1.0) & (profiles["longitude"] == 30.0)].iloc[0]
    assert row["tavg_30d"] == pytest.approx(21.0)  # mean of 20, 22
    assert row["rain_sum_90d"] == pytest.approx(150.0)  # mean of 100, 200


def test_build_location_profiles_missing_column_raises():
    df = pd.DataFrame({"latitude": [1.0], "longitude": [30.0]})
    with pytest.raises(KeyError):
        build_location_profiles(df)


def test_build_location_profiles_null_value_raises():
    df = pd.DataFrame(
        {
            "latitude": [1.0, 2.0],
            "longitude": [30.0, 31.0],
            "tavg_30d": [20.0, None],
            "rain_sum_90d": [100.0, 50.0],
            "elevation": [1000.0, 1200.0],
        }
    )
    with pytest.raises(ValueError):
        build_location_profiles(df)


def test_build_location_profiles_real_train_matches_forensics(train_full):
    # Stage 3 forensics: 43 unique (latitude, longitude) pairs in Train.
    profiles = build_location_profiles(train_full)
    assert len(profiles) == 43


# ---------------------------------------------------------------------------
# select_n_clusters
# ---------------------------------------------------------------------------


def test_select_n_clusters_recovers_two_obvious_blobs():
    df = _make_synthetic_two_blob_df()
    best_k, results = select_n_clusters(df, k_range=(2, 3, 4))
    assert best_k == 2
    assert all(isinstance(r, ClusterSelectionResult) for r in results)
    best_result = next(r for r in results if r.k == 2)
    assert best_result.silhouette > 0.9  # two tight, far-apart blobs -> near-perfect separation


def test_select_n_clusters_k_too_large_raises():
    df = _make_synthetic_two_blob_df(n_per_blob=3)  # 6 unique coordinates total
    with pytest.raises(ValueError):
        select_n_clusters(df, k_range=(10,))


def test_select_n_clusters_empty_k_range_raises():
    df = _make_synthetic_two_blob_df()
    with pytest.raises(ValueError):
        select_n_clusters(df, k_range=())


# ---------------------------------------------------------------------------
# SpatialClusterFeaturizer
# ---------------------------------------------------------------------------


def test_featurizer_recovers_two_obvious_blobs():
    df = _make_synthetic_two_blob_df()
    feat = SpatialClusterFeaturizer(n_clusters=2, random_state=RANDOM_STATE)
    labeled = feat.fit_transform(df)
    # every row in the first half (blob A) should share one label, second half (blob B)
    # the other -- order within each blob's rows doesn't matter, just a 2-way split.
    labels_a = set(labeled["spatial_cluster"].iloc[:30])
    labels_b = set(labeled["spatial_cluster"].iloc[30:])
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a != labels_b


def test_featurizer_same_coordinate_always_same_cluster(train_full):
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    labeled = feat.fit_transform(train_full)
    n_distinct_per_coord = labeled.groupby(["latitude", "longitude"])["spatial_cluster"].nunique()
    assert n_distinct_per_coord.max() == 1


def test_featurizer_assigns_unseen_test_coordinates_without_error(train_full, test_full):
    # The core promise of this module: a coordinate present only at transform time
    # (all 12 unique test coordinates have zero overlap with the 43 training ones, per
    # Stage 3 forensics) must still get a sensible cluster assignment, not an error.
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    feat.fit(train_full)
    test_labeled = feat.transform(test_full)
    assert test_labeled["spatial_cluster"].notna().all()
    assert set(test_labeled["spatial_cluster"].unique()) <= set(range(6))
    assert len(test_labeled) == len(test_full)


def test_featurizer_transform_preserves_row_count_and_index(train_full):
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    feat.fit(train_full)
    labeled = feat.transform(train_full)
    assert len(labeled) == len(train_full)
    assert list(labeled.index) == list(train_full.index)


def test_featurizer_transform_raises_on_output_col_collision():
    # Regression test: transform() used to silently merge into "spatial_cluster_x" /
    # "spatial_cluster_y" (pandas' default suffix behavior for a name collision)
    # instead of raising, when X already had a column named output_col -- e.g. from
    # re-transforming already-labeled data, or a stray column from an upstream step.
    # That would silently corrupt the output rather than fail loudly.
    df = _make_synthetic_two_blob_df(n_per_blob=3)
    df["spatial_cluster"] = "PRE_EXISTING"
    feat = SpatialClusterFeaturizer(n_clusters=2, random_state=RANDOM_STATE)
    feat.fit(df.drop(columns=["spatial_cluster"]))
    with pytest.raises(ValueError, match="spatial_cluster"):
        feat.transform(df)


def test_featurizer_transform_row_alignment_matches_input(train_full):
    # Stronger than the index-equality check alone: verify every row's own
    # coordinates in the transformed output still match that same row's coordinates
    # in the input, position for position -- this is what actually matters for
    # `out.index = X.index` to be safe, not just that the two index objects are equal
    # length/labels.
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    feat.fit(train_full)
    # shuffle rows and use a non-default index to stress-test the alignment logic
    shuffled = train_full.sample(frac=1.0, random_state=3)
    labeled = feat.transform(shuffled)
    assert list(labeled.index) == list(shuffled.index)
    assert (labeled["latitude"].to_numpy() == shuffled["latitude"].to_numpy()).all()
    assert (labeled["longitude"].to_numpy() == shuffled["longitude"].to_numpy()).all()


def test_fit_transform_rejects_unexpected_fit_params():
    # Regression test: fit_transform(**fit_params) used to silently swallow any
    # unexpected keyword arguments (e.g. a misspelled sample_weight meant for a
    # different pipeline step) since fit() doesn't accept them and they were never
    # passed through or validated.
    df = _make_synthetic_two_blob_df(n_per_blob=3)
    feat = SpatialClusterFeaturizer(n_clusters=2, random_state=RANDOM_STATE)
    with pytest.raises(TypeError):
        feat.fit_transform(df, sample_weight=[1] * len(df))


def test_featurizer_get_params_preserves_input_type():
    # sklearn convention: __init__ must not transform its arguments -- get_params()
    # (and therefore clone()) should return exactly what was passed in, not a
    # coerced type. Regression test for a prior version that cast coord_cols to a
    # tuple in __init__, so get_params()["coord_cols"] disagreed with a list input.
    feat = SpatialClusterFeaturizer(coord_cols=["latitude", "longitude"])
    assert feat.get_params()["coord_cols"] == ["latitude", "longitude"]
    assert isinstance(feat.get_params()["coord_cols"], list)


def test_featurizer_transform_before_fit_raises():
    feat = SpatialClusterFeaturizer(n_clusters=2)
    df = _make_synthetic_two_blob_df(n_per_blob=3)
    with pytest.raises(NotFittedError):
        feat.transform(df)


def test_featurizer_n_clusters_too_large_raises():
    df = _make_synthetic_two_blob_df(n_per_blob=3)  # 6 unique coordinates
    feat = SpatialClusterFeaturizer(n_clusters=10)
    with pytest.raises(ValueError):
        feat.fit(df)


def test_featurizer_missing_column_raises():
    df = pd.DataFrame({"latitude": [1.0], "longitude": [30.0]})
    feat = SpatialClusterFeaturizer(n_clusters=1)
    with pytest.raises(KeyError):
        feat.fit(df)


def test_featurizer_cluster_centers_original_scale_shape(train_full):
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    feat.fit(train_full)
    centers = feat.cluster_centers_original_scale()
    assert centers.shape == (6, 5)  # latitude, longitude, tavg_30d, rain_sum_90d, elevation
    assert list(centers.columns) == [
        "latitude",
        "longitude",
        "tavg_30d",
        "rain_sum_90d",
        "elevation",
    ]
    # sanity: centroids should be plausible Uganda coordinates, not scaled/garbage values
    assert centers["latitude"].between(-2.0, 5.0).all()
    assert centers["longitude"].between(29.0, 36.0).all()


def test_featurizer_real_train_no_coordinate_split_across_clusters(train_full):
    # Empirical smoke test mirroring the manual check run during development.
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    labeled = feat.fit_transform(train_full)
    assert labeled["spatial_cluster"].nunique() == 6
    assert labeled["spatial_cluster"].isna().sum() == 0


def test_featurizer_is_sklearn_compatible_clone():
    from sklearn.base import clone

    feat = SpatialClusterFeaturizer(n_clusters=3, random_state=7)
    cloned = clone(feat)
    assert cloned.n_clusters == 3
    assert cloned.random_state == 7
    assert not hasattr(cloned, "kmeans_")  # clone is unfitted


# ---------------------------------------------------------------------------
# summarize_clusters
# ---------------------------------------------------------------------------


def test_summarize_clusters_shape_and_columns(train_full):
    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    feat.fit(train_full)
    summary = summarize_clusters(feat, train_full)
    assert len(summary) == 6
    assert "n_records" in summary.columns
    assert "n_unique_locations" in summary.columns
    assert "dominant_zone" in summary.columns
    assert summary["n_records"].sum() == len(train_full)


# ---------------------------------------------------------------------------
# add_coordinate_polynomial_features
# ---------------------------------------------------------------------------


def test_add_coordinate_polynomial_features_correctness():
    df = pd.DataFrame({"latitude": [2.0, -1.0], "longitude": [30.0, 32.0]})
    out = add_coordinate_polynomial_features(df)
    assert out["latitude_sq"].tolist() == [4.0, 1.0]
    assert out["longitude_sq"].tolist() == [900.0, 1024.0]
    assert out["latitude_x_longitude"].tolist() == [60.0, -32.0]
    # original columns untouched
    assert out["latitude"].tolist() == [2.0, -1.0]


def test_add_coordinate_polynomial_features_missing_column_raises():
    df = pd.DataFrame({"latitude": [1.0]})
    with pytest.raises(KeyError):
        add_coordinate_polynomial_features(df)


def test_add_coordinate_polynomial_features_null_coordinate_raises():
    df = pd.DataFrame({"latitude": [1.0, None], "longitude": [30.0, 31.0]})
    with pytest.raises(ValueError, match="null"):
        add_coordinate_polynomial_features(df)


# ---------------------------------------------------------------------------
# location_fallback_tokens
# ---------------------------------------------------------------------------


def test_location_fallback_tokens_basic():
    df = pd.DataFrame({"location": ["Nakigo I, Iganga, Uganda"]})
    out = location_fallback_tokens(df, max_tokens=3)
    assert out["location_token_0"].iloc[0] == "Uganda"
    assert out["location_token_1"].iloc[0] == "Iganga"
    assert out["location_token_2"].iloc[0] == "Nakigo I"


def test_location_fallback_tokens_handles_missing_tokens():
    # only 2 comma-separated parts -> token_2 should be None, not raise
    df = pd.DataFrame({"location": ["Some Place, Uganda"]})
    out = location_fallback_tokens(df, max_tokens=3)
    assert out["location_token_0"].iloc[0] == "Uganda"
    assert out["location_token_1"].iloc[0] == "Some Place"
    assert out["location_token_2"].iloc[0] is None


def test_location_fallback_tokens_missing_column_raises():
    df = pd.DataFrame({"not_location": ["x"]})
    with pytest.raises(KeyError):
        location_fallback_tokens(df)


def test_location_fallback_tokens_null_location_raises_clear_error():
    # Regression test: a null location used to crash with a raw, confusing
    # "TypeError: 'NoneType' object is not iterable" from deep inside the split/strip
    # logic instead of a clear, explicit error like every other validation in this
    # module and the rest of the codebase (compute_twin_groups, build_location_profiles).
    df = pd.DataFrame({"location": ["A, B, Uganda", None]})
    with pytest.raises(ValueError, match="null"):
        location_fallback_tokens(df)


def test_location_fallback_tokens_real_data_no_error(train_full):
    # Stage 3/4 found location strings vary from 3-8 comma-separated tokens -- this
    # should never raise on the real data, however deep the string.
    out = location_fallback_tokens(train_full, max_tokens=3)
    assert out["location_token_0"].notna().all()  # trailing token ("Uganda") always present


# ---------------------------------------------------------------------------
# Integration with cv.py's tier2_splits — this is the whole point of Stage 7: retiring
# the placeholder k=8 coordinate-only clustering `tier2_splits` falls back to when no
# spatial_cluster_col is given.
# ---------------------------------------------------------------------------


def test_featurizer_output_retires_the_cv_placeholder_warning(train_df, recwarn):
    from climate_health.evaluation.cv import PLACEHOLDER_SPATIAL_CLUSTER_SOURCE, tier2_splits

    feat = SpatialClusterFeaturizer(n_clusters=6, random_state=RANDOM_STATE)
    labeled = feat.fit_transform(train_df)
    list(
        tier2_splits(
            labeled,
            spatial_cluster_col="spatial_cluster",
            spatial_cluster_source="spatial_py_kmeans_k6_coord_plus_climate",
            n_splits=5,
            n_repeats=1,
            random_state=RANDOM_STATE,
        )
    )
    placeholder_warnings = [
        w for w in recwarn.list if PLACEHOLDER_SPATIAL_CLUSTER_SOURCE in str(w.message)
    ]
    assert not placeholder_warnings, (
        "tier2_splits still fell back to the placeholder clustering — the real "
        "SpatialClusterFeaturizer output was not picked up as spatial_cluster_col."
    )


@pytest.fixture(scope="module")
def train_df(train_full) -> pd.DataFrame:
    # cv.py's twin/tier machinery expects the core Train.csv columns (location,
    # deathdate, is_climate_sensitive); train_full already is exactly that, merged
    # with climate_features.csv, so it satisfies both this module's and cv.py's needs.
    return train_full


# ---------------------------------------------------------------------------
# merge_core_and_climate (src/climate_health/data/loaders.py) -- no direct test
# coverage existed for its two explicit error paths before this stage's review;
# both were only reachable implicitly (happy path only) via load_train_full/
# load_test_full's real-data fixtures.
# ---------------------------------------------------------------------------


def test_merge_core_and_climate_missing_id_raises():
    from climate_health.data.loaders import DataContractError, merge_core_and_climate

    core = pd.DataFrame(
        {
            "ID": ["ID_AAAAAAAA", "ID_BBBBBBBB"],
            "deathdate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    climate = pd.DataFrame({"ID": ["ID_AAAAAAAA"], "deathdate": pd.to_datetime(["2020-01-01"])})
    with pytest.raises(DataContractError, match="no matching ID"):
        merge_core_and_climate(core, climate)


def test_merge_core_and_climate_mismatched_deathdate_raises():
    from climate_health.data.loaders import DataContractError, merge_core_and_climate

    core = pd.DataFrame({"ID": ["ID_AAAAAAAA"], "deathdate": pd.to_datetime(["2020-01-01"])})
    climate = pd.DataFrame({"ID": ["ID_AAAAAAAA"], "deathdate": pd.to_datetime(["2020-01-02"])})
    with pytest.raises(DataContractError, match="disagrees"):
        merge_core_and_climate(core, climate)


def test_merge_core_and_climate_happy_path():
    from climate_health.data.loaders import merge_core_and_climate

    core = pd.DataFrame(
        {
            "ID": ["ID_AAAAAAAA", "ID_BBBBBBBB"],
            "deathdate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    climate = pd.DataFrame(
        {
            "ID": ["ID_AAAAAAAA", "ID_BBBBBBBB"],
            "deathdate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "tavg_30d": [20.0, 21.0],
        }
    )
    merged = merge_core_and_climate(core, climate)
    assert len(merged) == 2
    assert "deathdate_cf" not in merged.columns
    assert merged["tavg_30d"].tolist() == [20.0, 21.0]
