"""Tests for Stage 8 — src/climate_health/features/climate.py.

Coverage mirrors the discipline established in test_spatial.py: synthetic
unit tests for every function/branch, plus real-data smoke tests that exercise
the empirically-observed sparsity/edge cases (zero-denominator rainfall
ratios, sparse cluster x month buckets) rather than only hypothetical ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from climate_health.features import climate as c
from climate_health.features.spatial import SpatialClusterFeaturizer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def real_train_climate():
    """Real train data merged with climate features and Stage 7 spatial cluster —
    the actual pipeline shape Stage 8 will run against."""
    from climate_health.data.loaders import load_train_full

    train = load_train_full()
    sf = SpatialClusterFeaturizer(n_clusters=6, random_state=42)
    return sf.fit_transform(train)


@pytest.fixture(scope="module")
def real_test_climate(real_train_climate):
    from climate_health.data.loaders import load_test_full

    test = load_test_full()
    sf = SpatialClusterFeaturizer(n_clusters=6, random_state=42)
    sf.fit(pd.DataFrame(real_train_climate))  # refit on train only, consistent w/ pipeline
    return sf.transform(test)


def _rain_df(**overrides):
    base = {
        "rain_sum_7d": [7.0, 0.0, 14.0],
        "rain_sum_30d": [30.0, 0.0, 60.0],
        "rain_sum_90d": [90.0, 0.0, 180.0],
        "rain_days_30d": [10, 0, 20],
        "max_daily_rain_30d": [5.0, 0.0, 10.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _anomaly_train_df():
    return pd.DataFrame(
        {
            "tavg_30d": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "rain_sum_90d": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
            "spatial_cluster": [0, 0, 0, 1, 1, 1],
            "deathdate": pd.to_datetime(
                ["2020-01-01", "2020-01-15", "2020-02-01", "2020-03-01", "2020-03-15", "2020-04-01"]
            ),
        }
    )


# =============================================================================
# _safe_ratio
# =============================================================================


class TestSafeRatio:
    def test_normal_division(self):
        result = c._safe_ratio(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
        assert list(result) == [5.0, 5.0]

    def test_zero_denominator_yields_nan(self):
        result = c._safe_ratio(pd.Series([10.0, 0.0]), pd.Series([0.0, 0.0]))
        assert result.isna().all()

    def test_mixed_zero_and_nonzero(self):
        result = c._safe_ratio(pd.Series([10.0, 5.0]), pd.Series([2.0, 0.0]))
        assert result.iloc[0] == 5.0
        assert np.isnan(result.iloc[1])


# =============================================================================
# Track A — drop_dead_columns
# =============================================================================


class TestDropDeadColumns:
    def test_drops_constant_column(self):
        df = pd.DataFrame({"hot_days_30d": [0, 0, 0], "other": [1, 2, 3]})
        out = c.drop_dead_columns(df)
        assert "hot_days_30d" not in out.columns
        assert "other" in out.columns

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(KeyError):
            c.drop_dead_columns(df)

    def test_raises_when_not_actually_constant(self):
        df = pd.DataFrame({"hot_days_30d": [0, 1, 0], "other": [1, 2, 3]})
        with pytest.raises(ValueError, match="NOT constant"):
            c.drop_dead_columns(df)

    def test_verify_constant_false_allows_override(self):
        df = pd.DataFrame({"hot_days_30d": [0, 1, 0], "other": [1, 2, 3]})
        out = c.drop_dead_columns(df, verify_constant=False)
        assert "hot_days_30d" not in out.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"hot_days_30d": [0, 0], "other": [1, 2]})
        c.drop_dead_columns(df)
        assert "hot_days_30d" in df.columns

    def test_real_data_hot_days_30d_is_constant_zero(self, real_train_climate):
        assert real_train_climate["hot_days_30d"].nunique(dropna=False) == 1
        assert real_train_climate["hot_days_30d"].iloc[0] == 0
        out = c.drop_dead_columns(real_train_climate)
        assert "hot_days_30d" not in out.columns


# =============================================================================
# Track A — add_rainfall_rate_features
# =============================================================================


class TestRainfallRateFeatures:
    def test_rates_are_sum_over_window_length(self):
        df = _rain_df()
        out = c.add_rainfall_rate_features(df)
        assert out["rain_rate_7d"].iloc[0] == pytest.approx(7.0 / 7)
        assert out["rain_rate_30d"].iloc[0] == pytest.approx(30.0 / 30)
        assert out["rain_rate_90d"].iloc[0] == pytest.approx(90.0 / 90)

    def test_ratio_features_correct(self):
        df = _rain_df()
        out = c.add_rainfall_rate_features(df)
        assert out["rain_ratio_acute_medium"].iloc[0] == pytest.approx(7.0 / 30.0)
        assert out["rain_ratio_medium_chronic"].iloc[0] == pytest.approx(30.0 / 90.0)
        assert out["rain_intensity_30d"].iloc[0] == pytest.approx(5.0 / 30.0)
        assert out["rain_day_fraction_30d"].iloc[0] == pytest.approx(10.0 / 30.0)

    def test_zero_denominator_rows_yield_nan_not_inf_or_zero(self):
        df = _rain_df()  # row 1 is all-zero
        out = c.add_rainfall_rate_features(df)
        for col in ["rain_ratio_acute_medium", "rain_ratio_medium_chronic", "rain_intensity_30d"]:
            assert np.isnan(out[col].iloc[1])
            assert not np.isinf(out[col].iloc[1])

    def test_missing_column_raises_keyerror(self):
        df = _rain_df().drop(columns=["rain_days_30d"])
        with pytest.raises(KeyError):
            c.add_rainfall_rate_features(df)

    def test_null_value_raises_valueerror(self):
        df = _rain_df()
        df.loc[0, "rain_sum_7d"] = np.nan
        with pytest.raises(ValueError, match="null values"):
            c.add_rainfall_rate_features(df)

    def test_output_collision_raises(self):
        df = _rain_df()
        df["rain_rate_7d"] = 999.0
        with pytest.raises(ValueError, match="already has column"):
            c.add_rainfall_rate_features(df)

    def test_does_not_mutate_input(self):
        df = _rain_df()
        before = df.copy()
        c.add_rainfall_rate_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_real_data_zero_denominator_rate(self, real_train_climate):
        out = c.add_rainfall_rate_features(real_train_climate)
        n_zero_30d = (real_train_climate["rain_sum_30d"] == 0).sum()
        assert n_zero_30d == out["rain_ratio_acute_medium"].isna().sum()
        assert n_zero_30d >= 1


# =============================================================================
# Track B — compute_heat_threshold / add_heat_exceedance_features
# =============================================================================


class TestHeatThreshold:
    def test_default_quantile_is_90th_percentile(self):
        df = pd.DataFrame({"tmax_30d": list(range(1, 101))})  # 1..100
        thresh = c.compute_heat_threshold(df)
        assert thresh == pytest.approx(90.1, abs=0.5)

    def test_invalid_quantile_raises(self):
        df = pd.DataFrame({"tmax_30d": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="quantile must be in"):
            c.compute_heat_threshold(df, quantile=1.5)
        with pytest.raises(ValueError, match="quantile must be in"):
            c.compute_heat_threshold(df, quantile=0.0)

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({"other": [1.0]})
        with pytest.raises(KeyError):
            c.compute_heat_threshold(df)

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"tmax_30d": [1.0, np.nan]})
        with pytest.raises(ValueError, match="null values"):
            c.compute_heat_threshold(df)

    def test_real_data_never_exceeds_35c(self, real_train_climate, real_test_climate):
        assert real_train_climate["tmax_30d"].max() < 35.0
        assert real_test_climate["tmax_30d"].max() < 35.0

    def test_real_data_threshold_and_flag_rates(self, real_train_climate, real_test_climate):
        thresh = c.compute_heat_threshold(real_train_climate, quantile=0.90)
        assert 28.0 < thresh < 33.0  # sanity band around the observed ~30.7
        train_out = c.add_heat_exceedance_features(real_train_climate, threshold=thresh)
        test_out = c.add_heat_exceedance_features(real_test_climate, threshold=thresh)
        train_rate = train_out["tmax_30d_exceeds_flag"].mean()
        test_rate = test_out["tmax_30d_exceeds_flag"].mean()
        assert 0.05 < train_rate < 0.15
        assert 0.05 < test_rate < 0.20


class TestHeatExceedanceFeatures:
    def test_flag_and_margin_correctness(self):
        df = pd.DataFrame({"tmax_30d": [10.0, 20.0, 30.0]})
        out = c.add_heat_exceedance_features(df, threshold=20.0)
        assert list(out["tmax_30d_exceeds_flag"]) == [0, 0, 1]
        assert out["tmax_30d_exceedance_margin"].tolist() == pytest.approx([-10.0, 0.0, 10.0])

    def test_output_collision_raises(self):
        df = pd.DataFrame({"tmax_30d": [10.0], "tmax_30d_exceeds_flag": [1]})
        with pytest.raises(ValueError, match="already has column"):
            c.add_heat_exceedance_features(df, threshold=5.0)

    def test_custom_column_name(self):
        df = pd.DataFrame({"tavg_90d": [10.0, 25.0]})
        out = c.add_heat_exceedance_features(df, threshold=20.0, col="tavg_90d")
        assert "tavg_90d_exceeds_flag" in out.columns
        assert "tavg_90d_exceedance_margin" in out.columns

    def test_nan_threshold_raises(self):
        df = pd.DataFrame({"tmax_30d": [10.0, 20.0]})
        with pytest.raises(ValueError, match="finite"):
            c.add_heat_exceedance_features(df, threshold=float("nan"))

    def test_infinite_threshold_raises(self):
        df = pd.DataFrame({"tmax_30d": [10.0, 20.0]})
        with pytest.raises(ValueError, match="finite"):
            c.add_heat_exceedance_features(df, threshold=float("inf"))


# =============================================================================
# ClimateAnomalyFeaturizer
# =============================================================================


class TestClimateAnomalyFeaturizer:
    def test_bare_string_value_cols_raises_typeerror(self):
        # A bare string is iterable char-by-char in Python — a classic footgun if
        # someone forgets to wrap a single column name in a list. Must raise clearly
        # rather than silently decomposing "tavg_30d" into 8 single-character "columns".
        with pytest.raises(TypeError, match="bare string"):
            c.ClimateAnomalyFeaturizer(value_cols="tavg_30d")

    def test_get_params_preserves_input_types(self):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], group_col="spatial_cluster")
        params = caf.get_params()
        assert isinstance(params["value_cols"], list)
        assert params["value_cols"] == ["tavg_30d"]

    def test_clone_roundtrip(self):
        from sklearn.base import clone

        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d", "rain_sum_90d"])
        cloned = clone(caf)
        assert cloned.get_params() == caf.get_params()

    def test_transform_before_fit_raises_not_fitted(self):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"])
        with pytest.raises(NotFittedError):
            caf.transform(_anomaly_train_df())

    def test_fit_missing_column_raises_keyerror(self):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["missing_col"])
        with pytest.raises(KeyError):
            caf.fit(_anomaly_train_df())

    def test_fit_null_raises_valueerror(self):
        df = _anomaly_train_df()
        df.loc[0, "tavg_30d"] = np.nan
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"])
        with pytest.raises(ValueError, match="null values"):
            caf.fit(df)

    def test_transform_output_collision_raises(self):
        train = _anomaly_train_df()
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf.fit(train)
        collide = train.copy()
        collide["tavg_30d_anomaly"] = 0.0
        with pytest.raises(ValueError, match="already has column"):
            caf.transform(collide)

    def test_fit_transform_rejects_unexpected_kwargs(self):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"])
        with pytest.raises(TypeError, match="unexpected fit_params"):
            caf.fit_transform(_anomaly_train_df(), sample_weight=[1] * 6)

    def test_group_month_tier_used_when_bucket_has_enough_rows(self):
        train = _anomaly_train_df()
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf.fit(train)
        out = caf.transform(train)
        assert (out["tavg_30d_anomaly_baseline_level"] == "group_month").all()

    def test_group_tier_used_when_bucket_too_sparse(self):
        train = _anomaly_train_df()
        # min_group_month_size=5 but every (cluster, month) bucket here has <=2 rows
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=5)
        caf.fit(train)
        out = caf.transform(train)
        assert (out["tavg_30d_anomaly_baseline_level"] == "group").all()
        # anomaly should be value minus the group (cluster) mean
        expected_cluster0_mean = train.loc[train["spatial_cluster"] == 0, "tavg_30d"].mean()
        row0_anomaly = (
            out.loc[out["spatial_cluster"] == 0, "tavg_30d"].iloc[0] - expected_cluster0_mean
        )
        assert out.loc[out["spatial_cluster"] == 0, "tavg_30d_anomaly"].iloc[0] == pytest.approx(
            row0_anomaly + 0  # sanity: recomputed the same way
        )

    def test_global_tier_used_for_novel_group_at_transform_time(self):
        train = _anomaly_train_df()
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=2)
        caf.fit(train)
        novel = pd.DataFrame(
            {
                "tavg_30d": [30.0],
                "spatial_cluster": [999],
                "deathdate": pd.to_datetime(["2020-05-01"]),
            }
        )
        out = caf.transform(novel)
        assert out["tavg_30d_anomaly_baseline_level"].iloc[0] == "global"
        assert out["tavg_30d_anomaly"].iloc[0] == pytest.approx(30.0 - caf.global_mean_["tavg_30d"])

    def test_group_tier_used_for_unseen_month_of_known_group(self):
        train = _anomaly_train_df()
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf.fit(train)
        unseen_month = pd.DataFrame(
            {
                "tavg_30d": [50.0],
                "spatial_cluster": [0],
                "deathdate": pd.to_datetime(["2020-12-01"]),  # cluster 0 never seen in Dec
            }
        )
        out = caf.transform(unseen_month)
        assert out["tavg_30d_anomaly_baseline_level"].iloc[0] == "group"

    def test_multiple_value_cols_independent_anomalies(self):
        train = _anomaly_train_df()
        caf = c.ClimateAnomalyFeaturizer(
            value_cols=["tavg_30d", "rain_sum_90d"], min_group_month_size=1
        )
        out = caf.fit_transform(train)
        assert "tavg_30d_anomaly" in out.columns
        assert "rain_sum_90d_anomaly" in out.columns
        assert "tavg_30d_anomaly_baseline_level" in out.columns
        assert "rain_sum_90d_anomaly_baseline_level" in out.columns

    def test_does_not_mutate_input(self):
        train = _anomaly_train_df()
        before = train.copy()
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf.fit_transform(train)
        pd.testing.assert_frame_equal(train, before)

    # -- Leave-one-out (LOO) baseline: fit_transform must exclude each training row's
    # -- own value from its own bucket statistic, unlike plain fit().transform().

    def test_fit_transform_uses_leave_one_out_not_in_sample_mean(self):
        # 3-row group: in-sample mean includes each row's own value; LOO must not.
        df = pd.DataFrame(
            {
                "tavg_30d": [10.0, 20.0, 30.0],
                "spatial_cluster": [0, 0, 0],
                "deathdate": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            }
        )
        # min_group_month_size=5 with only 1 row per (cluster, month) bucket forces
        # every row to the group tier, whose LOO exclusion is easy to hand-verify.
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=5)
        out = caf.fit_transform(df)
        assert (out["tavg_30d_anomaly_baseline_level"] == "group").all()
        # Row 0 (value=10): LOO baseline = mean(20, 30) = 25 -> anomaly = 10 - 25 = -15
        # Row 1 (value=20): LOO baseline = mean(10, 30) = 20 -> anomaly = 20 - 20 = 0
        # Row 2 (value=30): LOO baseline = mean(10, 20) = 15 -> anomaly = 30 - 15 = 15
        assert out["tavg_30d_anomaly"].tolist() == pytest.approx([-15.0, 0.0, 15.0])

    def test_fit_transform_matches_in_sample_mean_only_when_group_is_large(self):
        # For a large group, LOO and in-sample means converge (self-weight -> 0), so
        # both formulations should agree to several decimal places.
        rng = np.random.default_rng(0)
        n = 5000
        df = pd.DataFrame(
            {
                "tavg_30d": rng.normal(20, 2, size=n),
                "spatial_cluster": np.zeros(n, dtype=int),
                "deathdate": pd.to_datetime(["2020-01-01"] * n),
            }
        )
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        out = caf.fit_transform(df)
        in_sample_mean = df["tavg_30d"].mean()
        expected_in_sample_anomaly = df["tavg_30d"] - in_sample_mean
        # LOO differs from in-sample by a factor of n/(n-1) on the mean itself, which
        # is negligible at n=5000 but must NOT be exactly zero (i.e. this really is
        # computing something different, not silently falling back to in-sample).
        assert not np.allclose(out["tavg_30d_anomaly"], expected_in_sample_anomaly, atol=0)
        assert np.allclose(out["tavg_30d_anomaly"], expected_in_sample_anomaly, atol=1e-2)

    def test_fit_transform_differs_from_fit_then_transform(self):
        # Deliberate, documented divergence from the sklearn fit(X).transform(X)
        # default — this is the whole point of the LOO design (see class docstring).
        train = _anomaly_train_df()
        caf_ft = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=5)
        loo_out = caf_ft.fit_transform(train)

        caf_plain = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=5)
        caf_plain.fit(train)
        plain_out = caf_plain.transform(train)

        assert not np.allclose(
            loo_out["tavg_30d_anomaly"].to_numpy(), plain_out["tavg_30d_anomaly"].to_numpy()
        )

    def test_fit_transform_loo_falls_back_when_bucket_would_be_emptied(self):
        # min_group_month_size=1 admits a (cluster, month) bucket of exactly 1 row as
        # "valid" for plain transform() — but under LOO, excluding that row's own
        # value leaves zero others to average, so fit_transform must fall through to
        # the group tier for that specific row instead of dividing by zero rows.
        df = pd.DataFrame(
            {
                "tavg_30d": [10.0, 20.0, 99.0],  # cluster 0: Jan, Jan; cluster 0: Feb (alone)
                "spatial_cluster": [0, 0, 0],
                "deathdate": pd.to_datetime(["2020-01-01", "2020-01-15", "2020-02-01"]),
            }
        )
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        out = caf.fit_transform(df)
        levels = out["tavg_30d_anomaly_baseline_level"].tolist()
        # The two January rows share a (cluster, month) bucket of size 2 -> LOO count
        # 1, still usable -> "group_month". The lone February row's bucket has size 1
        # -> LOO count 0 -> must fall back to "group".
        assert levels[0] == "group_month"
        assert levels[1] == "group_month"
        assert levels[2] == "group"
        # February row's group-tier LOO baseline = mean of the OTHER two rows only.
        assert out["tavg_30d_anomaly"].iloc[2] == pytest.approx(99.0 - (10.0 + 20.0) / 2)

    def test_transform_on_external_data_unaffected_by_loo(self):
        # transform() on genuinely new data must be identical whether or not the
        # featurizer was fit via fit() or fit_transform() — LOO only changes how
        # fit_transform's OWN output (on the training rows) is computed.
        train = _anomaly_train_df()
        external = pd.DataFrame(
            {
                "tavg_30d": [50.0],
                "spatial_cluster": [0],
                "deathdate": pd.to_datetime(["2020-01-20"]),
            }
        )
        caf_a = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf_a.fit_transform(train)
        out_a = caf_a.transform(external)

        caf_b = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"], min_group_month_size=1)
        caf_b.fit(train)
        out_b = caf_b.transform(external)

        pd.testing.assert_frame_equal(out_a, out_b)

    def test_real_data_fit_transform_no_nan_and_all_tiers_reachable(
        self, real_train_climate, real_test_climate
    ):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d", "rain_sum_90d"])
        train_out = caf.fit_transform(real_train_climate)
        test_out = caf.transform(real_test_climate)

        assert train_out["tavg_30d_anomaly"].isna().sum() == 0
        assert test_out["tavg_30d_anomaly"].isna().sum() == 0

        # Empirically confirmed sparsity: some (cluster, month) buckets fall below
        # the default min_group_month_size=5, so the 'group' tier must be reachable
        # on real data, not just in a synthetic test.
        levels_seen = set(train_out["tavg_30d_anomaly_baseline_level"].unique()) | set(
            test_out["tavg_30d_anomaly_baseline_level"].unique()
        )
        assert "group_month" in levels_seen
        assert "group" in levels_seen

    def test_real_data_sparse_bucket_count_matches_expected_range(self, real_train_climate):
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d"])
        caf.fit(real_train_climate)
        assert caf.n_group_month_buckets_fit_ < caf.n_group_month_buckets_total_


# =============================================================================
# Track D — add_ndvi_trend_feature
# =============================================================================


class TestNdviTrendFeature:
    def test_positive_when_greening(self):
        df = pd.DataFrame({"ndvi_30d": [0.5], "ndvi_90d": [0.3]})
        out = c.add_ndvi_trend_feature(df)
        assert out["ndvi_trend_30_90"].iloc[0] == pytest.approx(0.2)

    def test_negative_when_browning(self):
        df = pd.DataFrame({"ndvi_30d": [0.2], "ndvi_90d": [0.4]})
        out = c.add_ndvi_trend_feature(df)
        assert out["ndvi_trend_30_90"].iloc[0] == pytest.approx(-0.2)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"ndvi_30d": [0.2]})
        with pytest.raises(KeyError):
            c.add_ndvi_trend_feature(df)

    def test_output_collision_raises(self):
        df = pd.DataFrame({"ndvi_30d": [0.2], "ndvi_90d": [0.1], "ndvi_trend_30_90": [0.0]})
        with pytest.raises(ValueError, match="already has column"):
            c.add_ndvi_trend_feature(df)

    def test_real_data_no_nan(self, real_train_climate):
        out = c.add_ndvi_trend_feature(real_train_climate)
        assert out["ndvi_trend_30_90"].isna().sum() == 0


# =============================================================================
# Provenance table
# =============================================================================


class TestProvenance:
    def test_render_returns_markdown_table(self):
        md = c.render_provenance_markdown()
        assert md.startswith("| Feature |")
        assert "hot_days_30d" in md

    def test_every_entry_renders_without_error(self):
        for entry in c.FEATURE_PROVENANCE:
            assert entry.feature
            assert entry.reference_period
            assert entry.available_at_prediction_time
            assert entry.risk

    def test_hot_days_30d_marked_dropped(self):
        entry = next(e for e in c.FEATURE_PROVENANCE if e.feature == "hot_days_30d")
        assert "dropped" in entry.available_at_prediction_time.lower()

    def test_dry_spell_limitation_documented(self):
        entry = next(e for e in c.FEATURE_PROVENANCE if "dry-spell" in e.feature.lower())
        assert "not computable" in entry.notes.lower() or "NOT computable" in entry.notes


# =============================================================================
# End-to-end pipeline smoke test (Track A -> B -> C -> D chained, real data)
# =============================================================================


class TestFullPipelineRealData:
    def test_full_track_chain_runs_clean_on_real_data(self, real_train_climate, real_test_climate):
        train = c.drop_dead_columns(real_train_climate)
        train = c.add_rainfall_rate_features(train)
        thresh = c.compute_heat_threshold(train)
        train = c.add_heat_exceedance_features(train, threshold=thresh)
        caf = c.ClimateAnomalyFeaturizer(value_cols=["tavg_30d", "rain_sum_90d"])
        train = caf.fit_transform(train)
        train = c.add_ndvi_trend_feature(train)

        test = c.drop_dead_columns(real_test_climate)
        test = c.add_rainfall_rate_features(test)
        test = c.add_heat_exceedance_features(test, threshold=thresh)
        test = caf.transform(test)
        test = c.add_ndvi_trend_feature(test)

        new_cols = [
            "rain_rate_7d",
            "rain_rate_30d",
            "rain_rate_90d",
            "rain_ratio_acute_medium",
            "rain_ratio_medium_chronic",
            "rain_intensity_30d",
            "rain_day_fraction_30d",
            "tmax_30d_exceeds_flag",
            "tmax_30d_exceedance_margin",
            "tavg_30d_anomaly",
            "rain_sum_90d_anomaly",
            "ndvi_trend_30_90",
        ]
        for col in new_cols:
            assert col in train.columns
            assert col in test.columns
        assert "hot_days_30d" not in train.columns
        assert "hot_days_30d" not in test.columns
