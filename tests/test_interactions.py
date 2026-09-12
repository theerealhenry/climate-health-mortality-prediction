"""Tests for Stage 9 — src/climate_health/features/interactions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_health.features import interactions as it


class TestAgeClimateInteraction:
    def test_product_correct(self):
        df = pd.DataFrame({"age": [0, 5, 10], "tavg_30d": [20.0, 25.0, 30.0]})
        out = it.add_age_climate_interaction(df)
        assert out["age_x_tavg_30d"].tolist() == [0.0, 125.0, 300.0]

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            it.add_age_climate_interaction(pd.DataFrame({"age": [1]}))

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"age": [1, np.nan], "tavg_30d": [1.0, 2.0]})
        with pytest.raises(ValueError, match="null values"):
            it.add_age_climate_interaction(df)

    def test_output_collision_raises(self):
        df = pd.DataFrame({"age": [1], "tavg_30d": [1.0], "age_x_tavg_30d": [0.0]})
        with pytest.raises(ValueError, match="already has column"):
            it.add_age_climate_interaction(df)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"age": [1, 2], "tavg_30d": [1.0, 2.0]})
        before = df.copy()
        it.add_age_climate_interaction(df)
        pd.testing.assert_frame_equal(df, before)


class TestClusterZoneInteraction:
    def test_combined_category_correct(self):
        df = pd.DataFrame({"spatial_cluster": [0, 1], "zone": ["Rural", "Peri_urban"]})
        out = it.add_cluster_zone_interaction(df)
        assert out["spatial_cluster_x_zone"].tolist() == ["0_Rural", "1_Peri_urban"]

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            it.add_cluster_zone_interaction(pd.DataFrame({"zone": ["Rural"]}))

    def test_output_collision_raises(self):
        df = pd.DataFrame(
            {"spatial_cluster": [0], "zone": ["Rural"], "spatial_cluster_x_zone": ["x"]}
        )
        with pytest.raises(ValueError, match="already has column"):
            it.add_cluster_zone_interaction(df)

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"spatial_cluster": [0, np.nan], "zone": ["Rural", "Peri_urban"]})
        with pytest.raises(ValueError, match="null values"):
            it.add_cluster_zone_interaction(df)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"spatial_cluster": [0, 1], "zone": ["Rural", "Peri_urban"]})
        before = df.copy()
        it.add_cluster_zone_interaction(df)
        pd.testing.assert_frame_equal(df, before)


class TestAnomalyAgeVulnerabilityInteraction:
    def test_under5_and_60plus_flags_correct(self):
        df = pd.DataFrame(
            {
                "tavg_30d_anomaly": [2.0, -1.0, 3.0, 5.0],
                "age_band": ["0-4", "5-17", "18-59", "60+"],
            }
        )
        out = it.add_anomaly_age_vulnerability_interaction(df, anomaly_col="tavg_30d_anomaly")
        assert out["tavg_30d_anomaly_x_under5"].tolist() == [2.0, 0.0, 0.0, 0.0]
        assert out["tavg_30d_anomaly_x_60plus"].tolist() == [0.0, 0.0, 0.0, 5.0]

    def test_unexpected_age_band_raises(self):
        df = pd.DataFrame({"tavg_30d_anomaly": [1.0], "age_band": ["teen"]})
        with pytest.raises(ValueError, match="unexpected age_band"):
            it.add_anomaly_age_vulnerability_interaction(df, anomaly_col="tavg_30d_anomaly")

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"tavg_30d_anomaly": [1.0, np.nan], "age_band": ["0-4", "60+"]})
        with pytest.raises(ValueError, match="null values"):
            it.add_anomaly_age_vulnerability_interaction(df, anomaly_col="tavg_30d_anomaly")

    def test_output_collision_raises(self):
        df = pd.DataFrame({"a_anomaly": [1.0], "age_band": ["0-4"], "a_anomaly_x_under5": [0.0]})
        with pytest.raises(ValueError, match="already has column"):
            it.add_anomaly_age_vulnerability_interaction(df, anomaly_col="a_anomaly")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"tavg_30d_anomaly": [1.0, 2.0], "age_band": ["0-4", "60+"]})
        before = df.copy()
        it.add_anomaly_age_vulnerability_interaction(df, anomaly_col="tavg_30d_anomaly")
        pd.testing.assert_frame_equal(df, before)


# =============================================================================
# Real-data smoke test — the full Stage 9 interaction chain on the actual pipeline
# shape (real_train_climate mirrors test_climate.py's fixture: real train data with
# Stage 7's spatial_cluster already attached).
# =============================================================================


@pytest.fixture(scope="module")
def real_train_interactions():
    from climate_health.data.loaders import load_train_full
    from climate_health.features.climate import ClimateAnomalyFeaturizer
    from climate_health.features.demographic import add_age_band_features
    from climate_health.features.spatial import SpatialClusterFeaturizer

    train = load_train_full()
    train = SpatialClusterFeaturizer(n_clusters=6, random_state=42).fit_transform(train)
    train = add_age_band_features(train)
    anomaly = ClimateAnomalyFeaturizer(value_cols=["tavg_30d"])
    train["tavg_30d_anomaly"] = anomaly.fit_transform(train)["tavg_30d_anomaly"]
    return train


class TestFullChainRealData:
    def test_all_three_interactions_run_clean_on_real_data(self, real_train_interactions):
        out = it.add_age_climate_interaction(real_train_interactions)
        out = it.add_cluster_zone_interaction(out)
        out = it.add_anomaly_age_vulnerability_interaction(out, anomaly_col="tavg_30d_anomaly")
        new_cols = [
            "age_x_tavg_30d",
            "spatial_cluster_x_zone",
            "tavg_30d_anomaly_x_under5",
            "tavg_30d_anomaly_x_60plus",
        ]
        for col in new_cols:
            assert col in out.columns
        assert out[new_cols].isna().sum().sum() == 0
