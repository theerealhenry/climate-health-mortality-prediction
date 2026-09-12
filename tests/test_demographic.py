"""Tests for Stage 6 — src/climate_health/features/demographic.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_health.features import demographic as d


@pytest.fixture(scope="module")
def real_train():
    from climate_health.data.loaders import load_train_full

    return load_train_full()


@pytest.fixture(scope="module")
def real_test():
    from climate_health.data.loaders import load_test_full

    return load_test_full()


# =============================================================================
# validate_age_column
# =============================================================================


class TestValidateAgeColumn:
    def test_valid_ages_pass(self):
        d.validate_age_column(pd.DataFrame({"age": [0, 5, 30, 110]}))  # no raise

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            d.validate_age_column(pd.DataFrame({"other": [1]}))

    def test_null_raises_valueerror(self):
        with pytest.raises(ValueError, match="null values"):
            d.validate_age_column(pd.DataFrame({"age": [1, np.nan]}))

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="negative"):
            d.validate_age_column(pd.DataFrame({"age": [-1]}))

    def test_fractional_age_raises(self):
        with pytest.raises(ValueError, match="non-integer"):
            d.validate_age_column(pd.DataFrame({"age": [4.5]}))

    def test_implausible_age_raises(self):
        with pytest.raises(ValueError, match="plausibility ceiling"):
            d.validate_age_column(pd.DataFrame({"age": [999]}))

    def test_boundary_plausible_age_passes(self):
        d.validate_age_column(pd.DataFrame({"age": [d.MAX_PLAUSIBLE_AGE]}))  # no raise

    def test_boundary_implausible_age_raises(self):
        with pytest.raises(ValueError, match="plausibility ceiling"):
            d.validate_age_column(pd.DataFrame({"age": [d.MAX_PLAUSIBLE_AGE + 1]}))


# =============================================================================
# add_age_band_features
# =============================================================================


class TestAgeBandFeatures:
    @pytest.mark.parametrize(
        "age,expected_band",
        [
            (0, "0-4"),
            (1, "0-4"),
            (4, "0-4"),
            (5, "5-17"),
            (17, "5-17"),
            (18, "18-59"),
            (59, "18-59"),
            (60, "60+"),
            (110, "60+"),
        ],
    )
    def test_boundary_ages_map_to_correct_band(self, age, expected_band):
        out = d.add_age_band_features(pd.DataFrame({"age": [age]}))
        assert out["age_band"].iloc[0] == expected_band

    def test_is_under5_flag_matches_band(self):
        out = d.add_age_band_features(pd.DataFrame({"age": [0, 4, 5, 60]}))
        assert out["is_under5"].tolist() == [1, 1, 0, 0]

    def test_all_four_bands_present_in_output_dtype(self):
        out = d.add_age_band_features(pd.DataFrame({"age": [0, 10, 30, 70]}))
        assert set(out["age_band"]) == {"0-4", "5-17", "18-59", "60+"}

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="negative"):
            d.add_age_band_features(pd.DataFrame({"age": [-1]}))

    def test_output_collision_raises(self):
        df = pd.DataFrame({"age": [1], "age_band": ["x"]})
        with pytest.raises(ValueError, match="already has column"):
            d.add_age_band_features(df)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"age": [1, 2, 3]})
        before = df.copy()
        d.add_age_band_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_real_data_band_distribution_and_rates(self, real_train):
        out = d.add_age_band_features(real_train)
        counts = out["age_band"].value_counts()
        # Verified directly against the real data.
        assert counts["0-4"] == 1733
        assert counts["5-17"] == 139
        assert counts["18-59"] == 628
        assert counts["60+"] == 646
        rates = out.groupby("age_band", observed=True)["is_climate_sensitive"].mean()
        assert rates["0-4"] > rates["60+"]  # directionally consistent with forensics
        assert rates["0-4"] == pytest.approx(0.836, abs=1e-2)
        assert rates["60+"] == pytest.approx(0.300, abs=1e-2)

    def test_real_data_age_zero_is_large_fraction(self, real_train):
        # The specific data characteristic flagged in the module docstring.
        frac = (real_train["age"] == 0).mean()
        assert frac == pytest.approx(0.354, abs=0.01)


# =============================================================================
# add_age_continuous_transforms
# =============================================================================


class TestAgeContinuousTransforms:
    def test_log1p_well_defined_at_zero(self):
        out = d.add_age_continuous_transforms(pd.DataFrame({"age": [0]}))
        assert out["age_log1p"].iloc[0] == 0.0

    def test_sqrt_well_defined_at_zero(self):
        out = d.add_age_continuous_transforms(pd.DataFrame({"age": [0]}))
        assert out["age_sqrt"].iloc[0] == 0.0

    def test_transforms_are_monotonic_increasing(self):
        df = pd.DataFrame({"age": [0, 5, 20, 50, 100]})
        out = d.add_age_continuous_transforms(df)
        assert out["age_log1p"].is_monotonic_increasing
        assert out["age_sqrt"].is_monotonic_increasing

    def test_known_values(self):
        out = d.add_age_continuous_transforms(pd.DataFrame({"age": [0, 3, 8]}))
        assert out["age_log1p"].tolist() == pytest.approx([np.log1p(0), np.log1p(3), np.log1p(8)])
        assert out["age_sqrt"].tolist() == pytest.approx([0.0, np.sqrt(3), np.sqrt(8)])

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="negative"):
            d.add_age_continuous_transforms(pd.DataFrame({"age": [-5]}))

    def test_output_collision_raises(self):
        df = pd.DataFrame({"age": [1], "age_log1p": [0.0]})
        with pytest.raises(ValueError, match="already has column"):
            d.add_age_continuous_transforms(df)

    def test_real_data_no_nan_no_inf(self, real_train, real_test):
        for df in (real_train, real_test):
            out = d.add_age_continuous_transforms(df)
            assert out["age_log1p"].isna().sum() == 0
            assert out["age_sqrt"].isna().sum() == 0
            assert np.isfinite(out["age_log1p"]).all()
            assert np.isfinite(out["age_sqrt"]).all()


# =============================================================================
# validate_categorical_demographics
# =============================================================================


class TestValidateCategoricalDemographics:
    def test_valid_categories_pass(self):
        df = pd.DataFrame({"zone": ["Rural", "Peri_urban"], "gender": ["Male", "Female"]})
        d.validate_categorical_demographics(df)  # no raise

    def test_unexpected_zone_raises(self):
        df = pd.DataFrame({"zone": ["Urban"], "gender": ["Male"]})
        with pytest.raises(ValueError, match="unexpected zone"):
            d.validate_categorical_demographics(df)

    def test_unexpected_gender_raises(self):
        df = pd.DataFrame({"zone": ["Rural"], "gender": ["Other"]})
        with pytest.raises(ValueError, match="unexpected gender"):
            d.validate_categorical_demographics(df)

    def test_missing_columns_raises_keyerror(self):
        with pytest.raises(KeyError):
            d.validate_categorical_demographics(pd.DataFrame({"other": [1]}))

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"zone": ["Rural", None], "gender": ["Male", "Female"]})
        with pytest.raises(ValueError, match="null values"):
            d.validate_categorical_demographics(df)

    def test_real_train_and_test_pass(self, real_train, real_test):
        d.validate_categorical_demographics(real_train)  # no raise
        d.validate_categorical_demographics(real_test)  # no raise


# =============================================================================
# End-to-end pipeline smoke test on real data
# =============================================================================


class TestFullPipelineRealData:
    def test_full_chain_runs_clean(self, real_train, real_test):
        for df in (real_train, real_test):
            out = d.add_age_band_features(df)
            out = d.add_age_continuous_transforms(out)
            d.validate_categorical_demographics(out)
            for col in ["age_band", "is_under5", "age_log1p", "age_sqrt"]:
                assert col in out.columns
            assert out[["is_under5", "age_log1p", "age_sqrt"]].isna().sum().sum() == 0
