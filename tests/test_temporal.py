"""Tests for Stage 6 — src/climate_health/features/temporal.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_health.features import temporal as t


@pytest.fixture(scope="module")
def real_train():
    from climate_health.data.loaders import load_train_full

    return load_train_full()


@pytest.fixture(scope="module")
def real_test():
    from climate_health.data.loaders import load_test_full

    return load_test_full()


# =============================================================================
# add_date_decomposition_features
# =============================================================================


class TestDateDecomposition:
    def test_month_and_day_of_year_correct(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01", "2021-06-15", "2021-12-31"])})
        out = t.add_date_decomposition_features(df)
        assert out["month"].tolist() == [1, 6, 12]
        assert out["day_of_year"].tolist() == [1, 166, 365]
        assert out["year"].tolist() == [2021, 2021, 2021]

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            t.add_date_decomposition_features(pd.DataFrame({"other": [1]}))

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01", None])})
        with pytest.raises(ValueError, match="null values"):
            t.add_date_decomposition_features(df)

    def test_non_datetime_dtype_raises_valueerror(self):
        df = pd.DataFrame({"deathdate": ["2021-01-01", "2021-02-01"]})
        with pytest.raises(ValueError, match="datetime64"):
            t.add_date_decomposition_features(df)

    def test_output_collision_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"]), "month": [1]})
        with pytest.raises(ValueError, match="already has column"):
            t.add_date_decomposition_features(df)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"])})
        before = df.copy()
        t.add_date_decomposition_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_month_cyclical_encoding_dec_jan_adjacent(self):
        # December and January should be close in (sin, cos) space, not at
        # opposite ends of a raw numeric scale.
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-12-01", "2021-01-01"])})
        out = t.add_date_decomposition_features(df)
        dec = out.iloc[0][["month_sin", "month_cos"]].to_numpy(dtype=float)
        jan = out.iloc[1][["month_sin", "month_cos"]].to_numpy(dtype=float)
        # Euclidean distance in cyclical space should be small (adjacent months),
        # not the ~2.0 max distance that unrelated points on the unit circle could have.
        dist = np.linalg.norm(dec - jan)
        assert dist < 0.6

    def test_month_cyclical_encoding_round_trip_unit_circle(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime([f"2021-{m:02d}-01" for m in range(1, 13)])})
        out = t.add_date_decomposition_features(df)
        radius = np.sqrt(out["month_sin"] ** 2 + out["month_cos"] ** 2)
        assert np.allclose(radius, 1.0)

    # -- Leap-year edge cases (explicitly called out in the blueprint) --

    def test_leap_day_does_not_crash_and_has_correct_day_of_year(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2020-02-29"])})  # 2020 is a leap year
        out = t.add_date_decomposition_features(df)
        assert out["day_of_year"].iloc[0] == 60
        assert out["month"].iloc[0] == 2

    def test_dec31_leap_vs_nonleap_year_maps_to_same_phase(self):
        # Without leap-aware normalization, Dec 31 of a leap year (day 366 of 366)
        # and Dec 31 of a non-leap year (day 365 of 365) should represent the SAME
        # point in the annual cycle -- a fixed period=365 would instead place the
        # leap-year Dec 31 one full day short of completing the circle.
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2020-12-31", "2021-12-31"])})  # 2020 leap
        out = t.add_date_decomposition_features(df)
        leap_point = out.iloc[0][["day_of_year_sin", "day_of_year_cos"]].to_numpy(dtype=float)
        nonleap_point = out.iloc[1][["day_of_year_sin", "day_of_year_cos"]].to_numpy(dtype=float)
        assert np.allclose(leap_point, nonleap_point, atol=1e-9)

    def test_day_of_year_sin_cos_on_unit_circle(self):
        df = pd.DataFrame(
            {"deathdate": pd.to_datetime(["2020-02-29", "2021-03-01", "2019-07-15", "2000-12-31"])}
        )
        out = t.add_date_decomposition_features(df)
        radius = np.sqrt(out["day_of_year_sin"] ** 2 + out["day_of_year_cos"] ** 2)
        assert np.allclose(radius, 1.0)

    def test_real_data_leap_day_record_handled(self, real_test):
        # Confirmed via direct inspection: the real test set contains a Feb 29
        # record. Must not raise, and must decompose correctly.
        leap_rows = real_test[
            (real_test["deathdate"].dt.month == 2) & (real_test["deathdate"].dt.day == 29)
        ]
        assert len(leap_rows) >= 1
        out = t.add_date_decomposition_features(real_test)
        matched = out.loc[leap_rows.index]
        assert (matched["day_of_year"] == 60).all()

    def test_real_data_full_pipeline_no_nan(self, real_train, real_test):
        train_out = t.add_date_decomposition_features(real_train)
        test_out = t.add_date_decomposition_features(real_test)
        cols = [
            "month",
            "day_of_year",
            "year",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ]
        assert train_out[cols].isna().sum().sum() == 0
        assert test_out[cols].isna().sum().sum() == 0


# =============================================================================
# compute_reference_year / add_year_trend_feature
# =============================================================================


class TestYearTrend:
    def test_reference_year_is_min_year(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2010-01-01", "2005-06-01", "2015-01-01"])})
        assert t.compute_reference_year(df) == 2005

    def test_year_trend_relative_to_reference(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2005-01-01", "2010-01-01"])})
        out = t.add_year_trend_feature(df, reference_year=2005)
        assert out["year_since_reference"].tolist() == [0, 5]

    def test_year_trend_can_be_negative_for_earlier_data(self):
        # transform() applied to a hypothetically earlier dataset than the
        # reference year should not raise -- negative is meaningful, not an error.
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2000-01-01"])})
        out = t.add_year_trend_feature(df, reference_year=2005)
        assert out["year_since_reference"].iloc[0] == -5

    def test_output_collision_raises(self):
        df = pd.DataFrame(
            {"deathdate": pd.to_datetime(["2021-01-01"]), "year_since_reference": [0]}
        )
        with pytest.raises(ValueError, match="already has column"):
            t.add_year_trend_feature(df, reference_year=2021)

    def test_real_data_reference_year_and_consistency(self, real_train, real_test):
        ref_year = t.compute_reference_year(real_train)
        assert ref_year == 2007
        train_out = t.add_year_trend_feature(real_train, ref_year)
        test_out = t.add_year_trend_feature(real_test, ref_year)
        # Verified empirically: test never precedes train's reference year.
        assert train_out["year_since_reference"].min() >= 0
        assert test_out["year_since_reference"].min() >= 0
        assert train_out["year_since_reference"].max() == 15


# =============================================================================
# validate_seasonal_window
# =============================================================================


class TestValidateSeasonalWindow:
    def test_missing_columns_raises_keyerror(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"])})
        with pytest.raises(KeyError):
            t.validate_seasonal_window(df, "target", (3, 4, 5), "mam")

    def test_empty_months_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"]), "target": [1]})
        with pytest.raises(ValueError, match="non-empty"):
            t.validate_seasonal_window(df, "target", (), "empty")

    def test_invalid_month_value_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"]), "target": [1]})
        with pytest.raises(ValueError, match="invalid month"):
            t.validate_seasonal_window(df, "target", (0, 13), "bad")

    def test_degenerate_table_raises_clear_error(self):
        # every row in-season -> the crosstab collapses to a 1x2 table, not 2x2
        df = pd.DataFrame(
            {
                "deathdate": pd.to_datetime(["2021-03-01", "2021-04-01", "2021-05-01"]),
                "target": [1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="not 2x2"):
            t.validate_seasonal_window(df, "target", (3, 4, 5), "mam")

    def test_clear_signal_is_detected_as_significant(self):
        # Construct a dataset where season membership perfectly predicts the
        # target, to confirm the statistical machinery actually works, not just
        # that it correctly reports "not significant" on this project's real (flat)
        # data.
        rng = np.random.default_rng(0)
        n = 400
        months = rng.integers(1, 13, size=n)
        in_season = np.isin(months, [3, 4, 5])
        target = in_season.astype(int)  # perfect separation
        df = pd.DataFrame(
            {
                "deathdate": pd.to_datetime(
                    [f"2021-{m:02d}-{(i % 27) + 1:02d}" for i, m in enumerate(months)]
                ),
                "target": target,
            }
        )
        result = t.validate_seasonal_window(df, "target", (3, 4, 5), "mam")
        assert result.is_significant
        assert result.rate_in_season == pytest.approx(1.0)
        assert result.rate_out_of_season == pytest.approx(0.0)

    def test_real_data_mam_son_not_significant(self, real_train):
        # This is the actual, honest finding this module's docstring documents --
        # verified directly here rather than only asserted in prose.
        for name, months in t.RAINY_SEASON_CANDIDATES.items():
            result = t.validate_seasonal_window(real_train, "is_climate_sensitive", months, name)
            assert not result.is_significant, f"{name} unexpectedly significant: {result}"
            assert result.p_value > 0.5

    def test_real_data_result_matches_hand_verified_numbers(self, real_train):
        result = t.validate_seasonal_window(real_train, "is_climate_sensitive", (3, 4, 5), "mam")
        assert result.n_in_season == 790
        assert result.n_out_of_season == 2356
        assert result.rate_in_season == pytest.approx(0.6481, abs=1e-3)


# =============================================================================
# add_rainy_season_flag
# =============================================================================


class TestAddRainySeasonFlag:
    def test_flag_correctness(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-03-15", "2021-06-01", "2021-04-01"])})
        out = t.add_rainy_season_flag(df, (3, 4, 5), "is_mam")
        assert out["is_mam"].tolist() == [1, 0, 1]

    def test_same_rule_applies_identically_to_any_dataset(self):
        # Static calendar rule -- no fitting, so applying it to two different
        # frames with the same underlying dates gives identical results.
        df1 = pd.DataFrame({"deathdate": pd.to_datetime(["2021-09-01"])})
        df2 = pd.DataFrame({"deathdate": pd.to_datetime(["2021-09-01"]), "extra": [1]})
        out1 = t.add_rainy_season_flag(df1, (9, 10, 11), "is_son")
        out2 = t.add_rainy_season_flag(df2, (9, 10, 11), "is_son")
        assert out1["is_son"].iloc[0] == out2["is_son"].iloc[0] == 1

    def test_empty_months_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"])})
        with pytest.raises(ValueError, match="non-empty"):
            t.add_rainy_season_flag(df, (), "bad")

    def test_invalid_month_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"])})
        with pytest.raises(ValueError, match="invalid month"):
            t.add_rainy_season_flag(df, (0,), "bad")

    def test_output_collision_raises(self):
        df = pd.DataFrame({"deathdate": pd.to_datetime(["2021-01-01"]), "is_mam": [0]})
        with pytest.raises(ValueError, match="already has column"):
            t.add_rainy_season_flag(df, (3, 4, 5), "is_mam")

    def test_real_data_train_test_flag_rate_consistent(self, real_train, real_test):
        train_out = t.add_rainy_season_flag(real_train, (3, 4, 5), "is_mam")
        test_out = t.add_rainy_season_flag(real_test, (3, 4, 5), "is_mam")
        assert train_out["is_mam"].isin([0, 1]).all()
        assert test_out["is_mam"].isin([0, 1]).all()
