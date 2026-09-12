"""Tests for src/climate_health/utils/validation.py — extracted during Stage 6 from
climate.py's originally-private helpers, now shared across feature modules."""

from __future__ import annotations

import pandas as pd
import pytest

from climate_health.utils.validation import (
    check_no_output_collision,
    validate_no_nulls,
    validate_required_columns,
)


class TestValidateRequiredColumns:
    def test_passes_when_all_present(self):
        validate_required_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"], "fn")  # no raise

    def test_raises_keyerror_listing_all_missing(self):
        with pytest.raises(KeyError) as exc_info:
            validate_required_columns(pd.DataFrame({"a": [1]}), ["a", "b", "c"], "fn")
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)


class TestValidateNoNulls:
    def test_passes_when_no_nulls(self):
        validate_no_nulls(pd.DataFrame({"a": [1, 2]}), ["a"], "fn")  # no raise

    def test_raises_valueerror_listing_all_bad_columns(self):
        df = pd.DataFrame({"a": [1, None], "b": [1, 2], "c": [None, None]})
        with pytest.raises(ValueError) as exc_info:
            validate_no_nulls(df, ["a", "b", "c"], "fn")
        assert "a" in str(exc_info.value)
        assert "c" in str(exc_info.value)
        assert "'b'" not in str(exc_info.value)


class TestCheckNoOutputCollision:
    def test_passes_when_no_collision(self):
        check_no_output_collision(pd.DataFrame({"a": [1]}), ["b", "c"], "fn")  # no raise

    def test_raises_valueerror_on_collision(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="already has column"):
            check_no_output_collision(df, ["b", "c"], "fn")
