"""Tests for Stage 11.1's shared training interface."""

import pytest

from climate_health.models.baselines import BASELINE_MODELS
from climate_health.models.interface import assert_valid_estimator_factory


def test_every_existing_baseline_satisfies_the_interface():
    for name, factory in BASELINE_MODELS.items():
        assert_valid_estimator_factory(factory, name)


def test_invalid_factory_raises():
    with pytest.raises(TypeError):
        assert_valid_estimator_factory(lambda: object(), "not_an_estimator")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
