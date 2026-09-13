"""Tests for Stage 13.2's Optuna tuning module (src/climate_health/models/tune.py).

Scope, deliberately: these prove the three things that would silently break Stage
13's guarantees if wrong — the locked holdout is genuinely excluded, the suggested
hyperparameter ranges are honored, and the tuning feature frame matches what actually
won in Stage 11.3 (branch_a, not branch_b). A full end-to-end CatBoost tuning run is
Stage 13.3's smoke test, not a unit test — fitting real trials here would make the
suite slow for no extra correctness signal beyond what's covered below.
"""

import optuna
import pytest

from climate_health.evaluation.cv import tier2_splits
from climate_health.models.baselines import RANDOM_STATE
from climate_health.models.tune import (
    LOCKED_HOLDOUT,
    build_tuning_frame,
    non_locked_tier2_splits,
    suggest_catboost_params,
)


@pytest.fixture(scope="module")
def tuning_frame():
    return build_tuning_frame()


def test_non_locked_tier2_splits_never_yields_the_locked_fold(tuning_frame):
    combined, _, _ = tuning_frame
    seen = {
        (s.repeat, s.fold)
        for s in non_locked_tier2_splits(
            combined, spatial_cluster_col="spatial_cluster", random_state=RANDOM_STATE
        )
    }
    assert LOCKED_HOLDOUT not in seen


def test_non_locked_tier2_splits_excludes_exactly_one_fold_relative_to_tier2_splits(tuning_frame):
    combined, _, _ = tuning_frame
    all_splits = {
        (s.repeat, s.fold)
        for s in tier2_splits(
            combined, spatial_cluster_col="spatial_cluster", random_state=RANDOM_STATE
        )
    }
    non_locked = {
        (s.repeat, s.fold)
        for s in non_locked_tier2_splits(
            combined, spatial_cluster_col="spatial_cluster", random_state=RANDOM_STATE
        )
    }
    assert LOCKED_HOLDOUT in all_splits, "locked fold must exist under the default seed"
    assert all_splits - non_locked == {LOCKED_HOLDOUT}


def test_build_tuning_frame_uses_native_categorical_branch(tuning_frame):
    """Must match Stage 11.3's winning representation (branch_a) — see
    test_zoo.py::test_catboost_native_actually_used_categorical_columns, which this
    mirrors for the tuning path specifically."""
    _, feature_cols, cat_idx = tuning_frame
    assert "zone" in feature_cols
    assert "zone_te" not in feature_cols
    assert len(cat_idx) > 0


def test_suggest_catboost_params_stays_within_declared_bounds():
    study = optuna.create_study(direction="maximize")
    for _ in range(25):
        trial = study.ask()
        params = suggest_catboost_params(trial)
        study.tell(trial, 0.0)  # value is irrelevant; only the suggested ranges matter

        assert 200 <= params["iterations"] <= 800
        assert 4 <= params["depth"] <= 8
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert 1.0 <= params["l2_leaf_reg"] <= 10.0
        assert 0.0 <= params["bagging_temperature"] <= 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
