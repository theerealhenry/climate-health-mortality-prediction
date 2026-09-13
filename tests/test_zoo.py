"""Tests for Stage 11.3's model zoo (src/climate_health/models/zoo.py)."""

import mlflow

from climate_health.models.zoo import ZOO_MODELS, run_zoo
from climate_health.tracking import TRACKING_URI


def test_run_zoo_returns_all_candidates_with_valid_scores():
    scorecard = run_zoo()

    assert set(scorecard) == set(ZOO_MODELS)
    for name, row in scorecard.items():
        for key in ("tier1_mean", "tier2_mean", "tier3_score"):
            assert 0.0 <= row[key] <= 1.0, f"{name}.{key} = {row[key]} out of [0,1]"


def test_catboost_native_actually_used_categorical_columns():
    """Sanity check that branch_a's category dtype columns reached CatBoost as real
    categoricals, not just numeric-coerced noise — the entire point of Branch A."""
    from climate_health.data.loaders import load_train_full
    from climate_health.features.pipeline import build_feature_matrix
    from climate_health.models.baselines import TARGET_COL
    from climate_health.models.zoo import _branch_frame, branch_a_native_categorical

    X_full, _ = build_feature_matrix(load_train_full(), fit=True, target_col=TARGET_COL)
    _, feature_cols = _branch_frame(X_full, branch_a_native_categorical)
    assert "zone" in feature_cols
    assert "zone_te" not in feature_cols


def test_runs_are_logged_to_mlflow():
    run_zoo()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("climate-health")
    runs = client.search_runs(exp.experiment_id, filter_string="params.model = 'catboost_native'")
    assert len(runs) >= 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
