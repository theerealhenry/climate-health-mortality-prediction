"""Tests for Stage 10.3's baseline scorecard (src/climate_health/models/baselines.py)."""

import mlflow
import pytest

from climate_health.models.baselines import BASELINE_MODELS, run_baselines
from climate_health.tracking import TRACKING_URI


def test_run_baselines_returns_all_three_models_with_valid_scores():
    scorecard = run_baselines()

    assert set(scorecard) == set(BASELINE_MODELS)
    for name, row in scorecard.items():
        for key in ("tier1_mean", "tier2_mean", "tier3_score"):
            assert 0.0 <= row[key] <= 1.0, f"{name}.{key} = {row[key]} out of [0,1]"


def test_lightgbm_beats_majority_baseline_on_tier1():
    """Sanity check on the low/high anchors — LightGBM has real signal (age, climate
    features) that a majority-class predictor structurally cannot have."""
    scorecard = run_baselines()
    lgbm_score = scorecard["lightgbm_default"]["tier1_mean"]
    majority_score = scorecard["majority_baseline"]["tier1_mean"]
    assert lgbm_score > majority_score


def test_runs_are_logged_to_mlflow():
    run_baselines()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("climate-health")
    runs = client.search_runs(exp.experiment_id, filter_string="params.model = 'lightgbm_default'")
    assert len(runs) >= 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
