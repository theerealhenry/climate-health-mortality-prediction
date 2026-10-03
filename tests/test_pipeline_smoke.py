"""Stage 18.2 -- end-to-end pipeline smoke test (docs/ideas/stage18-20-phase7-plan.md).

Runs the real `predict.py` path against the real `Train.csv`/`Test.csv` -- full
feature-engineering pipeline, full CatBoost retrain on `configs/model_best.yaml`
best params, and the production Platt calibrator -- then checks the output
against `SampleSubmission.csv`'s schema exactly. No pre-built submission file
and no mock: this proves the pipeline from raw CSVs forward, not only the
schema shape of something already generated (see tests/test_submission_schema.py
for that narrower, Stage-17 check against an archived file).

No sampling: both CSVs are small (~3,100 train rows; Test.csv in full), and a
single CatBoost fit with already-tuned params finishes in low single-digit
seconds -- fast enough for CI as-is. If that stops being true after a future
champion change, sample Train.csv/Test.csv here and document the sample size
and why it is representative, per this task's acceptance criteria.
"""

from __future__ import annotations

from climate_health.data.loaders import load_sample_submission
from climate_health.models.predict import (
    fit_production_calibrator,
    predict_test_set,
    retrain_champion_on_full_data,
    validate_against_sample,
)


def test_predict_pipeline_end_to_end_matches_sample_submission_schema():
    model, fitted_state, feature_cols, _ = retrain_champion_on_full_data()
    calibrator = fit_production_calibrator()
    submission = predict_test_set(model, calibrator, fitted_state, feature_cols)

    sample = load_sample_submission()
    # Raises on any mismatch: column names/order, row count, ID set,
    # TargetF1 domain, TargetRAUC range. A clean return is the pass condition.
    validate_against_sample(submission, sample)
