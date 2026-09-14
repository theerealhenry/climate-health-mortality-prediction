"""Tests for Stage 15.3's submission-generation logic (predict.py).

The full pipeline (retrain on real data, predict on Test.csv) needs the actual
data files and a real CatBoost fit — not a fast, deterministic unit test. What
IS pure logic, worth testing directly, is the two functions between "we have
calibrated probabilities" and "a validated submission.csv on disk":
building the two-target submission frame, and checking it against
SampleSubmission.csv's schema/shape before it's ever written out.
"""

import pandas as pd
import pytest

from climate_health.models.predict import build_submission_frame, validate_against_sample


def test_build_submission_frame_thresholds_at_0_5_for_targetf1():
    ids = ["ID_x0001", "ID_x0002", "ID_x0003"]
    probs = [0.2, 0.5, 0.81]
    submission = build_submission_frame(ids, probs)

    assert list(submission["TargetF1"]) == [0, 1, 1]
    assert list(submission["TargetRAUC"]) == probs
    assert list(submission["ID"]) == ids


def test_build_submission_frame_column_order_matches_sample_submission():
    submission = build_submission_frame(["ID_x0001"], [0.5])
    assert list(submission.columns) == ["ID", "TargetF1", "TargetRAUC"]


def test_validate_against_sample_passes_for_matching_ids_and_shape():
    sample = pd.DataFrame({"ID": ["a", "b"], "TargetF1": [0, 1], "TargetRAUC": [0.1, 0.9]})
    submission = pd.DataFrame({"ID": ["b", "a"], "TargetF1": [1, 0], "TargetRAUC": [0.9, 0.1]})
    validate_against_sample(submission, sample)  # should not raise


def test_validate_against_sample_raises_on_row_count_mismatch():
    sample = pd.DataFrame({"ID": ["a", "b"], "TargetF1": [0, 1], "TargetRAUC": [0.1, 0.9]})
    submission = pd.DataFrame({"ID": ["a"], "TargetF1": [0], "TargetRAUC": [0.1]})
    with pytest.raises(ValueError, match="row count"):
        validate_against_sample(submission, sample)


def test_validate_against_sample_raises_on_id_set_mismatch():
    sample = pd.DataFrame({"ID": ["a", "b"], "TargetF1": [0, 1], "TargetRAUC": [0.1, 0.9]})
    submission = pd.DataFrame({"ID": ["a", "c"], "TargetF1": [0, 1], "TargetRAUC": [0.1, 0.9]})
    with pytest.raises(ValueError, match="ID"):
        validate_against_sample(submission, sample)


def test_validate_against_sample_raises_on_out_of_range_probability():
    sample = pd.DataFrame({"ID": ["a"], "TargetF1": [0], "TargetRAUC": [0.1]})
    submission = pd.DataFrame({"ID": ["a"], "TargetF1": [0], "TargetRAUC": [1.5]})
    with pytest.raises(ValueError, match="TargetRAUC"):
        validate_against_sample(submission, sample)
