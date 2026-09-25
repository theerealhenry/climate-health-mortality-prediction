"""Stage 17, check 5 of 8 -- schema-equality gate between a generated submission and
SampleSubmission.csv, plus Test.csv ID coverage. Runs against the archived, actually
-submitted file (submissions/submission_745a64f.csv, C-001) so every future submission
gets this validation for free (see docs/submission_strategy_policy.md Section 7).

Value-domain note: SampleSubmission.csv itself is a placeholder (every row is 0/0), so
it can't tell us the real value domain. The domain checked here -- TargetF1 in {0, 1},
TargetRAUC a float in [0, 1] -- is the contract predict.py's own docstring and
build_submission_frame() define (Stage 15.3), confirmed against SampleSubmission.csv's
column names/order/dtype shape directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION_PATH = REPO_ROOT / "data" / "raw" / "SampleSubmission.csv"
TEST_PATH = REPO_ROOT / "data" / "raw" / "Test.csv"
ARCHIVED_SUBMISSION_PATH = REPO_ROOT / "submissions" / "submission_745a64f.csv"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"test_submission_schema: expected file missing: {path}")
    return pd.read_csv(path)


def test_column_names_and_order_match_sample_submission():
    submission = _load(ARCHIVED_SUBMISSION_PATH)
    sample = _load(SAMPLE_SUBMISSION_PATH)
    assert list(submission.columns) == list(sample.columns)


def test_row_count_and_id_set_match_test_csv_exactly():
    submission = _load(ARCHIVED_SUBMISSION_PATH)
    test_df = _load(TEST_PATH)

    submission_ids = submission["ID"]
    assert not submission_ids.duplicated().any(), "duplicate IDs in submission"
    assert len(submission) == len(test_df), "row count doesn't match Test.csv"
    assert set(submission_ids) == set(test_df["ID"]), "ID set doesn't match Test.csv exactly"


def test_target_f1_is_hard_zero_or_one():
    submission = _load(ARCHIVED_SUBMISSION_PATH)
    assert submission["TargetF1"].isin([0, 1]).all()


def test_target_rauc_is_a_probability_in_zero_one():
    submission = _load(ARCHIVED_SUBMISSION_PATH)
    rauc = submission["TargetRAUC"]
    assert rauc.between(0.0, 1.0).all()
    # A real probability column, not a smuggled-in copy of the hard 0/1 call.
    assert rauc.nunique() > 2


def test_no_missing_values():
    submission = _load(ARCHIVED_SUBMISSION_PATH)
    assert not submission.isnull().any().any()
