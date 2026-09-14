"""Tests for Stage 15.2's calibration comparison (src/climate_health/models/calibrate.py).

Synthetic cases where miscalibration and its fix are knowable by construction,
same rationale as test_ensemble.py: fast, deterministic unit tests over pure
calibration logic, not an expensive end-to-end retrain.
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from climate_health.models.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    compare_calibration_methods,
    cv_calibrate,
)


def competition_score_like(y_true, y_pred_proba) -> float:
    """Local stand-in for baselines.competition_score, avoiding the
    lightgbm/xgboost/catboost import chain for pure-calibration-logic tests."""
    f1 = f1_score(y_true, (y_pred_proba >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_pred_proba)
    return 0.60 * f1 + 0.40 * auc


# ---------------------------------------------------------------------------
# PlattCalibrator / IsotonicCalibrator
# ---------------------------------------------------------------------------


def test_platt_calibrator_fixes_a_systematically_shifted_probability():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 2000)
    # Systematically overconfident: true prob is y_true's own 0/1, squashed toward
    # the extremes — a classic sigmoid-shaped miscalibration Platt scaling targets.
    raw = np.clip(y_true * 0.9 + (1 - y_true) * 0.1 + rng.normal(0, 0.05, 2000), 0.001, 0.999)

    calibrator = PlattCalibrator().fit(raw, y_true)
    calibrated = calibrator.predict(raw)

    assert competition_score_like(y_true, calibrated) >= competition_score_like(y_true, raw) - 1e-9


def test_isotonic_calibrator_fixes_a_non_monotonic_bump():
    rng = np.random.RandomState(1)
    n = 3000
    true_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < true_prob).astype(int)
    # Non-monotonic distortion: a dip in the middle range, exactly the shape Platt
    # (a monotonic sigmoid) cannot fix but isotonic regression can.
    raw = np.clip(true_prob - 0.15 * np.exp(-((true_prob - 0.6) ** 2) / 0.01), 0.001, 0.999)

    calibrator = IsotonicCalibrator().fit(raw, y_true)
    calibrated = calibrator.predict(raw)

    # Isotonic should recover most of the AUC/F1 lost to the dip; Platt (monotonic)
    # structurally cannot, since it can only apply a monotonic transform to an
    # already partially non-monotonic reliability curve.
    platt_calibrated = PlattCalibrator().fit(raw, y_true).predict(raw)
    assert competition_score_like(y_true, calibrated) > competition_score_like(
        y_true, platt_calibrated
    )


def test_isotonic_calibrator_output_is_bounded_zero_to_one():
    rng = np.random.RandomState(2)
    raw = rng.uniform(0, 1, 200)
    y_true = rng.randint(0, 2, 200)
    calibrated = IsotonicCalibrator().fit(raw, y_true).predict(raw)
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


# ---------------------------------------------------------------------------
# cv_calibrate
# ---------------------------------------------------------------------------


def test_cv_calibrate_never_fits_on_the_fold_it_calibrates():
    """A calibrator that just memorizes (a lookup-table stand-in with no real
    generalization) should NOT reproduce the training labels when applied to a
    fold it never saw fit on — proves cv_calibrate is actually excluding each
    fold's own rows from that fold's calibrator fit, not just running fit-predict
    on everything at once."""

    class MemorizingCalibrator:
        def fit(self, raw, y_true):
            self._lookup = dict(zip(raw, y_true, strict=True))
            return self

        def predict(self, raw):
            # Rows never seen during fit fall back to 0.5 (uninformative) —
            # if cv_calibrate leaked the held-out fold into fit, every row
            # would hit the lookup table and this fallback would never fire.
            return np.array([self._lookup.get(r, 0.5) for r in raw])

    rng = np.random.RandomState(3)
    n = 100
    raw = rng.uniform(0, 1, n)
    y_true = rng.randint(0, 2, n)
    fold_id = np.repeat(np.arange(5), n // 5)

    calibrated = cv_calibrate(y_true, raw, fold_id, lambda: MemorizingCalibrator())

    assert len(calibrated) == n
    assert np.any(calibrated == 0.5), (
        "every row hit the lookup table — cv_calibrate is leaking each fold's "
        "own rows into its own calibrator fit"
    )


def test_cv_calibrate_covers_every_row_exactly_once():
    rng = np.random.RandomState(4)
    n = 50
    raw = rng.uniform(0, 1, n)
    y_true = rng.randint(0, 2, n)
    fold_id = np.repeat(np.arange(5), n // 5)

    calibrated = cv_calibrate(y_true, raw, fold_id, lambda: PlattCalibrator())

    assert len(calibrated) == n
    assert not np.isnan(calibrated).any()


# ---------------------------------------------------------------------------
# compare_calibration_methods
# ---------------------------------------------------------------------------


def test_compare_calibration_methods_reports_all_three_rows():
    rng = np.random.RandomState(5)
    n = 500
    true_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < true_prob).astype(int)
    raw = np.clip(true_prob * 0.8 + 0.1, 0.001, 0.999)
    fold_id = np.repeat(np.arange(5), n // 5)

    results = compare_calibration_methods(y_true, raw, fold_id)

    assert set(results) == {"raw", "platt", "isotonic"}
    for row in results.values():
        assert set(row) == {"f1_at_0.5", "auc", "competition_score", "brier"}


def test_compare_calibration_methods_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compare_calibration_methods(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), np.array([0, 0]))
