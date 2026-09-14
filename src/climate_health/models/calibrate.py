"""Stage 15.2 — Platt scaling vs. isotonic regression, compared against raw
(uncalibrated) predictions, per Stage 15.1's finding that `catboost_tuned`'s OOF
predictions are miscalibrated near p=0.5 (the fixed, un-tunable competition
threshold) and non-monotonically miscalibrated in the 0.5-0.7 range.

Both calibrators share a `fit(raw, y_true) -> self` / `predict(raw) -> calibrated`
contract, matching sklearn's own estimator shape — Platt scaling is literally
logistic regression on the raw probability as a single feature; isotonic is
sklearn's IsotonicRegression, which (unlike Platt) can fit a non-monotonic
reliability curve back onto the diagonal since it isn't limited to one monotonic
sigmoid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


class PlattCalibrator:
    """Logistic regression on the raw probability as its single feature — the
    standard "Platt scaling" recalibration, monotonic by construction."""

    def fit(self, raw: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        self._model = LogisticRegression().fit(np.asarray(raw).reshape(-1, 1), y_true)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """sklearn's IsotonicRegression — a free-form non-decreasing fit, able to
    correct a non-monotonic reliability curve that Platt's single sigmoid cannot."""

    def fit(self, raw: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            raw, y_true
        )
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._model.predict(raw)


def cv_calibrate(
    y_true: np.ndarray,
    raw: np.ndarray,
    fold_id: np.ndarray,
    calibrator_factory: Callable[[], object],
) -> np.ndarray:
    """Fits a fresh calibrator per fold on every OTHER fold's (raw, y_true) pairs
    and predicts that fold's own rows — so no row is ever calibrated by a model
    that saw its own label, the same OOF discipline the base model's own
    predictions were built under. `fold_id` reuses the existing 5-fold assignment
    the champion's OOF predictions were generated on (Stage 14.1's
    notebooks/artifacts/stage14_1_oof_predictions.npz)."""
    calibrated = np.full(len(y_true), np.nan)
    for fold in np.unique(fold_id):
        val_mask = fold_id == fold
        train_mask = ~val_mask
        calibrator = calibrator_factory().fit(raw[train_mask], y_true[train_mask])
        calibrated[val_mask] = calibrator.predict(raw[val_mask])
    return calibrated


def compare_calibration_methods(
    y_true: np.ndarray, raw: np.ndarray, fold_id: np.ndarray
) -> dict[str, dict[str, float]]:
    """Raw vs. Platt vs. isotonic, each scored on F1@0.5, AUC, the competition
    score, and Brier score — Platt/isotonic scored via `cv_calibrate` (never
    fit and evaluated on the same rows), raw scored as-is since there's nothing
    to fit. Returns whichever row scores best on `competition_score`, not
    whichever was tried first — per the blueprint's explicit instruction not to
    default to isotonic just because the miscalibration is non-monotonic."""
    y_true, raw, fold_id = np.asarray(y_true), np.asarray(raw), np.asarray(fold_id)
    if not (len(y_true) == len(raw) == len(fold_id)):
        raise ValueError(
            "compare_calibration_methods: y_true, raw, and fold_id must be the same length"
        )

    candidates = {
        "raw": raw,
        "platt": cv_calibrate(y_true, raw, fold_id, PlattCalibrator),
        "isotonic": cv_calibrate(y_true, raw, fold_id, IsotonicCalibrator),
    }
    results = {}
    for name, probs in candidates.items():
        f1 = f1_score(y_true, (probs >= 0.5).astype(int))
        auc = roc_auc_score(y_true, probs)
        results[name] = {
            "f1_at_0.5": f1,
            "auc": auc,
            "competition_score": 0.60 * f1 + 0.40 * auc,
            "brier": brier_score_loss(y_true, probs),
        }
    return results
