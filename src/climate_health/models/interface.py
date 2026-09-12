"""Stage 11.1 — the shared training interface every model-zoo candidate satisfies.

LightGBM/XGBoost/CatBoost/HistGradientBoosting/ExtraTrees/LogisticRegression already
implement the same shape (.fit(X, y) / .predict_proba(X)) via sklearn's estimator
API — exactly what evaluate_all_tiers's `make_estimator` parameter already requires
(see evaluation/cv.py). There's nothing to wrap: the interface is "a zero-argument
callable returning a fresh, unfitted object with those two methods," already
satisfied by every factory in Stage 10.3's BASELINE_MODELS. This module just names
the contract and gives Stage 11.3 one place to check a new candidate against it
before registering it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Estimator(Protocol):
    def fit(self, X, y): ...
    def predict_proba(self, X): ...


EstimatorFactory = Callable[[], Estimator]


def assert_valid_estimator_factory(factory: EstimatorFactory, name: str) -> None:
    """Fails loudly at registration time, not mid-CV-loop, if a candidate doesn't
    match the contract evaluate_all_tiers assumes."""
    est = factory()
    if not (hasattr(est, "fit") and hasattr(est, "predict_proba")):
        raise TypeError(
            f"{name}: estimator factory must return an object with fit()/predict_proba() "
            f"— got {type(est).__name__}"
        )
