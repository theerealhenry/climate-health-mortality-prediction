"""Stage 19.1 -- FastAPI inference service.

Wraps the exact Stage 15.3/17 champion path (climate_health.models.predict)
behind a single /predict endpoint. The champion model and calibrator are
loaded once at startup via retrain_champion_on_full_data() and
fit_production_calibrator() -- the same functions generate_submission()
calls -- and reused for every request. This module adds no prediction logic
of its own: build_feature_matrix, branch_a_native_categorical, and
build_submission_frame are the identical calls predict_test_set() makes,
just applied to one request row instead of the whole test set.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI

from app.api.schemas import PredictRequest, PredictResponse
from climate_health.features.pipeline import branch_a_native_categorical, build_feature_matrix
from climate_health.models.baselines import TARGET_COL
from climate_health.models.predict import (
    build_submission_frame,
    fit_production_calibrator,
    retrain_champion_on_full_data,
)

# Module-level, not app.state: lets tests inject the champion artifacts
# directly (bypassing a second, redundant retrain) by writing into this dict.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, fitted_state, feature_cols, _ = retrain_champion_on_full_data()
    _state["model"] = model
    _state["fitted_state"] = fitted_state
    _state["feature_cols"] = feature_cols
    _state["calibrator"] = fit_production_calibrator()
    yield
    _state.clear()


app = FastAPI(title="Climate Health Mortality Risk API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "model" in _state else "loading"}


# Note: this row's prediction is computed in isolation and is not guaranteed
# to bit-match a batch run (predict_test_set()) that also includes other
# rows at the same coordinate -- SpatialClusterFeaturizer (features/spatial.py)
# rebuilds each coordinate's climate-normal profile from whichever rows
# share the call, by design. See tests/test_api.py's module docstring for
# the full explanation.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    row = pd.DataFrame([request.model_dump()])
    row["deathdate"] = pd.to_datetime(row["deathdate"])

    X, _ = build_feature_matrix(
        row, fit=False, fitted_state=_state["fitted_state"], target_col=TARGET_COL
    )
    branch_df = branch_a_native_categorical(X, target_col=TARGET_COL)
    combined = X.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values

    feature_cols = _state["feature_cols"]
    raw_prob = _state["model"].predict_proba(combined[feature_cols].values)[:, 1]
    calibrated_prob = _state["calibrator"].predict(raw_prob)

    submission = build_submission_frame(list(combined["ID"]), calibrated_prob)
    result = submission.iloc[0]
    return PredictResponse(
        ID=result["ID"],
        TargetF1=int(result["TargetF1"]),
        TargetRAUC=float(result["TargetRAUC"]),
    )
