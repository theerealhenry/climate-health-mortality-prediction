"""Stage 19.1 -- /predict endpoint tests.

Equivalence note: this does NOT compare the API's single-row output against
predict_test_set()'s full-batch submission. It can't -- SpatialClusterFeaturizer
(features/spatial.py) rebuilds each coordinate's climate-normal profile from
whichever rows share that call's own X, rather than from a row-independent
fitted statistic. Every coordinate in Test.csv recurs (verified: 0 of 1,030
rows have a unique lat/lon), so a row's spatial_cluster assignment -- and
everything downstream of it -- can legitimately differ between "this row
alone" and "this row as part of the 1,030-row batch". That's an existing,
already-shipped property of features/spatial.py, out of scope for Phase 7's
no-model/no-feature-change constraint, not a bug in this endpoint.

What IS tested: that the API's request-handling glue -- JSON parse, one-row
DataFrame construction, column alignment -- calls the identical predict.py /
features.pipeline building blocks (build_feature_matrix,
branch_a_native_categorical, build_submission_frame) with the same arguments
app/api/main.py uses, reproducing their result for that row in isolation. A
bug in main.py's glue (wrong feature_cols order, raw vs. calibrated
probability, lost precision in the JSON round-trip) would still fail this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# app/ sits at the repo root, outside the installed src/climate_health
# package -- add the repo root to sys.path so `from app.api import main`
# resolves the same way `uvicorn app.api.main:app` resolves it from cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import main as api_main  # noqa: E402
from climate_health.data.loaders import load_test_full  # noqa: E402
from climate_health.features.pipeline import (  # noqa: E402
    branch_a_native_categorical,
    build_feature_matrix,
)
from climate_health.models.baselines import TARGET_COL  # noqa: E402
from climate_health.models.predict import (  # noqa: E402
    build_submission_frame,
    fit_production_calibrator,
    retrain_champion_on_full_data,
)

client = TestClient(api_main.app)


@pytest.fixture(scope="module")
def champion_artifacts():
    model, fitted_state, feature_cols, _ = retrain_champion_on_full_data()
    calibrator = fit_production_calibrator()
    api_main._state.update(
        model=model, fitted_state=fitted_state, feature_cols=feature_cols, calibrator=calibrator
    )
    yield model, fitted_state, feature_cols, calibrator
    api_main._state.clear()


def _row_to_payload(row: pd.Series) -> dict:
    payload = {}
    for key, value in row.items():
        if key == "deathdate":
            payload[key] = value.date().isoformat()
        elif isinstance(value, np.integer):
            payload[key] = int(value)
        elif isinstance(value, np.floating):
            payload[key] = float(value)
        else:
            payload[key] = value
    return payload


def _predict_isolated_row_via_predict_py(
    model, calibrator, fitted_state, feature_cols, row: pd.DataFrame
) -> pd.DataFrame:
    """Same three predict.py/pipeline calls predict_test_set() and
    app/api/main.py's handler both make, run on a caller-supplied one-row
    frame -- predict_test_set() itself has no such parameter (it always
    loads the full Test.csv), so this reuses its constituent calls rather
    than comparing against its batch-only output (see module docstring)."""
    X, _ = build_feature_matrix(row, fit=False, fitted_state=fitted_state, target_col=TARGET_COL)
    branch_df = branch_a_native_categorical(X, target_col=TARGET_COL)
    combined = X.copy()
    for col in branch_df.columns:
        combined[col] = branch_df[col].values
    raw_probs = model.predict_proba(combined[feature_cols].values)[:, 1]
    calibrated = calibrator.predict(raw_probs)
    return build_submission_frame(list(combined["ID"]), calibrated)


def test_predict_matches_predict_py_for_the_same_row_in_isolation(champion_artifacts):
    model, fitted_state, feature_cols, calibrator = champion_artifacts
    sample_row = load_test_full().iloc[[0]]
    expected = _predict_isolated_row_via_predict_py(
        model, calibrator, fitted_state, feature_cols, sample_row
    )
    expected_row = expected.iloc[0]

    response = client.post("/predict", json=_row_to_payload(sample_row.iloc[0]))

    assert response.status_code == 200
    body = response.json()
    assert body["ID"] == expected_row["ID"]
    assert body["TargetF1"] == int(expected_row["TargetF1"])
    assert body["TargetRAUC"] == pytest.approx(float(expected_row["TargetRAUC"]), abs=1e-9)


def test_predict_rejects_missing_field():
    payload = _row_to_payload(load_test_full().iloc[0])
    del payload["age"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    payload = _row_to_payload(load_test_full().iloc[0])
    payload["age"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_out_of_range_category():
    payload = _row_to_payload(load_test_full().iloc[0])
    payload["zone"] = "Not_A_Real_Zone"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_health_endpoint_exists():
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()
