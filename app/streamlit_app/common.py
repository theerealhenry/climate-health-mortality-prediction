"""Stage 19.2 -- shared helpers for the Streamlit demo: the exact blueprint
disclaimer, the FastAPI client, and cached sample-record loading. Every
prediction goes through the Task 19.1 /predict endpoint (never
climate_health.models directly), so the API stays the single source of
truth for inference -- see app/api/main.py's module docstring and
tests/test_api.py for why a single-row prediction isn't guaranteed to
bit-match a full-batch submission run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from climate_health.data.loaders import load_test_full, load_train_full

API_BASE_URL = os.environ.get("CLIMATE_HEALTH_API_URL", "http://127.0.0.1:8000")

# Exact wording from docs/PROJECT_BLUEPRINT.md -- required verbatim on every page.
DISCLAIMER = (
    "This model estimates whether a mortality record is likely to belong to "
    "the climate-sensitive category defined by the dataset. It is a "
    "research/ decision-support demonstration, not a clinical diagnostic or "
    "individual mortality-risk tool."
)


def render_disclaimer() -> None:
    st.warning(DISCLAIMER)


def row_to_payload(row: pd.Series) -> dict:
    """JSON-safe dict for POST /predict -- numpy scalar types aren't JSON
    serializable, and deathdate needs ISO-date-string form."""
    payload = {}
    for key, value in row.items():
        if key == "deathdate":
            payload[key] = pd.Timestamp(value).date().isoformat()
        elif isinstance(value, np.integer):
            payload[key] = int(value)
        elif isinstance(value, np.floating):
            payload[key] = float(value)
        else:
            payload[key] = value
    payload.pop("is_climate_sensitive", None)  # Train rows only; /predict doesn't accept it
    return payload


def call_predict(payload: dict) -> dict:
    """POSTs to the Task 19.1 /predict endpoint. Raises RuntimeError with a
    readable message on any failure -- the caller shows it with st.error
    rather than letting a raw exception crash the page."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach the prediction API at {API_BASE_URL}. "
            "Is it running? (uvicorn app.api.main:app)"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(f"Prediction API returned {response.status_code}: {response.text}")
    return response.json()


@st.cache_data(show_spinner=False)
def load_records(dataset: str) -> pd.DataFrame:
    """Cached Train.csv/Test.csv (+ climate_features.csv) load -- the record
    picker's source of real, schema-valid rows to select and perturb."""
    return load_train_full() if dataset == "Train" else load_test_full()
