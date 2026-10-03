"""Stage 19.2 -- Streamlit demo entry point (landing page).

Calls the Task 19.1 FastAPI service for every prediction -- see
app/streamlit_app/common.py and app/api/main.py's module docstrings for why
(one source of truth for inference).
"""

from __future__ import annotations

import streamlit as st
from common import render_disclaimer

st.set_page_config(page_title="Climate-Sensitive Mortality Risk Demo")
st.title("Climate-Sensitive Mortality Risk Demo")
render_disclaimer()

st.write(
    "This demo serves the Phase 7 production build of the Climate & Health "
    "Risk Prediction competition pipeline. It calls the same FastAPI "
    "`/predict` endpoint (Task 19.1) that wraps the champion model trained "
    "in earlier phases -- nothing here re-implements the model."
)

st.subheader("Pages")
st.markdown(
    "- **Record Explorer** -- pick a real Train.csv/Test.csv record and "
    "change its demographic fields (age, gender, zone) as a what-if, "
    "keeping its real location, date, and climate values fixed.\n"
    "- **Batch Upload** -- upload a CSV shaped like the competition's "
    "Test.csv (merged with climate_features.csv) and get predictions for "
    "every row, downloadable as a submission-shaped CSV."
)
st.caption("Use the sidebar to open either page.")
