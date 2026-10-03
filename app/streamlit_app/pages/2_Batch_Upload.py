"""Stage 19.2 -- batch CSV upload flow.

Uploads a CSV shaped like climate_health.data.loaders.load_test_full()'s
output (Test.csv's columns + climate_features.csv's columns, merged on ID)
and calls the Task 19.1 /predict endpoint once per row. Correctness here is
inherited from /predict's own proven equivalence to predict.py's building
blocks (tests/test_api.py) -- this page is a thin loop over that endpoint,
not a second implementation of the prediction path.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import call_predict, load_records, render_disclaimer, row_to_payload

st.set_page_config(page_title="Batch Upload")
st.title("Batch Upload")
render_disclaimer()

st.write(
    "Upload a CSV with the same columns as `Test.csv` merged with "
    "`climate_features.csv` (one row per record). Each row is sent to the "
    "same `/predict` endpoint the Record Explorer page uses."
)

use_sample = st.checkbox("Use a small sample of the real Test.csv instead of uploading")
uploaded = None if use_sample else st.file_uploader("CSV file", type="csv")

df = None
if use_sample:
    n = st.slider("Sample size", min_value=1, max_value=20, value=5)
    df = load_records("Test").head(n).reset_index(drop=True)
elif uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["deathdate"])

if df is not None:
    st.write(f"{len(df)} row(s) loaded.")
    if st.button("Run batch prediction", type="primary"):
        results = []
        errors = []
        progress = st.progress(0.0)
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                results.append(call_predict(row_to_payload(row)))
            except RuntimeError as exc:
                errors.append(f"{row.get('ID', f'row {i}')}: {exc}")
            progress.progress((i + 1) / len(df))

        if errors:
            st.error(f"{len(errors)} row(s) failed:\n" + "\n".join(errors))

        if results:
            submission = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(submission, use_container_width=True)
            st.download_button(
                "Download as submission.csv",
                submission.to_csv(index=False),
                file_name="submission.csv",
                mime="text/csv",
            )
