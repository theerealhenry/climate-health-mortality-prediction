"""Stage 19.2 -- record-selection + demographic-perturbation flow.

Picks a real Train.csv/Test.csv row (real location, date, real climate
values) and lets the person change only the demographic fields (age,
gender, zone) -- a genuine "what if this demographic profile had this
climate exposure" question, not a fabricated record.
"""

from __future__ import annotations

import streamlit as st
from common import call_predict, load_records, render_disclaimer, row_to_payload

st.set_page_config(page_title="Record Explorer")
st.title("Record Explorer")
render_disclaimer()

dataset = st.radio(
    "Dataset",
    ["Test", "Train"],
    horizontal=True,
    help="Train.csv rows also carry the historical label; Test.csv rows do not.",
)
records = load_records(dataset)

labels = {
    row["ID"]: f"{row['ID']} \u2014 {row['location']} \u2014 {row['deathdate'].date()}"
    for _, row in records.iterrows()
}
selected_id = st.selectbox("Record", options=list(labels), format_func=lambda i: labels[i])
record = records.loc[records["ID"] == selected_id].iloc[0]

st.subheader("Fixed fields (from the real record)")
fixed_cols = st.columns(4)
fixed_cols[0].metric("Location", record["location"])
fixed_cols[1].metric("Death date", str(record["deathdate"].date()))
fixed_cols[2].metric("Avg. temperature (C)", f"{record['avg_temperature']:.1f}")
fixed_cols[3].metric("90-day rainfall (mm)", f"{record['rain_sum_90d']:.0f}")

if dataset == "Train":
    st.caption(
        f"Historical label (is_climate_sensitive): **{int(record['is_climate_sensitive'])}**"
    )

st.subheader("Demographic what-if")
col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", min_value=0.0, max_value=120.0, value=float(record["age"]), step=1.0)
gender = col2.selectbox(
    "Gender", ["Male", "Female"], index=["Male", "Female"].index(record["gender"])
)
zone = col3.selectbox(
    "Zone", ["Rural", "Peri_urban"], index=["Rural", "Peri_urban"].index(record["zone"])
)

changed = []
if age != record["age"]:
    changed.append(f"age {record['age']:.0f} -> {age:.0f}")
if gender != record["gender"]:
    changed.append(f"gender {record['gender']} -> {gender}")
if zone != record["zone"]:
    changed.append(f"zone {record['zone']} -> {zone}")
if changed:
    st.caption("Changed from the real record: " + ", ".join(changed))

if st.button("Run prediction", type="primary"):
    perturbed = record.copy()
    perturbed["age"] = age
    perturbed["gender"] = gender
    perturbed["zone"] = zone
    payload = row_to_payload(perturbed)

    try:
        with st.spinner("Calling the prediction API..."):
            result = call_predict(payload)
    except RuntimeError as exc:
        st.error(str(exc))
    else:
        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Climate-sensitive (TargetF1)", "Yes" if result["TargetF1"] else "No")
        m2.metric("Predicted probability (TargetRAUC)", f"{result['TargetRAUC']:.3f}")
