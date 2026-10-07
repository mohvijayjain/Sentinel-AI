"""Prediction monitoring: recent rows from prediction_logs."""

import pandas as pd
import streamlit as st

from components.ui import fetch, page_header, show_error


page_header(
    "Prediction Monitoring",
    "Every successful /predict call is logged after the response is sent "
    "(inputs, engineered features, prediction, latency).",
)

limit = st.selectbox("Rows", [25, 50, 100, 200], index=1)

logs = fetch("get_prediction_logs", limit=limit)

if not logs.ok:
    show_error(logs, "Prediction logs")
    st.stop()

rows = logs.data

if not rows:
    st.info("No predictions logged yet. Send a request to POST /predict to see it here.")
    st.stop()

frame = pd.DataFrame(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Predictions shown", len(frame))
c2.metric("Mean prediction", f"{frame['prediction_minutes'].mean():.1f} min")
c3.metric("Mean latency", f"{frame['latency_ms'].mean():.1f} ms")

st.dataframe(
    pd.DataFrame({
        "Timestamp (UTC)": pd.to_datetime(frame["predicted_at"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Trip distance (mi)": frame["trip_distance"],
        "Pickup hour": frame["pickup_hour"],
        "Prediction (s)": frame["prediction_seconds"],
        "Prediction (min)": frame["prediction_minutes"],
        "Model version": frame["model_version"],
        "Latency (ms)": frame["latency_ms"],
    }),
    hide_index=True,
    use_container_width=True,
)

if len(frame) <= 1:
    st.caption("Only one prediction logged so far: run a prediction to see more.")
