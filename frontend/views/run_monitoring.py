"""Run New Monitoring: upload a production dataset, show the API's result."""

import streamlit as st

import api_client
from components.ui import (
    ACTION_BANDS, action_badge, clear_cache, page_header, show_error,
)


page_header(
    "Run New Monitoring",
    "Upload a production dataset (raw NYC TLC trip records). The API "
    "preprocesses it, compares it with the reference data using the "
    "statistical, SHAP and prediction detectors, scores it and saves it "
    "as a new monitoring run. Each upload is one independent run.",
)

uploaded = st.file_uploader("Dataset (.csv or .parquet)", type=["csv", "parquet"])
period = st.text_input("Period / label (optional)", placeholder="e.g. June 2026")

if st.button("Run Monitoring", type="primary", disabled=uploaded is None):

    with st.spinner(
        f"Running monitoring on {uploaded.name}: preprocessing, drift "
        "detection against the reference and scoring. Large files can "
        "take several minutes..."
    ):
        result = api_client.run_monitoring(
            uploaded.name, uploaded.getvalue(), period.strip()
        )

    if not result.ok:
        if result.error == api_client.TIMEOUT:
            st.warning(
                "No response before the client timeout. The run may still "
                "finish on the server: check the Monitoring page shortly."
            )
        else:
            show_error(result, "Monitoring run")
        st.stop()

    clear_cache()          # the new run should show up on the other pages
    run = result.data

    st.success(f"Monitoring completed: run #{run['run_id']}")

    st.markdown(
        f"<div class='sentinel-card'>"
        f"{action_badge(run['action'], '1.4rem')} &nbsp; "
        f"<span style='font-size:1.6rem;font-weight:700'>{run['overall_score']:.3f}</span>"
        f" <span class='sentinel-muted'>overall drift</span>"
        f"<div class='sentinel-muted' style='margin-top:6px'>{ACTION_BANDS}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Statistical drift", f"{run['statistical_score']:.3f}")
    c2.metric("SHAP drift", f"{run['shap_score']:.3f}")
    c3.metric("Prediction drift", f"{run['prediction_score']:.3f}")

    reference = run.get("reference") or {}
    st.markdown(
        f"- **File:** `{run['file_name']}`"
        + (f" ({run['label']})" if run.get("label") else "")
        + f"\n- **Rows:** {run['rows_uploaded']:,} uploaded, "
        f"{run['rows_processed']:,} after preprocessing"
        f"\n- **Reference:** `{reference.get('path', '—')}` "
        f"({reference.get('rows', 0):,} rows)"
        f"\n- **Drifted features:** "
        + (", ".join(run["drifted_features"]) or "none")
    )

    retraining = run.get("retraining")
    if retraining == "started":
        st.warning(
            "**RETRAIN decided.** The existing retraining pipeline was started "
            "in the background (training + validation gates take several "
            "minutes). The outcome will appear on the Retraining page."
        )
    elif retraining == "already_running":
        st.warning(
            "**RETRAIN recorded.** A retraining run is already in progress, "
            "so no new one was started."
        )

    if not run.get("indexed_for_assistant", True):
        st.caption("The run is saved, but could not be indexed for the AI "
                   "Assistant (it can be re-indexed later).")
