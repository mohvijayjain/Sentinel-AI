"""Run New Monitoring: upload a production dataset, show the API's result."""

import threading

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

JOB_KEY = "monitoring_job"        # the run in progress (this browser session)
RESULT_KEY = "monitoring_result"  # the last finished run, kept across reruns


def _start(name: str, content: bytes, label: str):
    """POST the upload from a background thread so the page stays usable."""

    box = {}

    def call():
        box["result"] = api_client.run_monitoring(name, content, label)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    st.session_state[JOB_KEY] = {
        "thread": thread, "box": box, "file": name, "stopping": False,
    }
    st.session_state.pop(RESULT_KEY, None)


@st.fragment(run_every=2)
def _progress():
    job = st.session_state.get(JOB_KEY)
    if job is None:
        return

    if not job["thread"].is_alive():
        # Finished: hand the result to the full page
        st.session_state[RESULT_KEY] = job["box"].get("result")
        st.session_state.pop(JOB_KEY, None)
        st.rerun()

    if job["stopping"]:
        st.info(f"Stopping the run on {job['file']}: it stops at the next "
                "stage, before anything is saved...")
    else:
        st.info(f"Running monitoring on {job['file']}: preprocessing, drift "
                "detection against the reference and scoring. Large files "
                "can take several minutes...")

    if st.button("Stop", icon="⏹️", disabled=job["stopping"]):
        stop = api_client.stop_monitoring()
        if stop.ok or stop.error == api_client.REJECTED:
            # REJECTED = nothing running any more: the run just finished
            job["stopping"] = True
        else:
            show_error(stop, "Stop")
        st.rerun(scope="fragment")


running = JOB_KEY in st.session_state

if st.button("Run Monitoring", type="primary",
             disabled=uploaded is None or running):
    _start(uploaded.name, uploaded.getvalue(), period.strip())
    st.rerun()

if running:
    _progress()
    st.stop()

result = st.session_state.get(RESULT_KEY)
if result is None:
    st.stop()

if not result.ok:
    if result.error == api_client.TIMEOUT:
        st.warning(
            "No response before the client timeout. The run may still "
            "finish on the server: check the Monitoring page shortly."
        )
    else:
        show_error(result, "Monitoring run")
    st.stop()

run = result.data

if run.get("status") == "stopped":
    st.warning(run.get("message") or "Monitoring run was stopped.")
    st.stop()

if not st.session_state.get("monitoring_cache_cleared") == run.get("run_id"):
    clear_cache()          # the new run should show up on the other pages
    st.session_state["monitoring_cache_cleared"] = run.get("run_id")


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
