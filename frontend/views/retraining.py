"""Retraining: latest Champion vs Challenger decision, gates, history."""

import pandas as pd
import streamlit as st

from components.ui import (
    DECISION_COLORS, badge, fetch, fmt_num, fmt_time, page_header,
    retraining_decision, show_error,
)


GATES = [
    ("Frozen test metrics", "Challenger vs Champion RMSE, MAE and R² on the frozen hold-out test set."),
    ("Bootstrap", "Paired bootstrap (1,000 resamples): the RMSE difference must be statistically significant."),
    ("Segment", "Per-segment error (busiest pickup zones, JFK / LaGuardia / Newark) must not degrade beyond 10% and 20 s."),
    ("Recent data", "Challenger error on the most recent labelled month vs its own frozen-test error."),
]


page_header(
    "Retraining",
    "When drift reaches RETRAIN (or retraining is triggered manually), a Challenger "
    "is trained and must pass every validation gate before it replaces the Champion.",
)

events = fetch("get_retraining_history", limit=20)

if not events.ok:
    show_error(events, "Retraining events")
    st.stop()

if not events.data:
    st.info("No retraining event has been recorded yet. Events appear here after "
            "the first retraining attempt.")
    st.stop()


event = events.data[0]
decision, decision_source = retraining_decision(event)
color = DECISION_COLORS.get(decision, "#94a3b8")


# ============================================================
# Final decision + trigger
# ============================================================

st.markdown(f"#### Latest Retraining Event · #{event['id']}")

left, right = st.columns([1, 2])

with left:
    st.markdown(
        f"<div class='sentinel-card'>"
        f"<div class='sentinel-muted'>Final decision</div>"
        f"<div style='margin:10px 0'>{badge(decision, color, '1.6rem')}</div>"
        f"<div class='sentinel-muted'>{fmt_time(event.get('triggered_at'))} UTC</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if decision_source == "promoted":
        st.caption("Derived from the `promoted` flag: this database has no "
                   "`status` column yet (migration 001 not applied).")

with right:
    st.markdown(
        f"<div class='sentinel-card'>"
        f"<div class='sentinel-muted'>Trigger</div>"
        f"<div style='font-size:1.15rem;font-weight:600'>"
        f"{event.get('triggered_reason') or 'not recorded'}</div>"
        f"<div class='sentinel-muted' style='margin-top:8px'>MLflow run</div>"
        f"<code>{event.get('mlflow_run_id') or 'not recorded'}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if event.get("error_message"):
        st.error(f"Failure: {event['error_message']}")


# ============================================================
# Champion vs Challenger
# ============================================================

st.markdown("#### Champion vs Challenger")

champion_rmse = event.get("champion_rmse")
challenger_rmse = event.get("new_model_rmse")

c1, c2, c3 = st.columns(3)
c1.metric("Champion RMSE", fmt_num(champion_rmse, 2, " s"))
if champion_rmse is not None and challenger_rmse is not None:
    c2.metric(
        "Challenger RMSE",
        fmt_num(challenger_rmse, 2, " s"),
        delta=f"{challenger_rmse - champion_rmse:+.2f} s vs Champion",
        delta_color="inverse",          # lower RMSE is better
    )
else:
    c2.metric("Challenger RMSE", fmt_num(challenger_rmse, 2, " s"))
c3.metric("Promoted", {True: "Yes", False: "No"}.get(event.get("promoted"), "—"))

st.caption("The retraining record stores RMSE for both models; MAE and R² are "
           "evaluated during promotion but not stored with the event.")


# ============================================================
# Validation gates
# ============================================================

st.markdown("#### Validation Gates")

st.dataframe(
    pd.DataFrame(
        [(name, "not recorded", why) for name, why in GATES],
        columns=["Gate", "Result", "What it checks"],
    ),
    hide_index=True,
    use_container_width=True,
)

note = ("Per-gate PASS/FAIL is computed during promotion but only written to the "
        "promotion log, not to the database, so the API cannot show it.")
if decision == "REJECTED":
    note += " A REJECTED decision means at least one gate failed."
elif decision == "PROMOTED":
    note += " A PROMOTED decision means every gate passed."
st.info(note)


# ============================================================
# History
# ============================================================

if len(events.data) > 1:
    st.markdown("#### All Retraining Events")

st.dataframe(
    pd.DataFrame([
        {
            "Event": e["id"],
            "Triggered (UTC)": fmt_time(e.get("triggered_at")),
            "Reason": e.get("triggered_reason"),
            "Champion RMSE": e.get("champion_rmse"),
            "Challenger RMSE": e.get("new_model_rmse"),
            "Decision": retraining_decision(e)[0],
        }
        for e in events.data
    ]),
    hide_index=True,
    use_container_width=True,
)
