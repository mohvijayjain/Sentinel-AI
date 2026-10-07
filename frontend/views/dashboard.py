"""Dashboard: model health, current decision, recent real activity."""

import streamlit as st

from components.ui import (
    ACTION_BANDS, DECISION_COLORS, action_badge, badge, fetch, fmt_num,
    fmt_time, retraining_decision, show_error,
)


st.markdown("# 🛡 Sentinel-AI")
st.caption("Self-Healing ML Model Monitoring & Retraining Platform")
st.markdown(
    "<div class='sentinel-flow'>MONITOR → DETECT DRIFT → DECIDE → RETRAIN → "
    "TRAIN CHALLENGER → VALIDATE → PROMOTE / REJECT → EXPLAIN WITH AI</div>",
    unsafe_allow_html=True,
)
st.write("")


# ============================================================
# Model health  (GET /model/info)
# ============================================================

st.markdown("#### Model Health")

with st.spinner("Loading model info..."):
    model = fetch("get_model_info")

if model.ok:
    info = model.data
    metrics = info.get("metrics") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Serving model", info.get("model_version", "—"),
              help="model_version reported by the API (/model/info)")
    c2.metric("R²", fmt_num(metrics.get("r2"), 4))
    c3.metric("MAE", fmt_num(metrics.get("mae"), 1, " s"))
    c4.metric("RMSE", fmt_num(metrics.get("rmse"), 1, " s"))

    st.caption(
        f"{info.get('model_type', '')} · target {info.get('target', '')} · "
        f"trained on {info.get('trained_on', '')} · loaded {fmt_time(info.get('loaded_at'))}"
    )
else:
    show_error(model, "Model health")

st.write("")


# ============================================================
# Current system status  (GET /monitoring/latest)
# ============================================================

st.markdown("#### Current System Status")

latest = fetch("get_drift_status")

if not latest.ok:
    show_error(latest, "System status")
elif latest.data is None:
    st.info("No monitoring run has been recorded yet.")
else:
    run = latest.data
    left, right = st.columns([1, 2])

    with left:
        st.markdown(
            f"<div class='sentinel-card'>"
            f"<div class='sentinel-muted'>Decision · run #{run['id']}</div>"
            f"<div style='margin:10px 0 6px'>{action_badge(run['action'], '1.6rem')}</div>"
            f"<div class='sentinel-muted'>Overall drift score</div>"
            f"<div style='font-size:2rem;font-weight:700'>{run['overall_score']:.3f}</div>"
            f"<div class='sentinel-muted'>{fmt_time(run.get('run_time'))}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        s1, s2, s3 = st.columns(3)
        s1.metric("Statistical drift", f"{run['statistical_score']:.3f}",
                  help="PSI / KS / JS on input features")
        s2.metric("SHAP drift", f"{run['shap_score']:.3f}",
                  help="Shift in feature importance")
        s3.metric("Prediction drift", f"{run['prediction_score']:.3f}",
                  help="Shift in the model's output distribution")
        st.caption(f"Action bands on the overall score: {ACTION_BANDS}")

st.write("")


# ============================================================
# Recent activity  (monitoring history + retraining events)
# ============================================================

st.markdown("#### Recent Activity")

history = fetch("get_monitoring_history", limit=5)
events = fetch("get_retraining_history", limit=5)

activity = []

if history.ok:
    for run in history.data:
        activity.append((
            run.get("run_time"),
            f"{action_badge(run['action'], '.75rem')} &nbsp; Monitoring run #{run['id']} "
            f"· overall drift {run['overall_score']:.3f}",
        ))

if events.ok:
    for event in events.data:
        decision, _ = retraining_decision(event)
        color = DECISION_COLORS.get(decision, "#94a3b8")
        activity.append((
            event.get("triggered_at"),
            f"{badge(decision, color, '.75rem')} &nbsp; Retraining event #{event['id']} "
            f"· {event.get('triggered_reason') or 'reason not recorded'} · challenger RMSE "
            f"{fmt_num(event.get('new_model_rmse'), 1)} vs champion "
            f"{fmt_num(event.get('champion_rmse'), 1)}",
        ))

if activity:
    activity.sort(key=lambda item: str(item[0] or ""), reverse=True)
    for when, text in activity[:8]:
        st.markdown(
            f"<div style='padding:6px 0'><span class='sentinel-muted'>"
            f"{fmt_time(when)}</span> &nbsp; {text}</div>",
            unsafe_allow_html=True,
        )
elif history.ok and events.ok:
    st.info("No monitoring runs or retraining events recorded yet.")

if not history.ok:
    show_error(history, "Monitoring history")
if not events.ok:
    show_error(events, "Retraining events")
