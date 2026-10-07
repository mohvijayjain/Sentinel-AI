"""Drift monitoring: latest run, detector scores, feature PSI, history."""

import pandas as pd
import streamlit as st

from components.ui import (
    ACTION_BANDS, action_badge, fetch, fmt_time, page_header, show_error,
)


def psi_severity(psi) -> str:
    """Same bands as stastical_drift.get_psi_severity()."""
    if pd.isna(psi):
        return "UNKNOWN"
    for cutoff, label in ((0.10, "NO_DRIFT"), (0.20, "LOW"), (0.25, "MEDIUM"), (0.50, "HIGH")):
        if psi < cutoff:
            return label
    return "CRITICAL"


page_header(
    "Drift Monitoring",
    "Three detectors are combined into one weighted overall score "
    "(statistical 0.4 · SHAP 0.3 · prediction 0.3), which decides the action.",
)


# ============================================================
# Latest run  (GET /monitoring/latest)
# ============================================================

latest = fetch("get_drift_status")

if not latest.ok:
    show_error(latest, "Latest monitoring run")
elif latest.data is None:
    st.info("No monitoring run has been recorded yet.")
else:
    run = latest.data

    st.markdown(
        f"<div class='sentinel-card'>"
        f"{action_badge(run['action'], '1.3rem')} &nbsp; "
        f"<span style='font-size:1.6rem;font-weight:700'>{run['overall_score']:.3f}</span>"
        f" <span class='sentinel-muted'>overall drift · run #{run['id']} · "
        f"{fmt_time(run.get('run_time'))}</span>"
        f"<div class='sentinel-muted' style='margin-top:6px'>{ACTION_BANDS}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    d1.metric("Statistical drift", f"{run['statistical_score']:.3f}")
    d1.caption("PSI, KS and JS divergence per input feature vs the reference data.")
    d2.metric("SHAP drift", f"{run['shap_score']:.3f}")
    d2.caption("Change in the model's feature importance (SHAP) on current data.")
    d3.metric("Prediction drift", f"{run['prediction_score']:.3f}")
    d3.caption("Change in the distribution of the model's predictions.")

    drifted = [f for f in (run.get("drifted_features") or "").split(",") if f]
    st.caption(
        "Drifted features in this run: " + (", ".join(drifted) if drifted else "none")
    )

st.write("")


# ============================================================
# Feature drift  (GET /monitoring/feature-scores)
# ============================================================

st.markdown("#### Feature Drift (PSI)")

scores = fetch("get_feature_drift")

if not scores.ok:
    show_error(scores, "Feature drift")
elif not scores.data:
    st.info("No feature drift scores recorded yet.")
else:
    frame = pd.DataFrame(scores.data)
    # The endpoint returns every run's scores: keep each feature's latest
    frame["run_date"] = pd.to_datetime(frame["run_date"])
    latest_scores = (
        frame.sort_values("run_date")
        .groupby("feature_name", as_index=False)
        .last()
        .sort_values("psi_score", ascending=False)
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": latest_scores["feature_name"],
            "PSI": latest_scores["psi_score"].round(4),
            "Severity": latest_scores["psi_score"].map(psi_severity),
            "Drifted": latest_scores["is_drifted"].map({True: "YES", False: "no"}),
            "Last scored": latest_scores["run_date"].dt.strftime("%Y-%m-%d %H:%M"),
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Latest stored PSI per feature. Severity uses the detector's PSI bands "
        "(NO_DRIFT < 0.10 · LOW < 0.20 · MEDIUM < 0.25 · HIGH < 0.50 · CRITICAL); "
        "Drifted is the stored flag (any severity above NO_DRIFT)."
    )

st.write("")


# ============================================================
# Drift over time  (GET /monitoring/history)
# ============================================================

st.markdown("#### Drift Over Time")

history = fetch("get_monitoring_history", limit=50)

if not history.ok:
    show_error(history, "Monitoring history")
elif len(history.data) < 2:
    st.info("At least two monitoring runs are needed for a trend.")
else:
    trend = pd.DataFrame(history.data)
    trend["run_time"] = pd.to_datetime(trend["run_time"])
    trend = trend.sort_values("run_time").set_index("run_time")

    st.line_chart(
        trend[["overall_score", "statistical_score", "shap_score", "prediction_score"]],
        height=280,
    )

    with st.expander(f"All {len(trend)} runs"):
        st.dataframe(
            trend.reset_index()[[
                "id", "run_time", "action", "overall_score",
                "statistical_score", "shap_score", "prediction_score",
            ]].sort_values("id", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
