"""
Sentinel-AI monitoring UI.

    streamlit run frontend/app.py

Streamlit -> HTTP -> the running Sentinel-AI FastAPI service. Every data
call sends X-API-Key from SENTINEL_API_KEY (server-side only).
"""

import streamlit as st

import config
from components.ui import clear_cache, fetch


st.set_page_config(
    page_title="Sentinel-AI",
    page_icon="🛡",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1300px; }
      div[data-testid="stMetric"] {
          background: #111827; border: 1px solid #1f2937;
          border-radius: 10px; padding: 14px 16px;
      }
      .sentinel-badge {
          display: inline-block; padding: 4px 14px; border-radius: 999px;
          font-weight: 700; letter-spacing: .04em;
      }
      .sentinel-card {
          background: #111827; border: 1px solid #1f2937;
          border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;
      }
      .sentinel-muted { color: #94a3b8; font-size: .9rem; }
      .sentinel-flow { color: #94a3b8; font-size: .85rem; letter-spacing: .03em; }
    </style>
    """,
    unsafe_allow_html=True,
)


pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("views/monitoring.py", title="Monitoring", icon="📈"),
    st.Page("views/run_monitoring.py", title="Run Monitoring", icon="📤"),
    st.Page("views/retraining.py", title="Retraining", icon="🔁"),
    st.Page("views/predictions.py", title="Predictions", icon="🎯"),
    st.Page("views/assistant.py", title="AI Assistant", icon="🤖"),
]

navigation = st.navigation(pages, position="hidden")


with st.sidebar:

    st.markdown("### 🛡 SENTINEL-AI")
    st.markdown(
        "<span style='color:#22c55e'>●</span> Production",
        unsafe_allow_html=True,
    )
    st.write("")

    for page in pages:
        st.page_link(page)

    st.write("")
    if st.button("↻ Refresh data", use_container_width=True):
        clear_cache()
        st.rerun()

    st.divider()

    health = fetch("get_health")
    if health.ok:
        st.markdown(
            "Backend: <span style='color:#22c55e'>● Connected</span>",
            unsafe_allow_html=True,
        )
        if not health.data.get("model_loaded"):
            st.caption("Model not loaded")
    else:
        st.markdown(
            "Backend: <span style='color:#ef4444'>● Offline</span>",
            unsafe_allow_html=True,
        )

    st.caption(f"API: `{config.API_URL}`")
    if not config.API_KEY:
        st.caption("⚠ SENTINEL_API_KEY not set")


if not config.API_KEY:
    st.warning(
        "Set `SENTINEL_API_KEY` to connect: every data endpoint requires the "
        "API's X-API-Key. See `frontend/.env.example`."
    )

navigation.run()
