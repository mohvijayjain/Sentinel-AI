"""
Shared rendering helpers: cached API access, error states, status badges.
"""

from datetime import datetime

import streamlit as st

import api_client
import config


# Mirrors drift_scorer.get_action(): overall score bands
ACTION_COLORS = {
    "WAIT": "#22c55e",
    "MONITOR": "#3b82f6",
    "ALERT": "#f59e0b",
    "RETRAIN": "#ef4444",
}

ACTION_BANDS = "WAIT < 0.25 · MONITOR < 0.50 · ALERT < 0.75 · RETRAIN ≥ 0.75"

DECISION_COLORS = {
    "PROMOTED": "#22c55e",
    "REJECTED": "#ef4444",
    "FAILED": "#f59e0b",
}


# ============================================================
# Cached API access (one call per endpoint per TTL)
# ============================================================

@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def fetch(name: str, **kwargs) -> api_client.ApiResult:
    """api_client.<name>(**kwargs), cached. Never raises."""
    return getattr(api_client, name)(**kwargs)


def clear_cache():
    fetch.clear()


# ============================================================
# Error / empty states
# ============================================================

def show_error(result: api_client.ApiResult, what: str):
    """One clear message per failure kind, scoped to one card."""

    if result.error == api_client.NO_KEY:
        st.warning(f"**{what}:** set `SENTINEL_API_KEY` to connect.")
    elif result.error == api_client.UNAUTHORIZED:
        st.error(f"**{what}:** not authenticated, check `SENTINEL_API_KEY`.")
    elif result.error == api_client.OFFLINE:
        st.error(f"**{what}:** backend offline at `{config.API_URL}`.")
    elif result.error == api_client.TIMEOUT:
        st.warning(f"**{what}:** the API did not answer in time. Try again.")
    elif result.error == api_client.REJECTED:
        st.error(f"**{what}:** {result.detail}")
    elif result.error == api_client.NOT_DEPLOYED:
        st.warning(
            f"**{what}:** `{result.detail}` is not available on the running API. "
            "Rebuild the API container to deploy it."
        )
    else:
        st.error(f"**{what}:** request failed ({result.detail or 'unknown error'}).")


# ============================================================
# Formatting
# ============================================================

def fmt_time(value) -> str:
    """ISO timestamp from the API -> 'YYYY-MM-DD HH:MM'."""
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value)


def fmt_num(value, digits=3, suffix="") -> str:
    if value is None:
        return "not recorded"
    return f"{float(value):,.{digits}f}{suffix}"


def badge(label: str, color: str, size: str = "1rem") -> str:
    return (
        f"<span class='sentinel-badge' style='background:{color}22;"
        f"color:{color};border:1px solid {color};font-size:{size}'>{label}</span>"
    )


def action_badge(action: str, size: str = "1rem") -> str:
    return badge(action, ACTION_COLORS.get(action, "#94a3b8"), size)


def page_header(title: str, caption: str):
    st.markdown(f"## {title}")
    st.caption(caption)


# ============================================================
# Retraining decision (read defensively: status may not exist)
# ============================================================

def retraining_decision(event: dict):
    """
    (label, source) for one retraining_events row.

    status is the source of truth when present. Databases without
    migration 001 have no status column: there, promoted True/False is
    PROMOTED/REJECTED, exactly as the migration labels those rows.
    """

    status = event.get("status")
    if status:
        return status.upper(), "status"

    promoted = event.get("promoted")
    if promoted is True:
        return "PROMOTED", "promoted"
    if promoted is False:
        return "REJECTED", "promoted"

    return "UNKNOWN", None
