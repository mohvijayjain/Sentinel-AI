"""
Frontend settings, read server-side from the environment.

SENTINEL_API_URL  base URL of the running Sentinel-AI FastAPI service.
SENTINEL_API_KEY  the API's X-API-Key. Required for every data endpoint;
                  it stays in this Streamlit process and is never rendered.

Locally: environment or frontend/.env (see frontend/.env.example).
Streamlit Cloud: the app's Secrets (Settings -> Secrets), same names.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _setting(name):
    """Environment / frontend/.env first; then Streamlit secrets
    (Streamlit Cloud: Manage app -> Settings -> Secrets)."""

    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:          # no secrets configured (local runs)
        return None


API_URL = (_setting("SENTINEL_API_URL") or "http://localhost:8000").strip().rstrip("/")

API_KEY = (_setting("SENTINEL_API_KEY") or "").strip() or None

# Most endpoints are quick database reads. /chat is bounded at ~52s on the
# server (NVIDIA timeout + one retry), so the client waits a little longer.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("SENTINEL_API_TIMEOUT", "10"))
CHAT_TIMEOUT_SECONDS = float(os.getenv("SENTINEL_CHAT_TIMEOUT", "65"))

# An uploaded monitoring run preprocesses the file and compares it with the
# full reference (millions of rows) synchronously: allow several minutes.
RUN_TIMEOUT_SECONDS = float(os.getenv("SENTINEL_RUN_TIMEOUT", "900"))

# Seconds an API response is reused across reruns / page switches
CACHE_TTL_SECONDS = 30
