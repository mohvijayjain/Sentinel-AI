"""
The only module that talks to the Sentinel-AI API.

Every call returns an ApiResult: either data, or a typed error the pages
render (offline, not authenticated, endpoint missing, timeout, ...).
Nothing here raises into the UI, and nothing here invents data.
"""

from dataclasses import dataclass
from typing import Any, Optional

import requests

import config


# Error kinds the pages distinguish
NO_KEY = "no_key"
OFFLINE = "offline"
UNAUTHORIZED = "unauthorized"
NOT_DEPLOYED = "not_deployed"
TIMEOUT = "timeout"
SERVER_ERROR = "server_error"
REJECTED = "rejected"          # 4xx: the API refused the input (detail says why)


@dataclass(frozen=True)
class ApiResult:
    data: Any = None
    error: Optional[str] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def _request(method, path, *, auth=True, timeout=None, **kwargs) -> ApiResult:

    headers = {}

    if auth:
        if not config.API_KEY:
            return ApiResult(error=NO_KEY)
        headers["X-API-Key"] = config.API_KEY

    try:
        response = requests.request(
            method,
            f"{config.API_URL}{path}",
            headers=headers,
            timeout=timeout or config.REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.Timeout:
        return ApiResult(error=TIMEOUT)
    except requests.RequestException:
        return ApiResult(error=OFFLINE)

    if response.status_code == 401:
        return ApiResult(error=UNAUTHORIZED)

    if response.status_code == 404:
        return ApiResult(error=NOT_DEPLOYED, detail=path)

    if response.status_code >= 400:
        # The API's own error detail (already safe for clients), if any
        try:
            detail = str(response.json().get("detail", ""))
        except (ValueError, AttributeError):
            detail = ""
        if response.status_code < 500 and detail:
            return ApiResult(error=REJECTED, detail=detail)
        return ApiResult(
            error=SERVER_ERROR,
            detail=f"HTTP {response.status_code} {detail}".strip(),
        )

    try:
        return ApiResult(data=response.json())
    except ValueError:
        return ApiResult(error=SERVER_ERROR, detail="response was not JSON")


# ============================================================
# Endpoints
# ============================================================

def get_health() -> ApiResult:
    """GET /health (public): {status, model_loaded}."""
    return _request("GET", "/health", auth=False, timeout=5)


def get_model_info() -> ApiResult:
    """GET /model/info: model_version, metrics{rmse, mae, r2, ...}, ..."""
    return _request("GET", "/model/info")


def get_drift_status() -> ApiResult:
    """GET /monitoring/latest: the latest run, or data=None if none yet."""
    result = _request("GET", "/monitoring/latest")
    if result.ok and "id" not in result.data:
        return ApiResult(data=None)        # {"message": "No monitoring data found"}
    return result


def get_monitoring_history(limit: int = 50) -> ApiResult:
    """GET /monitoring/history: monitoring runs, newest first."""
    return _request("GET", "/monitoring/history", params={"limit": limit})


def get_feature_drift() -> ApiResult:
    """GET /monitoring/feature-scores: drift_scores rows across all runs."""
    return _request("GET", "/monitoring/feature-scores")


def get_retraining_history(limit: int = 20) -> ApiResult:
    """GET /monitoring/retraining-events: retraining_events, newest first."""
    return _request("GET", "/monitoring/retraining-events", params={"limit": limit})


def get_prediction_logs(limit: int = 50) -> ApiResult:
    """GET /monitoring/prediction-logs: prediction_logs, newest first."""
    return _request("GET", "/monitoring/prediction-logs", params={"limit": limit})


def ask_assistant(question: str) -> ApiResult:
    """POST /chat: {answer}. Slow (NVIDIA LLM), hence the long timeout."""
    return _request(
        "POST", "/chat",
        json={"question": question},
        timeout=config.CHAT_TIMEOUT_SECONDS,
    )


def run_monitoring(file_name: str, content: bytes, period: str = "") -> ApiResult:
    """
    POST /monitoring/run (multipart): one monitoring run for an uploaded
    dataset. Runs synchronously on the API, so the timeout is long.
    """
    return _request(
        "POST", "/monitoring/run",
        files={"file": (file_name, content)},
        data={"period": period} if period else None,
        timeout=config.RUN_TIMEOUT_SECONDS,
    )
