"""
frontend/api_client.py: every call returns data or a typed error, sends
X-API-Key, and never raises into the UI. requests is faked: no server.
"""

import importlib
import os
import sys

import pytest
import requests


FRONTEND = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "frontend"))

FAKE_KEY = "sntl_live_FAKE_ui_0123456789abcdef"


class FakeResponse:
    def __init__(self, status_code, payload=None, json_error=False):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def client(monkeypatch):

    monkeypatch.syspath_prepend(FRONTEND)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("SENTINEL_API_URL", "http://api.test:8000/")
    monkeypatch.setenv("SENTINEL_API_KEY", FAKE_KEY)

    for name in ("config", "api_client"):
        sys.modules.pop(name, None)
    module = importlib.import_module("api_client")

    calls = []

    def respond(outcome):
        def fake_request(method, url, headers=None, timeout=None, **kwargs):
            calls.append({"method": method, "url": url, "headers": headers,
                          "timeout": timeout, **kwargs})
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        monkeypatch.setattr(module.requests, "request", fake_request)

    module.respond = respond
    module.calls = calls
    yield module

    for name in ("config", "api_client"):
        sys.modules.pop(name, None)


def test_sends_key_and_url(client):

    client.respond(FakeResponse(200, {"metrics": {"r2": 0.85}}))

    result = client.get_model_info()

    assert result.ok and result.data == {"metrics": {"r2": 0.85}}
    [call] = client.calls
    assert call["url"] == "http://api.test:8000/model/info"
    assert call["headers"] == {"X-API-Key": FAKE_KEY}


def test_health_is_public(client):

    client.respond(FakeResponse(200, {"status": "healthy", "model_loaded": True}))

    assert client.get_health().ok
    assert client.calls[0]["headers"] == {}


def test_missing_key_never_calls_the_api(client, monkeypatch):

    monkeypatch.setattr(client.config, "API_KEY", None)
    client.respond(FakeResponse(200, {}))

    assert client.get_model_info().error == client.NO_KEY
    assert client.calls == []


@pytest.mark.parametrize("outcome, kind", [
    (FakeResponse(401, {"detail": "Invalid or missing API key"}), "unauthorized"),
    (FakeResponse(404, {"detail": "Not Found"}), "not_deployed"),
    (FakeResponse(500, {"detail": "RAG request failed"}), "server_error"),
    (FakeResponse(200, json_error=True), "server_error"),
    (requests.ConnectionError("refused"), "offline"),
    (requests.Timeout("slow"), "timeout"),
])
def test_errors_are_typed_not_raised(client, outcome, kind):

    client.respond(outcome)

    result = client.get_feature_drift()

    assert not result.ok
    assert result.error == kind
    assert FAKE_KEY not in result.detail


def test_no_monitoring_run_is_empty_not_error(client):

    client.respond(FakeResponse(200, {"message": "No monitoring data found"}))

    result = client.get_drift_status()

    assert result.ok and result.data is None


def test_chat_uses_long_timeout_and_posts_question(client):

    client.respond(FakeResponse(200, {"answer": "WAIT"}))

    result = client.ask_assistant("What is the current drift status?")

    assert result.data == {"answer": "WAIT"}
    [call] = client.calls
    assert call["method"] == "POST"
    assert call["json"] == {"question": "What is the current drift status?"}
    assert call["timeout"] >= 60


def test_limits_are_passed_through(client):

    client.respond(FakeResponse(200, []))

    client.get_prediction_logs(limit=25)
    client.get_retraining_history(limit=5)

    assert client.calls[0]["params"] == {"limit": 25}
    assert client.calls[0]["url"].endswith("/monitoring/prediction-logs")
    assert client.calls[1]["url"].endswith("/monitoring/retraining-events")


def test_key_not_in_frontend_sources():

    for dirpath, _, filenames in os.walk(FRONTEND):
        for filename in filenames:
            if filename.endswith((".py", ".toml", ".example", ".md")):
                with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                    text = f.read()
                assert "sntl_live" not in text
                assert not any(line.startswith("SENTINEL_API_KEY=") and
                               line.strip() != "SENTINEL_API_KEY=change-me"
                               for line in text.splitlines())


# ============================================================
# run_monitoring: multipart upload to POST /monitoring/run
# ============================================================

def test_run_monitoring_uploads_file_with_key_and_long_timeout(client):

    client.respond(FakeResponse(200, {"run_id": 12, "action": "WAIT"}))

    result = client.run_monitoring("june.parquet", b"PAR1...", "June 2026")

    assert result.ok and result.data == {"run_id": 12, "action": "WAIT"}
    [call] = client.calls
    assert call["method"] == "POST"
    assert call["url"].endswith("/monitoring/run")
    assert call["headers"] == {"X-API-Key": FAKE_KEY}
    assert call["files"] == {"file": ("june.parquet", b"PAR1...")}
    assert call["data"] == {"period": "June 2026"}
    assert call["timeout"] >= 600


def test_run_monitoring_without_period_sends_no_form_field(client):

    client.respond(FakeResponse(200, {"run_id": 1}))

    client.run_monitoring("june.csv", b"a,b\n")

    assert client.calls[0]["data"] is None


@pytest.mark.parametrize("status, detail", [
    (415, "Unsupported file type: upload a .csv or .parquet file."),
    (422, "Missing required columns: PULocationID, RatecodeID"),
    (422, "The dataset is empty."),
    (409, "A monitoring run is already in progress. Try again when it finishes."),
])
def test_run_monitoring_rejections_carry_the_api_message(client, status, detail):

    client.respond(FakeResponse(status, {"detail": detail}))

    result = client.run_monitoring("june.csv", b"x")

    assert result.error == client.REJECTED
    assert result.detail == detail


def test_run_monitoring_processing_failure_is_server_error(client):

    client.respond(FakeResponse(500, {"detail": "Monitoring run failed while processing the dataset."}))

    result = client.run_monitoring("june.csv", b"x")

    assert result.error == client.SERVER_ERROR
    assert "500" in result.detail


@pytest.mark.parametrize("outcome, kind", [
    (requests.ConnectionError("refused"), "offline"),
    (requests.Timeout("slow"), "timeout"),
    (FakeResponse(401, {"detail": "Invalid or missing API key"}), "unauthorized"),
])
def test_run_monitoring_connection_and_auth_failures(client, outcome, kind):

    client.respond(outcome)

    assert client.run_monitoring("june.csv", b"x").error == kind
