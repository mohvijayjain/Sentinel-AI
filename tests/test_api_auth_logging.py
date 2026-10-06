"""
/predict authentication must not log credentials.

src/serving/routes.py logged both the received AND the server's expected
API key on every request. It now logs only the outcome (success /
failure, reason, method, route). Authentication behaviour is unchanged.

All secrets below are fake.
"""

import logging
import os
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.serving import routes


SERVER_KEY = "sntl_live_FAKE_9f8e7d6c5b4a3f2e1d0c"
WRONG_KEY = "sntl_live_FAKE_0000aaaa1111bbbb2222"
BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.ZmFrZXNpZ25hdHVyZQ"

PAYLOAD = {
    "trip_distance": 3.5,
    "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230,
    "dolocationid": 161,
    "payment_type": 1,
    "vendorid": 2,
    "ratecodeid": 1,
}


@pytest.fixture
def client(monkeypatch):

    monkeypatch.setattr(routes, "API_KEY", SERVER_KEY)

    calls = []

    def fake_predict_trip(**kwargs):
        calls.append(kwargs)
        return {
            "prediction_seconds": 600.0,
            "prediction_minutes": 10.0,
            "model_version": "v1",
            "latency_ms": 1.0,
            "timestamp": "2026-03-02T08:30:00",
        }

    monkeypatch.setattr(routes, "predict_trip", fake_predict_trip)

    app = FastAPI()
    app.include_router(routes.router)
    app.state.model = object()
    app.state.features = []
    app.state.model_version = "v1"

    with TestClient(app) as test_client:
        test_client.predict_calls = calls
        yield test_client


def _post(client, headers):
    return client.post("/predict", json=PAYLOAD, headers=headers)


def _assert_no_secret(text):
    for secret in (SERVER_KEY, WRONG_KEY, BEARER):
        assert secret not in text


# ============================================================
# Behaviour unchanged
# ============================================================

def test_valid_key_accepted(client):

    response = _post(client, {"x-api-key": SERVER_KEY})

    assert response.status_code == 200
    assert response.json()["prediction_seconds"] == 600.0
    assert len(client.predict_calls) == 1


def test_wrong_key_rejected(client):

    response = _post(client, {"x-api-key": WRONG_KEY})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert client.predict_calls == []


def test_missing_key_rejected(client):

    response = _post(client, {})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert client.predict_calls == []


# ============================================================
# No credential in logs
# ============================================================

@pytest.mark.parametrize(
    "headers, status",
    [
        ({"x-api-key": SERVER_KEY}, 200),
        ({"x-api-key": WRONG_KEY}, 401),
        ({}, 401),
        ({"x-api-key": SERVER_KEY, "Authorization": f"Bearer {BEARER}"}, 200),
        ({"x-api-key": WRONG_KEY, "Authorization": f"Bearer {BEARER}"}, 401),
    ],
    ids=["valid", "wrong", "missing", "valid+bearer", "wrong+bearer"],
)
def test_no_key_or_header_in_logs(client, caplog, headers, status):

    with caplog.at_level(logging.DEBUG):
        response = _post(client, headers)

    assert response.status_code == status
    _assert_no_secret(caplog.text)
    assert "Authorization" not in caplog.text
    assert "Bearer" not in caplog.text


def test_outcome_is_still_observable(client, caplog):

    with caplog.at_level(logging.INFO):
        _post(client, {"x-api-key": SERVER_KEY})
        _post(client, {"x-api-key": WRONG_KEY})
        _post(client, {})

    text = caplog.text

    assert "API authentication success | POST /predict" in text
    assert "API authentication failure | POST /predict | reason=invalid | status=401" in text
    assert "API authentication failure | POST /predict | reason=missing | status=401" in text


# ============================================================
# Static guard: no log line in the serving / API layers mentions keys
# ============================================================

_LOG_CALL = re.compile(r"\b(logger\.\w+|logging\.\w+|print)\(")

# What actually leaks: a credential interpolated into the message, passed
# as a bare argument, or the raw request headers. Using the key only to
# decide an outcome (e.g. "x_api_key is None") is fine.
_LEAK = re.compile(
    r"\{[^}]*(api_key|authorization|bearer|token|headers)[^}]*\}"
    r"|[,(]\s*(x_api_key|API_KEY|authorization)\s*[,)]"
    r"|request\.headers",
    re.IGNORECASE,
)


def _leaking_log_lines(lines):

    offending = []

    for number, line in enumerate(lines, 1):
        if not _LOG_CALL.search(line):
            continue
        # The call plus its continuation lines, up to the closing ")"
        window = " ".join(lines[number - 1:number + 6])
        call = window[_LOG_CALL.search(window).start():]
        depth, end = 0, len(call)
        for i, ch in enumerate(call):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 0 and ch == ")":
                end = i + 1
                break
        if _LEAK.search(call[:end]):
            offending.append(f"{number}: {line.strip()}")

    return offending


def test_guard_catches_the_original_leak():

    original = [
        '    logger.info(f"Received API Key: {x_api_key}")',
        '    logger.info(f"Expected API Key: {API_KEY}")',
        '    logger.info("headers=%s", request.headers)',
        '    logger.debug("key %s", x_api_key)',
    ]

    assert len(_leaking_log_lines(original)) == 4


def test_no_log_statement_references_credentials():

    root = os.path.join(os.path.dirname(__file__), os.pardir, "src")
    offending = []

    for layer in ("serving", "api"):
        for dirpath, _, filenames in os.walk(os.path.join(root, layer)):
            for filename in filenames:
                if filename.endswith(".py"):
                    path = os.path.join(dirpath, filename)
                    with open(path, encoding="utf-8") as f:
                        lines = f.read().splitlines()
                    offending += [f"{filename}:{hit}"
                                  for hit in _leaking_log_lines(lines)]

    assert offending == []
