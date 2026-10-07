"""
Read-only endpoints added for the Streamlit UI:

    GET /monitoring/retraining-events?limit=   (retraining_events, newest first)
    GET /monitoring/prediction-logs?limit=     (prediction_logs, newest first)

They sit on the monitoring router, so they inherit its X-API-Key
dependency (and test_endpoint_auth's every-route guard). Rows are returned
as stored: a retraining_events table from before migration 001 has no
status column, and that must pass through, not fail.
"""

import os
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import SERVING_FAKE_KEY
from test_repository_success_reporting import (  # noqa: F401
    TransactionalEngine,
    load_repository,
)


AUTH = {"x-api-key": SERVING_FAKE_KEY}

LEGACY_EVENT = {   # live shape before migration 001
    "id": 2, "triggered_at": datetime(2026, 10, 4, 16, 32, 8, 538367),
    "triggered_reason": "manual_retraining", "new_model_rmse": 347.63,
    "champion_rmse": 326.56, "promoted": False, "mlflow_run_id": "run-bbb",
}

LOG = {
    "id": 1, "predicted_at": datetime(2026, 10, 7, 7, 19, 15, 694546),
    "trip_distance": 3.5, "pickup_hour": 8, "prediction_seconds": 1662.15,
    "model_version": "v1",
}


@pytest.fixture
def client(serving_app, serving_artifacts, monkeypatch):

    import src.api.routes.monitoring as monitoring

    calls = {}

    def events(limit):
        calls["events"] = limit
        return [SimpleNamespace(_mapping=LEGACY_EVENT)]

    def logs(limit):
        calls["logs"] = limit
        return [SimpleNamespace(_mapping=LOG)]

    monkeypatch.setattr(monitoring, "get_retraining_events", events)
    monkeypatch.setattr(monitoring, "get_prediction_logs", logs)

    with TestClient(serving_app.app) as test_client:
        test_client.calls = calls
        yield test_client


@pytest.mark.parametrize("path", ["/monitoring/retraining-events",
                                  "/monitoring/prediction-logs"])
@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}], ids=["missing", "wrong"])
def test_require_api_key(client, path, headers):

    response = client.get(path, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_retraining_events_returned_as_stored(client):

    response = client.get("/monitoring/retraining-events", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == [{
        **LEGACY_EVENT, "triggered_at": "2026-10-04T16:32:08.538367"}]
    assert "status" not in response.json()[0]          # legacy row, passed through
    assert client.calls["events"] == 20                 # default limit


def test_prediction_logs_returned_as_stored(client):

    response = client.get("/monitoring/prediction-logs?limit=5", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == [{**LOG, "predicted_at": "2026-10-07T07:19:15.694546"}]
    assert client.calls["logs"] == 5


@pytest.mark.parametrize("path, bad", [
    ("/monitoring/retraining-events", 0), ("/monitoring/retraining-events", 201),
    ("/monitoring/prediction-logs", 0), ("/monitoring/prediction-logs", 501),
])
def test_limit_is_bounded(client, path, bad):

    assert client.get(f"{path}?limit={bad}", headers=AUTH).status_code == 422


# ============================================================
# Repository: read-only SELECTs, newest first, bound limit
# ============================================================

class RecordingEngine(TransactionalEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statements = []

    def connect(self):
        return self.begin()

    def execute(self, query, params=None):
        self.statements.append((str(query), params))

        class Result:
            def fetchall(self):
                return ["row"]

        return Result()


@pytest.mark.parametrize("function, table", [
    ("get_retraining_events", "retraining_events"),
    ("get_prediction_logs", "prediction_logs"),
])
def test_repository_reads_are_select_only(load_repository, function, table):

    engine = RecordingEngine()

    rows = getattr(load_repository(engine), function)(limit=7)

    assert rows == ["row"]
    [(sql, params)] = engine.statements
    normalized = " ".join(sql.split())
    assert normalized == f"SELECT * FROM {table} ORDER BY id DESC LIMIT :limit"
    assert params == {"limit": 7}
