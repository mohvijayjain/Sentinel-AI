"""
Phase 2.6: /predict writes prediction_logs (best effort, after the response).

Before: prediction_logs existed in Sentinel.sql but nothing wrote it; /predict
only emitted a logger.info line. Now a successful prediction schedules
log_prediction() as a background task: it runs after the response is sent
and never raises, so a slow or failing database cannot delay or fail
/predict. The repository insert is tested against a recording engine and
the schema text; the endpoint against the real app. No real database.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import SERVING_FAKE_KEY
from test_repository_success_reporting import (  # noqa: F401
    TransactionalEngine,
    load_repository,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

PAYLOAD = {
    "trip_distance": 3.5, "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
    "vendorid": 2, "ratecodeid": 1,
}

DB_PASSWORD = "Pg5ecretFakePw"
BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.ZmFrZXNpZ25hdHVyZQ"


def _schema_columns():
    """prediction_logs columns as PostgreSQL creates them (unquoted -> lower)."""

    with open(os.path.join(REPO_ROOT, "Sentinel.sql"), encoding="utf-8") as f:
        sql = f.read()

    body = re.search(
        r"CREATE TABLE IF NOT EXISTS prediction_logs \((.*?)\n\);", sql, re.S
    ).group(1)

    columns = {}
    for line in body.strip().splitlines():
        name, sql_type = line.split()[:2]
        columns[name.lower()] = sql_type.rstrip(",").upper()

    return columns


class RecordingEngine(TransactionalEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statements = []

    def execute(self, query, params=None):
        self.statements.append((str(query), params))
        return super().execute(query, params)


def _row(**overrides):
    row = {
        "predicted_at": "2026-10-07T06:00:00+00:00",
        "trip_distance": 3.5, "pickup_hour": 8, "pickup_day_of_week": 0,
        "pickup_month": 3, "is_weekend": 0, "is_rush_hour": 1,
        "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
        "vendorid": 2, "ratecodeid": 1.0, "prediction_seconds": 600.0,
        "prediction_minutes": 10.0, "model_version": "v1", "latency_ms": 1.5,
    }
    row.update(overrides)
    return row


# ============================================================
# Repository: insert_prediction_log against the schema
# ============================================================

def test_insert_matches_schema_columns_exactly(load_repository):

    engine = RecordingEngine(returned_id=5)

    log_id = load_repository(engine).insert_prediction_log(**_row())

    assert log_id == 5
    assert engine.events == ["begin", "commit"]

    [(sql, params)] = engine.statements
    inserted = re.search(r"INSERT INTO prediction_logs\s*\((.*?)\)", sql, re.S).group(1)
    inserted = [c.strip() for c in inserted.split(",")]

    schema = _schema_columns()
    assert set(inserted) == set(schema) - {"id"}       # every column, no id
    assert inserted == [c for c in inserted if c == c.lower()]   # folded names
    assert set(params) == set(inserted)


def test_values_have_the_schema_types(load_repository):

    engine = RecordingEngine()
    load_repository(engine).insert_prediction_log(**_row())

    [(_, params)] = engine.statements
    schema = _schema_columns()

    for column, value in params.items():
        if column == "predicted_at":
            continue
        expected = {"INT": int, "FLOAT": float}.get(schema[column], str)
        if schema[column].startswith("VARCHAR"):
            assert len(value) <= int(re.search(r"\((\d+)\)", schema[column]).group(1))
            expected = str
        assert type(value) is expected, (column, value)


def test_predicted_at_is_written_as_utc_like_retraining_events(load_repository):

    engine = RecordingEngine()
    load_repository(engine).insert_prediction_log(**_row())

    [(sql, params)] = engine.statements
    assert "CAST(:predicted_at AS TIMESTAMPTZ) AT TIME ZONE 'UTC'" in sql
    assert params["predicted_at"] == "2026-10-07T06:00:00+00:00"
    assert "NOW()" not in sql


def test_insert_failure_rolls_back_and_raises(load_repository):

    engine = RecordingEngine(fail_on_execute=ConnectionError("db down"))

    with pytest.raises(ConnectionError):
        load_repository(engine).insert_prediction_log(**_row())

    assert engine.events == ["begin", "rollback"]


def test_insert_requires_every_field(load_repository):

    row = _row()
    del row["model_version"]

    with pytest.raises(TypeError):
        load_repository(RecordingEngine()).insert_prediction_log(**row)


# ============================================================
# Row construction
# ============================================================

def _request(**overrides):
    from src.serving.schemas import PredictRequest
    return PredictRequest(**{**PAYLOAD, **overrides})


def _response(**overrides):
    from src.serving.schemas import PredictResponse
    values = {"prediction_seconds": 612.34, "prediction_minutes": 10.21,
              "model_version": "v1", "latency_ms": 4.2,
              "timestamp": "2026-03-02T08:30:01"}
    return PredictResponse(**{**values, **overrides})


def test_row_holds_request_inputs_features_and_prediction():

    from src.serving.prediction_logger import build_prediction_log

    row = build_prediction_log(_request(), _response(), "2026-10-07T06:00:00+00:00")

    assert row == {
        "predicted_at": "2026-10-07T06:00:00+00:00",
        "trip_distance": 3.5,
        "pickup_hour": 8, "pickup_day_of_week": 0, "pickup_month": 3,
        "is_weekend": 0, "is_rush_hour": 1,          # Monday 08:30
        "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
        "vendorid": 2, "ratecodeid": 1.0,
        "prediction_seconds": 612.34, "prediction_minutes": 10.21,
        "model_version": "v1", "latency_ms": 4.2,
    }
    assert set(row) == set(_schema_columns()) - {"id"}


def test_features_come_from_the_shared_engineer_features():

    from src.serving.prediction_logger import build_prediction_log

    row = build_prediction_log(
        _request(pickup_datetime="2026-01-10T22:15:00"), _response(), "t")

    assert (row["pickup_hour"], row["pickup_day_of_week"], row["pickup_month"],
            row["is_weekend"], row["is_rush_hour"]) == (22, 5, 1, 1, 0)


def test_log_prediction_reports_failure_without_raising(monkeypatch, caplog):

    from src.serving import prediction_logger

    def failing_insert(**row):
        raise ConnectionError(
            f"could not connect: postgresql://sentinel:{DB_PASSWORD}@postgres:5432/x")

    monkeypatch.setattr(prediction_logger.drift_repository,
                        "insert_prediction_log", failing_insert)

    with caplog.at_level(logging.ERROR):
        ok = prediction_logger.log_prediction(_request(), _response(), "t")

    assert ok is False
    assert "Prediction log not written; the prediction itself was served." in caplog.text
    assert DB_PASSWORD not in caplog.text


# ============================================================
# /predict through the real app and lifespan
# ============================================================

@pytest.fixture
def inserts(serving_app, monkeypatch):

    from src.serving import prediction_logger

    rows = []
    monkeypatch.setattr(prediction_logger.drift_repository,
                        "insert_prediction_log", lambda **row: rows.append(row) or 1)
    return rows


@pytest.fixture
def client(serving_app, serving_artifacts, inserts):

    with TestClient(serving_app.app) as test_client:
        yield test_client


def _post(client, headers=None, payload=PAYLOAD):
    return client.post("/predict", json=payload,
                       headers={"x-api-key": SERVING_FAKE_KEY} if headers is None else headers)


def test_successful_prediction_writes_one_matching_log(client, inserts):

    before = datetime.now(timezone.utc)
    response = _post(client)
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    body = response.json()

    [row] = inserts
    assert row["prediction_seconds"] == body["prediction_seconds"] == 600.0
    assert row["prediction_minutes"] == body["prediction_minutes"]
    assert row["model_version"] == body["model_version"]
    assert row["latency_ms"] == body["latency_ms"]
    assert row["trip_distance"] == PAYLOAD["trip_distance"]
    assert row["pulocationid"] == PAYLOAD["pulocationid"]
    assert row["ratecodeid"] == float(PAYLOAD["ratecodeid"])

    predicted_at = datetime.fromisoformat(row["predicted_at"])
    assert predicted_at.utcoffset() == timedelta(0)
    assert before <= predicted_at <= after


def test_no_credentials_or_headers_persisted(client, inserts):

    response = _post(client, headers={
        "x-api-key": SERVING_FAKE_KEY,
        "Authorization": f"Bearer {BEARER}",
        "Cookie": "session=abc",
    })

    assert response.status_code == 200
    stored = json.dumps(inserts)
    for secret in (SERVING_FAKE_KEY, BEARER, "Bearer", "session=abc"):
        assert secret not in stored
    assert set(inserts[0]) == set(_schema_columns()) - {"id"}


def test_logging_failure_does_not_fail_or_change_the_prediction(
    serving_app, serving_artifacts, monkeypatch, caplog
):
    from src.serving import prediction_logger

    with TestClient(serving_app.app) as client:
        monkeypatch.setattr(prediction_logger.drift_repository,
                            "insert_prediction_log", lambda **row: 1)
        baseline = _post(client).json()

        def failing_insert(**row):
            raise ConnectionError(
                f"could not connect: postgresql://sentinel:{DB_PASSWORD}@postgres/x")

        monkeypatch.setattr(prediction_logger.drift_repository,
                            "insert_prediction_log", failing_insert)

        with caplog.at_level(logging.ERROR):
            response = _post(client)

    assert response.status_code == 200
    body = response.json()
    for key in ("prediction_seconds", "prediction_minutes", "model_version"):
        assert body[key] == baseline[key]
    assert "Prediction log not written" in caplog.text
    assert DB_PASSWORD not in caplog.text


def test_log_is_written_after_the_response_is_sent(serving_app, serving_artifacts, monkeypatch):

    from src.serving import prediction_logger

    events = []
    monkeypatch.setattr(prediction_logger.drift_repository, "insert_prediction_log",
                        lambda **row: events.append("insert") or 1)

    body = json.dumps(PAYLOAD).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/predict", "raw_path": b"/predict",
        "root_path": "", "query_string": b"", "server": ("testserver", 80),
        "client": ("testclient", 5000),
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-api-key", SERVING_FAKE_KEY.encode())],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            events.append(("status", message["status"]))
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            events.append("response sent")

    with TestClient(serving_app.app):            # real lifespan: app.state
        asyncio.run(serving_app.app(scope, receive, send))

    assert events == [("status", 200), "response sent", "insert"]


def test_failed_prediction_writes_no_log(client, inserts, serving_app):

    from src.serving import routes
    from fastapi import HTTPException

    def failing_predict(**kwargs):
        raise HTTPException(status_code=500, detail="Prediction failed")

    original = routes.predict_trip
    routes.predict_trip = failing_predict
    try:
        response = _post(client)
    finally:
        routes.predict_trip = original

    assert response.status_code == 500
    assert inserts == []


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}], ids=["missing", "wrong"])
def test_auth_unchanged_and_rejected_requests_not_logged(client, inserts, headers):

    response = _post(client, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert inserts == []


def test_invalid_request_unchanged_and_not_logged(client, inserts):

    response = _post(client, payload={**PAYLOAD, "trip_distance": -1})

    assert response.status_code == 422
    assert inserts == []


def test_response_schema_unchanged(client):

    assert set(_post(client).json()) == {
        "prediction_seconds", "prediction_minutes", "model_version",
        "latency_ms", "timestamp"}
