"""
Phase 2.2: operational endpoints require the existing X-API-Key.

/monitoring/*, /model/info and the Prometheus /metrics were public. They
now use routes.verify_api_key, the same check /predict performs (same
comparison, same 401 detail, same redacted outcome logging). /health and
/ stay public. Tests run the real app through its real lifespan with temp
artifacts; DB getters and the RAG answer are stubs.
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import SERVING_FAKE_KEY, SERVING_FEATURES, SERVING_METRICS


WRONG_KEY = "sntl_live_FAKE_ffff0000eeee1111dddd"
DETAIL = {"detail": "Invalid or missing API key"}

PREDICT_PAYLOAD = {
    "trip_distance": 3.5, "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
    "vendorid": 2, "ratecodeid": 1,
}

# Routes deliberately reachable without a key. Anything else must 401.
PUBLIC_PATHS = {"/", "/health"}

# Valid bodies, so an unauthenticated request reaches the key check
# (/predict and /chat validate the body first, as before)
BODIES = {"/predict": PREDICT_PAYLOAD, "/chat": {"question": "hi"}}


def _row(**values):
    return SimpleNamespace(_mapping=values)


@pytest.fixture
def client(serving_app, serving_artifacts, monkeypatch):

    import src.api.routes.monitoring as monitoring
    import src.api.routes.rag as rag

    latest = _row(run_id=7, overall_score=0.42, action="MONITOR")
    monkeypatch.setattr(monitoring, "get_latest_monitoring_run", lambda: latest)
    monkeypatch.setattr(monitoring, "get_monitoring_history",
                        lambda limit: [_row(run_id=i) for i in range(limit)])
    monkeypatch.setattr(monitoring, "get_drifted_features",
                        lambda: [_row(feature="trip_distance", severity="HIGH")])
    monkeypatch.setattr(monitoring, "get_feature_drift_scores",
                        lambda: [_row(feature="pickup_hour", psi=0.3)])
    monkeypatch.setattr(rag, "ask", lambda question, top_k: f"answer to {question}")

    with TestClient(serving_app.app) as test_client:
        yield test_client


# ============================================================
# Newly protected endpoints: 401 / 401 / original behaviour
# ============================================================

PROTECTED = [
    "/monitoring/latest",
    "/monitoring/history",
    "/monitoring/drifted-features",
    "/monitoring/feature-scores",
    "/model/info",
    "/metrics",
]


@pytest.mark.parametrize("path", PROTECTED)
def test_missing_key_rejected(client, path):

    response = client.get(path)

    assert response.status_code == 401
    assert response.json() == DETAIL


@pytest.mark.parametrize("path", PROTECTED)
@pytest.mark.parametrize("key", [WRONG_KEY, "", "sentinel-secret"],
                         ids=["wrong", "empty", "old-default"])
def test_invalid_key_rejected(client, path, key):

    response = client.get(path, headers={"x-api-key": key})

    assert response.status_code == 401
    assert response.json() == DETAIL


@pytest.mark.parametrize("path", PROTECTED)
def test_unset_server_key_rejects_everything(client, monkeypatch, path):

    from src.serving import routes

    monkeypatch.setattr(routes, "API_KEY", None)

    assert client.get(path).status_code == 401
    assert client.get(path, headers={"x-api-key": SERVING_FAKE_KEY}).status_code == 401


def _get(client, path):
    return client.get(path, headers={"x-api-key": SERVING_FAKE_KEY})


def test_valid_key_monitoring_bodies_unchanged(client):

    assert _get(client, "/monitoring/latest").json() == {
        "run_id": 7, "overall_score": 0.42, "action": "MONITOR"}
    assert _get(client, "/monitoring/history?limit=3").json() == [
        {"run_id": 0}, {"run_id": 1}, {"run_id": 2}]
    assert _get(client, "/monitoring/drifted-features").json() == [
        {"feature": "trip_distance", "severity": "HIGH"}]
    assert _get(client, "/monitoring/feature-scores").json() == [
        {"feature": "pickup_hour", "psi": 0.3}]


def test_valid_key_monitoring_latest_empty_message(client, monkeypatch):

    import src.api.routes.monitoring as monitoring

    monkeypatch.setattr(monitoring, "get_latest_monitoring_run", lambda: None)

    response = _get(client, "/monitoring/latest")

    assert response.status_code == 200
    assert response.json() == {"message": "No monitoring data found"}


def test_valid_key_model_info_body(client):

    response = _get(client, "/model/info")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model_type", "target", "features", "best_params",
                         "metrics", "trained_on", "model_version", "loaded_at"}
    assert body["model_type"] == "LightGBM Regressor"
    assert body["features"] == SERVING_FEATURES
    assert body["metrics"] == SERVING_METRICS
    assert body["model_version"] == "v1"


def test_valid_key_metrics_is_prometheus_text(client):

    response = _get(client, "/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text or "# HELP" in response.text


# ============================================================
# /predict and /chat unchanged; /health and / public
# ============================================================

@pytest.mark.parametrize("path", ["/predict", "/chat"])
@pytest.mark.parametrize(
    "headers, status",
    [({"x-api-key": SERVING_FAKE_KEY}, 200), ({"x-api-key": WRONG_KEY}, 401), ({}, 401)],
    ids=["valid", "wrong", "missing"],
)
def test_predict_and_chat_auth_unchanged(client, path, headers, status):

    response = client.post(path, json=BODIES[path], headers=headers)

    assert response.status_code == status
    if status == 401:
        assert response.json() == DETAIL
    elif path == "/chat":
        assert response.json() == {"answer": "answer to hi"}
    else:
        assert response.json()["prediction_seconds"] == 600.0


@pytest.mark.parametrize("path", sorted(PUBLIC_PATHS))
def test_public_paths_need_no_key(client, path):

    assert client.get(path).status_code == 200


# ============================================================
# Guard: no route ships without the key unless listed as public
# ============================================================

def test_every_non_public_route_requires_the_key(serving_app, client):

    paths = serving_app.app.openapi()["paths"]

    # the enumeration really sees the routers (not an empty schema)
    assert {"/predict", "/chat", "/model/info", "/metrics",
            "/monitoring/latest"} <= set(paths)

    unprotected = []

    for path, operations in paths.items():
        if path in PUBLIC_PATHS:
            continue
        for method in operations:
            response = client.request(method.upper(), path, json=BODIES.get(path))
            if response.status_code != 401:
                unprotected.append(f"{method.upper()} {path} -> {response.status_code}")

    assert unprotected == []


def test_monitoring_router_carries_the_auth_dependency():

    import src.api.routes.monitoring as monitoring
    from src.serving.routes import verify_api_key

    # router-level: routes added to this router later inherit it
    assert [d.dependency for d in monitoring.router.dependencies] == [verify_api_key]


# ============================================================
# No credential in logs, outcome still observable
# ============================================================

@pytest.mark.parametrize("path", PROTECTED)
def test_auth_logs_contain_no_key(client, caplog, path):

    with caplog.at_level(logging.DEBUG):
        client.get(path, headers={"x-api-key": SERVING_FAKE_KEY})
        client.get(path, headers={"x-api-key": WRONG_KEY})
        client.get(path)

    text = caplog.text
    assert SERVING_FAKE_KEY not in text
    assert WRONG_KEY not in text
    assert f"API authentication success | GET {path}" in text
    assert f"API authentication failure | GET {path} | reason=invalid | status=401" in text
    assert f"API authentication failure | GET {path} | reason=missing | status=401" in text


@pytest.mark.parametrize("path", PROTECTED)
def test_auth_failure_response_leaks_nothing(client, path):

    response = client.get(path, headers={"x-api-key": WRONG_KEY})

    assert SERVING_FAKE_KEY not in response.text
    assert WRONG_KEY not in response.text
    assert "Traceback" not in response.text
