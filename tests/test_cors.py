"""
Phase 2.3: CORS is an explicit allowlist.

app.py had allow_origins=["*"] with allow_credentials=True (any page could
call the API; and the wildcard + credentials pair is invalid per the CORS
spec). Origins now come from CORS_ALLOWED_ORIGINS (comma-separated);
unset/blank -> http://localhost:3000 only; "*" is refused. Checked on the
actual response headers, not only on config values.
"""

import importlib.util
import os

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from conftest import SERVING_FAKE_KEY


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

LOCAL = "http://localhost:3000"
EVIL = "https://evil.example"

PAYLOAD = {
    "trip_distance": 3.5, "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
    "vendorid": 2, "ratecodeid": 1,
}


def _load_config(monkeypatch, raw):
    """A fresh src/serving/config.py with CORS_ALLOWED_ORIGINS=raw."""

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    if raw is None:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", raw)

    spec = importlib.util.spec_from_file_location(
        "sentinel_test_cors_config",
        os.path.join(REPO_ROOT, "src", "serving", "config.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cors_middleware(app):
    [middleware] = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    return middleware


def _app_with_origins(serving_app, origins):
    """A tiny app carrying the real app's CORS settings, origins swapped."""

    kwargs = {**_cors_middleware(serving_app.app).kwargs, "allow_origins": origins}

    app = FastAPI()
    app.add_middleware(CORSMiddleware, **kwargs)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


def _preflight(client, path, origin, method="GET"):
    return client.options(path, headers={
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "x-api-key, content-type",
    })


# ============================================================
# Parsing
# ============================================================

@pytest.mark.parametrize("raw", [None, "", "   ", " , ,, "],
                         ids=["unset", "empty", "blank", "only-commas"])
def test_unset_or_blank_defaults_to_local_dev_origin(monkeypatch, raw):

    assert _load_config(monkeypatch, raw).CORS_ALLOWED_ORIGINS == [LOCAL]


def test_multiple_origins_whitespace_and_empties(monkeypatch):

    config = _load_config(
        monkeypatch, " http://localhost:3000 ,, https://app.example.com/ ,")

    assert config.CORS_ALLOWED_ORIGINS == [LOCAL, "https://app.example.com"]


@pytest.mark.parametrize("raw", ["*", "http://localhost:3000, *"])
def test_wildcard_is_refused(monkeypatch, raw):

    with pytest.raises(RuntimeError, match="explicit origins"):
        _load_config(monkeypatch, raw)


def test_real_app_has_no_wildcard_and_credentials_only_with_explicit_origins(serving_app):

    from src.serving.config import CORS_ALLOWED_ORIGINS

    kwargs = _cors_middleware(serving_app.app).kwargs

    assert kwargs["allow_origins"] == CORS_ALLOWED_ORIGINS
    assert "*" not in kwargs["allow_origins"]
    assert kwargs["allow_origins"]
    assert kwargs["allow_credentials"] is True


# ============================================================
# Response headers: the real app (default allowlist in tests)
# ============================================================

@pytest.fixture
def client(serving_app, serving_artifacts):

    with TestClient(serving_app.app) as test_client:
        yield test_client


def test_real_app_grants_configured_origin(client):

    from src.serving.config import CORS_ALLOWED_ORIGINS

    origin = CORS_ALLOWED_ORIGINS[0]
    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"


def test_real_app_does_not_grant_unconfigured_origin(client):

    response = client.get("/health", headers={"Origin": EVIL})

    assert response.status_code == 200            # same response, no grant
    assert "access-control-allow-origin" not in response.headers

    preflight = _preflight(client, "/predict", EVIL, "POST")

    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


def test_real_app_preflight_for_configured_origin(client):

    from src.serving.config import CORS_ALLOWED_ORIGINS

    origin = CORS_ALLOWED_ORIGINS[0]
    response = _preflight(client, "/predict", origin, "POST")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "x-api-key" in response.headers["access-control-allow-headers"].lower()


def test_same_origin_requests_unaffected(client):

    # No Origin header (curl, server-to-server, same-origin GET)
    response = client.post("/predict", json=PAYLOAD,
                           headers={"x-api-key": SERVING_FAKE_KEY})

    assert response.status_code == 200
    assert response.json()["prediction_seconds"] == 600.0
    assert "access-control-allow-origin" not in response.headers


def test_predict_with_allowed_origin_unchanged(client):

    from src.serving.config import CORS_ALLOWED_ORIGINS

    origin = CORS_ALLOWED_ORIGINS[0]
    response = client.post("/predict", json=PAYLOAD,
                           headers={"x-api-key": SERVING_FAKE_KEY, "Origin": origin})

    assert response.status_code == 200
    assert response.json()["prediction_seconds"] == 600.0
    assert response.headers["access-control-allow-origin"] == origin


def test_chat_with_allowed_origin_unchanged(client, monkeypatch):

    import src.api.routes.rag as rag
    from src.serving.config import CORS_ALLOWED_ORIGINS

    monkeypatch.setattr(rag, "ask", lambda question, top_k: f"answer to {question}")
    origin = CORS_ALLOWED_ORIGINS[0]

    response = client.post("/chat", json={"question": "hi"},
                           headers={"x-api-key": SERVING_FAKE_KEY, "Origin": origin})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer to hi"}
    assert response.headers["access-control-allow-origin"] == origin


# ============================================================
# Response headers: allowlists parsed from env
# ============================================================

def test_multiple_configured_origins_each_granted(serving_app, monkeypatch):

    origins = _load_config(
        monkeypatch, "http://localhost:3000, https://app.example.com").CORS_ALLOWED_ORIGINS
    client = _app_with_origins(serving_app, origins)

    for origin in origins:
        response = client.get("/ping", headers={"Origin": origin})
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["access-control-allow-credentials"] == "true"

    denied = client.get("/ping", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in denied.headers


def test_whitespace_entry_matches_exact_origin(serving_app, monkeypatch):

    origins = _load_config(monkeypatch, "  https://app.example.com/  ").CORS_ALLOWED_ORIGINS
    client = _app_with_origins(serving_app, origins)

    response = client.get("/ping", headers={"Origin": "https://app.example.com"})

    assert response.headers["access-control-allow-origin"] == "https://app.example.com"


def test_default_never_echoes_arbitrary_origin(serving_app, monkeypatch):

    origins = _load_config(monkeypatch, None).CORS_ALLOWED_ORIGINS
    client = _app_with_origins(serving_app, origins)

    for origin in (EVIL, "null", "http://localhost:3001", "http://127.0.0.1:3000"):
        response = client.get("/ping", headers={"Origin": origin})
        assert "access-control-allow-origin" not in response.headers
        assert response.headers.get("access-control-allow-origin") != "*"
