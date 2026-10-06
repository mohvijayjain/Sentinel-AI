"""
No built-in API key.

src/serving/config.py used API_KEY = os.getenv("API_KEY", "sentinel-secret"),
so a deployment without API_KEY silently accepted a guessable key. Now:
  * unset / blank API_KEY -> None, and startup fails fast (lifespan);
  * while it is None every request is rejected (an unset key must never
    match a missing header, None == None);
  * with a key configured, accept / reject / 401 are unchanged.
Fake keys only.
"""

import importlib.util
import os
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FAKE_KEY = "sntl_live_FAKE_1g_7c6b5a4f3e2d1c0b"

PAYLOAD = {
    "trip_distance": 3.5, "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
    "vendorid": 2, "ratecodeid": 1,
}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, *relative.split("/"))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_config(monkeypatch):

    def _factory(api_key=None):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        if api_key is None:
            monkeypatch.delenv("API_KEY", raising=False)
        else:
            monkeypatch.setenv("API_KEY", api_key)
        return _load("sentinel_test_serving_config", "src/serving/config.py")

    return _factory


# ============================================================
# Config: no default, fail fast
# ============================================================

def test_unset_key_has_no_default(load_config):

    config = load_config(None)

    assert config.API_KEY is None
    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        config.require_api_key()


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_key_counts_as_unset(load_config, blank):

    config = load_config(blank)

    assert config.API_KEY is None
    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        config.require_api_key()


def test_configured_key_is_used(load_config):

    config = load_config(FAKE_KEY)

    assert config.API_KEY == FAKE_KEY
    assert config.require_api_key() == FAKE_KEY


def test_startup_fails_before_serving_without_key(monkeypatch):

    from src.serving import config, lifespan

    loaded = []
    monkeypatch.setattr(config, "API_KEY", None)
    monkeypatch.setattr(lifespan, "load_model", lambda app: loaded.append(app))

    app = FastAPI(lifespan=lifespan.lifespan)

    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        with TestClient(app):
            pass

    assert loaded == []                     # model never loaded


def test_startup_succeeds_with_key(monkeypatch):

    from src.serving import config, lifespan

    loaded = []
    monkeypatch.setattr(config, "API_KEY", FAKE_KEY)
    monkeypatch.setattr(lifespan, "load_model", lambda app: loaded.append(app))

    with TestClient(FastAPI(lifespan=lifespan.lifespan)):
        pass

    assert len(loaded) == 1


# ============================================================
# /predict while unset: nothing authenticates
# ============================================================

@pytest.fixture
def predict_client(monkeypatch):

    from src.serving import routes

    calls = []
    monkeypatch.setattr(routes, "predict_trip", lambda **kw: calls.append(kw) or {
        "prediction_seconds": 1.0, "prediction_minutes": 0.02,
        "model_version": "v1", "latency_ms": 1.0, "timestamp": "t",
    })

    app = FastAPI()
    app.include_router(routes.router)
    app.state.model, app.state.features, app.state.model_version = object(), [], "v1"

    with TestClient(app) as client:
        client.routes_module = routes
        client.calls = calls
        yield client


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-api-key": "sentinel-secret"}, {"x-api-key": ""}],
    ids=["missing", "old-default", "empty"],
)
def test_unset_key_rejects_everything_on_predict(predict_client, monkeypatch, headers):

    monkeypatch.setattr(predict_client.routes_module, "API_KEY", None)

    response = predict_client.post("/predict", json=PAYLOAD, headers=headers)

    assert response.status_code == 401
    assert predict_client.calls == []


# ============================================================
# /chat: unset rejects, configured unchanged
# ============================================================

@pytest.fixture
def chat_client(monkeypatch):
    """The real /chat route with a stub ask() (no NVIDIA client)."""

    fake_rag = types.ModuleType("src.rag.rag")
    fake_rag.ask = lambda question, top_k: f"answer to {question}"
    monkeypatch.setitem(sys.modules, "src.rag.rag", fake_rag)

    module = _load("sentinel_test_rag_route", "src/api/routes/rag.py")

    app = FastAPI()
    app.include_router(module.router)

    with TestClient(app) as client:
        client.route_module = module
        yield client


@pytest.mark.parametrize(
    "headers", [{}, {"x-api-key": "sentinel-secret"}], ids=["missing", "old-default"]
)
def test_unset_key_rejects_everything_on_chat(chat_client, monkeypatch, headers):

    monkeypatch.setattr(chat_client.route_module, "API_KEY", None)

    response = chat_client.post("/chat", json={"question": "hi"}, headers=headers)

    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers, status",
    [({"x-api-key": FAKE_KEY}, 200), ({"x-api-key": "wrong"}, 401), ({}, 401)],
    ids=["valid", "wrong", "missing"],
)
def test_configured_key_behaviour_unchanged_on_chat(
    chat_client, monkeypatch, headers, status
):
    monkeypatch.setattr(chat_client.route_module, "API_KEY", FAKE_KEY)

    response = chat_client.post("/chat", json={"question": "hi"}, headers=headers)

    assert response.status_code == status
    if status == 401:
        assert response.json() == {"detail": "Invalid or missing API key"}
    else:
        assert response.json() == {"answer": "answer to hi"}


# ============================================================
# Static
# ============================================================

def test_old_default_key_is_gone_from_source():

    hits = []
    for dirpath, _, filenames in os.walk(os.path.join(REPO_ROOT, "src")):
        for filename in filenames:
            if filename.endswith(".py"):
                with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                    if "sentinel-secret" in f.read():
                        hits.append(filename)

    assert hits == []


def test_env_example_marks_api_key_required():

    with open(os.path.join(REPO_ROOT, ".env.example"), encoding="utf-8") as f:
        text = f.read()

    assert "REQUIRED" in text.split("API_KEY=")[0].rsplit("# ---", 1)[-1]
    assert "API_KEY=change-me" in text
    assert "sentinel-secret" not in text
