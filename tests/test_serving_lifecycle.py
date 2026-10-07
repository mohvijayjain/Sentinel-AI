"""
Phase 2.1: one canonical FastAPI lifecycle.

src/serving/app.py used to define a second `async def lifespan` and a
local load_model() writing module globals (model, features, model_metrics,
loaded_at), with metrics read from FEATURES_PATH. FastAPI was constructed
before that definition, so it registered the imported lifespan.py; the
duplicate was dead but shadowed `src.serving.app.lifespan` and left stale
globals that could diverge from app.state. It is deleted. These tests start
the real app through its real lifespan (TestClient as a context manager)
with temp model artifacts and a fake key: no services, no NVIDIA key.
"""

import os
import re

import pytest
from fastapi.testclient import TestClient

from conftest import (
    SERVING_FAKE_KEY, SERVING_FEATURES, SERVING_METRICS, FixedModel,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

PAYLOAD = {
    "trip_distance": 3.5, "pickup_datetime": "2026-03-02T08:30:00",
    "pulocationid": 230, "dolocationid": 161, "payment_type": 1,
    "vendorid": 2, "ratecodeid": 1,
}

AUTH = {"x-api-key": SERVING_FAKE_KEY}


def _read(relative):
    with open(os.path.join(REPO_ROOT, *relative.split("/")), encoding="utf-8") as f:
        return f.read()


# ============================================================
# Exactly one lifespan, one loader, no stale globals
# ============================================================

def test_app_module_exposes_the_canonical_lifespan(serving_app):

    from src.serving import lifespan

    assert serving_app.lifespan is lifespan.lifespan


def test_no_duplicate_lifespan_or_loader_in_app_module(serving_app):

    source = _read("src/serving/app.py")

    assert "async def lifespan" not in source
    assert "asynccontextmanager" not in source
    assert "def load_model" not in source
    assert "pickle" not in source

    for stale in ("model", "features", "model_metrics", "loaded_at",
                  "model_version", "load_model"):
        assert not hasattr(serving_app, stale), stale


def test_only_lifespan_py_defines_a_lifespan():

    hits = []
    for dirpath, _, filenames in os.walk(os.path.join(REPO_ROOT, "src")):
        for filename in filenames:
            if filename.endswith(".py"):
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as f:
                    if re.search(r"^\s*async def lifespan\b", f.read(), re.M):
                        hits.append(os.path.relpath(path, REPO_ROOT))

    assert hits == [os.path.join("src", "serving", "lifespan.py")]


def test_metrics_are_loaded_from_metrics_path_not_features_path():

    loader = _read("src/serving/model_loader.py")
    config = _read("src/serving/config.py")

    assert "open(METRICS_PATH" in loader
    assert "getenv" not in loader
    assert 'METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")' in config


# ============================================================
# Startup runs require_api_key() then the canonical loader
# ============================================================

def test_startup_invokes_canonical_loader_and_populates_state(
    serving_app, serving_artifacts, monkeypatch
):
    from src.serving import lifespan

    order = []
    real_require, real_load = lifespan.require_api_key, lifespan.load_model

    def require():
        order.append("require_api_key")
        return real_require()

    def load(app):
        order.append(("load_model", app))
        return real_load(app)

    monkeypatch.setattr(lifespan, "require_api_key", require)
    monkeypatch.setattr(lifespan, "load_model", load)

    app = serving_app.app

    with TestClient(app):
        assert order == ["require_api_key", ("load_model", app)]

        assert isinstance(app.state.model, FixedModel)
        assert app.state.features == SERVING_FEATURES
        assert app.state.model_metrics == SERVING_METRICS
        assert app.state.model_version == "v1"
        assert isinstance(app.state.loaded_at, str) and app.state.loaded_at


def test_routes_read_the_state_startup_populated(serving_app, serving_artifacts):

    app = serving_app.app

    with TestClient(app) as client:
        info = client.get("/model/info", headers=AUTH)
        health = client.get("/health")

    assert info.status_code == 200
    body = info.json()
    assert body["features"] == app.state.features == SERVING_FEATURES
    assert body["metrics"] == app.state.model_metrics == SERVING_METRICS
    assert body["model_version"] == app.state.model_version
    assert body["loaded_at"] == app.state.loaded_at

    assert health.json() == {"status": "healthy", "model_loaded": True}


def test_predict_unchanged_through_real_lifespan(serving_app, serving_artifacts):

    app = serving_app.app

    with TestClient(app) as client:
        response = client.post("/predict", json=PAYLOAD, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_seconds"] == 600.0
    assert body["prediction_minutes"] == 10.0
    assert body["model_version"] == "v1"
    assert set(body) == {"prediction_seconds", "prediction_minutes",
                         "model_version", "latency_ms", "timestamp"}
    # the loaded feature list, in order, is what the model received
    assert app.state.model.seen_columns == SERVING_FEATURES


def test_predict_still_validates_before_auth(serving_app, serving_artifacts):

    with TestClient(serving_app.app) as client:
        response = client.post("/predict", json={"trip_distance": -1})

    # unchanged ordering: body validation (422) precedes the key check
    assert response.status_code == 422


# ============================================================
# Fail fast: never serve on a bad start
# ============================================================

def test_unset_api_key_fails_startup_before_loading(
    serving_app, serving_artifacts, monkeypatch
):
    from src.serving import config, lifespan

    loaded = []
    monkeypatch.setattr(config, "API_KEY", None)
    monkeypatch.setattr(lifespan, "load_model", lambda app: loaded.append(app))

    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        with TestClient(serving_app.app):
            pass

    assert loaded == []


@pytest.mark.parametrize("artifact", ["MODEL_PATH", "FEATURES_PATH", "METRICS_PATH"])
def test_missing_artifact_fails_startup(serving_app, serving_artifacts, artifact):

    serving_artifacts[artifact].unlink()

    with pytest.raises(FileNotFoundError):
        with TestClient(serving_app.app):
            pass


@pytest.mark.parametrize("artifact", ["MODEL_PATH", "FEATURES_PATH", "METRICS_PATH"])
def test_corrupt_artifact_fails_startup(serving_app, serving_artifacts, artifact):

    serving_artifacts[artifact].write_bytes(b"\x00 not a pickle / not json {")

    with pytest.raises(Exception):
        with TestClient(serving_app.app):
            pass


# ============================================================
# /health: public, readiness only
# ============================================================

def test_health_is_public_and_minimal(serving_app, serving_artifacts):

    with TestClient(serving_app.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_health_reports_not_loaded_without_state(serving_app):

    from fastapi import FastAPI
    from src.serving.routes import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "model not loaded", "model_loaded": False}
