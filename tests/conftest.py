"""
Shared test setup for the Sentinel-AI drift scoring suite.

drift_scorer imports, at module top, three modules that drag in live
infrastructure on import (DB engine, NVIDIA/OpenAI embedding client that
raises without NVIDIA_API_KEY, mlflow/lightgbm). We register fake
modules under those names in sys.modules BEFORE anything imports
drift_scorer, so the suite runs with no services, no env vars and no
network. Python's import system returns a cached sys.modules entry
without importing its parent packages, so the real modules never load.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ============================================================
# Make `src.` importable regardless of how pytest is launched
# ============================================================

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ============================================================
# Stub infra-bound modules imported by drift_scorer
# ============================================================

STUBBED_MODULES = {
    "src.database.drift_repository": [
        "insert_monitoring_run",
        "insert_drift_scores",
        "insert_prediction_log",
    ],
    "src.rag.monitoring_updater": [
        "upsert_monitoring_run",
    ],
    "src.training.orchestrator": [
        "run_retraining_pipeline",
    ],
}


def _install_stubs():

    for module_name, attributes in STUBBED_MODULES.items():

        stub = types.ModuleType(module_name)
        stub.__file__ = f"<stub {module_name}>"

        for attribute in attributes:
            setattr(stub, attribute, MagicMock(name=attribute))

        sys.modules[module_name] = stub


# Runs at conftest import time, i.e. before any test module is collected
_install_stubs()


# ============================================================
# CSV headers, matching the real files in reports/
# ============================================================

STATISTICAL_COLUMNS = [
    "feature", "type", "psi", "ks_statistic", "ks_pvalue",
    "js_divergence", "wasserstein", "severity",
]

SHAP_COLUMNS = [
    "feature", "ref_importance", "cur_importance", "absolute_shift",
    "relative_shift_%", "direction", "ref_rank", "cur_rank",
    "rank_shift", "is_drifted", "severity",
]

PREDICTION_COLUMNS = [
    "psi", "ks_statistic", "ks_pvalue", "js_divergence",
    "wasserstein", "severity",
]


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def drift_scorer():
    """The drift_scorer module, imported against the stubs."""

    from src.monitoring import drift_scorer as module

    return module


@pytest.fixture
def write_report(tmp_path):
    """
    Write a report CSV with the real header into tmp_path.

    Usage: write_report("statistical", [{"severity": "LOW"}, ...])
    Columns not given in a row are filled with neutral placeholders.
    """

    columns_by_kind = {
        "statistical": STATISTICAL_COLUMNS,
        "shap": SHAP_COLUMNS,
        "prediction": PREDICTION_COLUMNS,
    }

    def _write(kind, rows):

        columns = columns_by_kind[kind]
        defaults = {column: 0 for column in columns}
        defaults.update(feature="f", type="numerical", direction="INCREASED")

        records = [
            {column: {**defaults, **row}.get(column) for column in columns}
            for row in rows
        ]

        path = tmp_path / f"{kind}_drift.csv"

        pd.DataFrame(records, columns=columns).to_csv(path, index=False)

        return str(path)

    return _write


# ============================================================
# The real FastAPI app (serving lifecycle / auth / CORS tests)
# ============================================================

@pytest.fixture(scope="session")
def serving_app():
    """
    Import src.serving.app exactly once, against stubs.

    Once only: the Prometheus Instrumentator registers its metrics in the
    process-global registry at import time. While importing, the monitoring
    router's DB getters come from the drift_repository stub above, and
    src.rag.rag (whose retriever builds the NVIDIA client) is a fake, so no
    database, ChromaDB or NVIDIA key is needed. Tests patch the names the
    routers bound (e.g. monitoring.get_latest_monitoring_run, rag.ask).
    """

    getters = [
        "get_latest_monitoring_run", "get_monitoring_history",
        "get_drifted_features", "get_feature_drift_scores",
        "get_retraining_events", "get_prediction_logs",
    ]

    fake_rag = types.ModuleType("src.rag.rag")
    fake_rag.ask = MagicMock(name="ask")

    with pytest.MonkeyPatch.context() as mp:
        repo = sys.modules["src.database.drift_repository"]
        for name in getters:
            mp.setattr(repo, name, MagicMock(name=name), raising=False)
        mp.setitem(sys.modules, "src.rag.rag", fake_rag)

        from src.serving import app as app_module

    return app_module


class FixedModel:
    """Picklable stand-in for the LightGBM model: always 600 seconds."""

    def __init__(self):
        self.seen_columns = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return [600.0]


SERVING_FAKE_KEY = "sntl_live_FAKE_2x_a1b2c3d4e5f60718"

SERVING_FEATURES = ["trip_distance", "pickup_hour", "is_weekend"]

SERVING_METRICS = {"rmse": 300.0, "rmse_min": 5.0, "mae": 200.0, "r2": 0.85}


@pytest.fixture
def serving_artifacts(serving_app, tmp_path, monkeypatch):
    """
    Point the canonical loader at temp model/features/metrics files and set
    a fake API key everywhere it is read (config for startup, routes and
    the /chat route for requests). Never touches the real model/ dir.
    """

    import json
    import pickle

    from src.serving import config, model_loader, routes

    paths = {
        "MODEL_PATH": tmp_path / "model.pkl",
        "FEATURES_PATH": tmp_path / "features.json",
        "METRICS_PATH": tmp_path / "metrics.json",
    }

    paths["MODEL_PATH"].write_bytes(pickle.dumps(FixedModel()))
    paths["FEATURES_PATH"].write_text(json.dumps(SERVING_FEATURES))
    paths["METRICS_PATH"].write_text(json.dumps(SERVING_METRICS))

    for name, path in paths.items():
        monkeypatch.setattr(model_loader, name, str(path))

    for module in (config, routes, sys.modules["src.api.routes.rag"]):
        monkeypatch.setattr(module, "API_KEY", SERVING_FAKE_KEY)

    return paths
