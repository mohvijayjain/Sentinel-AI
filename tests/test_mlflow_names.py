"""
MLflow experiment / registered-model names are configurable
(MLFLOW_EXPERIMENT_NAME, MLFLOW_REGISTERED_MODEL_NAME) with defaults equal
to the original literals; the Champion alias stays fixed.

Training / promotion modules import mlflow, which the rest of the suite
keeps out of the session, so those checks run in a fresh interpreter with
an unreachable dummy tracking URI and every MLflow call stubbed: nothing is
contacted, nothing is written.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _load_names(monkeypatch, **env):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for key in ("MLFLOW_EXPERIMENT_NAME", "MLFLOW_REGISTERED_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "sentinel_test_mlflow_names",
        os.path.join(REPO_ROOT, "src", "training", "mlflow_names.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# The names module itself (no mlflow import)
# ============================================================

def test_defaults_equal_original_literals(monkeypatch):

    names = _load_names(monkeypatch)

    assert names.MLFLOW_EXPERIMENT_NAME == "Sentinel-AI"
    assert names.MLFLOW_REGISTERED_MODEL_NAME == "sentinel-ai-champion"
    assert names.CHAMPION_ALIAS == "Champion"


def test_env_override(monkeypatch):

    names = _load_names(
        monkeypatch,
        MLFLOW_EXPERIMENT_NAME="sentinel_e2e_test",
        MLFLOW_REGISTERED_MODEL_NAME="sentinel-test-model",
    )

    assert names.MLFLOW_EXPERIMENT_NAME == "sentinel_e2e_test"
    assert names.MLFLOW_REGISTERED_MODEL_NAME == "sentinel-test-model"
    assert names.CHAMPION_ALIAS == "Champion"


def test_literals_live_only_in_names_module():

    training = os.path.join(REPO_ROOT, "src", "training")
    hits = []

    for filename in os.listdir(training):
        if filename.endswith(".py") and filename != "mlflow_names.py":
            with open(os.path.join(training, filename), encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    for literal in ('"Sentinel-AI"', '"sentinel-ai-champion"',
                                    '"Champion"'):
                        if literal in code:
                            hits.append(f"{filename}:{number}: {line.strip()}")

    assert hits == []


# ============================================================
# Training and promotion use the configured names (fresh interpreter)
# ============================================================

_SCRIPT = r"""
import inspect, json, os, sys
from unittest.mock import MagicMock
sys.path.insert(0, os.getcwd())

import mlflow
recorded = {"set_experiment": [], "register_model": [], "alias": [], "lookup": []}
mlflow.set_tracking_uri = lambda uri: None
mlflow.set_experiment = lambda name: recorded["set_experiment"].append(name)

from src.training import mlflow_logger, promote, retrain

# Real MLflowLogger default experiment
mlflow_logger.MlflowClient = MagicMock()
mlflow_logger.MLflowLogger()

# retrain.log_challenger builds its logger with the configured experiment
built_with = []
class RecordingLogger(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Child mocks (ml_logger.start_run, ...) also run __init__; only
        # the real construction passes experiment_name
        if "experiment_name" in kwargs:
            built_with.append(kwargs["experiment_name"])
retrain.MLflowLogger = RecordingLogger
try:
    retrain.log_challenger(model=None, metrics={})
except Exception:
    pass   # stops after construction; only the experiment name matters here

# promote: register / alias / champion lookup
promote.mlflow.register_model = lambda model_uri, name: (
    recorded["register_model"].append(name) or MagicMock(version="5"))
client = MagicMock()
client.set_registered_model_alias.side_effect = (
    lambda name, alias, version: recorded["alias"].append([name, alias, version]))
client.get_model_version_by_alias.side_effect = (
    lambda name, alias: recorded["lookup"].append([name, alias]) or (_ for _ in ()).throw(RuntimeError("stop")))
promote.client = client

promote.register_challenger("run-1")
promote.promote("5")
try:
    promote.load_champion()
except RuntimeError:
    pass

print("RESULT " + json.dumps({
    "recorded": recorded,
    "retrain_built_with": built_with,
    "logger_defaults": {
        "experiment": inspect.signature(mlflow_logger.MLflowLogger.__init__).parameters["experiment_name"].default,
        "model": inspect.signature(mlflow_logger.MLflowLogger.register_model).parameters["model_name"].default,
        "alias": inspect.signature(mlflow_logger.MLflowLogger.set_champion).parameters["alias"].default,
    },
    "promote": [promote.EXPERIMENT_NAME, promote.MODEL_NAME, promote.CHAMPION_ALIAS],
    "retrain": retrain.EXPERIMENT_NAME,
}))
"""


def _run(env_overrides):

    env = {
        k: v for k, v in os.environ.items()
        if k not in ("MLFLOW_EXPERIMENT_NAME", "MLFLOW_REGISTERED_MODEL_NAME")
    }
    env.update({"MLFLOW_TRACKING_URI": "http://127.0.0.1:9",
                "PYTHONIOENCODING": "utf-8", **env_overrides})

    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    line = [l for l in completed.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.fixture(scope="module")
def default_run():
    return _run({})


@pytest.fixture(scope="module")
def override_run():
    return _run({
        "MLFLOW_EXPERIMENT_NAME": "sentinel_e2e_test",
        "MLFLOW_REGISTERED_MODEL_NAME": "sentinel-test-model",
        # Not a supported knob: must have no effect on the alias
        "MLFLOW_CHAMPION_ALIAS": "SomethingElse",
    })


def test_defaults_preserve_behaviour(default_run):

    assert default_run["promote"] == ["Sentinel-AI", "sentinel-ai-champion", "Champion"]
    assert default_run["retrain"] == "Sentinel-AI"
    assert default_run["logger_defaults"] == {
        "experiment": "Sentinel-AI", "model": "sentinel-ai-champion",
        "alias": "Champion",
    }
    assert default_run["recorded"]["register_model"] == ["sentinel-ai-champion"]
    assert default_run["recorded"]["alias"] == [["sentinel-ai-champion", "Champion", "5"]]


def test_training_uses_configured_experiment(override_run):

    assert override_run["retrain_built_with"] == ["sentinel_e2e_test"]
    assert override_run["recorded"]["set_experiment"] == ["sentinel_e2e_test"]


def test_promotion_uses_configured_registered_model(override_run):

    recorded = override_run["recorded"]

    assert recorded["register_model"] == ["sentinel-test-model"]
    assert recorded["alias"] == [["sentinel-test-model", "Champion", "5"]]
    assert recorded["lookup"] == [["sentinel-test-model", "Champion"]]


def test_champion_alias_unchanged(override_run):

    assert override_run["promote"][2] == "Champion"
    assert override_run["logger_defaults"]["alias"] == "Champion"
