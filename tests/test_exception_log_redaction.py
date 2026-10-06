"""
Exception logging on credential-capable paths is redacted.

  * RAG indexing (src/rag/monitoring_updater.py): NVIDIA embedding and
    Chroma failures were logged with logger.exception (raw traceback).
  * MLflow promotion (src/training/promote.py): load / register / alias
    failures were logged raw. MLflow errors embed tracking and artifact
    URIs, which can carry user:password@ or tokens; promote runs inside
    automated retraining, so this bypassed the orchestrator's redaction.

They now log a redacted traceback. The exception objects are untouched:
indexing still returns False, promote still raises the same RuntimeError
chained to the original error. All secrets are fake.
"""

import logging
import os
import re
import sys

import pytest

from src.common.error_redaction import ERROR_MESSAGE_MAX_LENGTH, REDACTED

# Real monitoring_updater over fake embeddings / in-memory Chroma
from test_retraining_events import EVENT_ID, rag, store  # noqa: F401


URL_PASSWORD = "Pg5ecretPw77"
BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.ZmFrZXNpZ25hdHVyZTk"
NVIDIA_KEY = "nvapi-FAKEfakeFAKEfake1234567890abcd"
PWD = "Hunter2Fake"
JWT = "eyJ0eXAiOiJKV1QifQ.eyJyb2xlIjoiYWRtaW4ifQ.c2lnbmF0dXJlMTIz"

SECRETS = (URL_PASSWORD, BEARER, NVIDIA_KEY, PWD, JWT)


class ProviderError(Exception):
    """Shaped like an OpenAI-compatible SDK error from the NVIDIA API."""


def _provider_error():
    return ProviderError(
        "Error code: 401 - {'error': {'message': 'Invalid credentials', "
        "'type': 'authentication_error'}, 'request': {'headers': "
        f"{{'Authorization': 'Bearer {BEARER}'}}, 'api_key': '{NVIDIA_KEY}'}}}}"
    )


def _db_error():
    return ConnectionError(
        f"could not connect: postgresql://sentinel:{URL_PASSWORD}@db:5432/x "
        f"(dsn: host=db user=sentinel password={PWD}; token={JWT})"
    )


def _assert_clean(text):
    for secret in SECRETS:
        assert secret not in text, secret


# ============================================================
# RAG: NVIDIA embedding / Chroma failures
# ============================================================

def test_embedding_failure_log_is_redacted(rag, monkeypatch, caplog):

    original = _provider_error()

    def failing_embed(documents):
        raise original

    monkeypatch.setattr(rag, "embed_passages", failing_embed)

    with caplog.at_level(logging.DEBUG):
        ok = rag.upsert_retraining_event(
            EVENT_ID, "2026-10-06T12:00:00+00:00", "drift_detected",
            4.1, 4.6, True, "run-1", status="promoted",
        )

    assert ok is False                                  # behaviour unchanged
    _assert_clean(caplog.text)
    assert "Failed to index retraining event 17" in caplog.text
    assert "ProviderError: Error code: 401" in caplog.text
    assert "authentication_error" in caplog.text        # useful info kept
    assert "Traceback (most recent call last)" in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
    assert NVIDIA_KEY in str(original)                  # exception untouched


def test_chroma_failure_log_is_redacted(rag, store, monkeypatch, caplog):

    def failing_upsert(**kwargs):
        raise _db_error()

    monkeypatch.setattr(store, "upsert_documents", failing_upsert)

    with caplog.at_level(logging.DEBUG):
        ok = rag.upsert_monitoring_run(
            run_id=3, statistical_score=0.5, shap_score=0.25,
            prediction_score=0.0, overall_score=0.275, action="MONITOR",
        )

    assert ok is False
    _assert_clean(caplog.text)
    assert "Failed to index monitoring run 3" in caplog.text
    assert "postgresql://sentinel:" in caplog.text and "@db:5432" in caplog.text


def test_failed_event_chroma_text_is_redacted_and_bounded(rag, store):

    raw = f"ProviderError: {_provider_error()} | {_db_error()} " + "z" * 900

    assert rag.upsert_retraining_event(
        EVENT_ID, "2026-10-06T12:00:00+00:00", "drift_detected",
        None, None, None, None, status="failed", error_message=raw,
    ) is True

    record = store.get(f"retraining_event:{EVENT_ID}")
    stored = record["metadata"]["error_message"]

    _assert_clean(record["document"] + str(record["metadata"]))
    assert len(stored) <= ERROR_MESSAGE_MAX_LENGTH
    assert REDACTED in stored
    assert stored.startswith("ProviderError: Error code: 401")


# ============================================================
# MLflow promotion failures (fresh interpreter)
# ============================================================
#
# Importing the real promote module loads mlflow, which the rest of the
# suite asserts is never imported (conftest isolation). So these cases run
# in a fresh interpreter against an unreachable dummy tracking URI with an
# in-memory MlflowClient: nothing is contacted.

_PROMOTE_SCRIPT = r"""
import io, json, logging, os, sys
from unittest.mock import MagicMock
sys.path.insert(0, os.getcwd())

buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

from src.training import promote

client = MagicMock(name="MlflowClient")
client.list_artifacts.return_value = [MagicMock(path="model")]
promote.client = client

error_text = sys.argv[1]
results = {}

def run(name, call):
    original = RuntimeError(error_text)
    buffer.seek(0); buffer.truncate()
    outcome = {}
    try:
        call(original)
    except RuntimeError as raised:
        outcome = {"raised": str(raised),
                   "cause_is_original": raised.__cause__ is original}
    outcome["log"] = buffer.getvalue()
    results[name] = outcome

def register(original):
    def fail(*a, **k): raise original
    promote.mlflow.register_model = fail
    promote.register_challenger("run-123")

def alias(original):
    client.set_registered_model_alias.side_effect = original
    promote.promote("8")

def load(original):
    def fail(*a, **k): raise original
    promote.mlflow.lightgbm.load_model = fail
    promote.load_challenger("run-123")

run("register", register)
run("alias", alias)
run("load", load)
print("RESULT " + json.dumps(results))
"""


@pytest.fixture(scope="module")
def promote_results():

    import json
    import subprocess

    error_text = (
        f"API request to http://admin:{URL_PASSWORD}@mlflow:5000/api/2.0/mlflow "
        f"failed with Authorization: Bearer {BEARER}"
    )

    env = {**os.environ, "MLFLOW_TRACKING_URI": "http://127.0.0.1:9",
           "PYTHONIOENCODING": "utf-8"}

    completed = subprocess.run(
        [sys.executable, "-c", _PROMOTE_SCRIPT, error_text],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
        env=env, capture_output=True, text=True, timeout=300,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]

    line = [l for l in completed.stdout.splitlines() if l.startswith("RESULT ")][-1]
    results = json.loads(line[len("RESULT "):])
    results["_stderr"] = completed.stderr
    return results


@pytest.mark.parametrize(
    "case, raised, logged",
    [
        ("register", "Could not register Challenger", "Failed to register Challenger"),
        ("alias", "Could not promote version 8", "Failed to promote version 8"),
        ("load", "Could not load Challenger", "Failed to load Challenger"),
    ],
)
def test_mlflow_failure_logs_are_redacted(promote_results, case, raised, logged):

    outcome = promote_results[case]

    # Behaviour unchanged: same RuntimeError, chained to the original
    assert raised in outcome["raised"]
    if case != "load":
        assert outcome["cause_is_original"] is True

    _assert_clean(outcome["log"] + promote_results["_stderr"])
    assert logged in outcome["log"]
    assert "mlflow:5000/api/2.0/mlflow" in outcome["log"]     # context kept
    assert "Traceback (most recent call last)" in outcome["log"]


# ============================================================
# Static guard: no raw logger.exception on credential-capable layers
# ============================================================

@pytest.mark.parametrize(
    "path",
    ["src/rag", "src/training", "src/database", "src/monitoring", "src/api"],
)
def test_no_raw_logger_exception(path):

    root = os.path.join(os.path.dirname(__file__), os.pardir, *path.split("/"))
    offending = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                    for number, line in enumerate(f, 1):
                        code = line.split("#", 1)[0]
                        if re.search(r"\blogger\.exception\(|exc_info\s*=\s*True", code):
                            offending.append(f"{filename}:{number}")

    assert offending == []
