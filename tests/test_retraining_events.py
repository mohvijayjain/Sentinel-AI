"""
Retraining event auditing (failed attempts) and the canonical
triggered_at, across the real orchestrator, the real repository
insert_retraining_event and the real RAG upsert_retraining_event.

conftest stubs the orchestrator, drift_repository and monitoring_updater
module names, and the real modules import infra (SQLAlchemy engine,
ChromaDB, NVIDIA embeddings). Each real module is therefore executed
from its source file under a private name, after registering fakes for
exactly the infra it imports:

  orchestrator         fake src.training.retrain / src.training.promote
  drift_repository     fake src.database.postgres.engine (records SQL)
  monitoring_updater   fake src.rag.chroma_store / src.rag.embeddings
                       (an in-memory store that can be read back)
"""

import importlib.util
import logging
import os
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

RUN_ID = "challenger-run-abc123"
EVENT_ID = 17

RETRAIN_RESULT = {
    "run_id": RUN_ID,
    "metrics": {"rmse": 4.10, "mae": 2.05, "r2": 0.912},
}

PROMOTED = {
    "promoted": True,
    "mlflow_run_id": RUN_ID,
    "new_model_rmse": 4.10,
    "new_model_mae": 2.05,
    "new_model_r2": 0.912,
    "champion_rmse": 4.60,
    "champion_mae": 2.31,
    "champion_r2": 0.894,
    "champion_version": "7",
    "new_version": "8",
}

REJECTED = {**PROMOTED, "promoted": False, "new_model_rmse": 4.90,
            "new_version": None}


def _load_source(private_name, relative_path):

    spec = importlib.util.spec_from_file_location(
        private_name, os.path.join(REPO_ROOT, *relative_path.split("/"))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# Real repository against a recording engine
# ============================================================

class RecordingEngine:
    """Stands in for the SQLAlchemy engine; records each execute()."""

    def __init__(self, event_id=EVENT_ID, error=None):
        self.statements = []
        self.event_id = event_id
        self.error = error

    def begin(self):
        engine = self

        class _Transaction:
            def __enter__(self):
                return engine

            def __exit__(self, *exc):
                return False

        return _Transaction()

    connect = begin

    def execute(self, query, params=None):
        if self.error is not None:
            raise self.error
        self.statements.append((str(query), dict(params or {})))
        result = MagicMock()
        result.scalar_one.return_value = self.event_id
        return result


@pytest.fixture
def engine(monkeypatch):

    recording = RecordingEngine()

    fake_postgres = types.ModuleType("src.database.postgres")
    fake_postgres.engine = recording
    monkeypatch.setitem(sys.modules, "src.database.postgres", fake_postgres)

    return recording


@pytest.fixture
def repository(engine):

    return _load_source(
        "sentinel_real_drift_repository", "src/database/drift_repository.py"
    )


# ============================================================
# Real RAG updater against an in-memory store
# ============================================================

class InMemoryStore:
    """Minimal ChromaStore double: upsert by id, read back by id."""

    def __init__(self):
        self.records = {}

    def upsert_documents(self, documents, embeddings, ids, metadatas=None):
        for i, record_id in enumerate(ids):
            self.records[record_id] = {
                "document": documents[i],
                "embedding": embeddings[i],
                "metadata": (metadatas or [{}] * len(ids))[i],
            }

    def get(self, record_id):
        return self.records.get(record_id)


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def rag(monkeypatch, store):

    fake_chroma = types.ModuleType("src.rag.chroma_store")
    fake_chroma.ChromaStore = lambda: store

    fake_embeddings = types.ModuleType("src.rag.embeddings")
    fake_embeddings.embed_passages = lambda docs: [[0.0, 1.0] for _ in docs]

    monkeypatch.setitem(sys.modules, "src.rag.chroma_store", fake_chroma)
    monkeypatch.setitem(sys.modules, "src.rag.embeddings", fake_embeddings)

    return _load_source(
        "sentinel_real_monitoring_updater", "src/rag/monitoring_updater.py"
    )


# ============================================================
# Real orchestrator against fake retrain / promote
# ============================================================

@pytest.fixture
def orchestrator(monkeypatch):

    fake_retrain = types.ModuleType("src.training.retrain")
    fake_retrain.main = MagicMock(name="retrain.main")

    fake_promote = types.ModuleType("src.training.promote")
    fake_promote.main = MagicMock(name="promote.main")

    monkeypatch.setitem(sys.modules, "src.training.retrain", fake_retrain)
    monkeypatch.setitem(sys.modules, "src.training.promote", fake_promote)

    monkeypatch.setattr(
        sys.modules["src.database.drift_repository"],
        "insert_retraining_event", MagicMock(), raising=False,
    )
    monkeypatch.setattr(
        sys.modules["src.rag.monitoring_updater"],
        "upsert_retraining_event", MagicMock(), raising=False,
    )

    module = _load_source(
        "sentinel_real_orchestrator", "src/training/orchestrator.py"
    )

    # Count every timestamp the orchestrator generates
    clock = MagicMock(wraps=datetime)
    monkeypatch.setattr(module, "datetime", clock)
    module.clock = clock

    return module


@pytest.fixture
def fakes(orchestrator, monkeypatch):
    """Mock collaborators logging into one ordered call list."""

    calls = []

    def recorder(name, return_value):
        def side_effect(*args, **kwargs):
            calls.append(name)
            return return_value
        return MagicMock(name=name, side_effect=side_effect)

    ns = types.SimpleNamespace(
        calls=calls,
        retrain_main=recorder("retrain.main", RETRAIN_RESULT),
        promote_model=recorder("promote_model", PROMOTED),
        insert_event=recorder("insert_retraining_event", EVENT_ID),
        upsert_event=recorder("upsert_retraining_event", True),
    )

    monkeypatch.setattr(orchestrator.retrain, "main", ns.retrain_main)
    monkeypatch.setattr(orchestrator, "promote_model", ns.promote_model)
    monkeypatch.setattr(orchestrator, "insert_retraining_event", ns.insert_event)
    monkeypatch.setattr(orchestrator, "upsert_retraining_event", ns.upsert_event)

    return ns


@pytest.fixture
def wired(orchestrator, repository, rag, engine, store, monkeypatch):
    """
    Orchestrator wired to the REAL repository and RAG functions, with
    only retrain / promote / engine / Chroma faked underneath.
    """

    retrain_main = MagicMock(return_value=RETRAIN_RESULT)
    promote_model = MagicMock(return_value=PROMOTED)

    monkeypatch.setattr(orchestrator.retrain, "main", retrain_main)
    monkeypatch.setattr(orchestrator, "promote_model", promote_model)
    monkeypatch.setattr(orchestrator, "insert_retraining_event",
                        repository.insert_retraining_event)
    monkeypatch.setattr(orchestrator, "upsert_retraining_event",
                        rag.upsert_retraining_event)

    return types.SimpleNamespace(
        run=orchestrator.run_retraining_pipeline,
        clock=orchestrator.clock,
        retrain_main=retrain_main,
        promote_model=promote_model,
        engine=engine,
        store=store,
    )


def _postgres_row(engine):
    """The single INSERT INTO retraining_events executed, as (sql, params)."""

    inserts = [s for s in engine.statements if "retraining_events" in s[0]]
    assert len(inserts) == 1, inserts
    return inserts[0]


def _chroma_record(store):
    records = [r for k, r in store.records.items()
               if k.startswith("retraining_event:")]
    assert len(records) == 1, records
    return records[0]


# ============================================================
# Repository: insert_retraining_event
# ============================================================

def test_insert_writes_caller_timestamp_status_and_error(repository, engine):

    event_id = repository.insert_retraining_event(
        triggered_reason="drift_detected",
        new_model_rmse=None,
        champion_rmse=None,
        promoted=None,
        mlflow_run_id=None,
        triggered_at="2026-10-06T12:00:00.123456+00:00",
        status="failed",
        error_message="RuntimeError: boom",
    )

    sql, params = _postgres_row(engine)

    assert event_id == EVENT_ID
    assert params == {
        "triggered_at": "2026-10-06T12:00:00.123456+00:00",
        "triggered_reason": "drift_detected",
        "new_model_rmse": None,
        "champion_rmse": None,
        "promoted": None,
        "mlflow_run_id": None,
        "status": "failed",
        "error_message": "RuntimeError: boom",
    }

    # Stored as UTC wall-clock in the existing TIMESTAMP column
    assert "CAST(:triggered_at AS TIMESTAMPTZ) AT TIME ZONE 'UTC'" in sql


def test_insert_generates_no_timestamp_of_its_own(repository, engine):

    repository.insert_retraining_event(
        "drift_detected", 4.1, 4.6, True, RUN_ID,
        triggered_at="2026-10-06T12:00:00+00:00", status="promoted",
    )

    sql, params = _postgres_row(engine)

    for clock_sql in ("NOW()", "CURRENT_TIMESTAMP", "LOCALTIMESTAMP"):
        assert clock_sql not in sql.upper()

    assert params["error_message"] is None


@pytest.mark.parametrize("missing", ["triggered_at", "status"])
def test_insert_requires_timestamp_and_status(repository, missing):
    """Keyword-only and required: no caller can fall back to NOW()."""

    kwargs = {"triggered_at": "2026-10-06T12:00:00+00:00", "status": "promoted"}
    del kwargs[missing]

    with pytest.raises(TypeError):
        repository.insert_retraining_event(
            "drift_detected", 4.1, 4.6, True, RUN_ID, **kwargs
        )


# ============================================================
# RAG: upsert_retraining_event
# ============================================================

def test_promoted_document_text_unchanged(rag):
    """Completed events render exactly as before the status change."""

    document = rag.build_retraining_document(
        event_id=5, triggered_at="2026-10-06T12:00:00+00:00",
        triggered_reason="drift_detected", new_model_rmse=4.1,
        champion_rmse=4.6, status="promoted", mlflow_run_id=RUN_ID,
    )

    assert document == (
        "Sentinel-AI Retraining Event 5.\n"
        "Triggered at: 2026-10-06T12:00:00+00:00.\n"
        "Triggered reason: drift_detected.\n"
        "Challenger model RMSE: 4.100.\n"
        "Champion model RMSE: 4.600.\n"
        "Promotion decision: PROMOTED.\n"
        f"MLflow run ID: {RUN_ID}."
    )


def test_failed_event_indexes_and_reads_back(rag, store):

    ok = rag.upsert_retraining_event(
        event_id=EVENT_ID,
        triggered_at="2026-10-06T12:00:00+00:00",
        triggered_reason="drift_detected",
        new_model_rmse=None,
        champion_rmse=None,
        promoted=None,
        mlflow_run_id=None,
        status="failed",
        error_message="RuntimeError: optuna blew up",
    )

    assert ok is True

    record = store.get(f"retraining_event:{EVENT_ID}")

    assert record["metadata"] == {
        "source": "retraining_events",
        "event_id": EVENT_ID,
        "triggered_at": "2026-10-06T12:00:00+00:00",
        "triggered_reason": "drift_detected",
        "status": "failed",
        "error_message": "RuntimeError: optuna blew up",
    }
    assert None not in record["metadata"].values()
    assert "Promotion decision: FAILED." in record["document"]
    assert "Challenger model RMSE: not available." in record["document"]
    assert "Failure: RuntimeError: optuna blew up" in record["document"]


def test_unknown_status_is_not_indexed_and_never_raises(rag, store):

    ok = rag.upsert_retraining_event(
        EVENT_ID, "2026-10-06T12:00:00+00:00", "drift_detected",
        4.1, 4.6, True, RUN_ID, status="maybe",
    )

    assert ok is False
    assert store.records == {}


# ============================================================
# Issue #2: failed attempts are auditable
# ============================================================

def test_retrain_failure_persists_failed_event(orchestrator, fakes):

    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    kwargs = fakes.insert_event.call_args.kwargs

    assert kwargs["status"] == "failed"
    assert kwargs["triggered_reason"] == "drift_detected"
    assert kwargs["error_message"] == "RuntimeError: optuna blew up"

    # Nothing invented for an attempt that produced no model
    for key in ("new_model_rmse", "champion_rmse", "promoted", "mlflow_run_id"):
        assert kwargs[key] is None


def test_original_exception_propagates_unchanged(orchestrator, fakes):

    original = RuntimeError("optuna blew up")
    fakes.retrain_main.side_effect = original

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert excinfo.value is original


def test_no_promotion_when_retrain_fails(orchestrator, fakes):

    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    fakes.promote_model.assert_not_called()


def test_promote_failure_records_real_run_id(orchestrator, fakes):
    """Retraining did produce an MLflow run; that real id is kept."""

    fakes.promote_model.side_effect = RuntimeError("mlflow unreachable")

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    kwargs = fakes.insert_event.call_args.kwargs

    assert kwargs["status"] == "failed"
    assert kwargs["mlflow_run_id"] == RUN_ID
    assert kwargs["new_model_rmse"] is None


@pytest.mark.parametrize(
    "fail",
    ["retrain", "promote", "bad-promote", "no-run-id", "none"],
)
def test_exactly_one_event_per_attempt(orchestrator, fakes, fail):

    if fail == "retrain":
        fakes.retrain_main.side_effect = RuntimeError("x")
    elif fail == "promote":
        fakes.promote_model.side_effect = RuntimeError("x")
    elif fail == "bad-promote":
        fakes.promote_model.side_effect = None
        fakes.promote_model.return_value = 1
    elif fail == "no-run-id":
        fakes.retrain_main.side_effect = None
        fakes.retrain_main.return_value = {"metrics": {}}

    try:
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")
    except Exception:
        pass

    assert fakes.insert_event.call_count == 1
    assert fakes.upsert_event.call_count == 1
    assert fakes.calls[-2:] == ["insert_retraining_event",
                                "upsert_retraining_event"]


def test_failed_event_indexed_with_its_postgres_id(orchestrator, fakes):

    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    kwargs = fakes.upsert_event.call_args.kwargs

    assert kwargs["event_id"] == EVENT_ID
    assert kwargs["status"] == "failed"
    assert kwargs["error_message"] == "RuntimeError: optuna blew up"
    assert fakes.calls[-2:] == ["insert_retraining_event",
                                "upsert_retraining_event"]


def test_failed_event_end_to_end_in_both_stores(wired):

    wired.retrain_main.side_effect = RuntimeError("optuna blew up")

    with pytest.raises(RuntimeError, match="optuna blew up"):
        wired.run(triggered_reason="drift_detected")

    _, params = _postgres_row(wired.engine)
    record = _chroma_record(wired.store)

    assert params["status"] == "failed"
    assert params["promoted"] is None
    assert record["metadata"]["status"] == "failed"
    assert record["metadata"]["event_id"] == EVENT_ID


def test_failed_record_insert_failure_keeps_original_error(
    orchestrator, fakes, caplog
):
    """Postgres down while recording: no false persistence, no Chroma."""

    original = RuntimeError("optuna blew up")
    fakes.retrain_main.side_effect = original
    fakes.insert_event.side_effect = ConnectionError("postgres down")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            orchestrator.run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

    assert excinfo.value is original
    fakes.upsert_event.assert_not_called()
    assert "Could not record failed retraining attempt" in caplog.text


def test_success_insert_failure_writes_no_failed_row(orchestrator, fakes):
    """Recording a completed attempt fails: one insert, no 'failed' retry."""

    fakes.insert_event.side_effect = ConnectionError("postgres down")

    with pytest.raises(ConnectionError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert fakes.insert_event.call_count == 1
    assert fakes.insert_event.call_args.kwargs["status"] == "promoted"
    fakes.upsert_event.assert_not_called()


def test_chroma_failure_leaves_postgres_row_and_original_error(
    orchestrator, fakes
):
    original = RuntimeError("optuna blew up")
    fakes.retrain_main.side_effect = original
    fakes.upsert_event.side_effect = RuntimeError("chroma down")

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert excinfo.value is original
    fakes.insert_event.assert_called_once()


def test_error_message_is_bounded_single_line(orchestrator, fakes):

    fakes.retrain_main.side_effect = RuntimeError("line one\nline two " * 200)

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    message = fakes.insert_event.call_args.kwargs["error_message"]

    assert len(message) <= orchestrator.ERROR_MESSAGE_MAX_LENGTH == 500
    assert "\n" not in message
    assert message.startswith("RuntimeError: line one line two")
    assert message.endswith("...")


@pytest.mark.parametrize(
    "kwargs, reason",
    [({"triggered_reason": "drift_detected"}, "drift_detected"),
     ({}, "manual_retraining")],
    ids=["auto", "manual"],
)
def test_failed_event_keeps_triggered_reason(orchestrator, fakes, kwargs, reason):

    fakes.retrain_main.side_effect = RuntimeError("x")

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(**kwargs)

    assert fakes.insert_event.call_args.kwargs["triggered_reason"] == reason
    assert fakes.upsert_event.call_args.kwargs["triggered_reason"] == reason


@pytest.mark.parametrize(
    "promotion, status",
    [(PROMOTED, "promoted"), (REJECTED, "rejected")],
    ids=["promoted", "rejected"],
)
def test_completed_attempts_record_their_status(
    orchestrator, fakes, promotion, status
):
    fakes.promote_model.side_effect = None
    fakes.promote_model.return_value = promotion

    result = orchestrator.run_retraining_pipeline(
        triggered_reason="drift_detected"
    )

    assert fakes.insert_event.call_args.kwargs["status"] == status
    assert fakes.upsert_event.call_args.kwargs["status"] == status
    assert fakes.insert_event.call_args.kwargs.get("error_message") is None
    assert result["promoted"] is promotion["promoted"]
    assert result["new_version"] == promotion["new_version"]


# ============================================================
# Issue #3: one canonical triggered_at
# ============================================================

def _outcome(wired, outcome):

    if outcome == "rejected":
        wired.promote_model.return_value = REJECTED
    elif outcome == "failed":
        wired.retrain_main.side_effect = RuntimeError("optuna blew up")

    try:
        wired.run(triggered_reason="drift_detected")
    except RuntimeError:
        assert outcome == "failed"


OUTCOMES = ["promoted", "rejected", "failed"]


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_exactly_one_timestamp_per_attempt(wired, outcome):

    _outcome(wired, outcome)

    assert wired.clock.now.call_count == 1
    wired.clock.now.assert_called_once_with(timezone.utc)


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_same_timestamp_in_postgres_and_chroma(
    wired, orchestrator, monkeypatch, outcome
):
    # Capture the value the orchestrator actually generates
    generated = []
    real_canonical = orchestrator._canonical_triggered_at

    def capturing_canonical():
        generated.append(real_canonical())
        return generated[-1]

    monkeypatch.setattr(orchestrator, "_canonical_triggered_at",
                        capturing_canonical)

    _outcome(wired, outcome)

    _, params = _postgres_row(wired.engine)
    record = _chroma_record(wired.store)

    postgres_at = params["triggered_at"]
    chroma_at = record["metadata"]["triggered_at"]

    assert postgres_at == chroma_at
    assert f"Triggered at: {postgres_at}." in record["document"]

    # One tz-aware UTC instant, ISO-8601
    parsed = datetime.fromisoformat(postgres_at)
    assert parsed.utcoffset().total_seconds() == 0

    # ...and it is exactly the one value generated for this attempt
    assert generated == [postgres_at]


def test_timestamp_taken_at_start_of_attempt(orchestrator, fakes, monkeypatch):
    """Generated before retraining runs, not after it finishes."""

    order = []

    real_canonical = orchestrator._canonical_triggered_at

    def tracking_canonical():
        order.append("timestamp")
        return real_canonical()

    monkeypatch.setattr(orchestrator, "_canonical_triggered_at",
                        tracking_canonical)
    fakes.retrain_main.side_effect = (
        lambda: order.append("retrain") or RETRAIN_RESULT
    )

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert order == ["timestamp", "retrain"]


def test_each_attempt_gets_its_own_timestamp(orchestrator, fakes):

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")
    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    first, second = (
        call.kwargs["triggered_at"]
        for call in fakes.insert_event.call_args_list
    )

    assert orchestrator.clock.now.call_count == 2
    assert first <= second
