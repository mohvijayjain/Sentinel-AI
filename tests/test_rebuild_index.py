"""
Phase 2.4: PostgreSQL -> ChromaDB rebuild (src/rag/rebuild_index.py).

Runs the real rebuild, the real monitoring_updater upserts, the real
ChromaStore (a temp PersistentClient directory, never data/chroma) and the
real retriever. Only the NVIDIA embedding module is faked: a deterministic
hashed bag-of-words vector, so similar text lands close together and the
retriever can be exercised for real. "PostgreSQL" is a temp SQLite file
with the monitoring_runs / retraining_events columns the rebuild reads.
"""

import hashlib
import importlib.util
import math
import os
import re
import sys
import types

import pytest
from sqlalchemy import create_engine, text


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DIM = 64

MONITORING_ROWS = [
    # id, statistical, shap, prediction, overall, action
    (1, 0.10, 0.05, 0.02, 0.0670, "WAIT"),
    (2, 0.30, 0.40, 0.20, 0.3000, "MONITOR"),
    (3, 0.90, 0.80, 0.70, 0.8100, "RETRAIN"),
]

RETRAINING_ROWS = [
    # id, triggered_at, reason, new_rmse, champion_rmse, promoted,
    # mlflow_run_id, status, error_message
    (1, "2026-10-04T15:47:04.372230+00:00", "manual_retraining",
     347.63, 326.56, False, "run-aaa", None, None),          # legacy row
    (2, "2026-10-05T09:00:00+00:00", "drift_detected",
     300.10, 326.56, True, "run-bbb", "promoted", None),
    (3, "2026-10-06T10:00:00+00:00", "drift_detected",
     None, None, None, None, "failed", "Training data unavailable"),
    (4, "2026-10-06T11:00:00+00:00", "manual_retraining",
     None, None, None, None, None, None),                     # unlabelled
]

EXPECTED_IDS = {
    "monitoring_run:1", "monitoring_run:2", "monitoring_run:3",
    "retraining_event:1", "retraining_event:2", "retraining_event:3",
}


def _load(private_name, relative):
    spec = importlib.util.spec_from_file_location(
        private_name, os.path.join(REPO_ROOT, *relative.split("/"))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_vector(text_value):
    """Hashed bag of words, L2-normalised: deterministic, no network."""

    vector = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text_value.lower()):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbeddings:
    """embed_passages / embed_query, with an optional failure switch."""

    def __init__(self):
        self.calls = 0
        self.fail_on_calls = set()
        self.fail_always = False

    def embed_passages(self, texts):
        self.calls += 1
        if self.fail_always or self.calls in self.fail_on_calls:
            raise ConnectionError("NVIDIA embedding API unavailable")
        return [fake_vector(t) for t in texts]

    def embed_query(self, query):
        return fake_vector(query)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def embeddings(monkeypatch):

    fake = FakeEmbeddings()
    module = types.ModuleType("src.rag.embeddings")
    module.embed_passages = fake.embed_passages
    module.embed_query = fake.embed_query
    monkeypatch.setitem(sys.modules, "src.rag.embeddings", module)
    return fake


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):

    from src.rag import chroma_store

    path = tmp_path / "chroma"
    monkeypatch.setattr(chroma_store, "CHROMA_DIR", path)
    return path


@pytest.fixture
def updater(embeddings, chroma_dir):
    """The real monitoring_updater (conftest stubs its module name)."""

    return _load("sentinel_real_updater_rebuild", "src/rag/monitoring_updater.py")


@pytest.fixture
def rebuild(updater, monkeypatch):

    module = _load("sentinel_real_rebuild_index", "src/rag/rebuild_index.py")
    monkeypatch.setattr(module, "monitoring_updater", updater)
    return module


@pytest.fixture
def store(updater):
    """The same ChromaStore instance the upserts write to."""

    return updater._get_store()


def _create_db(path, monitoring=MONITORING_ROWS, retraining=RETRAINING_ROWS):

    engine = create_engine(f"sqlite:///{path}")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE monitoring_runs (
                id INTEGER PRIMARY KEY, run_time TEXT,
                statistical_score REAL NOT NULL, shap_score REAL NOT NULL,
                prediction_score REAL NOT NULL, overall_score REAL NOT NULL,
                action TEXT NOT NULL, drifted_features TEXT, report_path TEXT)
        """))
        conn.execute(text("""
            CREATE TABLE retraining_events (
                id INTEGER PRIMARY KEY, triggered_at TEXT NOT NULL,
                triggered_reason TEXT, new_model_rmse REAL,
                champion_rmse REAL, promoted BOOLEAN, mlflow_run_id TEXT,
                status TEXT, error_message TEXT)
        """))
        for row in monitoring:
            conn.execute(text(
                "INSERT INTO monitoring_runs (id, statistical_score, shap_score, "
                "prediction_score, overall_score, action) "
                "VALUES (:i, :s, :h, :p, :o, :a)"),
                dict(zip("ishpoa", row)))
        for row in retraining:
            conn.execute(text(
                "INSERT INTO retraining_events (id, triggered_at, "
                "triggered_reason, new_model_rmse, champion_rmse, promoted, "
                "mlflow_run_id, status, error_message) "
                "VALUES (:i, :t, :r, :n, :c, :p, :m, :s, :e)"),
                dict(zip("itrncpmse", row)))

    return engine


@pytest.fixture
def db(tmp_path):
    return _create_db(tmp_path / "pg.sqlite")


def _snapshot(engine):
    """Every PostgreSQL row, to prove the rebuild never writes to it."""

    with engine.connect() as conn:
        return (
            conn.execute(text("SELECT * FROM monitoring_runs ORDER BY id")).all(),
            conn.execute(text("SELECT * FROM retraining_events ORDER BY id")).all(),
        )


def _indexed(store):
    got = store.get()
    return dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))


# ============================================================
# Rebuild contents, ids, metadata
# ============================================================

def test_rebuild_indexes_all_history_with_deterministic_ids(rebuild, store, db):

    report = rebuild.rebuild_index(db)

    assert report.ok
    assert report.monitoring_runs_indexed == [1, 2, 3]
    assert report.retraining_events_indexed == [1, 2, 3]
    assert report.retraining_events_skipped == [4]
    assert set(_indexed(store)) == EXPECTED_IDS
    assert store.count() == len(EXPECTED_IDS)


def test_rebuilt_documents_and_metadata_match_the_builders(rebuild, updater, store, db):

    rebuild.rebuild_index(db)
    indexed = _indexed(store)

    document, metadata = indexed["monitoring_run:3"]
    assert document == updater.build_monitoring_document(3, 0.9, 0.8, 0.7, 0.81, "RETRAIN")
    assert metadata == {"source": "monitoring_runs", "run_id": 3,
                        "action": "RETRAIN", "overall_score": 0.81}

    document, metadata = indexed["retraining_event:2"]
    assert document == updater.build_retraining_document(
        2, "2026-10-05T09:00:00+00:00", "drift_detected", 300.10, 326.56,
        "promoted", "run-bbb")
    assert metadata == {
        "source": "retraining_events", "event_id": 2,
        "triggered_at": "2026-10-05T09:00:00+00:00",
        "triggered_reason": "drift_detected", "status": "promoted",
        "promoted": True, "new_model_rmse": 300.10, "champion_rmse": 326.56,
        "mlflow_run_id": "run-bbb",
    }

    document, metadata = indexed["retraining_event:3"]
    assert "Promotion decision: FAILED." in document
    assert "Failure: Training data unavailable" in document
    assert metadata["status"] == "failed"
    assert "promoted" not in metadata and "new_model_rmse" not in metadata


def test_legacy_row_without_status_is_labelled_like_migration_001(rebuild, store, db):

    rebuild.rebuild_index(db)

    document, metadata = _indexed(store)["retraining_event:1"]
    assert metadata["status"] == "rejected"
    assert metadata["promoted"] is False
    assert "Promotion decision: REJECTED." in document


def test_rebuild_matches_the_normal_write_through(rebuild, updater, store, db):

    # what the monitoring pipeline writes for run 2...
    assert updater.upsert_monitoring_run(2, 0.30, 0.40, 0.20, 0.30, "MONITOR")
    before = _indexed(store)["monitoring_run:2"]

    rebuild.rebuild_index(db)

    # ...is exactly what the rebuild writes, under the same id
    assert _indexed(store)["monitoring_run:2"] == before
    assert store.count() == len(EXPECTED_IDS)


def test_iso_utc_renders_naive_utc_timestamps_like_the_orchestrator(rebuild):

    from datetime import datetime, timezone

    naive = datetime(2026, 10, 4, 16, 32, 8, 538367)

    assert rebuild._iso_utc(naive) == "2026-10-04T16:32:08.538367+00:00"
    assert rebuild._iso_utc(naive.replace(tzinfo=timezone.utc)) \
        == "2026-10-04T16:32:08.538367+00:00"


# ============================================================
# Idempotency
# ============================================================

def test_double_run_creates_no_duplicates(rebuild, store, db):

    rebuild.rebuild_index(db)
    first = _indexed(store)

    report = rebuild.rebuild_index(db)

    assert report.ok
    assert store.count() == len(EXPECTED_IDS)
    assert _indexed(store) == first


def test_rebuild_after_index_loss_restores_everything(rebuild, store, db):

    rebuild.rebuild_index(db)
    expected = _indexed(store)

    store.clear()                           # Chroma lost / wiped
    assert store.count() == 0

    assert rebuild.rebuild_index(db).ok
    assert _indexed(store) == expected


def test_empty_history_is_handled(rebuild, store, tmp_path):

    empty = _create_db(tmp_path / "empty.sqlite", monitoring=[], retraining=[])

    report = rebuild.rebuild_index(empty)

    assert report.ok
    assert report.monitoring_runs_indexed == []
    assert report.retraining_events_indexed == []
    assert store.count() == 0


# ============================================================
# Failure safety
# ============================================================

def test_embedding_failure_reports_failure_and_leaves_pg_untouched(
    rebuild, store, db, embeddings
):
    before = _snapshot(db)
    embeddings.fail_always = True

    report = rebuild.rebuild_index(db)

    assert not report.ok
    assert report.monitoring_runs_failed == [1, 2, 3]
    assert report.retraining_events_failed == [1, 2, 3]
    assert store.count() == 0
    assert _snapshot(db) == before


def test_chroma_failure_reports_failure_and_leaves_pg_untouched(
    rebuild, store, db, monkeypatch
):
    before = _snapshot(db)

    def broken_upsert(*args, **kwargs):
        raise RuntimeError("chroma: disk I/O error")

    monkeypatch.setattr(store, "upsert_documents", broken_upsert)

    report = rebuild.rebuild_index(db)

    assert not report.ok
    assert len(report.monitoring_runs_failed) == 3
    assert len(report.retraining_events_failed) == 3
    assert _snapshot(db) == before


def test_partial_failure_then_rerun_converges_without_duplicates(
    rebuild, store, db, embeddings
):
    # Reference: a clean rebuild into a fresh index
    rebuild.rebuild_index(db)
    expected = _indexed(store)
    store.clear()
    embeddings.calls = 0

    # Embedding API fails midway: call 3 (monitoring run 3) and call 5
    # (retraining event 2) fail, the rest are indexed
    embeddings.fail_on_calls = {3, 5}
    partial = rebuild.rebuild_index(db)

    assert not partial.ok
    assert partial.monitoring_runs_failed == [3]
    assert partial.retraining_events_failed == [2]
    assert store.count() == len(EXPECTED_IDS) - 2

    # Retry: everything is re-upserted under the same ids
    embeddings.fail_on_calls = set()
    retry = rebuild.rebuild_index(db)

    assert retry.ok
    assert store.count() == len(EXPECTED_IDS)
    assert _indexed(store) == expected


def test_unreadable_postgres_raises_before_any_chroma_write(rebuild, store):

    class DownEngine:
        def connect(self):
            raise ConnectionError("could not connect to server")

    with pytest.raises(ConnectionError):
        rebuild.rebuild_index(DownEngine())

    assert store.count() == 0


def test_rebuild_never_deletes_other_documents(rebuild, store, db):

    store.upsert_documents(
        documents=["Sentinel-AI drift monitoring report: {}."],
        embeddings=[fake_vector("drift report")],
        ids=["sentinel_chunk_0"],
        metadatas=[{"source": "reports/drift_summary.json"}],
    )

    rebuild.rebuild_index(db)

    assert "sentinel_chunk_0" in _indexed(store)
    assert store.count() == len(EXPECTED_IDS) + 1


# ============================================================
# CLI
# ============================================================

def _fake_postgres(monkeypatch, engine):
    module = types.ModuleType("src.database.postgres")
    module.engine = engine
    monkeypatch.setitem(sys.modules, "src.database.postgres", module)


def test_cli_exit_zero_on_success(rebuild, db, monkeypatch, capsys):

    _fake_postgres(monkeypatch, db)

    assert rebuild.main() == 0
    out = capsys.readouterr().out
    assert "monitoring_runs:   3 indexed, 0 failed" in out
    assert "1 skipped (unlabelled)" in out
    assert "REBUILD OK" in out


def test_cli_exit_one_on_partial_failure(rebuild, db, embeddings, monkeypatch, capsys):

    _fake_postgres(monkeypatch, db)
    embeddings.fail_on_calls = {1}

    assert rebuild.main() == 1
    assert "REBUILD INCOMPLETE" in capsys.readouterr().out


def test_cli_exit_one_when_postgres_unreadable(rebuild, monkeypatch, caplog):

    class DownEngine:
        def connect(self):
            raise ConnectionError(
                "could not connect: postgresql://sentinel:Pg5ecretFake@db/x")

    _fake_postgres(monkeypatch, DownEngine())

    assert rebuild.main() == 1
    assert "Pg5ecretFake" not in caplog.text


# ============================================================
# Retrievable by the existing retriever
# ============================================================

def test_rebuilt_documents_are_retrievable(rebuild, embeddings, chroma_dir, db):

    rebuild.rebuild_index(db)

    retriever = _load("sentinel_real_retriever_rebuild", "src/rag/retriever.py")

    results = retriever.RAGRetriever(top_k=3).retrieve(
        "Retraining Event 3 promotion decision failed failure")

    assert results[0]["id"] == "retraining_event:3"
    assert results[0]["metadata"]["source"] == "retraining_events"

    results = retriever.RAGRetriever(top_k=1).retrieve(
        "Monitoring Run 2 recommended action MONITOR")

    assert results[0]["id"] == "monitoring_run:2"
