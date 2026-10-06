"""
Post-commit success reporting can no longer turn a saved row into a
reported failure.

insert_monitoring_run / insert_retraining_event printed an emoji AFTER the
transaction committed. On a stdout that cannot encode it (Windows cp1252,
piped output) the print raised UnicodeEncodeError, so the caller saw an
exception for a row that was in fact persisted. They now log an ASCII
line instead. Transaction semantics and return values are unchanged.

The encoding check forces a strict-ASCII stdout, so it is platform
independent.
"""

import contextlib
import importlib.util
import io
import logging
import os
import sys
import types

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPOSITORY_FILE = os.path.join(REPO_ROOT, "src", "database", "drift_repository.py")


class TransactionalEngine:
    """Fake SQLAlchemy engine: begin() commits on success, rolls back on error."""

    def __init__(self, returned_id=41, fail_on_execute=None):
        self.returned_id = returned_id
        self.fail_on_execute = fail_on_execute
        self.events = []

    def begin(self):
        engine = self

        class Transaction:
            def __enter__(self):
                engine.events.append("begin")
                return engine

            def __exit__(self, exc_type, exc, tb):
                engine.events.append("rollback" if exc_type else "commit")
                return False

        return Transaction()

    def execute(self, query, params=None):
        if self.fail_on_execute is not None:
            raise self.fail_on_execute
        engine = self

        class Result:
            def scalar_one(self):
                return engine.returned_id

        return Result()


@pytest.fixture
def load_repository(monkeypatch):

    def _load(engine):
        fake_postgres = types.ModuleType("src.database.postgres")
        fake_postgres.engine = engine
        monkeypatch.setitem(sys.modules, "src.database.postgres", fake_postgres)

        spec = importlib.util.spec_from_file_location(
            "sentinel_real_drift_repository_1c", REPOSITORY_FILE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


def ascii_stdout():
    """
    Redirect stdout to a stream that raises on any non-ASCII character,
    like cp1252 on an emoji. Used inside the test body: pytest re-installs
    its own capture stdout at the start of the call phase, so a fixture
    that swapped sys.stdout during setup would silently not apply.
    """

    stream = io.TextIOWrapper(
        io.BytesIO(), encoding="ascii", errors="strict", write_through=True
    )
    return contextlib.redirect_stdout(stream)


def _insert_event(repository):
    return repository.insert_retraining_event(
        "drift_detected", 4.1, 4.6, True, "run-1",
        triggered_at="2026-10-06T12:00:00+00:00", status="promoted",
    )


def _insert_run(repository):
    return repository.insert_monitoring_run(
        statistical_score=0.5, shap_score=0.25, prediction_score=0.0,
        overall_score=0.275, action="MONITOR", drifted_features=["f"],
        report_path="reports/drift_summary.json",
    )


# ============================================================
# Success returns the id, committed
# ============================================================

@pytest.mark.parametrize("insert", [_insert_event, _insert_run],
                         ids=["retraining_event", "monitoring_run"])
def test_successful_insert_returns_id(load_repository, insert):

    engine = TransactionalEngine(returned_id=41)

    assert insert(load_repository(engine)) == 41
    assert engine.events == ["begin", "commit"]


# ============================================================
# Unencodable stdout after commit cannot turn success into failure
# ============================================================

@pytest.mark.parametrize("insert", [_insert_event, _insert_run],
                         ids=["retraining_event", "monitoring_run"])
def test_ascii_stdout_cannot_fail_a_committed_insert(load_repository, insert):

    engine = TransactionalEngine(returned_id=7)
    repository = load_repository(engine)

    with ascii_stdout():
        returned = insert(repository)

    assert returned == 7
    assert engine.events == ["begin", "commit"]


def test_the_hazard_is_real_for_the_old_emoji_print():
    """What the old success line did on such a stream (proves the harness)."""

    with pytest.raises(UnicodeEncodeError):
        with ascii_stdout():
            print("✅ Retraining event saved to PostgreSQL: event_id=7")


@pytest.mark.parametrize(
    "insert, expected",
    [(_insert_event, "Retraining event inserted successfully: event_id=9"),
     (_insert_run, "Monitoring run inserted successfully: run_id=9")],
    ids=["retraining_event", "monitoring_run"],
)
def test_success_is_logged_in_ascii(load_repository, caplog, insert, expected):

    with caplog.at_level(logging.INFO):
        insert(load_repository(TransactionalEngine(returned_id=9)))

    assert expected in caplog.text
    caplog.text.encode("ascii")          # every line is ASCII-safe


# ============================================================
# Failures before commit still raise and roll back
# ============================================================

@pytest.mark.parametrize("insert", [_insert_event, _insert_run],
                         ids=["retraining_event", "monitoring_run"])
def test_pre_commit_failure_raises_and_rolls_back(load_repository, caplog, insert):

    error = ConnectionError("server closed the connection unexpectedly")
    engine = TransactionalEngine(fail_on_execute=error)

    with caplog.at_level(logging.INFO):
        with pytest.raises(ConnectionError) as excinfo:
            insert(load_repository(engine))

    assert excinfo.value is error
    assert engine.events == ["begin", "rollback"]
    assert "inserted successfully" not in caplog.text


# ============================================================
# Static guard
# ============================================================

def test_repository_has_no_print_or_non_ascii_output():

    with open(REPOSITORY_FILE, encoding="utf-8") as f:
        source = f.read()

    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    assert "print(" not in code
    code.encode("ascii")
