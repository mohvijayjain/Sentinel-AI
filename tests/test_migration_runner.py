"""
src/database/migrate.py at the logic level (no live PostgreSQL).

  * run_migrations() against an in-memory backend that simulates
    transactional apply: fresh / existing / already-applied databases,
    deterministic order, failure not recorded, idempotent re-run.
  * PostgresBackend.apply() against a fake DB-API connection: the exact
    transaction sequence (lock, re-check, body, record, commit) and
    rollback-without-record on failure.

These do NOT prove the SQL runs on a real server; see the Part B report.
"""

import os

import pytest

from src.database import migrate
from src.database.migrate import (
    Migration,
    MigrationError,
    PostgresBackend,
    discover_migrations,
    run_migrations,
    transaction_body,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ============================================================
# In-memory backend: apply is all-or-nothing, like a transaction
# ============================================================

class FakeBackend:

    def __init__(self, applied=None, fail_on=()):
        self.base_applied = 0
        self.tracking_exists = applied is not None
        self.records = dict(applied or {})
        self.executed = []
        self.fail_on = set(fail_on)

    def apply_base_schema(self, sql):
        self.base_applied += 1

    def ensure_tracking_table(self):
        self.tracking_exists = True

    def applied(self):
        assert self.tracking_exists
        return dict(self.records)

    def apply(self, migration):
        body = migration.body          # validated like the real backend
        if migration.filename in self.fail_on:
            # Transaction rolled back: nothing executed, nothing recorded
            raise RuntimeError(f"syntax error in {migration.filename}")
        self.executed.append(migration.filename)
        self.records[migration.filename] = migration.checksum
        return True


def _migrations(*names):
    return [Migration(name, f"BEGIN;\nSELECT '{name}';\nCOMMIT;\n")
            for name in names]


ABC = ("001_a.sql", "002_b.sql", "003_c.sql")


# ============================================================
# Discovery and ordering
# ============================================================

def test_discovers_real_migration():

    found = discover_migrations()

    assert [m.filename for m in found][0] == "001_retraining_events_status.sql"


def test_numeric_deterministic_order(tmp_path):

    for name in ("010_late.sql", "002_mid.sql", "001_first.sql", "README.md"):
        (tmp_path / name).write_text("SELECT 1;")

    assert [m.filename for m in discover_migrations(str(tmp_path))] == [
        "001_first.sql", "002_mid.sql", "010_late.sql",
    ]


@pytest.mark.parametrize("bad", ["1_short.sql", "001-Upper.sql", "abc.sql"])
def test_rejects_bad_filenames(tmp_path, bad):

    (tmp_path / bad).write_text("SELECT 1;")

    with pytest.raises(MigrationError, match="must look like"):
        discover_migrations(str(tmp_path))


def test_rejects_duplicate_numbers(tmp_path):

    (tmp_path / "001_a.sql").write_text("SELECT 1;")
    (tmp_path / "001_b.sql").write_text("SELECT 2;")

    with pytest.raises(MigrationError, match="Duplicate"):
        discover_migrations(str(tmp_path))


# ============================================================
# Transaction wrapper handling
# ============================================================

def test_real_migration_wrapper_is_stripped():

    migration = discover_migrations()[0]
    body = migration.body

    assert "BEGIN;" in migration.sql and "COMMIT;" in migration.sql
    assert "BEGIN;" not in body and "COMMIT;" not in body
    assert "ADD COLUMN IF NOT EXISTS status" in body


@pytest.mark.parametrize(
    "sql",
    [
        "BEGIN;\nSELECT 1;\nCOMMIT;\nSELECT 2; COMMIT;\n",
        "SELECT 1; ROLLBACK;",
        "START TRANSACTION;\nSELECT 1;",
    ],
)
def test_rejects_inner_transaction_control(sql):

    with pytest.raises(MigrationError, match="manages its own transaction"):
        transaction_body(sql, "009_x.sql")


def test_plpgsql_blocks_are_allowed():

    sql = "DO $$\nBEGIN\n  PERFORM 1;\nEND;\n$$;\n"

    assert transaction_body(sql) == sql


def test_wrapper_lines_inside_comments_do_not_trip_the_check():

    sql = "-- run with BEGIN; ... COMMIT; via psql\nSELECT 1;\n"

    assert "SELECT 1;" in transaction_body(sql)


# ============================================================
# run_migrations: fresh / existing / already applied
# ============================================================

def test_fresh_database_applies_all_in_order():

    backend = FakeBackend(applied=None)

    result = run_migrations(backend, _migrations(*ABC), base_schema_sql="--")

    assert backend.base_applied == 1
    assert backend.tracking_exists
    assert result.applied == list(ABC)
    assert backend.executed == list(ABC)
    assert set(backend.records) == set(ABC)


def test_existing_database_applies_only_pending():

    migrations = _migrations(*ABC)
    backend = FakeBackend(applied={"001_a.sql": migrations[0].checksum})

    result = run_migrations(backend, migrations)

    assert result.applied == ["002_b.sql", "003_c.sql"]
    assert result.skipped == ["001_a.sql"]
    assert backend.executed == ["002_b.sql", "003_c.sql"]


def test_already_applied_is_a_no_op():

    migrations = _migrations(*ABC)
    backend = FakeBackend(applied={m.filename: m.checksum for m in migrations})

    result = run_migrations(backend, migrations)

    assert result.applied == []
    assert result.skipped == list(ABC)
    assert backend.executed == []


def test_rerun_is_idempotent():

    backend = FakeBackend(applied=None)
    migrations = _migrations(*ABC)

    run_migrations(backend, migrations)
    records_after_first = dict(backend.records)

    second = run_migrations(backend, migrations)

    assert second.applied == []
    assert backend.records == records_after_first
    assert backend.executed == list(ABC)       # each ran exactly once


def test_failed_migration_not_recorded_and_run_stops():

    backend = FakeBackend(applied=None, fail_on={"002_b.sql"})

    with pytest.raises(MigrationError, match="002_b.sql failed") as excinfo:
        run_migrations(backend, _migrations(*ABC))

    assert "NOT recorded" in str(excinfo.value)
    assert set(backend.records) == {"001_a.sql"}   # earlier success kept
    assert backend.executed == ["001_a.sql"]       # 003 never attempted


def test_rerun_after_fix_applies_the_rest():

    backend = FakeBackend(applied=None, fail_on={"002_b.sql"})

    with pytest.raises(MigrationError):
        run_migrations(backend, _migrations(*ABC))

    backend.fail_on.clear()
    result = run_migrations(backend, _migrations(*ABC))

    assert result.applied == ["002_b.sql", "003_c.sql"]
    assert set(backend.records) == set(ABC)


def test_changed_applied_migration_warns_and_is_not_rerun(caplog):

    original, = _migrations("001_a.sql")
    edited = Migration("001_a.sql", original.sql + "SELECT 'edited';\n")
    backend = FakeBackend(applied={"001_a.sql": original.checksum})

    result = run_migrations(backend, [edited])

    assert result.applied == []
    assert backend.executed == []
    assert "changed after it was applied" in caplog.text


# ============================================================
# PostgresBackend.apply: real transaction code, fake DB-API
# ============================================================

class FakeCursor:

    def __init__(self, log, already_applied=False, fail_on_body=False):
        self.log = log
        self.already_applied = already_applied
        self.fail_on_body = fail_on_body
        self._last = None

    def execute(self, sql, params=None):
        self.log.append(("execute", sql.strip().split()[0].upper(), params))
        self._last = sql
        if self.fail_on_body and "BROKEN" in sql:
            raise RuntimeError("syntax error at or near BROKEN")

    def fetchone(self):
        if "FROM schema_migrations" in self._last and self.already_applied:
            return (1,)
        return None


class FakeConnection:

    def __init__(self, cursor, log):
        self._cursor = cursor
        self.log = log

    def cursor(self):
        return self._cursor

    def commit(self):
        self.log.append(("commit",))

    def rollback(self):
        self.log.append(("rollback",))

    def close(self):
        self.log.append(("close",))


def _backend(monkeypatch, **cursor_kwargs):

    log = []
    backend = PostgresBackend.__new__(PostgresBackend)
    backend.engine = type("E", (), {})()
    backend.engine.raw_connection = lambda: FakeConnection(
        FakeCursor(log, **cursor_kwargs), log
    )
    return backend, log


def test_apply_runs_lock_recheck_body_record_commit(monkeypatch):

    backend, log = _backend(monkeypatch)
    migration = Migration("001_a.sql", "BEGIN;\nALTER TABLE t ADD COLUMN c int;\nCOMMIT;\n")

    assert backend.apply(migration) is True

    statements = [entry for entry in log if entry[0] == "execute"]

    assert [s[1] for s in statements] == ["SELECT", "SELECT", "ALTER", "INSERT"]
    assert statements[0][2] == (migrate.ADVISORY_LOCK_KEY,)
    assert statements[2][2] is None                  # body sent without params
    assert statements[3][2] == ("001_a.sql", migration.checksum)
    assert log[-2:] == [("commit",), ("close",)]


def test_apply_failure_rolls_back_and_records_nothing(monkeypatch):

    backend, log = _backend(monkeypatch, fail_on_body=True)
    migration = Migration("002_b.sql", "ALTER TABLE BROKEN;")

    with pytest.raises(RuntimeError, match="BROKEN"):
        backend.apply(migration)

    assert ("commit",) not in log
    assert ("rollback",) in log
    assert not any(e[0] == "execute" and e[1] == "INSERT" for e in log)


def test_apply_skips_when_another_runner_won(monkeypatch):

    backend, log = _backend(monkeypatch, already_applied=True)

    assert backend.apply(Migration("001_a.sql", "ALTER TABLE t;")) is False
    assert not any(e[0] == "execute" and e[1] in ("ALTER", "INSERT") for e in log)


# ============================================================
# Base schema and CLI
# ============================================================

def test_base_schema_is_rerunnable_on_a_legacy_table():
    """
    Sentinel.sql runs before migrations on every run, including against
    a legacy retraining_events without status / error_message.
    """

    with open(os.path.join(REPO_ROOT, "Sentinel.sql"), encoding="utf-8") as f:
        active = "\n".join(l.split("--", 1)[0] for l in f.read().splitlines())

    upper = active.upper()

    assert "BEGIN;" not in upper and "COMMIT;" not in upper
    assert upper.count("CREATE TABLE ") == upper.count("CREATE TABLE IF NOT EXISTS")
    assert upper.count("CREATE INDEX ") == upper.count("CREATE INDEX IF NOT EXISTS")

    # COMMENT ON a column the legacy table lacks would abort the run
    assert "retraining_events.status" not in active
    assert "retraining_events.error_message" not in active


def test_cli_without_url_exits_2(monkeypatch, capsys):

    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    assert migrate.main([]) == 2
    assert "No database URL" in capsys.readouterr().err


def test_cli_failure_message_is_redacted(monkeypatch, capsys):

    class Boom:
        def __init__(self, url):
            raise RuntimeError(
                "could not connect to postgresql://sentinel:Pa55w0rdX@db:5432/x"
            )

    monkeypatch.setattr(migrate, "PostgresBackend", Boom)

    code = migrate.main(["--database-url", "postgresql://u:Pa55w0rdX@db/x"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Pa55w0rdX" not in captured.out + captured.err
    assert "Migration failed: RuntimeError" in captured.err
