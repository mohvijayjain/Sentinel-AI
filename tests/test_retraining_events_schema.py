"""
Static checks on the retraining_events schema SQL. No PostgreSQL is
available to these tests, so they prove consistency of the SQL TEXT
(Sentinel.sql, migrations/001, the repository INSERT and the app's
constants), not that the migration runs: that still has to be verified
on a dev database.
"""

import os
import re

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

SCHEMA = os.path.join(REPO_ROOT, "Sentinel.sql")
MIGRATION = os.path.join(REPO_ROOT, "migrations",
                         "001_retraining_events_status.sql")
REPOSITORY = os.path.join(REPO_ROOT, "src", "database", "drift_repository.py")
ORCHESTRATOR = os.path.join(REPO_ROOT, "src", "training", "orchestrator.py")
RAG = os.path.join(REPO_ROOT, "src", "rag", "monitoring_updater.py")

# retraining_events as deployed before migration 001 (what a live DB has)
LEGACY_COLUMNS = {
    "id", "triggered_at", "triggered_reason", "new_model_rmse",
    "champion_rmse", "promoted", "mlflow_run_id",
}

STATUSES = {"promoted", "rejected", "failed"}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _active_sql(path):
    """SQL with -- comments removed (what PostgreSQL would execute)."""

    return "\n".join(
        line.split("--", 1)[0] for line in _read(path).splitlines()
    )


def _squash(text):
    return " ".join(text.split())


def _create_columns(sql):
    """{column: definition} from CREATE TABLE ... retraining_events (...)."""

    start = re.search(
        r"CREATE TABLE IF NOT EXISTS retraining_events\s*\(", sql
    ).end()

    depth, i = 1, start
    while depth:
        depth += {"(": 1, ")": -1}.get(sql[i], 0)
        i += 1
    body = sql[start:i - 1]

    parts, depth, current = [], 0, ""
    for ch in body:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)

    columns = {}
    for part in parts:
        name, _, definition = part.strip().partition(" ")
        columns[name] = _squash(definition)
    return columns


def _migration_added_columns(sql):
    return {
        name: _squash(definition)
        for name, definition in re.findall(
            r"ADD COLUMN IF NOT EXISTS (\w+)\s+(.*?)(?=,\s*ADD COLUMN|;)",
            sql, flags=re.S,
        )
    }


def _insert_columns():
    match = re.search(
        r"INSERT INTO retraining_events\s*\((.*?)\)", _read(REPOSITORY), re.S
    )
    return [c.strip() for c in match.group(1).split(",")]


# ============================================================
# Fresh database (Sentinel.sql) == migrated database (001)
# ============================================================

def test_fresh_and_migrated_column_definitions_match():

    created = _create_columns(_active_sql(SCHEMA))
    added = _migration_added_columns(_active_sql(MIGRATION))

    assert set(added) == {"status", "error_message"}

    for column, definition in added.items():
        assert created[column] == definition


def test_insert_columns_exist_on_migrated_legacy_table():
    """The repository INSERT works against an old DB once 001 has run."""

    migrated = LEGACY_COLUMNS | set(
        _migration_added_columns(_active_sql(MIGRATION))
    )

    assert set(_insert_columns()) <= migrated


def test_insert_columns_exist_on_fresh_table():

    assert set(_insert_columns()) <= set(_create_columns(_active_sql(SCHEMA)))


def test_new_columns_are_nullable_text():

    added = _migration_added_columns(_active_sql(MIGRATION))

    for definition in added.values():
        assert definition.startswith("TEXT")
        assert "NOT NULL" not in definition


def test_status_set_matches_application():

    definition = _migration_added_columns(_active_sql(MIGRATION))["status"]
    in_sql = set(re.findall(r"'(\w+)'", definition))

    rag_set = re.search(
        r"RETRAINING_STATUSES = frozenset\(\{(.*?)\}\)", _read(RAG), re.S
    ).group(1)

    assert in_sql == STATUSES
    assert set(re.findall(r'"(\w+)"', rag_set)) == STATUSES
    assert "DEFAULT NULL" in definition


def test_error_message_bound_matches_orchestrator():

    definition = _migration_added_columns(
        _active_sql(MIGRATION)
    )["error_message"]

    sql_bound = int(re.search(r"<= (\d+)", definition).group(1))

    # Defined once in the shared redaction module, used by orchestrator
    from src.common.error_redaction import ERROR_MESSAGE_MAX_LENGTH

    assert sql_bound == ERROR_MESSAGE_MAX_LENGTH == 500
    assert "ERROR_MESSAGE_MAX_LENGTH" in _read(ORCHESTRATOR)


# ============================================================
# Migration is idempotent, additive, separated
# ============================================================

def test_migration_is_non_destructive():

    sql = _active_sql(MIGRATION).upper()

    for forbidden in ("DROP ", "TRUNCATE", "DELETE ", "RENAME",
                      "SET NOT NULL", "ADD CONSTRAINT", " TYPE ",
                      "CREATE TABLE"):
        assert forbidden not in sql, forbidden


def test_every_added_column_is_guarded():

    sql = _active_sql(MIGRATION)

    assert len(re.findall(r"ADD COLUMN", sql)) == len(
        re.findall(r"ADD COLUMN IF NOT EXISTS", sql)
    ) == 2


def test_migration_runs_in_one_transaction():

    statements = [s.strip() for s in _active_sql(MIGRATION).split(";")
                  if s.strip()]

    assert statements[0] == "BEGIN"
    assert statements[-1] == "COMMIT"


def test_backfill_is_deterministic_and_never_failed():

    updates = re.findall(
        r"UPDATE retraining_events(.*?);", _active_sql(MIGRATION), re.S
    )

    assert len(updates) == 1

    update = _squash(updates[0])

    assert update.startswith(
        "SET status = CASE WHEN promoted THEN 'promoted' ELSE 'rejected' END"
    )
    assert "WHERE status IS NULL AND promoted IS NOT NULL" in update
    assert "'failed'" not in update
    assert "error_message" not in update


def test_timezone_normalization_is_optional_and_commented():

    raw = _read(MIGRATION)
    active = _active_sql(MIGRATION)

    assert "<OLD_SERVER_TZ>" in raw
    assert "<OLD_SERVER_TZ>" not in active
    assert not re.search(r"SET\s+triggered_at", active)


def test_schema_file_no_longer_folds_in_the_migration():
    """The ALTER / backfill lives only in migrations/, not in Sentinel.sql."""

    sql = _active_sql(SCHEMA)

    assert "ALTER TABLE retraining_events" not in sql
    assert "UPDATE retraining_events" not in sql
    assert "migrations/001_retraining_events_status.sql" in _read(SCHEMA)


# ============================================================
# UTC convention is stated everywhere it matters
# ============================================================

@pytest.mark.parametrize("path", [SCHEMA, MIGRATION], ids=["schema", "migration"])
def test_triggered_at_convention_is_utc(path):

    raw = _read(path)
    active = _squash(_active_sql(path))

    assert "triggered_at represents UTC" in raw
    assert "COMMENT ON COLUMN retraining_events.triggered_at IS" in active
    assert "Represents UTC" in active

    # A writer that omits triggered_at gets UTC too, not server-local
    assert "(NOW() AT TIME ZONE 'UTC')" in active


def test_orchestrator_documents_convention_at_generation():

    source = _read(ORCHESTRATOR)
    generator = source[source.index("def _canonical_triggered_at"):]
    generator = generator[:generator.index("\ndef ")]

    assert "retraining_events.triggered_at represents UTC" in generator
    assert "datetime.now(timezone.utc)" in generator
