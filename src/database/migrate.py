"""
Minimal schema migration runner for Sentinel-AI.

Usage (from the repository root):

    python -m src.database.migrate                      # POSTGRES_URL from .env
    python -m src.database.migrate --database-url URL   # e.g. a disposable test DB

Each run:
  1. applies the base schema (Sentinel.sql). It is idempotent
     (CREATE ... IF NOT EXISTS only), so it creates a brand-new database
     and leaves an existing one alone;
  2. ensures the schema_migrations tracking table exists;
  3. applies every pending migrations/NNN_name.sql in numeric order, each
     in ONE transaction together with its schema_migrations row. A failed
     migration rolls back completely and is NOT recorded; the run stops
     there, and earlier migrations stay applied.

Migration files may wrap themselves in BEGIN; / COMMIT; lines so they also
run standalone with psql; the runner strips that outer wrapper and uses
its own transaction instead. A file that manages transactions anywhere
else is rejected.

Re-running is safe: applied migrations are skipped by filename. If an
applied file's content changed since, a warning is logged and it is NOT
re-run (write a new migration instead).
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List

from src.common.error_redaction import describe_error


logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

DEFAULT_MIGRATIONS_DIR = os.path.join(REPO_ROOT, "migrations")
BASE_SCHEMA_PATH = os.path.join(REPO_ROOT, "Sentinel.sql")

MIGRATION_FILENAME = re.compile(r"^(\d{3,})_[a-z0-9_\-]+\.sql$")

# applied_at follows the project's UTC convention for TIMESTAMP columns
TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
"""

# Serializes concurrent runners for the duration of one migration
ADVISORY_LOCK_KEY = 0x53454E54  # "SENT"

_WRAPPER_LINE = re.compile(r"^[ \t]*(BEGIN|COMMIT)[ \t]*;[ \t]*$",
                           re.IGNORECASE | re.MULTILINE)

# END; is deliberately not matched: it also closes PL/pgSQL blocks
_TRANSACTION_CONTROL = re.compile(
    r"\b(BEGIN|COMMIT|ROLLBACK|START\s+TRANSACTION)\s*;",
    re.IGNORECASE,
)


class MigrationError(RuntimeError):
    """A migration failed; it was rolled back and not recorded."""


@dataclass(frozen=True)
class Migration:
    filename: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()

    @property
    def body(self) -> str:
        return transaction_body(self.sql, self.filename)


def _strip_comments(sql: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def transaction_body(sql: str, filename: str = "<migration>") -> str:
    """
    The migration without its standalone BEGIN; / COMMIT; wrapper lines,
    ready to run inside the runner's transaction.
    """

    body = _WRAPPER_LINE.sub("", sql)

    if _TRANSACTION_CONTROL.search(_strip_comments(body)):
        raise MigrationError(
            f"{filename} manages its own transaction outside a single "
            "BEGIN;/COMMIT; wrapper; the runner cannot apply it atomically."
        )

    return body


def discover_migrations(directory: str = DEFAULT_MIGRATIONS_DIR) -> List[Migration]:
    """All NNN_name.sql files, in numeric order. Other files are ignored."""

    found = []

    for filename in os.listdir(directory):

        if not filename.endswith(".sql"):
            continue

        match = MIGRATION_FILENAME.match(filename)

        if not match:
            raise MigrationError(
                f"Migration filename {filename!r} must look like "
                "NNN_lowercase_name.sql."
            )

        with open(os.path.join(directory, filename), encoding="utf-8") as f:
            found.append((int(match.group(1)), filename, f.read()))

    found.sort()

    numbers = [number for number, _, _ in found]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})

    if duplicates:
        raise MigrationError(
            f"Duplicate migration numbers: {duplicates}."
        )

    return [Migration(filename, sql) for _, filename, sql in found]


# ============================================================
# PostgreSQL backend
# ============================================================

class PostgresBackend:
    """
    Executes through a raw psycopg2 connection: multi-statement SQL runs
    as-is (no SQLAlchemy bind-parameter parsing of ':' in COMMENT text,
    no '%' formatting because no parameters are passed).
    """

    def __init__(self, database_url: str):

        from sqlalchemy import create_engine

        self.engine = create_engine(database_url)

    def _run_in_transaction(self, steps):

        connection = self.engine.raw_connection()

        try:
            cursor = connection.cursor()
            result = steps(cursor)
            connection.commit()
            return result

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def apply_base_schema(self, sql: str) -> None:
        self._run_in_transaction(lambda cursor: cursor.execute(sql))

    def ensure_tracking_table(self) -> None:
        self._run_in_transaction(
            lambda cursor: cursor.execute(TRACKING_TABLE_SQL)
        )

    def applied(self) -> Dict[str, str]:

        def read(cursor):
            cursor.execute("SELECT filename, checksum FROM schema_migrations")
            return dict(cursor.fetchall())

        return self._run_in_transaction(read)

    def apply(self, migration: Migration) -> bool:
        """
        Apply one migration and record it, atomically. Returns False if
        another runner applied it first (checked under the lock).
        """

        body = migration.body

        def steps(cursor):

            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,)
            )

            cursor.execute(
                "SELECT 1 FROM schema_migrations WHERE filename = %s",
                (migration.filename,),
            )

            if cursor.fetchone():
                return False

            # No parameters: the SQL text is sent exactly as written
            cursor.execute(body)

            cursor.execute(
                "INSERT INTO schema_migrations (filename, checksum) "
                "VALUES (%s, %s)",
                (migration.filename, migration.checksum),
            )

            return True

        return self._run_in_transaction(steps)


# ============================================================
# Runner
# ============================================================

@dataclass
class RunResult:
    applied: List[str]
    skipped: List[str]


def run_migrations(
    backend,
    migrations: List[Migration],
    base_schema_sql: str = None,
) -> RunResult:

    if base_schema_sql is not None:
        backend.apply_base_schema(base_schema_sql)

    backend.ensure_tracking_table()

    already = backend.applied()
    result = RunResult(applied=[], skipped=[])

    for migration in migrations:

        if migration.filename in already:

            if already[migration.filename] != migration.checksum:
                logger.warning(
                    "%s changed after it was applied; not re-running it. "
                    "Write a new migration instead.",
                    migration.filename,
                )

            result.skipped.append(migration.filename)
            continue

        logger.info("Applying %s", migration.filename)

        try:
            newly_applied = backend.apply(migration)

        except Exception as error:
            raise MigrationError(
                f"{migration.filename} failed and was rolled back; it is "
                f"NOT recorded as applied. {describe_error(error)}"
            ) from error

        if newly_applied:
            result.applied.append(migration.filename)
        else:
            result.skipped.append(migration.filename)

    return result


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        description="Apply pending Sentinel-AI schema migrations."
    )
    parser.add_argument(
        "--database-url",
        help="SQLAlchemy URL; defaults to POSTGRES_URL from the environment/.env",
    )
    parser.add_argument(
        "--migrations-dir", default=DEFAULT_MIGRATIONS_DIR,
    )
    parser.add_argument(
        "--no-base-schema", action="store_true",
        help="Do not apply Sentinel.sql first",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    database_url = args.database_url

    if not database_url:
        from dotenv import load_dotenv

        load_dotenv()
        database_url = os.getenv("POSTGRES_URL")

    if not database_url:
        print("No database URL: pass --database-url or set POSTGRES_URL.",
              file=sys.stderr)
        return 2

    try:

        migrations = discover_migrations(args.migrations_dir)

        base_sql = None
        if not args.no_base_schema:
            with open(BASE_SCHEMA_PATH, encoding="utf-8") as f:
                base_sql = f.read()

        result = run_migrations(
            PostgresBackend(database_url), migrations, base_sql
        )

    except Exception as error:
        # Never echo the URL or a raw driver message (may hold credentials)
        print(f"Migration failed: {describe_error(error)}", file=sys.stderr)
        return 1

    for filename in result.applied:
        print(f"applied   {filename}")

    for filename in result.skipped:
        print(f"up to date {filename}")

    if not result.applied:
        print("Nothing to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
