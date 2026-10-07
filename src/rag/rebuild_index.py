"""
src/rag/rebuild_index.py

PostgreSQL -> ChromaDB rebuild of the monitoring/retraining history.

PostgreSQL is the source of truth; ChromaDB is a derived index. If the
index is lost, corrupted or missed a write (the write-through upserts are
best-effort), this re-indexes every monitoring_runs and retraining_events
row from PostgreSQL.

What it does:
  1. Reads all rows from monitoring_runs and retraining_events (read-only:
     one connection, nothing written, transaction rolled back on close).
  2. Passes each row to the SAME upsert_monitoring_run /
     upsert_retraining_event the pipelines use, so the document text,
     embedding call, metadata and id (monitoring_run:<id>,
     retraining_event:<id>) are identical to the normal write-through.
  3. Reports per-record success/failure. Never deletes anything: not from
     PostgreSQL, and not from ChromaDB (report-file chunks written by
     src/rag/ingest.py are left alone).

Idempotent: ids are the PostgreSQL ids and the write is an upsert, so a
second run overwrites the same records instead of adding new ones. After a
partial failure (e.g. the embedding API fails midway), just run it again:
already indexed records are overwritten in place, missing ones are added.

Usage (inside the API container, where POSTGRES_URL and data/chroma are
configured):

    docker exec sentinel-api python -m src.rag.rebuild_index

Exit code 0 when every record was indexed, 1 otherwise.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import text

from src.common.error_redaction import redacted_traceback
from src.rag import monitoring_updater


logger = logging.getLogger(__name__)


@dataclass
class RebuildReport:
    """Outcome of one rebuild. ok only when nothing failed."""

    monitoring_runs_indexed: List[int] = field(default_factory=list)
    monitoring_runs_failed: List[int] = field(default_factory=list)
    retraining_events_indexed: List[int] = field(default_factory=list)
    retraining_events_failed: List[int] = field(default_factory=list)
    # Legacy rows with neither status nor promoted: no truthful outcome
    # to index (migration 001 leaves them unlabelled too).
    retraining_events_skipped: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.monitoring_runs_failed or self.retraining_events_failed)


def _iso_utc(value) -> str:
    """triggered_at is UTC wall-clock (TIMESTAMP): render it as the
    orchestrator does, an ISO-8601 string with +00:00."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    return str(value)


def _retraining_status(row) -> Optional[str]:
    """status column, or for legacy rows (before migration 001) the same
    derivation the migration applies: promoted -> 'promoted'/'rejected'."""

    status = row.get("status")
    if status:
        return status

    promoted = row.get("promoted")
    if promoted is None:
        return None

    return "promoted" if promoted else "rejected"


def _read_history(engine):

    with engine.connect() as conn:
        runs = conn.execute(
            text("SELECT * FROM monitoring_runs ORDER BY id")
        ).mappings().all()
        events = conn.execute(
            text("SELECT * FROM retraining_events ORDER BY id")
        ).mappings().all()

    return runs, events


def rebuild_index(engine=None) -> RebuildReport:
    """
    Re-index all monitoring runs and retraining events from PostgreSQL.

    Raises only if PostgreSQL cannot be read (nothing has been written to
    ChromaDB at that point). Per-record ChromaDB / embedding failures are
    collected in the report, never raised.
    """

    if engine is None:
        from src.database.postgres import engine

    runs, events = _read_history(engine)

    report = RebuildReport()

    for row in runs:

        ok = monitoring_updater.upsert_monitoring_run(
            run_id=row["id"],
            statistical_score=row["statistical_score"],
            shap_score=row["shap_score"],
            prediction_score=row["prediction_score"],
            overall_score=row["overall_score"],
            action=row["action"],
        )

        (report.monitoring_runs_indexed if ok
         else report.monitoring_runs_failed).append(row["id"])

    for row in events:

        status = _retraining_status(row)

        if status is None:
            logger.warning(
                "Retraining event %s has no status and no promoted value; "
                "not indexed.",
                row["id"],
            )
            report.retraining_events_skipped.append(row["id"])
            continue

        ok = monitoring_updater.upsert_retraining_event(
            event_id=row["id"],
            triggered_at=_iso_utc(row["triggered_at"]),
            triggered_reason=row["triggered_reason"],
            new_model_rmse=row["new_model_rmse"],
            champion_rmse=row["champion_rmse"],
            promoted=row["promoted"],
            mlflow_run_id=row["mlflow_run_id"],
            status=status,
            error_message=row.get("error_message"),
        )

        (report.retraining_events_indexed if ok
         else report.retraining_events_failed).append(row["id"])

    return report


def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        report = rebuild_index()
    except Exception as error:
        logger.error(
            "Rebuild failed: could not read PostgreSQL. Nothing was "
            "indexed.\n%s",
            redacted_traceback(error),
        )
        return 1

    print(
        f"monitoring_runs:   {len(report.monitoring_runs_indexed)} indexed, "
        f"{len(report.monitoring_runs_failed)} failed "
        f"{report.monitoring_runs_failed or ''}"
    )
    print(
        f"retraining_events: {len(report.retraining_events_indexed)} indexed, "
        f"{len(report.retraining_events_failed)} failed "
        f"{report.retraining_events_failed or ''}, "
        f"{len(report.retraining_events_skipped)} skipped (unlabelled)"
    )
    print("REBUILD OK" if report.ok else
          "REBUILD INCOMPLETE: re-run to retry the failed records")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
