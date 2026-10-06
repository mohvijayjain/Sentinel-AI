# src/training/orchestrator.py

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from src.common.error_redaction import (
    ERROR_MESSAGE_MAX_LENGTH,
    redacted_traceback,
    safe_error_message,
)
from src.training import retrain
from src.training.promote import main as promote_model
from src.database.drift_repository import insert_retraining_event
from src.rag.monitoring_updater import upsert_retraining_event

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 70

# Keys promote.main() must return (dict contract, not the old int).
# Validated up front so a stale promote.main fails with a clear message
# instead of a downstream KeyError.
_REQUIRED_PROMOTION_KEYS = (
    "promoted",
    "mlflow_run_id",
    "new_model_rmse",
    "new_model_mae",
    "new_model_r2",
    "champion_rmse",
    "champion_mae",
    "champion_r2",
    "champion_version",
    "new_version",
)


def _require_promotion_dict(
    promotion_result: Any
) -> Dict[str, Any]:
    """
    Validate the shape of promote.main()'s return value.

    Guards against the legacy int return (0/1): a clear error here beats
    an opaque 'int object is not subscriptable' later.
    """

    if not isinstance(promotion_result, dict):
        raise TypeError(
            "promote.main() must return a dict, got "
            f"{type(promotion_result).__name__}. "
            "Upgrade promote.main() to the structured return."
        )

    missing = [
        key
        for key in _REQUIRED_PROMOTION_KEYS
        if key not in promotion_result
    ]

    if missing:
        raise KeyError(
            "promote.main() result is missing keys: "
            f"{', '.join(missing)}."
        )

    return promotion_result


# ERROR_MESSAGE_MAX_LENGTH (imported above) matches the CHECK on
# retraining_events.error_message


def _canonical_triggered_at() -> str:
    """
    The one ISO-8601 UTC timestamp for a retraining attempt. Called once
    at the start of the attempt; PostgreSQL and ChromaDB both receive
    exactly this value, whatever the outcome.

    Convention: retraining_events.triggered_at represents UTC (see
    Sentinel.sql and migrations/001_retraining_events_status.sql).
    """

    return datetime.now(timezone.utc).isoformat()


def _bounded_error_message(error: BaseException) -> str:
    """
    '<Type>: <message>' on one line, secrets redacted, capped for the DB
    column. Builds a new string; the exception itself is never modified.
    """

    return safe_error_message(
        f"{type(error).__name__}: {error}"
    )


def _record_failed_attempt(
    triggered_reason: str,
    triggered_at: str,
    run_id: Any,
    error: BaseException,
):
    """
    Persist a failed attempt for audit: PostgreSQL first, then ChromaDB.

    No metrics are invented; mlflow_run_id is only set when retraining
    produced a real run before the failure. Never raises, so the caller
    can always re-raise the original error.
    """

    error_message = _bounded_error_message(error)

    try:

        event_id = insert_retraining_event(
            triggered_reason=triggered_reason,
            new_model_rmse=None,
            champion_rmse=None,
            promoted=None,
            mlflow_run_id=run_id or None,
            triggered_at=triggered_at,
            status="failed",
            error_message=error_message,
        )

    except Exception as insert_error:

        # Not persisted, so not indexed either. Traceback is logged
        # redacted: logger.exception would emit the raw message.
        logger.error(
            "Could not record failed retraining attempt in PostgreSQL.\n%s",
            redacted_traceback(insert_error),
        )

        return None

    try:

        upsert_retraining_event(
            event_id=event_id,
            triggered_at=triggered_at,
            triggered_reason=triggered_reason,
            new_model_rmse=None,
            champion_rmse=None,
            promoted=None,
            mlflow_run_id=run_id or None,
            status="failed",
            error_message=error_message,
        )

    except Exception as index_error:

        logger.error(
            "Could not index failed retraining event %s in ChromaDB; "
            "PostgreSQL row is intact.\n%s",
            event_id,
            redacted_traceback(index_error),
        )

    return event_id


def run_retraining_pipeline(
    triggered_reason: str = "manual_retraining"
) -> dict:
    """
    Execute the complete retraining -> evaluation -> promotion workflow.

    Flow:
        Retrain Challenger
            ↓
        MLflow run
            ↓
        Promotion gates
            ↓
        Promote / Reject
            ↓
        Save retraining event to PostgreSQL
            ↓
        Index retraining event in ChromaDB
    """

    started_at = time.monotonic()

    # One canonical timestamp per attempt, taken at the start, shared by
    # PostgreSQL and ChromaDB for promoted, rejected and failed events.
    triggered_at = _canonical_triggered_at()

    run_id = None

    try:

        try:

            # ----------------------------------------------------
            # 1. Retrain Challenger
            # ----------------------------------------------------

            logger.info(
                "Starting Challenger retraining..."
            )

            retrain_result = retrain.main()

            run_id = retrain_result.get("run_id")

            if not run_id:
                raise ValueError(
                    "Retraining did not return a valid MLflow run_id."
                )

            logger.info(
                "Challenger training completed | run_id=%s",
                run_id,
            )

            # ----------------------------------------------------
            # 2. Evaluate Challenger + Promotion Gates
            # ----------------------------------------------------

            logger.info(
                "Starting Challenger evaluation and promotion gates..."
            )

            promotion_result = _require_promotion_dict(
                promote_model(run_id)
            )

        except Exception as error:

            # Make the failed attempt auditable, then re-raise it
            _record_failed_attempt(
                triggered_reason,
                triggered_at,
                run_id,
                error,
            )

            raise

        status = (
            "promoted"
            if promotion_result["promoted"]
            else "rejected"
        )

        # --------------------------------------------------------
        # 3. Save Retraining Event to PostgreSQL
        #
        # Outside the failure handler above: if this insert fails the
        # error propagates without a second ("failed") row, since the
        # attempt itself completed and may already have promoted.
        # --------------------------------------------------------

        event_id = insert_retraining_event(
            triggered_reason=triggered_reason,
            new_model_rmse=promotion_result["new_model_rmse"],
            champion_rmse=promotion_result["champion_rmse"],
            promoted=promotion_result["promoted"],
            mlflow_run_id=promotion_result["mlflow_run_id"],
            triggered_at=triggered_at,
            status=status,
        )

        # --------------------------------------------------------
        # 4. Index Retraining Event in ChromaDB
        # --------------------------------------------------------

        upsert_retraining_event(
            event_id=event_id,
            triggered_at=triggered_at,
            triggered_reason=triggered_reason,
            new_model_rmse=promotion_result["new_model_rmse"],
            champion_rmse=promotion_result["champion_rmse"],
            promoted=promotion_result["promoted"],
            mlflow_run_id=promotion_result["mlflow_run_id"],
            status=status,
        )

        # --------------------------------------------------------
        # 5. Final result
        # --------------------------------------------------------

        result = {
            "run_id": run_id,
            "retrain_metrics": retrain_result.get(
                "metrics",
                {}
            ),
            "event_id": event_id,
            "promoted": promotion_result["promoted"],
            "mlflow_run_id": promotion_result[
                "mlflow_run_id"
            ],
            "new_model_rmse": promotion_result[
                "new_model_rmse"
            ],
            "new_model_mae": promotion_result[
                "new_model_mae"
            ],
            "new_model_r2": promotion_result[
                "new_model_r2"
            ],
            "champion_rmse": promotion_result[
                "champion_rmse"
            ],
            "champion_mae": promotion_result[
                "champion_mae"
            ],
            "champion_r2": promotion_result[
                "champion_r2"
            ],
            "champion_version": promotion_result[
                "champion_version"
            ],
            "new_version": promotion_result[
                "new_version"
            ],
        }

    except Exception as pipeline_error:
        elapsed = time.monotonic() - started_at

        # Redacted traceback instead of logger.exception (raw message);
        # the original exception object is re-raised untouched
        logger.error(
            "Automated retraining pipeline failed after %.1fs.\n%s",
            elapsed,
            redacted_traceback(pipeline_error),
        )

        raise

    elapsed = time.monotonic() - started_at

    logger.info(_SEPARATOR)

    if result["promoted"]:
        logger.info(
            "AUTOMATED RETRAINING RESULT: PROMOTED"
        )

        logger.info(
            "New Champion version: %s",
            result["new_version"],
        )

    else:
        logger.warning(
            "AUTOMATED RETRAINING RESULT: REJECTED"
        )

        logger.warning(
            "Champion remains version %s",
            result["champion_version"],
        )

    logger.info(
        "Pipeline completed in %.1fs.",
        elapsed,
    )

    logger.info(_SEPARATOR)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_retraining_pipeline()