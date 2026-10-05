# src/training/orchestrator.py

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

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

    # Capture the event timestamp once so PostgreSQL and ChromaDB
    # can refer to the same retraining event timestamp.
    triggered_at = datetime.now(timezone.utc).isoformat()

    try:

        # --------------------------------------------------------
        # 1. Retrain Challenger
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # 2. Evaluate Challenger + Promotion Gates
        # --------------------------------------------------------

        logger.info(
            "Starting Challenger evaluation and promotion gates..."
        )

        promotion_result = _require_promotion_dict(
            promote_model(run_id)
        )

        # --------------------------------------------------------
        # 3. Save Retraining Event to PostgreSQL
        # --------------------------------------------------------

        event_id = insert_retraining_event(
            triggered_reason=triggered_reason,
            new_model_rmse=promotion_result["new_model_rmse"],
            champion_rmse=promotion_result["champion_rmse"],
            promoted=promotion_result["promoted"],
            mlflow_run_id=promotion_result["mlflow_run_id"],
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

    except Exception:
        elapsed = time.monotonic() - started_at

        logger.exception(
            "Automated retraining pipeline failed after %.1fs.",
            elapsed,
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