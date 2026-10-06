"""
src/rag/monitoring_updater.py

Index Sentinel-AI monitoring runs into ChromaDB for semantic retrieval.

Design contract
---------------
  * PostgreSQL (monitoring_runs) is the source of truth.
    ChromaDB is a derived, rebuildable index.
  * upsert_monitoring_run() is BEST-EFFORT: a ChromaDB or NVIDIA failure
    is logged and swallowed, never raised into the monitoring pipeline,
    because the index can always be rebuilt from Postgres.
  * The Chroma id is deterministic (monitoring_run:<id>), so re-running
    the same run overwrites its vector instead of duplicating it.
"""

import logging
from typing import Optional

from src.common.error_redaction import redacted_traceback, safe_error_message
from src.rag.chroma_store import ChromaStore
from src.rag.embeddings import embed_passages

logger = logging.getLogger(__name__)

# Mirrors drift_scorer.get_action(). Unknown values are still indexed,
# but logged, so a scorer change that forgets this layer is visible.
KNOWN_ACTIONS = frozenset({"WAIT", "MONITOR", "ALERT", "RETRAIN"})

# One persistent client per process. Instantiating ChromaStore per call
# reopens the on-disk client every time, which is slow and wasteful.
_store: Optional[ChromaStore] = None


def _get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


# ============================================================
# Document rendering (pure)
# ============================================================

def build_monitoring_document(
    run_id: int,
    statistical_score: float,
    shap_score: float,
    prediction_score: float,
    overall_score: float,
    action: str,
) -> str:
    """
    Convert a monitoring result into searchable RAG text.

    Pure function: no I/O. Scores are formatted to 3 decimals so the
    embedded text stays stable and readable regardless of float noise.
    """

    return (
        f"Sentinel-AI Monitoring Run {run_id}.\n"
        f"Statistical drift score: {statistical_score:.3f}.\n"
        f"SHAP drift score: {shap_score:.3f}.\n"
        f"Prediction drift score: {prediction_score:.3f}.\n"
        f"Overall drift score: {overall_score:.3f}.\n"
        f"Recommended action: {action}."
    )


# ============================================================
# Retraining Event Document
# ============================================================

RETRAINING_STATUSES = frozenset({"promoted", "rejected", "failed"})


def _format_rmse(value: Optional[float]) -> str:
    return "not available" if value is None else f"{float(value):.3f}"


def build_retraining_document(
    event_id: int,
    triggered_at: str,
    triggered_reason: str,
    new_model_rmse: Optional[float],
    champion_rmse: Optional[float],
    status: str,
    mlflow_run_id: Optional[str],
    error_message: Optional[str] = None,
) -> str:
    """
    Convert a retraining event into searchable RAG text.

    Pure function: no I/O. A failed attempt has no metrics (and may have
    no MLflow run), so those render as "not available".
    """

    document = (
        f"Sentinel-AI Retraining Event {event_id}.\n"
        f"Triggered at: {triggered_at}.\n"
        f"Triggered reason: {triggered_reason}.\n"
        f"Challenger model RMSE: {_format_rmse(new_model_rmse)}.\n"
        f"Champion model RMSE: {_format_rmse(champion_rmse)}.\n"
        f"Promotion decision: {status.upper()}.\n"
        f"MLflow run ID: {mlflow_run_id or 'not available'}."
    )

    if error_message:
        document += f"\nFailure: {error_message}"

    return document

# ============================================================
# Index one run (best-effort write-through)
# ============================================================

def upsert_monitoring_run(
    run_id: int,
    statistical_score: float,
    shap_score: float,
    prediction_score: float,
    overall_score: float,
    action: str,
) -> bool:
    """
    Embed one monitoring run and upsert it into ChromaDB.

    Call this right after the Postgres insert succeeds. Returns True on
    success and False on failure; never raises, so it cannot break the
    monitoring pipeline.
    """

    try:
        if run_id is None:
            raise ValueError("run_id is required.")

        if not action:
            raise ValueError("action is required.")

        if action not in KNOWN_ACTIONS:
            logger.warning(
                "Unexpected action '%s' for run %s; indexing anyway.",
                action,
                run_id,
            )

        document = build_monitoring_document(
            run_id=run_id,
            statistical_score=float(statistical_score),
            shap_score=float(shap_score),
            prediction_score=float(prediction_score),
            overall_score=float(overall_score),
            action=action,
        )

        embedding = embed_passages([document])[0]

        _get_store().upsert_documents(
            documents=[document],
            embeddings=[embedding],
            ids=[f"monitoring_run:{run_id}"],
            metadatas=[
                {
                    "source": "monitoring_runs",
                    "run_id": int(run_id),
                    "action": str(action),
                    "overall_score": float(overall_score),
                }
            ],
        )

        logger.info("Monitoring run indexed in ChromaDB: %s", run_id)
        return True

    except Exception as error:
        # Postgres already holds the truth; the index can be rebuilt.
        # NVIDIA / Chroma errors can echo credentials: redacted traceback.
        logger.error(
            "Failed to index monitoring run %s; "
            "Postgres remains the source of truth.\n%s",
            run_id,
            redacted_traceback(error),
        )
        return False
    
# ============================================================
# Index one retraining event (best-effort)
# ============================================================

def upsert_retraining_event(
    event_id: int,
    triggered_at: str,
    triggered_reason: str,
    new_model_rmse: Optional[float],
    champion_rmse: Optional[float],
    promoted: Optional[bool],
    mlflow_run_id: Optional[str],
    *,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    """
    Embed one retraining event and upsert it into ChromaDB.

    PostgreSQL remains the source of truth.
    ChromaDB is only the derived semantic-search index.

    status is 'promoted', 'rejected' or 'failed'. A failed attempt has
    no metrics; None fields are left out of the metadata.

    Returns True on success and False on failure.
    Never raises into the training pipeline.
    """

    try:

        if event_id is None:
            raise ValueError(
                "event_id is required."
            )

        if not triggered_reason:
            raise ValueError(
                "triggered_reason is required."
            )

        if status not in RETRAINING_STATUSES:
            raise ValueError(
                f"Unknown retraining status {status!r}."
            )

        # Never embed a raw secret; idempotent for already-safe text
        if error_message is not None:
            error_message = safe_error_message(error_message)

        document = build_retraining_document(
            event_id=event_id,
            triggered_at=str(triggered_at),
            triggered_reason=str(triggered_reason),
            new_model_rmse=new_model_rmse,
            champion_rmse=champion_rmse,
            status=status,
            mlflow_run_id=mlflow_run_id,
            error_message=error_message,
        )

        metadata = {
            "source": "retraining_events",
            "event_id": int(event_id),
            "triggered_at": str(triggered_at),
            "triggered_reason": str(triggered_reason),
            "status": status,
            "promoted": promoted,
            "new_model_rmse": new_model_rmse,
            "champion_rmse": champion_rmse,
            "mlflow_run_id": mlflow_run_id,
            "error_message": error_message,
        }

        # Chroma metadata values cannot be None
        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

        for key in ("new_model_rmse", "champion_rmse"):
            if key in metadata:
                metadata[key] = float(metadata[key])

        if "promoted" in metadata:
            metadata["promoted"] = bool(metadata["promoted"])

        embedding = embed_passages(
            [document]
        )[0]

        _get_store().upsert_documents(
            documents=[document],
            embeddings=[embedding],
            ids=[
                f"retraining_event:{event_id}"
            ],
            metadatas=[
                metadata
            ],
        )

        logger.info(
            "Retraining event indexed in ChromaDB: %s",
            event_id,
        )

        return True

    except Exception as error:

        # NVIDIA / Chroma errors can echo credentials: redacted traceback
        logger.error(
            "Failed to index retraining event %s; "
            "Postgres remains the source of truth.\n%s",
            event_id,
            redacted_traceback(error),
        )

        return False