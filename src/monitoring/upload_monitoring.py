"""
src/monitoring/upload_monitoring.py

One monitoring cycle for an uploaded production dataset. Glue only: every
stage is the existing pipeline code, called in the order the batch run
uses it:

    raw file   -> ingest_data.read_raw_file / preprocess_raw
                  (clean_and_engineer: same features, same CamelCase schema
                  as the reference)
    detection  -> drift_runner.run_drift_detection, current vs the existing
                  reference, reports written to a PRIVATE temp directory
    scoring    -> drift_scorer.score_reports on those private reports
    persist    -> insert_drift_scores + insert_monitoring_run (PostgreSQL),
                  upsert_monitoring_run (RAG index), as drift_scorer does
    RETRAIN    -> start_retraining() runs the existing
                  run_retraining_pipeline (the caller schedules it after
                  responding, since training takes minutes)

Isolation: each run gets its own directory for the uploaded file and the
three reports, deleted afterwards, so runs never touch the committed
reports/ files or each other's. Runs are also serialized (one at a time):
detection loads the full reference (~7M rows), so parallel runs would
multiply memory, not throughput.
"""

import logging
import os
import shutil
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Baseline for every uploaded run; never modified here
REFERENCE_PATH = "data/reference/reference_data.parquet"

ALLOWED_SUFFIXES = (".csv", ".parquet")

MAX_UPLOAD_BYTES = int(os.getenv("MONITORING_MAX_UPLOAD_MB", "300")) * 1024 * 1024

_run_lock = threading.Lock()
_retrain_lock = threading.Lock()


class UploadError(Exception):
    """A client-side problem with the upload: safe to show as-is."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class RunInProgress(Exception):
    """Another monitoring run is still executing."""


def new_run_dir() -> str:
    return tempfile.mkdtemp(prefix="sentinel-monitoring-")


def check_suffix(filename: str) -> str:
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(
            415, "Unsupported file type: upload a .csv or .parquet file."
        )
    return suffix


def missing_columns(path: str):
    """Required raw columns absent from the file (header / schema only)."""

    import pandas as pd
    import pyarrow.parquet as pq

    from src.ingestion.ingest_data import RAW_REQUIRED_COLUMNS

    try:
        if path.lower().endswith(".csv"):
            columns = list(pd.read_csv(path, nrows=0).columns)
        else:
            columns = pq.read_schema(path).names
    except Exception:
        raise UploadError(422, "The file could not be read as CSV / Parquet.")

    return [c for c in RAW_REQUIRED_COLUMNS if c not in columns]


def run_uploaded_dataset(
    file_path: str,
    reports_dir: str,
    *,
    model,
    file_name: str,
    label: Optional[str] = None,
) -> dict:
    """
    Validate, preprocess, detect, score and persist one uploaded dataset.

    Raises UploadError for bad input, RunInProgress if another run holds
    the lock; anything else is a processing failure for the caller to log
    (redacted) and report generically.
    """

    if not _run_lock.acquire(blocking=False):
        raise RunInProgress()

    try:
        return _run(file_path, reports_dir, model=model,
                    file_name=file_name, label=label)
    finally:
        _run_lock.release()


def _run(file_path, reports_dir, *, model, file_name, label):

    from src.ingestion.ingest_data import (
        RAW_REQUIRED_COLUMNS, preprocess_raw, read_raw_file,
    )
    from src.monitoring import drift_runner, drift_scorer
    from src.database.drift_repository import (
        insert_drift_scores, insert_monitoring_run,
    )
    from src.rag.monitoring_updater import upsert_monitoring_run

    # ---- Validate --------------------------------------------------
    missing = missing_columns(file_path)
    if missing:
        raise UploadError(
            422, "Missing required columns: " + ", ".join(missing)
        )

    # Only the columns preprocessing uses: same result, less memory
    current = read_raw_file(file_path, columns=RAW_REQUIRED_COLUMNS)
    rows_uploaded = len(current)

    if rows_uploaded == 0:
        raise UploadError(422, "The dataset is empty.")

    # ---- Preprocess (shared pipeline) ------------------------------
    current = preprocess_raw(current, month_name=label or file_name)
    rows_processed = len(current)

    if rows_processed == 0:
        raise UploadError(
            422,
            "No valid rows after preprocessing (every row was filtered "
            "as invalid or an outlier).",
        )

    # ---- Detect: current vs the existing reference -----------------
    reference = drift_runner.load_reference(REFERENCE_PATH)
    reference_rows = len(reference)

    drift_runner.run_drift_detection(
        reference=reference,
        current=current,
        model=model,
        reports_dir=reports_dir,
    )
    del reference, current

    # ---- Score the private reports ---------------------------------
    statistical_path = os.path.join(reports_dir, "statistical_drift.csv")

    scores = drift_scorer.score_reports(
        statistical_path=statistical_path,
        shap_path=os.path.join(reports_dir, "shap_drift.csv"),
        prediction_path=os.path.join(reports_dir, "prediction_drift.csv"),
    )

    # ---- Persist, exactly like drift_scorer's run ------------------
    statistical_df = drift_scorer.read_drift_report(statistical_path)
    if statistical_df is None:
        import pandas as pd
        statistical_df = pd.DataFrame()

    insert_drift_scores(statistical_df)
    drifted_features = drift_scorer.extract_drifted_features(statistical_df)

    # monitoring_runs has no label column: report_path records the source
    source = f"upload:{file_name}" + (f" ({label})" if label else "")

    run_id = insert_monitoring_run(
        statistical_score=scores["statistical_score"],
        shap_score=scores["shap_score"],
        prediction_score=scores["prediction_score"],
        overall_score=scores["overall_score"],
        action=scores["action"],
        drifted_features=drifted_features,
        report_path=source,
    )

    indexed = upsert_monitoring_run(
        run_id=run_id,
        statistical_score=scores["statistical_score"],
        shap_score=scores["shap_score"],
        prediction_score=scores["prediction_score"],
        overall_score=scores["overall_score"],
        action=scores["action"],
    )

    return {
        "run_id": run_id,
        "file_name": file_name,
        "label": label,
        "rows_uploaded": rows_uploaded,
        "rows_processed": rows_processed,
        "reference": {"path": REFERENCE_PATH, "rows": reference_rows},
        "statistical_score": scores["statistical_score"],
        "shap_score": scores["shap_score"],
        "prediction_score": scores["prediction_score"],
        "overall_score": round(scores["overall_score"], 3),
        "action": scores["action"],
        "drifted_features": drifted_features,
        "indexed_for_assistant": bool(indexed),
    }


def cleanup(run_dir: str):
    shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# RETRAIN: the existing pipeline, never two at once
# ============================================================

def retraining_running() -> bool:
    return _retrain_lock.locked()


def start_retraining(triggered_reason: str = "drift_detected_upload"):
    """
    Run the existing retraining + promotion pipeline (minutes). Meant to
    run after the HTTP response; the outcome is recorded in
    retraining_events by the pipeline itself. Never raises.
    """

    if not _retrain_lock.acquire(blocking=False):
        logger.warning("Retraining already running; not started again.")
        return

    try:
        from src.common.error_redaction import redacted_traceback
        from src.training.orchestrator import run_retraining_pipeline

        try:
            result = run_retraining_pipeline(triggered_reason=triggered_reason)
            logger.info(
                "Retraining finished: promoted=%s",
                (result or {}).get("promoted"),
            )
        except Exception as error:
            logger.error(
                "Retraining pipeline failed; Champion unchanged.\n%s",
                redacted_traceback(error),
            )
    finally:
        _retrain_lock.release()
