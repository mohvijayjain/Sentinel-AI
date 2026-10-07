import logging
import os
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query,
    Request, UploadFile,
)

from src.common.error_redaction import redacted_traceback

from src.database.drift_repository import (
    get_latest_monitoring_run,
    get_monitoring_history,
    get_drifted_features,
    get_feature_drift_scores,
    get_retraining_events,
    get_prediction_logs,
)
from src.serving.routes import verify_api_key
from src.monitoring import upload_monitoring


# Router-level: every /monitoring route requires the X-API-Key,
# including any added later.
router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("/latest")
def latest_monitoring():

    result = get_latest_monitoring_run()

    if result is None:
        return {
            "message": "No monitoring data found"
        }

    return dict(result._mapping)



@router.get("/history")
def monitoring_history(
    limit: int = 20
):

    results = get_monitoring_history(
        limit
    )

    return [
        dict(row._mapping)
        for row in results
    ]



@router.get("/drifted-features")
def drifted_features():

    results = get_drifted_features()

    return [
        dict(row._mapping)
        for row in results
    ]



@router.get("/feature-scores")
def feature_scores():

    results = get_feature_drift_scores()

    return [
        dict(row._mapping)
        for row in results
    ]



# Read-only views for the monitoring UI. Row columns are returned as
# stored (retraining_events may predate migration 001: no status column).

@router.get("/retraining-events")
def retraining_events(
    limit: int = Query(20, ge=1, le=200)
):

    return [
        dict(row._mapping)
        for row in get_retraining_events(limit)
    ]



@router.get("/prediction-logs")
def prediction_logs(
    limit: int = Query(50, ge=1, le=500)
):

    return [
        dict(row._mapping)
        for row in get_prediction_logs(limit)
    ]



# ============================================================
# Run monitoring on an uploaded production dataset
# ============================================================

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 1024 * 1024


def _save_upload(upload: UploadFile, path: str):
    """Stream the upload to disk, refusing files over the size limit."""

    written = 0
    with open(path, "wb") as out:
        while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > upload_monitoring.MAX_UPLOAD_BYTES:
                raise upload_monitoring.UploadError(
                    413,
                    "File too large: the limit is "
                    f"{upload_monitoring.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            out.write(chunk)


@router.post("/run")
def run_monitoring(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    period: Optional[str] = Form(None),
):
    """
    One monitoring run for one uploaded dataset (CSV / Parquet, raw NYC
    TLC columns): preprocess, detect drift against the reference, score,
    persist and index it. Synchronous; a RETRAIN decision starts the
    existing retraining pipeline after the response is sent.
    """

    file_name = os.path.basename(file.filename or "")
    label = (period or "").strip()[:100] or None

    try:
        suffix = upload_monitoring.check_suffix(file_name)
    except upload_monitoring.UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    run_dir = upload_monitoring.new_run_dir()

    try:
        upload_path = os.path.join(run_dir, "upload" + suffix)
        _save_upload(file, upload_path)

        result = upload_monitoring.run_uploaded_dataset(
            upload_path,
            run_dir,
            model=request.app.state.model,
            file_name=file_name,
            label=label,
        )

    except upload_monitoring.UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    except upload_monitoring.RunInProgress:
        raise HTTPException(
            status_code=409,
            detail="A monitoring run is already in progress. Try again when it finishes.",
        )

    except Exception as error:
        # Full detail server-side (redacted); never the data or a trace
        logger.error(
            "Uploaded monitoring run failed.\n%s",
            redacted_traceback(error),
        )
        raise HTTPException(
            status_code=500,
            detail="Monitoring run failed while processing the dataset.",
        )

    finally:
        upload_monitoring.cleanup(run_dir)

    if result["action"] != "RETRAIN":
        result["retraining"] = "not_required"
    elif upload_monitoring.retraining_running():
        result["retraining"] = "already_running"
    else:
        background_tasks.add_task(upload_monitoring.start_retraining)
        result["retraining"] = "started"

    return result
