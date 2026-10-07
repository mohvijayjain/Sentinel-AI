from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Request
from .schemas import PredictRequest, PredictResponse
from .logger import logger
from .config import API_KEY
import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone
from .predictor import predict_trip
from .prediction_logger import log_prediction

router = APIRouter()

@router.get("/")
def root():
    return {
        "title": "Sentinel AI",
        "description":"Autonomous MLOps Platform — Trip Duration Predictor",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "model_info": "GET /model/info",
            "metrics": "GET /metrics",
            "docs": "GET /docs"
        }
    }
    
@router.get("/health")
def health(request: Request):
    # Public readiness probe: no metrics, features or version details
    # (those are on the authenticated /model/info).
    model = getattr(request.app.state, "model", None)

    return {
        "status": "healthy" if model else "model not loaded",
        "model_loaded": model is not None,
    }


def verify_api_key(
    request: Request,
    x_api_key: str = Header(default=None)
):
    """
    X-API-Key check shared by /predict, /model/info, /metrics and
    /monitoring/*. Usable as a dependency or called directly.
    """
    # Never log the received or expected key, nor any header: only the
    # outcome of the attempt. Comparison and status codes are unchanged.
    # API_KEY None (unset) must never match a missing header (None)
    if API_KEY is None or x_api_key !=API_KEY:
        logger.warning(
            "API authentication failure | %s %s | reason=%s | status=401",
            request.method,
            request.url.path,
            "missing" if x_api_key is None else "invalid",
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    logger.info(
        "API authentication success | %s %s",
        request.method,
        request.url.path,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    data: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=None)
):
    # Called in the handler (not as a dependency) so request validation
    # still runs first, exactly as before.
    verify_api_key(request, x_api_key)
    response = predict_trip(
        data = data,
        model = request.app.state.model,
        features = request.app.state.features,
        model_version = request.app.state.model_version,
    )
    # Successful predictions only. Runs after the response has been sent
    # and never raises, so the database can neither delay nor fail
    # /predict. Timestamp taken now, in UTC.
    background_tasks.add_task(
        log_prediction,
        data,
        response,
        datetime.now(timezone.utc).isoformat(),
    )
    return response
        
@router.get("/model/info", dependencies=[Depends(verify_api_key)])
def model_info(request: Request):
    metrics = request.app.state.model_metrics
    features = request.app.state.features
    model_version = request.app.state.model_version
    loaded_at = request.app.state.loaded_at

    try:
        with open("../../model/best_params.json", "r") as f:
            best_params = json.load(f)
    except Exception:
        best_params = {}

    return {
        "model_type": "LightGBM Regressor",
        "target": "trip_duration (seconds)",
        "features": features,
        "best_params": best_params,
        "metrics": metrics,
        "trained_on": "NYC Yellow Taxi 2026 (Jan-Mar)",
        "model_version": model_version,
        "loaded_at": loaded_at,
    }