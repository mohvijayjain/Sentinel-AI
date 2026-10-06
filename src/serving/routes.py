from fastapi import APIRouter, HTTPException, Header, Request
from .schemas import PredictRequest, PredictResponse
from .logger import logger
from .config import API_KEY
import pandas as pd
import json
import numpy as np
from datetime import datetime
from .predictor import predict_trip

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
    model = request.app.state.model
    metrics = request.app.state.model_metrics
    features = request.app.state.features
    model_version = request.app.state.model_version
    loaded_at = request.app.state.loaded_at

    return {
        "status": "healthy" if model else "model not loaded",
        "model_version": model_version,
        "loaded_at": loaded_at,
        "metrics": {
            "rmse_seconds": metrics.get("rmse"),
            "rmse_minutes": metrics.get("rmse_min"),
            "mae_seconds": metrics.get("mae"),
            "r2": metrics.get("r2"),
        },
        "features": features,
        "timestamp": datetime.now().isoformat(),
    }
    

@router.post("/predict", response_model=PredictResponse)
def predict(
    data: PredictRequest,
    request: Request,
    x_api_key: str = Header(default=None)
):
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
    return predict_trip(
        data = data,
        model = request.app.state.model,
        features = request.app.state.features,
        model_version = request.app.state.model_version,
    )    
        
@router.get("/model/info")
def model_info(request: Request):
    metrics = request.app.state.model_metrics
    features = request.app.state.features
    model_version = request.app.state.model_version

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
    }