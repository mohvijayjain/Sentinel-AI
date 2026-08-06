import json 
import pickle 
from datetime import datetime 

from fastapi import FastAPI 

from .config import (
    MODEL_PATH,
    FEATURES_PATH,
    METRICS_PATH,
)

from .logger import logger

def load_model(app: FastAPI):
    logger.info("Loading Model")
    
    with open(MODEL_PATH, "rb") as f:
        app.state.model = pickle.load(f)
        
    with open(FEATURES_PATH, "r") as f:
        app.state.features = json.load(f)
        
    with open(METRICS_PATH, "r") as f:
        app.state.model_metrics = json.load(f)
        
        
    app.state.loaded_at = datetime.now().isoformat()
    app.state.model_version = "v1"
    
    logger.info("Model loaded Successfully")
    logger.info(f"Features: {app.state.features}")
    logger.info(f"RMSE: {app.state.model_metrics.get('rmse')}")
    logger.info(f"R^2: {app.state.model_metrics.get('r2')}")