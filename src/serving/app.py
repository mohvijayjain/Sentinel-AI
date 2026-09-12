import pickle
import os
import json

from pydantic import BaseModel,Field
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator


from datetime import datetime
from contextlib import asynccontextmanager

from .lifespan import lifespan
from .routes import router
from .logger import logger

from src.api.routes.monitoring import router as monitoring_router
from src.api.routes.rag import router as rag_router




app = FastAPI(
    title="Sentinel AI",
    description="Autonomous MLOps Platform — Trip Duration Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(router)
app.include_router(
    monitoring_router
)
app.include_router(rag_router)

model = None
features = None
model_metrics = {}
model_version = "v1"
loaded_at = None

def load_model():
    global model, features, model_metrics, loaded_at
    logger.info("Loading Model......")
    model_path = os.getenv("MODEL_PATH", "model/model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    features_path = os.getenv("FEATURES_PATH","model/features.json")
    with open(features_path, "rb") as f:
        features = json.load(f)
        
    metrics_path = os.getenv("FEATURES_PATH","model/features.json")
    with open(metrics_path, "rb") as f:
        model_metrics = json.load(f)
        
    loaded_at = datetime.now().isoformat()
    
    logger.info("Model loaded successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup  code
    logger.info("Starting Sentinel-AI")
    load_model()
    
    yield
    
    logger.info("Server stopped")
