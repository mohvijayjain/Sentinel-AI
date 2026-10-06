from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import require_api_key
from .logger import logger
from .model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting sentinel_AI")
    # Fail fast: never serve without a configured API key
    require_api_key()
    load_model(app)
    yield
    logger.info("shutting down")