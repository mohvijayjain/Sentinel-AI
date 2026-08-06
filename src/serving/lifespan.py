from contextlib import asynccontextmanager
from fastapi import FastAPI

from .logger import logger
from .model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting sentinel_AI")
    load_model(app)
    yield
    logger.info("shutting down")