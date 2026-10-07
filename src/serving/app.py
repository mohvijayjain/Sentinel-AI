from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

# The single canonical lifecycle: lifespan.py runs require_api_key() and
# then model_loader.load_model(app), which populates app.state.* (what the
# routes read). Do not define another lifespan or loader in this module.
from .lifespan import lifespan
from .config import CORS_ALLOWED_ORIGINS
from .routes import router, verify_api_key

from src.api.routes.monitoring import router as monitoring_router
from src.api.routes.rag import router as rag_router


app = FastAPI(
    title="Sentinel AI",
    description="Autonomous MLOps Platform — Trip Duration Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


# Explicit allowlist from CORS_ALLOWED_ORIGINS (never "*"; see config.py)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /metrics requires the same X-API-Key as /predict: nothing in this repo
# scrapes it, and port 8000 is published on the host. A Prometheus scrape
# job must send the header (scrape_config http_headers).
Instrumentator().instrument(app).expose(
    app, dependencies=[Depends(verify_api_key)]
)

app.include_router(router)
app.include_router(
    monitoring_router
)
app.include_router(rag_router)
