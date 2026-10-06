"""
MLflow experiment / registered-model names, in one place.

Configurable via environment; unset means the original literals, so the
default behaviour is unchanged. Names only: artifact locations are decided
by the tracking server (--default-artifact-root) or, for a local file
store, ./mlruns relative to the working directory, exactly as before.

MLFLOW_EXPERIMENT_NAME is also the variable MLflow itself reads for its
default experiment, so both agree.

The Champion alias is deliberately NOT configurable: promotion semantics
and the serving lookup depend on it.
"""

import os

from dotenv import load_dotenv


# Callers may import this before running their own load_dotenv()
load_dotenv()


MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "Sentinel-AI",
)

MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "sentinel-ai-champion",
)

CHAMPION_ALIAS = "Champion"
