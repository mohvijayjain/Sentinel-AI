# src/training/mlflow_logger.py

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

from src.training.mlflow_names import (
    CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
)


logger = logging.getLogger(__name__)


class MLflowLogger:

    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name

        self.tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
        )

        if not self.tracking_uri:
            raise RuntimeError(
                "MLFLOW_TRACKING_URI is not set"
            )

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        mlflow.set_experiment(
            self.experiment_name
        )

        self.client = MlflowClient(
            tracking_uri=self.tracking_uri
        )

    # --------------------------------------------------------
    # Run management
    # --------------------------------------------------------

    def start_run(self, run_name: str | None = None):
        return mlflow.start_run(
            run_name=run_name
        )

    def end_run(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float]):
        mlflow.log_metrics(metrics)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def log_model(self, model):
        """
        Save the LightGBM model locally and log it as a
        traditional MLflow RUN artifact.

        This intentionally uses the classic:
            runs:/<run_id>/model

        URI because the Sentinel-AI promotion pipeline
        loads the exact model artifact from the run.
        """

        if mlflow.active_run() is None:
            raise RuntimeError(
                "Cannot log model without an active MLflow run"
            )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="sentinel_mlflow_model_"
            )
        )

        model_dir = temp_dir / "model"

        try:

            logger.info(
                "Saving LightGBM model to temporary directory: %s",
                model_dir,
            )

            mlflow.lightgbm.save_model(
                lgb_model=model,
                path=str(model_dir),
            )

            logger.info(
                "Logging model as run artifact: model"
            )

            mlflow.log_artifacts(
                str(model_dir),
                artifact_path="model",
            )

            run_id = mlflow.active_run().info.run_id

            model_uri = (
                f"runs:/{run_id}/model"
            )

            logger.info(
                "Model logged successfully: %s",
                model_uri,
            )

            return model_uri

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    # --------------------------------------------------------
    # Generic artifact
    # --------------------------------------------------------

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    # --------------------------------------------------------
    # JSON artifact
    # --------------------------------------------------------

    def log_json(
        self,
        data: dict[str, Any],
        filename: str,
    ):
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="sentinel_mlflow_json_"
            )
        )

        try:

            json_path = (
                temp_dir / filename
            )

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=4,
                )

            mlflow.log_artifact(
                str(json_path)
            )

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    # --------------------------------------------------------
    # Registry
    # --------------------------------------------------------

    def register_model(
        self,
        run_id: str,
        model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    ) -> str:

        model_uri = (
            f"runs:/{run_id}/model"
        )

        logger.info(
            "Registering model from %s",
            model_uri,
        )

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        version = str(
            registered.version
        )

        logger.info(
            "Registered model %s version %s",
            model_name,
            version,
        )

        return version

    # --------------------------------------------------------
    # Champion alias
    # --------------------------------------------------------

    def set_champion(
        self,
        model_name: str,
        model_version: str,
        alias: str = CHAMPION_ALIAS,
    ):

        logger.info(
            "Setting %s alias -> %s version %s",
            alias,
            model_name,
            model_version,
        )

        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(model_version),
        )