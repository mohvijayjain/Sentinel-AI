import json
from pathlib import Path
import pickle
import mlflow
import mlflow.lightgbm
import os
import tempfile
from typing import Any

class MLflowLogger:
    def __init__(
        self,
        experiment_name: str="Sentinel-AI",
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
    def start_run(self,run_name: str = None):
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: dict):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: dict):
        """ Log evaluation metrics to mlflow. """
        mlflow.log_metrics(metrics)
            
    def log_model(self, model: Any):
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name = "model"
        )
    def log_artifact(self, path: str):
        mlflow.log_artifact(path)
        
    def log_json(self, data: dict, filename: str):
        """Create a JSON file from a dictionary and log it as an MLflow artifact."""

        temp_dir = Path(tempfile.gettempdir())
        json_path = temp_dir / filename

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        mlflow.log_artifact(str(json_path))
        
    def end_run(self):
        mlflow.end_run()
    
    def register_model(
        self,
        run_id: str,
        model_name: str = "sentinel-ai-champion"
    )->str:
        """Register model in mlflow Model registry"""
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        return mv.version
    def promote_to_production(
        self,
        model_name: str = "sentinel-ai-champion",
        model_version: str = "1"
    ):
        """Promote model version to Production stage"""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version,
            stage = "Production",
            archive_existing_version=True
        )
        print(f"Model v{model_version} promoted to Production")
        
    