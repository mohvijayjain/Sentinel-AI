# src/training/log_baseline.py

import json
import logging
import os
import pickle

import mlflow
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.dataset import load_frozen_test
from src.training.mlflow_logger import MLflowLogger


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MODEL_NAME = "sentinel-ai-champion"
EXPERIMENT_NAME = "Sentinel-AI"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Load Baseline Model
# ============================================================

def load_baseline_model():
    """
    Load the existing baseline model from model/model.pkl.
    """

    model_path = "model/model.pkl"

    logger.info(
        "Loading baseline model from %s",
        model_path,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Baseline model not found at {model_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(
        "Baseline model loaded successfully"
    )

    return model


# ============================================================
# Evaluate Baseline
# ============================================================

def evaluate_baseline(model, test_df):
    """
    Evaluate baseline model on the frozen test set.

    The baseline model was trained with its original feature
    names/order, so the frozen test data is explicitly aligned
    to model.feature_name_ before prediction.
    """

    logger.info(
        "Evaluating baseline on frozen test set"
    )

    if "trip_duration" not in test_df.columns:
        raise ValueError(
            "Frozen test set does not contain "
            "'trip_duration' target column"
        )

    y = test_df["trip_duration"]

    # --------------------------------------------------------
    # Get exact features expected by the trained model
    # --------------------------------------------------------

    model_features = list(model.feature_name_)

    logger.info(
        "Model expects %d features",
        len(model_features),
    )

    # --------------------------------------------------------
    # Map normalized column names to original model names
    # --------------------------------------------------------

    feature_aliases = {
        "pulocationid": "PULocationID",
        "dolocationid": "DOLocationID",
        "vendorid": "VendorID",
        "ratecodeid": "RatecodeID",
    }

    X = test_df.drop(
        columns=["trip_duration", "row_key"],
        errors="ignore",
    ).copy()

    X = X.rename(
        columns=feature_aliases
    )

    # --------------------------------------------------------
    # Validate feature set
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in X.columns
    ]

    extra_features = [
        feature
        for feature in X.columns
        if feature not in model_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if extra_features:
        logger.warning(
            "Ignoring non-model columns: %s",
            extra_features,
        )

    # --------------------------------------------------------
    # Exact feature selection + exact order
    # --------------------------------------------------------

    X = X[model_features]

    logger.info(
        "Feature alignment successful: %d features",
        len(X.columns),
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(X)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    mae = mean_absolute_error(
        y,
        predictions,
    )

    r2 = r2_score(
        y,
        predictions,
    )

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    logger.info(
        "Baseline metrics | RMSE=%.4f | MAE=%.4f | R2=%.4f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    return metrics


# ============================================================
# Load Model Metadata
# ============================================================

def load_model_metadata():
    """
    Load parameters and feature information associated
    with the baseline model.
    """

    with open("model/best_params.json", "r") as f:
        params = json.load(f)

    with open("model/features.json", "r") as f:
        features = json.load(f)

    logger.info(
        "Loaded model metadata | features=%d",
        len(features),
    )

    return params, features


# ============================================================
# Log Baseline to MLflow
# ============================================================

def log_baseline_model():

    logger.info("=" * 60)
    logger.info("SENTINEL-AI BASELINE / CHAMPION BOOTSTRAP")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load baseline model
    # --------------------------------------------------------

    model = load_baseline_model()

    # --------------------------------------------------------
    # Step 2: Load frozen test set
    # --------------------------------------------------------

    logger.info(
        "Loading frozen test set"
    )

    test_df = load_frozen_test()

    logger.info(
        "Frozen test rows: %d",
        len(test_df),
    )

    # --------------------------------------------------------
    # Step 3: Evaluate baseline
    # --------------------------------------------------------

    metrics = evaluate_baseline(
        model,
        test_df,
    )

    # --------------------------------------------------------
    # Step 4: Load metadata
    # --------------------------------------------------------

    params, features = load_model_metadata()

    # --------------------------------------------------------
    # Step 5: Initialize MLflow logger
    # --------------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set"
        )

    logger.info(
        "MLflow tracking URI: %s",
        tracking_uri,
    )

    mlflow_logger = MLflowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=tracking_uri,
    )

    # --------------------------------------------------------
    # Step 6: Create MLflow run
    # --------------------------------------------------------

    logger.info(
        "Logging baseline model to MLflow"
    )

    with mlflow_logger.start_run(
        run_name="baseline_champion_v1"
    ):

        # Hyperparameters
        mlflow_logger.log_params(
            params
        )

        # Evaluation metrics
        mlflow_logger.log_metrics(
            metrics
        )

        # Model metadata
        mlflow_logger.log_params(
            {
                "model_type": "LightGBM",
                "target": "trip_duration",
                "trained_on": "NYC Taxi 2026 Jan-Mar",
                "evaluation_dataset": "frozen_test.parquet",
                "evaluation_rows": len(test_df),
                "n_features": len(features),
            }
        )

        # Existing artifacts
        mlflow_logger.log_artifact(
            "model/features.json"
        )

        mlflow_logger.log_artifact(
            "model/feature_importance.csv"
        )

        # Feature definition
        mlflow_logger.log_json(
            {
                "features": features
            },
            "features.json",
        )

        # Model
        mlflow_logger.log_model(
            model
        )

        run_id = mlflow.active_run().info.run_id

        logger.info(
            "Baseline MLflow run created: %s",
            run_id,
        )

    # --------------------------------------------------------
    # Step 7: Register model
    # --------------------------------------------------------

    logger.info(
        "Registering baseline as '%s'",
        MODEL_NAME,
    )

    version = mlflow_logger.register_model(
        run_id=run_id,
        model_name=MODEL_NAME,
    )

    logger.info(
        "Baseline registered as version %s",
        version,
    )

    # --------------------------------------------------------
    # Step 8: Set Champion alias
    # --------------------------------------------------------

    logger.info(
        "Setting Champion alias → version %s",
        version,
    )

    mlflow_logger.set_champions(
        model_name=MODEL_NAME,
        model_version=version,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    logger.info("=" * 60)
    logger.info("BASELINE CHAMPION CREATED SUCCESSFULLY")
    logger.info("=" * 60)

    logger.info(
        "Model: %s",
        MODEL_NAME,
    )

    logger.info(
        "Version: %s",
        version,
    )

    logger.info(
        "Champion alias: %s",
        "Champion",
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "RMSE: %.4f",
        metrics["rmse"],
    )

    logger.info(
        "MAE: %.4f",
        metrics["mae"],
    )

    logger.info(
        "R2: %.4f",
        metrics["r2"],
    )

    return {
        "run_id": run_id,
        "version": version,
        "metrics": metrics,
    }


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        log_baseline_model()

    except Exception:

        logger.exception(
            "Failed to bootstrap Champion model"
        )

        raise