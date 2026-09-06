# src/training/promote.py

import logging
import os
import sys
from typing import Dict, Tuple

import mlflow
import mlflow.lightgbm
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.dataset import load_frozen_test


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MODEL_NAME = "sentinel-ai-champion"
CHAMPION_ALIAS = "Champion"

# Challenger must improve RMSE by at least 1%
RMSE_IMPROVEMENT_THRESHOLD = 0.01


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MLflow Configuration
# ============================================================

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


# ============================================================
# Types
# ============================================================

Metrics = Dict[str, float]


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(
    model,
    test_df: pd.DataFrame,
) -> Metrics:
    """
    Evaluate a model on the frozen test dataset.

    Metrics:
        RMSE - lower is better
        MAE  - lower is better
        R2   - higher is better
    """

    logger.info(
        "Evaluating model on frozen test set: %d rows",
        len(test_df),
    )

    if "trip_duration" not in test_df.columns:
        raise ValueError(
            "Frozen test dataset does not contain target column "
            "'trip_duration'"
        )

    X = test_df.drop(columns=["trip_duration"])
    y = test_df["trip_duration"]

    predictions = model.predict(X)

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
        "Evaluation complete | RMSE=%.4f | MAE=%.4f | R2=%.4f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    return metrics


# ============================================================
# Load Champion
# ============================================================

def load_champion() -> Tuple[object, str]:
    """
    Load the model currently assigned to the Champion alias.

    Returns:
        model
        champion_version
    """

    logger.info(
        "Looking up Champion model: %s @ %s",
        MODEL_NAME,
        CHAMPION_ALIAS,
    )

    try:
        champion_version = client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )

    except Exception as exc:
        logger.exception(
            "Failed to find Champion alias '%s' for model '%s'",
            CHAMPION_ALIAS,
            MODEL_NAME,
        )

        raise RuntimeError(
            f"Could not find Champion alias for '{MODEL_NAME}'"
        ) from exc

    version = champion_version.version

    logger.info(
        "Champion version found: %s",
        version,
    )

    model_uri = f"models:/{MODEL_NAME}/{CHAMPION_ALIAS}"

    logger.info(
        "Loading Champion model from: %s",
        model_uri,
    )

    try:
        model = mlflow.lightgbm.load_model(model_uri)

    except Exception as exc:
        logger.exception(
            "Failed to load Champion model"
        )

        raise RuntimeError(
            "Could not load Champion model"
        ) from exc

    logger.info(
        "Champion model loaded successfully"
    )

    return model, version


# ============================================================
# Load Challenger
# ============================================================

def load_challenger(run_id: str):
    """
    Load Challenger model directly from its MLflow run.
    """

    model_uri = f"runs:/{run_id}/model"

    logger.info(
        "Loading Challenger model from run: %s",
        run_id,
    )

    try:
        model = mlflow.lightgbm.load_model(model_uri)

    except Exception as exc:
        logger.exception(
            "Failed to load Challenger model from run %s",
            run_id,
        )

        raise RuntimeError(
            f"Could not load Challenger model from run '{run_id}'"
        ) from exc

    logger.info(
        "Challenger model loaded successfully"
    )

    return model


# ============================================================
# Promotion Gates
# ============================================================

def promotion_gates(
    champion_metrics: Metrics,
    challenger_metrics: Metrics,
) -> dict:
    """
    Apply deterministic promotion rules.

    Gate 1:
        Challenger RMSE must improve by at least 1%.

    Gate 2:
        Challenger MAE must not be worse.

    Gate 3:
        Challenger R2 must not be worse.

    All gates must pass for promotion.
    """

    champion_rmse = champion_metrics["rmse"]
    challenger_rmse = challenger_metrics["rmse"]

    champion_mae = champion_metrics["mae"]
    challenger_mae = challenger_metrics["mae"]

    champion_r2 = champion_metrics["r2"]
    challenger_r2 = challenger_metrics["r2"]

    # --------------------------------------------------------
    # Gate 1: RMSE improvement
    # --------------------------------------------------------

    required_rmse = (
        champion_rmse
        * (1 - RMSE_IMPROVEMENT_THRESHOLD)
    )

    rmse_pass = challenger_rmse <= required_rmse

    # --------------------------------------------------------
    # Gate 2: MAE must not regress
    # --------------------------------------------------------

    mae_pass = challenger_mae <= champion_mae

    # --------------------------------------------------------
    # Gate 3: R2 must not regress
    # --------------------------------------------------------

    r2_pass = challenger_r2 >= champion_r2

    all_pass = (
        rmse_pass
        and mae_pass
        and r2_pass
    )

    gates = {
        "rmse_gate": rmse_pass,
        "mae_gate": mae_pass,
        "r2_gate": r2_pass,
        "all_pass": all_pass,
        "required_rmse": required_rmse,
    }

    logger.info(
        "Gate 1 - RMSE >= %.2f%% improvement: %s",
        RMSE_IMPROVEMENT_THRESHOLD * 100,
        "PASS" if rmse_pass else "FAIL",
    )

    logger.info(
        "Gate 2 - MAE not worse: %s",
        "PASS" if mae_pass else "FAIL",
    )

    logger.info(
        "Gate 3 - R2 not worse: %s",
        "PASS" if r2_pass else "FAIL",
    )

    logger.info(
        "Promotion decision: %s",
        "PASS" if all_pass else "REJECT",
    )

    return gates


# ============================================================
# Register Challenger
# ============================================================

def register_challenger(run_id: str) -> str:
    """
    Register Challenger model in MLflow Model Registry.

    Returns:
        Registered model version.
    """

    model_uri = f"runs:/{run_id}/model"

    logger.info(
        "Registering Challenger model"
    )

    logger.info(
        "Model URI: %s",
        model_uri,
    )

    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )

    except Exception as exc:
        logger.exception(
            "Failed to register Challenger model"
        )

        raise RuntimeError(
            "Could not register Challenger model"
        ) from exc

    version = str(registered_model.version)

    logger.info(
        "Challenger registered successfully as version %s",
        version,
    )

    return version


# ============================================================
# Promote Model
# ============================================================

def promote(version: str) -> None:
    """
    Move the Champion alias to the supplied model version.
    """

    logger.info(
        "Promoting model version %s to Champion",
        version,
    )

    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=version,
        )

    except Exception as exc:
        logger.exception(
            "Failed to assign Champion alias to version %s",
            version,
        )

        raise RuntimeError(
            f"Could not promote model version {version}"
        ) from exc

    logger.info(
        "Model version %s is now Champion",
        version,
    )


# ============================================================
# Model Comparison Logging
# ============================================================

def log_model_comparison(
    champion_version: str,
    champion_metrics: Metrics,
    challenger_metrics: Metrics,
) -> None:
    """
    Log a clear Champion vs Challenger comparison.
    """

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    logger.info(
        "Champion version: %s",
        champion_version,
    )

    logger.info(
        "Champion   | RMSE=%.4f | MAE=%.4f | R2=%.4f",
        champion_metrics["rmse"],
        champion_metrics["mae"],
        champion_metrics["r2"],
    )

    logger.info(
        "Challenger | RMSE=%.4f | MAE=%.4f | R2=%.4f",
        challenger_metrics["rmse"],
        challenger_metrics["mae"],
        challenger_metrics["r2"],
    )

    # RMSE improvement percentage
    rmse_improvement = (
        (
            champion_metrics["rmse"]
            - challenger_metrics["rmse"]
        )
        / champion_metrics["rmse"]
    ) * 100

    # MAE improvement percentage
    mae_improvement = (
        (
            champion_metrics["mae"]
            - challenger_metrics["mae"]
        )
        / champion_metrics["mae"]
    ) * 100

    # R2 change
    r2_change = (
        challenger_metrics["r2"]
        - champion_metrics["r2"]
    )

    logger.info(
        "RMSE improvement: %.4f%%",
        rmse_improvement,
    )

    logger.info(
        "MAE improvement: %.4f%%",
        mae_improvement,
    )

    logger.info(
        "R2 change: %.6f",
        r2_change,
    )

    logger.info("=" * 60)


# ============================================================
# Main Promotion Pipeline
# ============================================================

def main(run_id: str) -> int:
    """
    Complete Champion vs Challenger promotion pipeline.

    Returns:
        0 -> promotion successful
        1 -> challenger rejected
    """

    logger.info("=" * 60)
    logger.info("SENTINEL-AI MODEL PROMOTION")
    logger.info("=" * 60)

    logger.info(
        "Challenger Run ID: %s",
        run_id,
    )

    # --------------------------------------------------------
    # Step 1: Load frozen test set
    # --------------------------------------------------------

    logger.info(
        "Loading frozen test dataset"
    )

    try:
        test_df = load_frozen_test()

    except Exception:
        logger.exception(
            "Failed to load frozen test dataset"
        )
        raise

    logger.info(
        "Frozen test rows: %d",
        len(test_df),
    )

    # --------------------------------------------------------
    # Step 2: Load Champion
    # --------------------------------------------------------

    champion_model, champion_version = load_champion()

    # --------------------------------------------------------
    # Step 3: Load Challenger
    # --------------------------------------------------------

    challenger_model = load_challenger(run_id)

    # --------------------------------------------------------
    # Step 4: Evaluate Champion
    # --------------------------------------------------------

    logger.info(
        "Evaluating Champion model"
    )

    champion_metrics = evaluate_model(
        champion_model,
        test_df,
    )

    # --------------------------------------------------------
    # Step 5: Evaluate Challenger
    # --------------------------------------------------------

    logger.info(
        "Evaluating Challenger model"
    )

    challenger_metrics = evaluate_model(
        challenger_model,
        test_df,
    )

    # --------------------------------------------------------
    # Step 6: Compare
    # --------------------------------------------------------

    log_model_comparison(
        champion_version,
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # Step 7: Apply promotion gates
    # --------------------------------------------------------

    gates = promotion_gates(
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # Step 8: Reject if any gate fails
    # --------------------------------------------------------

    if not gates["all_pass"]:

        logger.warning("=" * 60)
        logger.warning("CHALLENGER REJECTED")
        logger.warning("=" * 60)

        logger.warning(
            "No changes were made to the Champion model"
        )

        return 1

    # --------------------------------------------------------
    # Step 9: All gates passed
    # --------------------------------------------------------

    logger.info("=" * 60)
    logger.info("ALL PROMOTION GATES PASSED")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Step 10: Register Challenger
    # --------------------------------------------------------

    new_version = register_challenger(
        run_id
    )

    # --------------------------------------------------------
    # Step 11: Promote Challenger
    # --------------------------------------------------------

    promote(
        new_version
    )

    logger.info("=" * 60)
    logger.info(
        "MODEL PROMOTION SUCCESSFUL"
    )
    logger.info(
        "Champion version: %s",
        new_version,
    )
    logger.info("=" * 60)

    return 0


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        logger.error(
            "Invalid arguments"
        )

        logger.error(
            "Usage: "
            "python -m src.training.promote <challenger_run_id>"
        )

        sys.exit(1)

    challenger_run_id = sys.argv[1]

    try:

        exit_code = main(
            challenger_run_id
        )

    except Exception:

        logger.exception(
            "Model promotion pipeline failed"
        )

        exit_code = 1

    sys.exit(exit_code)