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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.training.dataset import load_frozen_test


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MODEL_NAME = "sentinel-ai-champion"

CHAMPION_ALIAS = "Champion"

EXPERIMENT_NAME = "Sentinel-AI"

RMSE_IMPROVEMENT_THRESHOLD = 0.01


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# MLflow
# ============================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

if not TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is not set"
    )

mlflow.set_tracking_uri(
    TRACKING_URI
)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)


# ============================================================
# Types
# ============================================================

Metrics = Dict[str, float]


# ============================================================
# Evaluate
# ============================================================

def evaluate_model(
    model,
    test_df: pd.DataFrame,
) -> Metrics:
    """
    Evaluate a model on the frozen test dataset.

    The evaluation automatically aligns the frozen-test columns
    with the exact feature names expected by the loaded model.
    """

    logger.info(
        "Evaluating model on %d frozen-test rows",
        len(test_df),
    )

    if "trip_duration" not in test_df.columns:
        raise ValueError(
            "Frozen test dataset does not contain "
            "'trip_duration'"
        )

    y = test_df["trip_duration"]

    X = test_df.drop(
        columns=[
            "trip_duration",
            "row_key",
        ],
        errors="ignore",
    ).copy()

    if not hasattr(model, "feature_name_"):
        raise ValueError(
            "Loaded model does not expose "
            "LightGBM feature_name_"
        )

    model_features = list(
        model.feature_name_
    )

    logger.info(
        "Model expects features: %s",
        model_features,
    )

    # --------------------------------------------------------
    # Case-insensitive feature alignment
    # --------------------------------------------------------

    column_lookup = {
        column.lower(): column
        for column in X.columns
    }

    missing_features = []

    selected_columns = []

    for feature in model_features:

        source_column = column_lookup.get(
            feature.lower()
        )

        if source_column is None:
            missing_features.append(
                feature
            )
        else:
            selected_columns.append(
                source_column
            )

    if missing_features:
        raise ValueError(
            f"Missing model features: "
            f"{missing_features}"
        )

    X = X[
        selected_columns
    ].copy()

    # Rename to EXACT model feature names
    X.columns = model_features

    logger.info(
        "Feature alignment successful: %d features",
        len(X.columns),
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = (
        mean_squared_error(
            y,
            predictions,
        )
        ** 0.5
    )

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
        "Metrics | RMSE=%.4f | MAE=%.4f | R2=%.4f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    return metrics


# ============================================================
# Load Champion
# ============================================================

def load_champion() -> Tuple[object, str]:

    logger.info(
        "Looking up Champion alias"
    )

    try:

        champion_version = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                CHAMPION_ALIAS,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not find Champion alias "
            f"for '{MODEL_NAME}'"
        ) from exc

    version = str(
        champion_version.version
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{version}"
    )

    logger.info(
        "Loading Champion: %s",
        model_uri,
    )

    try:

        model = (
            mlflow.lightgbm.load_model(
                model_uri
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load Champion version "
            f"{version}"
        ) from exc

    logger.info(
        "Champion version %s loaded",
        version,
    )

    return model, version


# ============================================================
# Load Challenger
# ============================================================

def load_challenger(
    run_id: str,
) -> object:

    model_uri = (
        f"runs:/{run_id}/model"
    )

    logger.info(
        "Loading Challenger from: %s",
        model_uri,
    )

    # --------------------------------------------------------
    # Verify run exists
    # --------------------------------------------------------

    try:

        run = client.get_run(
            run_id
        )

    except Exception as exc:

        raise RuntimeError(
            f"Challenger run '{run_id}' "
            "does not exist"
        ) from exc

    logger.info(
        "Challenger run found: %s",
        run.info.run_id,
    )

    # --------------------------------------------------------
    # Verify model artifact exists
    # --------------------------------------------------------

    try:

        artifacts = client.list_artifacts(
            run_id,
            "model",
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not inspect model artifacts "
            f"for run '{run_id}'"
        ) from exc

    if not artifacts:

        raise RuntimeError(
            f"No model artifact found in "
            f"Challenger run '{run_id}'. "
            "Retrain the Challenger using the "
            "updated MLflow logging code."
        )

    logger.info(
        "Challenger artifacts found:"
    )

    for artifact in artifacts:

        logger.info(
            "  %s",
            artifact.path,
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = (
            mlflow.lightgbm.load_model(
                model_uri
            )
        )

    except Exception as exc:

        logger.exception(
            "Failed to load Challenger"
        )

        raise RuntimeError(
            f"Could not load Challenger "
            f"from '{model_uri}'"
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

    champion_rmse = (
        champion_metrics["rmse"]
    )

    challenger_rmse = (
        challenger_metrics["rmse"]
    )

    champion_mae = (
        champion_metrics["mae"]
    )

    challenger_mae = (
        challenger_metrics["mae"]
    )

    champion_r2 = (
        champion_metrics["r2"]
    )

    challenger_r2 = (
        challenger_metrics["r2"]
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    required_rmse = (
        champion_rmse
        * (1 - RMSE_IMPROVEMENT_THRESHOLD)
    )

    rmse_pass = (
        challenger_rmse
        <= required_rmse
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae_pass = (
        challenger_mae
        <= champion_mae
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    r2_pass = (
        challenger_r2
        >= champion_r2
    )

    all_pass = (
        rmse_pass
        and mae_pass
        and r2_pass
    )

    logger.info(
        "RMSE gate: %s",
        "PASS" if rmse_pass else "FAIL",
    )

    logger.info(
        "MAE gate: %s",
        "PASS" if mae_pass else "FAIL",
    )

    logger.info(
        "R2 gate: %s",
        "PASS" if r2_pass else "FAIL",
    )

    logger.info(
        "Promotion decision: %s",
        "PROMOTE" if all_pass else "REJECT",
    )

    return {
        "rmse_gate": rmse_pass,
        "mae_gate": mae_pass,
        "r2_gate": r2_pass,
        "all_pass": all_pass,
        "required_rmse": float(
            required_rmse
        ),
    }


# ============================================================
# Register Challenger
# ============================================================

def register_challenger(
    run_id: str,
) -> str:

    model_uri = (
        f"runs:/{run_id}/model"
    )

    logger.info(
        "Registering Challenger: %s",
        model_uri,
    )

    try:

        registered = (
            mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME,
            )
        )

    except Exception as exc:

        logger.exception(
            "Failed to register Challenger"
        )

        raise RuntimeError(
            "Could not register Challenger"
        ) from exc

    version = str(
        registered.version
    )

    logger.info(
        "Challenger registered as version %s",
        version,
    )

    return version


# ============================================================
# Promote
# ============================================================

def promote(
    version: str,
) -> None:

    logger.info(
        "Setting Champion alias -> version %s",
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
            "Failed to promote version %s",
            version,
        )

        raise RuntimeError(
            f"Could not promote version {version}"
        ) from exc

    logger.info(
        "Version %s is now Champion",
        version,
    )


# ============================================================
# Comparison
# ============================================================

def log_model_comparison(
    champion_version: str,
    champion_metrics: Metrics,
    challenger_metrics: Metrics,
) -> None:

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

    rmse_improvement = (
        (
            champion_metrics["rmse"]
            - challenger_metrics["rmse"]
        )
        / champion_metrics["rmse"]
    ) * 100

    mae_improvement = (
        (
            champion_metrics["mae"]
            - challenger_metrics["mae"]
        )
        / champion_metrics["mae"]
    ) * 100

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
# Main
# ============================================================

def main(
    run_id: str,
) -> int:

    logger.info("=" * 60)
    logger.info(
        "SENTINEL-AI MODEL PROMOTION"
    )
    logger.info("=" * 60)

    logger.info(
        "Challenger Run ID: %s",
        run_id,
    )

    # --------------------------------------------------------
    # 1. Frozen test
    # --------------------------------------------------------

    test_df = load_frozen_test()

    logger.info(
        "Frozen test rows: %d",
        len(test_df),
    )

    # --------------------------------------------------------
    # 2. Champion
    # --------------------------------------------------------

    champion_model, champion_version = (
        load_champion()
    )

    # --------------------------------------------------------
    # 3. Challenger
    # --------------------------------------------------------

    challenger_model = load_challenger(
        run_id
    )

    # --------------------------------------------------------
    # 4. Evaluate Champion
    # --------------------------------------------------------

    logger.info(
        "Evaluating Champion"
    )

    champion_metrics = evaluate_model(
        champion_model,
        test_df,
    )

    # --------------------------------------------------------
    # 5. Evaluate Challenger
    # --------------------------------------------------------

    logger.info(
        "Evaluating Challenger"
    )

    challenger_metrics = evaluate_model(
        challenger_model,
        test_df,
    )

    # --------------------------------------------------------
    # 6. Comparison
    # --------------------------------------------------------

    log_model_comparison(
        champion_version,
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # 7. Gates
    # --------------------------------------------------------

    gates = promotion_gates(
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # 8. Reject
    # --------------------------------------------------------

    if not gates["all_pass"]:

        logger.warning("=" * 60)
        logger.warning(
            "CHALLENGER REJECTED"
        )
        logger.warning("=" * 60)

        logger.warning(
            "Champion remains version %s",
            champion_version,
        )

        return 1

    # --------------------------------------------------------
    # 9. Register
    # --------------------------------------------------------

    logger.info(
        "All gates passed."
    )

    new_version = (
        register_challenger(
            run_id
        )
    )

    # --------------------------------------------------------
    # 10. Promote
    # --------------------------------------------------------

    promote(
        new_version
    )

    logger.info("=" * 60)
    logger.info(
        "MODEL PROMOTION SUCCESSFUL"
    )
    logger.info(
        "Previous Champion: %s",
        champion_version,
    )
    logger.info(
        "New Champion: %s",
        new_version,
    )
    logger.info("=" * 60)

    return 0


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        logger.error(
            "Usage: "
            "python -m src.training.promote "
            "<challenger_run_id>"
        )

        sys.exit(1)

    challenger_run_id = (
        sys.argv[1]
    )

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