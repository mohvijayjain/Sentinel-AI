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
from src.evaluation.bootstrap import bootstrap_gate
from src.evaluation.segment import evaluate_segments
from src.evaluation.recent_test import evaluate_recent_test
from src.ingestion.preprocess import clean_and_engineer


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MODEL_NAME = "sentinel-ai-champion"

CHAMPION_ALIAS = "Champion"

EXPERIMENT_NAME = "Sentinel-AI"

RMSE_IMPROVEMENT_THRESHOLD = 0.01

RECENT_DATA_PATH = os.getenv(
    "RECENT_TEST_DATA_PATH",
    "/app/data/raw/yellow_tripdata_2026-04.parquet",
)

RECENT_DATA_MONTH = os.getenv(
    "RECENT_TEST_MONTH",
    "April 2026",
)


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

def _prepare_features(
    model,
    test_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Align test data with the exact feature names expected by the
    loaded LightGBM model.
    """

    if "trip_duration" not in test_df.columns:
        raise ValueError(
            "Test dataset does not contain 'trip_duration'"
        )

    if not hasattr(model, "feature_name_"):
        raise ValueError(
            "Loaded model does not expose LightGBM feature_name_"
        )

    y = test_df["trip_duration"]

    X = test_df.drop(
        columns=["trip_duration", "row_key"],
        errors="ignore",
    ).copy()

    model_features = list(model.feature_name_)

    column_lookup = {
        column.lower(): column
        for column in X.columns
    }

    missing_features = []
    selected_columns = []

    for feature in model_features:
        source_column = column_lookup.get(feature.lower())

        if source_column is None:
            missing_features.append(feature)
        else:
            selected_columns.append(source_column)

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = X[selected_columns].copy()
    X.columns = model_features

    logger.info(
        "Feature alignment successful: %d features",
        len(X.columns),
    )

    return y, X


def predict_model(
    model,
    test_df: pd.DataFrame,
) -> tuple[pd.Series, object]:
    """
    Generate predictions once and return the target plus predictions.

    This is shared by all evaluation gates so Champion and Challenger
    are not repeatedly scored on the same dataset.
    """

    logger.info(
        "Generating predictions on %d rows",
        len(test_df),
    )

    y, X = _prepare_features(model, test_df)

    predictions = model.predict(X)

    logger.info(
        "Prediction complete: %d rows",
        len(predictions),
    )

    return y, predictions


def calculate_metrics(
    y,
    predictions,
) -> Metrics:
    """Calculate RMSE, MAE and R2 from existing predictions."""

    rmse = (
        mean_squared_error(y, predictions)
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


def evaluate_model(
    model,
    test_df: pd.DataFrame,
) -> Metrics:
    """
    Evaluate a model on a test dataset.

    Kept as a compatibility wrapper. The promotion pipeline itself
    generates predictions once and reuses them across all gates.
    """

    y, predictions = predict_model(
        model,
        test_df,
    )

    return calculate_metrics(
        y,
        predictions,
    )


def load_recent_test() -> pd.DataFrame:
    """
    Load and preprocess the latest labeled recent-test dataset.

    For the current Sentinel-AI milestone, this is the April 2026
    drift-month dataset. It is completed-trip data, so trip_duration
    labels are available.
    """

    logger.info(
        "Loading Recent Test source: %s",
        RECENT_DATA_PATH,
    )

    if not os.path.exists(RECENT_DATA_PATH):
        raise RuntimeError(
            f"Recent Test data not found: "
            f"{RECENT_DATA_PATH}"
        )

    try:
        recent_raw = pd.read_parquet(
            RECENT_DATA_PATH
        )

        recent_df = clean_and_engineer(
            recent_raw,
            RECENT_DATA_MONTH,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load/process Recent Test data "
            f"from '{RECENT_DATA_PATH}'"
        ) from exc

    if "trip_duration" not in recent_df.columns:
        raise ValueError(
            "Recent Test dataset does not contain "
            "'trip_duration' after preprocessing."
        )

    logger.info(
        "Recent Test rows after preprocessing: %d",
        len(recent_df),
    )

    return recent_df


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
) -> dict:

    logger.info("=" * 70)
    logger.info("SENTINEL-AI MODEL PROMOTION")
    logger.info("=" * 70)

    logger.info(
        "Challenger Run ID: %s",
        run_id,
    )

    # --------------------------------------------------------
    # 1. Load Frozen Test
    # --------------------------------------------------------

    frozen_df = load_frozen_test()

    logger.info(
        "Frozen test rows: %d",
        len(frozen_df),
    )

    # --------------------------------------------------------
    # 2. Load Champion
    # --------------------------------------------------------

    champion_model, champion_version = (
        load_champion()
    )

    # --------------------------------------------------------
    # 3. Load Challenger
    # --------------------------------------------------------

    challenger_model = load_challenger(
        run_id
    )

    # --------------------------------------------------------
    # 4. Generate Frozen predictions ONCE
    # --------------------------------------------------------

    logger.info(
        "Generating Champion frozen predictions"
    )

    y_frozen, champion_frozen_pred = predict_model(
        champion_model,
        frozen_df,
    )

    logger.info(
        "Generating Challenger frozen predictions"
    )

    _, challenger_frozen_pred = predict_model(
        challenger_model,
        frozen_df,
    )

    # --------------------------------------------------------
    # 5. Frozen metrics
    # --------------------------------------------------------

    champion_metrics = calculate_metrics(
        y_frozen,
        champion_frozen_pred,
    )

    challenger_metrics = calculate_metrics(
        y_frozen,
        challenger_frozen_pred,
    )

    # --------------------------------------------------------
    # 6. Model comparison
    # --------------------------------------------------------

    log_model_comparison(
        champion_version,
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # 7. Existing metric gates
    # --------------------------------------------------------

    metric_gates = promotion_gates(
        champion_metrics,
        challenger_metrics,
    )

    # --------------------------------------------------------
    # 8. Bootstrap Gate
    #
    # Default gate_on is RMSE only. MAE/R2 are still reported
    # by the Bootstrap implementation.
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("BOOTSTRAP GATE")
    logger.info("=" * 70)

    bootstrap_result = bootstrap_gate(
        y_true=y_frozen,
        champion_pred=champion_frozen_pred,
        challenger_pred=challenger_frozen_pred,
    )

    logger.info(
        "Bootstrap: %s",
        "PASS" if bootstrap_result["passed"] else "FAIL",
    )

    logger.info(
        "Bootstrap reason: %s",
        bootstrap_result["reason"],
    )

    # --------------------------------------------------------
    # 9. Segment Gate
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("SEGMENT GATE")
    logger.info("=" * 70)

    segment_result = evaluate_segments(
        test_df=frozen_df,
        y_true=y_frozen,
        champion_pred=champion_frozen_pred,
        challenger_pred=challenger_frozen_pred,
    )

    logger.info(
        "Segment: %s",
        "PASS" if segment_result["passed"] else "FAIL",
    )

    logger.info(
        "Segment reason: %s",
        segment_result["reason"],
    )

    logger.info(
        "Segment evaluated=%d skipped=%d failed=%d",
        segment_result.get("evaluated_segments", 0),
        segment_result.get("skipped_segments", 0),
        segment_result.get("failed_segments", 0),
    )

    # --------------------------------------------------------
    # 10. Recent Test Gate
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info(
        "RECENT TEST GATE | %s",
        RECENT_DATA_MONTH,
    )
    logger.info("=" * 70)

    recent_df = load_recent_test()

    y_recent, champion_recent_pred = predict_model(
        champion_model,
        recent_df,
    )

    _, challenger_recent_pred = predict_model(
        challenger_model,
        recent_df,
    )

    recent_result = evaluate_recent_test(
        recent_df=recent_df,
        y_recent=y_recent,
        challenger_recent_pred=challenger_recent_pred,
        challenger_frozen_metrics=challenger_metrics,
        champion_frozen_metrics=champion_metrics,
        champion_recent_pred=champion_recent_pred,
    )

    logger.info(
        "Recent Test: %s",
        "PASS" if recent_result["passed"] else "FAIL",
    )

    logger.info(
        "Recent Test reason: %s",
        recent_result["reason"],
    )

    logger.info(
        "Recent Test rows: %d",
        recent_result["rows"],
    )

    if recent_result.get("evaluated"):
        challenger_recent = recent_result["challenger"]
        champion_context = recent_result["champion_context"]

        logger.info(
            "Challenger Recent | RMSE=%.4f | MAE=%.4f",
            challenger_recent["recent"]["rmse"],
            challenger_recent["recent"]["mae"],
        )

        logger.info(
            "Challenger degradation | RMSE=%.3f%% | MAE=%.3f%%",
            challenger_recent["degradation"]["rmse"]["relative_pct"],
            challenger_recent["degradation"]["mae"]["relative_pct"],
        )

        logger.info(
            "Champion context degradation | RMSE=%.3f%% | MAE=%.3f%%",
            champion_context["degradation"]["rmse"]["relative_pct"],
            champion_context["degradation"]["mae"]["relative_pct"],
        )

    # --------------------------------------------------------
    # 11. Final combined decision
    # --------------------------------------------------------

    all_gates_pass = (
        metric_gates["all_pass"]
        and bootstrap_result["passed"]
        and segment_result["passed"]
        and recent_result["passed"]
    )

    logger.info("=" * 70)
    logger.info("FINAL PROMOTION GATE SUMMARY")
    logger.info("=" * 70)

    logger.info(
        "Frozen metric gates : %s",
        "PASS" if metric_gates["all_pass"] else "FAIL",
    )

    logger.info(
        "Bootstrap gate      : %s",
        "PASS" if bootstrap_result["passed"] else "FAIL",
    )

    logger.info(
        "Segment gate        : %s",
        "PASS" if segment_result["passed"] else "FAIL",
    )

    logger.info(
        "Recent Test gate    : %s",
        "PASS" if recent_result["passed"] else "FAIL",
    )

    logger.info(
        "FINAL DECISION      : %s",
        "PROMOTE" if all_gates_pass else "REJECT",
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # 12. Reject
    #
    # IMPORTANT:
    # No registration happens before every gate passes.
    # --------------------------------------------------------

    if not all_gates_pass:

        logger.warning(
            "CHALLENGER REJECTED"
        )

        logger.warning(
            "Champion remains version %s",
            champion_version,
        )

        return {
            "promoted": False,
            "mlflow_run_id": run_id,
            "new_model_rmse": challenger_metrics["rmse"],
            "new_model_mae": challenger_metrics["mae"],
            "new_model_r2": challenger_metrics["r2"],
            "champion_rmse": champion_metrics["rmse"],
            "champion_mae": champion_metrics["mae"],
            "champion_r2": champion_metrics["r2"],
            "champion_version": champion_version,
            "new_version": None,
        }

    # --------------------------------------------------------
    # 13. Register Challenger
    # --------------------------------------------------------

    logger.info(
        "All promotion gates passed."
    )

    new_version = register_challenger(
        run_id
    )

    # --------------------------------------------------------
    # 14. Promote
    # --------------------------------------------------------

    promote(
        new_version
    )

    logger.info("=" * 70)
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
    logger.info("=" * 70)

    return {
        "promoted": True,
        "mlflow_run_id": run_id,
        "new_model_rmse": challenger_metrics["rmse"],
        "new_model_mae": challenger_metrics["mae"],
        "new_model_r2": challenger_metrics["r2"],
        "champion_rmse": champion_metrics["rmse"],
        "champion_mae": champion_metrics["mae"],
        "champion_r2": champion_metrics["r2"],
        "champion_version": champion_version,
        "new_version": new_version,
    }


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

        result = main(
        challenger_run_id
        )

        exit_code = 0 if result["promoted"] else 1

    except Exception:

        logger.exception(
            "Model promotion pipeline failed"
        )

        exit_code = 1

    sys.exit(exit_code)