# src/training/retrain.py

import logging

from dotenv import load_dotenv

load_dotenv()

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.ingestion.preprocess import FEATURES

from src.training.dataset import (
    ROW_KEY,
    SEED,
    load_all_row_keys,
    load_frozen_test,
    load_rows_by_key,
)

from src.training.mlflow_logger import MLflowLogger
from src.training.mlflow_names import MLFLOW_EXPERIMENT_NAME


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

TARGET = "trip_duration"

EXPERIMENT_NAME = MLFLOW_EXPERIMENT_NAME

SAMPLE_SIZE = 1_000_000


MODEL_PARAMS = {
    "n_estimators": 957,
    "learning_rate": 0.29707317286046714,
    "max_depth": 7,
    "num_leaves": 98,
    "min_child_samples": 95,
    "subsample": 0.9721735959159156,
    "colsample_bytree": 0.8636645677703403,
    "reg_alpha": 0.01103925459764792,
    "reg_lambda": 0.06296116597075296,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ============================================================
# Load Training Data
# ============================================================

def load_training_data() -> pd.DataFrame:
    """
    Load a bounded training sample from processed Parquet.

    Frozen-test rows are excluded before sampling.
    """

    logger.info(
        "Loading %s-row training sample...",
        f"{SAMPLE_SIZE:,}",
    )

    all_keys = load_all_row_keys()

    test_keys = load_frozen_test()[
        ROW_KEY
    ].to_numpy(
        dtype=np.int64
    )

    logger.info(
        "Corpus rows: %s",
        f"{len(all_keys):,}",
    )

    logger.info(
        "Frozen test rows: %s",
        f"{len(test_keys):,}",
    )

    candidates = all_keys[
        ~np.isin(
            all_keys,
            test_keys,
            assume_unique=True,
        )
    ]

    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(candidates):,} rows available "
            f"outside frozen test set; "
            f"need {SAMPLE_SIZE:,}."
        )

    rng = np.random.default_rng(
        SEED
    )

    train_keys = np.sort(
        rng.choice(
            candidates,
            size=SAMPLE_SIZE,
            replace=False,
        )
    )

    df = load_rows_by_key(
        train_keys
    )

    logger.info(
        "Loaded %s rows",
        f"{len(df):,}",
    )

    required_columns = (
        FEATURES + [TARGET]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df[
        required_columns + [ROW_KEY]
    ]


# ============================================================
# Leakage Check
# ============================================================

def assert_disjoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    train_keys = train_df[
        ROW_KEY
    ].to_numpy(
        dtype=np.int64
    )

    test_keys = test_df[
        ROW_KEY
    ].to_numpy(
        dtype=np.int64
    )

    overlap = np.intersect1d(
        train_keys,
        test_keys,
    )

    logger.info(
        "Training rows: %s",
        f"{len(train_keys):,}",
    )

    logger.info(
        "Frozen test rows: %s",
        f"{len(test_keys):,}",
    )

    logger.info(
        "Overlap: %s",
        f"{len(overlap):,}",
    )

    if len(overlap):

        raise ValueError(
            f"LEAKAGE: {len(overlap):,} rows "
            "appear in both training and frozen test."
        )


# ============================================================
# Train
# ============================================================

def train_model(
    X_train,
    y_train,
) -> lgb.LGBMRegressor:

    logger.info(
        "Training Challenger model..."
    )

    model = lgb.LGBMRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Challenger training completed."
    )

    return model


# ============================================================
# Evaluate
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
) -> dict:

    predictions = model.predict(
        X_test
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ============================================================
# Log Challenger
# ============================================================

def log_challenger(
    model,
    metrics: dict,
):

    ml_logger = MLflowLogger(
        experiment_name=EXPERIMENT_NAME
    )

    run_id = None

    try:

        ml_logger.start_run(
            run_name="Challenger"
        )

        run_id = (
            mlflow.active_run()
            .info.run_id
        )

        logger.info(
            "MLflow run started: %s",
            run_id,
        )

        # --------------------------------------------
        # Params
        # --------------------------------------------

        ml_logger.log_params(
            MODEL_PARAMS
        )

        # --------------------------------------------
        # Metrics
        # --------------------------------------------

        ml_logger.log_metrics(
            metrics
        )

        # --------------------------------------------
        # Model
        # --------------------------------------------

        model_uri = (
            ml_logger.log_model(
                model
            )
        )

        logger.info(
            "Challenger model URI: %s",
            model_uri,
        )

        # --------------------------------------------
        # Verify artifact exists
        # --------------------------------------------

        artifacts = mlflow.MlflowClient().list_artifacts(
            run_id,
            "model",
        )

        if not artifacts:

            raise RuntimeError(
                "MLflow model artifact verification failed. "
                "The 'model' artifact directory is empty."
            )

        logger.info(
            "MLflow model artifact verified."
        )

        for artifact in artifacts:

            logger.info(
                "Artifact: %s",
                artifact.path,
            )

    finally:

        ml_logger.end_run()

    logger.info(
        "Challenger logged successfully | run_id=%s",
        run_id,
    )

    return run_id


# ============================================================
# Main
# ============================================================

def main() -> dict:

    # --------------------------------------------------------
    # 1. Training data
    # --------------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------------
    # 2. Frozen test
    # --------------------------------------------------------

    test_df = load_frozen_test()

    assert_disjoint(
        train_df,
        test_df,
    )

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # 3. Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 4. Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    logger.info(
        "Challenger metrics | "
        "RMSE=%.4f | MAE=%.4f | R2=%.4f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    # --------------------------------------------------------
    # 5. MLflow
    # --------------------------------------------------------

    run_id = log_challenger(
        model,
        metrics,
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model": model,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    main()