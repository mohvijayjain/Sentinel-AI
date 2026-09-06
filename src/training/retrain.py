import logging

# Loaded before the MLflow import: MLflowLogger reads MLFLOW_TRACKING_URI at
# import time. This used to happen as a side effect of importing
# src.database.postgres, which no longer belongs in the training path.
from dotenv import load_dotenv

load_dotenv()

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ingestion.preprocess import FEATURES
from src.training.dataset import (
    ROW_KEY,
    SEED,
    load_all_row_keys,
    load_frozen_test,
    load_rows_by_key,
)
from src.training.mlflow_logger import MLflowLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

TARGET = "trip_duration"

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


SAMPLE_SIZE = 1_000_000


def load_training_data() -> pd.DataFrame:
    """
    Load a bounded training sample from processed Parquet.

    Rows belonging to data/frozen_test.parquet are excluded BEFORE sampling,
    so the training sample and the evaluation set are disjoint by
    construction. main() then proves it.

    PostgreSQL is no longer queried here; it remains responsible for
    monitoring runs, drift scores and model decision metadata.
    """

    logger.info(
        "Loading %s-row training sample from processed Parquet...",
        f"{SAMPLE_SIZE:,}"
    )

    all_keys = load_all_row_keys()
    test_keys = load_frozen_test()[ROW_KEY].to_numpy(dtype=np.int64)

    logger.info("Corpus rows:      %s", f"{len(all_keys):,}")
    logger.info("Frozen test rows: %s", f"{len(test_keys):,}")

    candidates = all_keys[~np.isin(all_keys, test_keys, assume_unique=True)]

    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(candidates):,} rows available outside the frozen "
            f"test set, need {SAMPLE_SIZE:,}."
        )

    rng = np.random.default_rng(SEED)

    train_keys = np.sort(
        rng.choice(candidates, size=SAMPLE_SIZE, replace=False)
    )

    df = load_rows_by_key(train_keys)

    logger.info(
        "Loaded %s rows",
        f"{len(df):,}"
    )

    required_columns = FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns from processed Parquet: {missing}"
        )

    logger.info("Training columns:")
    logger.info("%s", required_columns)

    return df[required_columns + [ROW_KEY]]


def assert_disjoint(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Prove the training sample shares no row with the frozen test set.

    Reproducibility is not sufficient - a reproducible split can still leak.
    This fails loudly, before any model is trained or logged to MLflow.
    """

    train_keys = train_df[ROW_KEY].to_numpy(dtype=np.int64)
    test_keys = test_df[ROW_KEY].to_numpy(dtype=np.int64)

    overlap = np.intersect1d(train_keys, test_keys)

    logger.info("Training rows:    %s", f"{len(train_keys):,}")
    logger.info("Frozen test rows: %s", f"{len(test_keys):,}")
    logger.info("Overlap:          %s", f"{len(overlap):,}")

    if len(overlap):
        raise ValueError(
            f"LEAKAGE: {len(overlap):,} rows appear in BOTH the training "
            "sample and data/frozen_test.parquet. Refusing to train or log. "
            f"First offending row_keys: {overlap[:10].tolist()}"
        )


def train_model(X_train, y_train) -> lgb.LGBMRegressor:
    """Train a fresh Challenger model."""

    logger.info("Training Challenger model...")

    model = lgb.LGBMRegressor(**MODEL_PARAMS)

    model.fit(
        X_train,
        y_train
    )

    logger.info("Challenger training completed.")

    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate Challenger using RMSE, MAE and R²."""

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def log_challenger(model, metrics: dict):
    """Log Challenger model and metrics to MLflow."""

    ml_logger = MLflowLogger(
        experiment_name="Sentinel-AI"
    )

    run_id = None

    try:
        ml_logger.start_run(
            run_name="Challenger"
        )

        run_id = mlflow.active_run().info.run_id

        ml_logger.log_params(
            MODEL_PARAMS
        )

        ml_logger.log_metrics(
            metrics
        )

        ml_logger.log_model(
            model
        )

    finally:
        ml_logger.end_run()

    logger.info(
        "Challenger logged to MLflow | run_id=%s",
        run_id
    )

    return run_id


def main() -> dict:

    # --------------------------------------------------
    # 1. Load processed data
    # --------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------
    # 2. Load THE frozen evaluation set + prove disjoint
    # --------------------------------------------------

    test_df = load_frozen_test()

    assert_disjoint(train_df, test_df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # --------------------------------------------------
    # 3. Train fresh Challenger
    # --------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 4. Evaluate Challenger
    # --------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    logger.info(
        "Challenger metrics | "
        "RMSE: %.4f | MAE: %.4f | R2: %.4f",
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    # --------------------------------------------------
    # 5. Log Challenger to MLflow
    # --------------------------------------------------

    run_id = log_challenger(
        model,
        metrics
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model": model,
    }


if __name__ == "__main__":
    main()