import logging

import lightgbm as lgb
import mlflow
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.database.postgres import engine
from src.ingestion.preprocess import FEATURES
from src.training.mlflow_logger import MLflowLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET = "trip_duration"
KEY_COLUMN = "trip_id"        # <-- set to your table's stable primary key
TEST_BUCKET_CUTOFF = 80       # buckets 0-79 train, 80-99 test (~20% holdout)

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


def load_training_data() -> pd.DataFrame:
    """Load features + target from PostgreSQL with a deterministic train/test bucket.

    The bucket is derived from a hash of the primary key, so a given row is
    ALWAYS in the same split regardless of row order or table growth. The
    comparison file must use the identical predicate to score the champion.
    """
    logger.info("Loading training data from PostgreSQL...")

    columns = ", ".join(FEATURES + [TARGET])
    query = f"""
        SELECT {columns},
            abs(hashtext({KEY_COLUMN}::text)) % 100 AS _bucket
        FROM taxi_trips
    """

    df = pd.read_sql(text(query), engine)
    logger.info("Loaded %s rows", f"{len(df):,}")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from query: {missing}")

    return df


def split_data(df: pd.DataFrame):
    """Deterministic split by frozen hash bucket (no random shuffle)."""
    train_df = df[df["_bucket"] < TEST_BUCKET_CUTOFF]
    test_df = df[df["_bucket"] >= TEST_BUCKET_CUTOFF]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    logger.info("Training rows: %s", f"{len(X_train):,}")
    logger.info("Test rows:     %s", f"{len(X_test):,}")

    if len(X_test) == 0 or len(X_train) == 0:
        raise ValueError("Empty train or test split — check KEY_COLUMN and data volume.")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train) -> lgb.LGBMRegressor:
    logger.info("Training challenger model...")
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def main() -> dict:
    df = load_training_data()
    X_train, y_train, X_test, y_test = split_data(df)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    logger.info("Challenger metrics | RMSE: %.4f | MAE: %.4f | R2: %.4f",
                metrics["rmse"], metrics["mae"], metrics["r2"])

    ml_logger = MLflowLogger(experiment_name="Sentinel-AI")
    run_id = None
    try:
        ml_logger.start_run(run_name="Challenger")
        run_id = mlflow.active_run().info.run_id
        ml_logger.log_params(MODEL_PARAMS)
        ml_logger.log_metrics(metrics)
        ml_logger.log_model(model)
    finally:
        ml_logger.end_run()

    logger.info("Challenger logged to MLflow | run_id=%s", run_id)

    # Returned so the comparison file can locate THIS challenger by run_id.
    return {"run_id": run_id, "metrics": metrics, "model": model}


if __name__ == "__main__":
    main()