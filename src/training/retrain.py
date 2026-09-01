import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

from src.ingestion.preprocess import clean_and_engineer
from src.training.mlflow_logger import MLflowLogger


# ============================================================
# Configuration
# ============================================================

RAW_DIR = "data/raw"
MODEL_DIR = "model"

TARGET = "trip_duration"

FEATURES = [
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "VendorID",
    "RatecodeID",
]

# Best parameters obtained from previous Optuna search.
# We DO NOT run Optuna again during automatic retraining.
BEST_PARAMS = {
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
# 1. Load training data
# ============================================================

def load_training_data():

    print("\n" + "=" * 60)
    print("LOADING RETRAINING DATA")
    print("=" * 60)

    paths = [
        f"{RAW_DIR}/yellow_tripdata_2026-01.parquet",
        f"{RAW_DIR}/yellow_tripdata_2026-02.parquet",
        f"{RAW_DIR}/yellow_tripdata_2026-03.parquet",
    ]

    dfs = []

    for path in paths:

        print(f"Loading: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training data not found: {path}"
            )

        df = pd.read_parquet(path)

        print(f"Rows: {len(df):,}")

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal raw rows: {len(df):,}")

    return df


# ============================================================
# 2. Preprocess
# ============================================================

def preprocess_data(df):

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # IMPORTANT:
    # Reuse the exact preprocessing pipeline used by
    # the original model.

    df = clean_and_engineer(
        df,
        month_name="Retraining"
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features after preprocessing: "
            f"{missing_features}"
        )

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nFeatures:")
    for feature in FEATURES:
        print(f"  - {feature}")

    return X, y


# ============================================================
# 3. Train model
# ============================================================

def train_model(X_train, y_train):

    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)

    print("\nUsing parameters:")
    for key, value in BEST_PARAMS.items():
        print(f"  {key}: {value}")

    model = LGBMRegressor(**BEST_PARAMS)

    model.fit(
        X_train,
        y_train
    )

    print("\n✅ Training complete")

    return model


# ============================================================
# 4. Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

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

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return metrics


# ============================================================
# 5. Save model artifacts
# ============================================================

def save_artifacts(model, metrics):

    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model
    model_path = os.path.join(
        MODEL_DIR,
        "model_retrained.pkl"
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved: {model_path}")

    # Parameters
    params_path = os.path.join(
        MODEL_DIR,
        "best_params_retrained.json"
    )

    with open(params_path, "w") as f:
        json.dump(
            BEST_PARAMS,
            f,
            indent=4
        )

    print(f"✅ Parameters saved: {params_path}")

    # Metrics
    metrics_path = os.path.join(
        MODEL_DIR,
        "metrics_retrained.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(f"✅ Metrics saved: {metrics_path}")

    # Features
    features_path = os.path.join(
        MODEL_DIR,
        "features_retrained.json"
    )

    with open(features_path, "w") as f:
        json.dump(
            FEATURES,
            f,
            indent=4
        )

    print(f"✅ Features saved: {features_path}")

    return {
        "model": model_path,
        "params": params_path,
        "metrics": metrics_path,
        "features": features_path,
    }


# ============================================================
# 6. Main retraining pipeline
# ============================================================

def retrain():

    print("\n")
    print("=" * 70)
    print("              SENTINEL AI - RETRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    X, y = preprocess_data(df)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    artifacts = save_artifacts(
        model,
        metrics
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOGGING TO MLFLOW")
    print("=" * 60)

    logger = MLflowLogger(
        experiment_name="Sentinel-AI"
    )

    with logger.start_run(
        run_name="automatic-retraining"
    ):

        # Parameters
        logger.log_params(
            BEST_PARAMS
        )

        # Metrics
        logger.log_metrics(
            metrics
        )

        # Artifacts
        logger.log_artifact(
            artifacts["params"]
        )

        logger.log_artifact(
            artifacts["metrics"]
        )

        logger.log_artifact(
            artifacts["features"]
        )

        # Model
        logger.log_model(
            model,
            artifact_path="model"
        )

    print("\n✅ MLflow logging complete")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("              RETRAINING COMPLETE")
    print("=" * 70)

    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    result = retrain()

    print("\nFinal metrics:")
    print(
        json.dumps(
            result["metrics"],
            indent=4
        )
    )