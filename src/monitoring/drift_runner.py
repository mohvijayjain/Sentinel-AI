import os
import pickle
import pandas as pd

from src.monitoring.stastical_drift import run_statistical_drift
from src.monitoring.shap_drift import run_shap_drift
from src.monitoring.prediction_drift import (
    generate_predictions,
    run_prediction_drift,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model/model.pkl"

REFERENCE_PATH = "data/reference/reference_data.parquet"

CURRENT_PATHS = [
    "data/raw/yellow_tripdata_2026-06.parquet",
    "data/raw/yellow_tripdata_2026-07.parquet",
]

REPORTS_DIR = "reports"


# ============================================================
# Load Model
# ============================================================

def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded from {model_path}")
    return model


# ============================================================
# Load Reference Data
# ============================================================

def load_reference(reference_path: str = REFERENCE_PATH):
    reference = pd.read_parquet(reference_path)

    print(f"📚 Reference data shape: {reference.shape}")

    return reference


# ============================================================
# Load Current Monitoring Window
# ============================================================

def load_current_data(current_paths=None):
    if current_paths is None:
        current_paths = CURRENT_PATHS

    frames = []

    for path in current_paths:
        print(f"📥 Loading current data: {path}")

        df = pd.read_parquet(path)

        print(f"   Shape: {df.shape}")

        frames.append(df)

    current = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"📊 Combined current data shape: "
        f"{current.shape}"
    )

    return current


# ============================================================
# Save Reports
# ============================================================

def save_reports(
    statistical_results,
    shap_results,
    prediction_results
):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    statistical_results.to_csv(
        os.path.join(
            REPORTS_DIR,
            "statistical_drift.csv"
        ),
        index=False
    )

    shap_results.to_csv(
        os.path.join(
            REPORTS_DIR,
            "shap_drift.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [prediction_results]
    ).to_csv(
        os.path.join(
            REPORTS_DIR,
            "prediction_drift.csv"
        ),
        index=False
    )

    print("\n✅ Drift reports generated:")
    print("   → reports/statistical_drift.csv")
    print("   → reports/shap_drift.csv")
    print("   → reports/prediction_drift.csv")


# ============================================================
# Run Complete Drift Detection
# ============================================================

def run_drift_detection():
    print("=" * 70)
    print(" Sentinel-AI — Drift Detection Runner")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data and model
    # --------------------------------------------------------

    reference = load_reference()
    current = load_current_data()
    model = load_model()

    # --------------------------------------------------------
    # Statistical Drift
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("📊 Running Statistical Drift")
    print("=" * 70)

    statistical_results, _, statistical_drifted = (
        run_statistical_drift(
            reference,
            current
        )
    )

    print(
        f"\nStatistical drifted features: "
        f"{statistical_drifted or 'None'}"
    )

    # --------------------------------------------------------
    # SHAP Drift
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("🧠 Running SHAP Drift")
    print("=" * 70)

    shap_results, _, shap_drifted = (
        run_shap_drift(
            reference,
            current,
            model
        )
    )

    print(
        f"\nSHAP drifted features: "
        f"{shap_drifted or 'None'}"
    )

    # --------------------------------------------------------
    # Prediction Drift
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("🔮 Running Prediction Drift")
    print("=" * 70)

    print("\nGenerating reference predictions...")

    reference_predictions = generate_predictions(
        model,
        reference
    )

    print("Generating current predictions...")

    current_predictions = generate_predictions(
        model,
        current
    )

    prediction_results = run_prediction_drift(
        reference_predictions,
        current_predictions
    )

    print("\nPrediction drift results:")

    for key, value in prediction_results.items():
        print(f"   {key}: {value}")

    # --------------------------------------------------------
    # Save all reports
    # --------------------------------------------------------

    save_reports(
        statistical_results,
        shap_results,
        prediction_results
    )

    print("\n" + "=" * 70)
    print("✅ Drift Detection Runner Complete")
    print("=" * 70)

    return {
        "statistical": statistical_results,
        "shap": shap_results,
        "prediction": prediction_results,
    }


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_drift_detection()