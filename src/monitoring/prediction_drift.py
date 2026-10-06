import numpy as np
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from src.monitoring.stastical_drift import get_psi_severity


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model/model.pkl"

REFERENCE_PATH = "data/reference/reference_data.parquet"
CURRENT_PATH = "data/processed/yellow_tripdata_2026-04_clean.parquet"


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


# ============================================================
# Load Model
# ============================================================

def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f" Model loaded from {MODEL_PATH}")

    return model


# ============================================================
# Load Data
# ============================================================

def load_data():

    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_parquet(CURRENT_PATH)

    print(f" Reference shape: {reference.shape}")
    print(f" Current shape: {current.shape}")

    return reference, current



# ============================================================
# Prediction Generation
# ============================================================

def generate_predictions(
    model,
    data
):

    X = data[FEATURES]

    predictions = model.predict(X)

    return predictions



# ============================================================
# Shared Binning (PSI and JS)
# ============================================================

def _binned_counts(
    reference,
    current,
    bins=10
):
    """
    Histogram reference and current on the SAME edges: equal-width over
    the reference range, with the outer edges opened to -inf / +inf so
    current values outside the reference range land in the edge bins
    instead of being dropped by np.histogram (same convention as
    stastical_drift.calculate_psi). NaN values are ignored.

    Returns (reference_counts, current_counts), or None when either side
    has no usable values.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return None

    bin_edges = np.histogram_bin_edges(
        reference,
        bins=bins
    )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=bin_edges
    )

    current_counts, _ = np.histogram(
        current,
        bins=bin_edges
    )

    return reference_counts, current_counts



# ============================================================
# PSI Calculation
# ============================================================

def calculate_psi(
    reference,
    current,
    bins=10
):

    counts = _binned_counts(
        reference,
        current,
        bins
    )


    # Nothing to compare: NaN, which the scorer refuses to score as 0
    if counts is None:
        return np.nan


    reference_counts, current_counts = counts


    epsilon = 1e-10


    reference_pct = (
        reference_counts + epsilon
    ) / (
        reference_counts.sum()
    )


    current_pct = (
        current_counts + epsilon
    ) / (
        current_counts.sum()
    )


    psi = np.sum(
        (current_pct-reference_pct)
        *
        np.log(current_pct/reference_pct)
    )


    return float(psi)



# ============================================================
# JS Divergence
# ============================================================

def calculate_js(
    reference,
    current,
    bins=10
):

    # Same bins as PSI, so out-of-range values count here too
    counts = _binned_counts(
        reference,
        current,
        bins
    )


    # Nothing to compare: NaN, as for PSI
    if counts is None:
        return np.nan


    reference_counts, current_counts = counts


    epsilon = 1e-10


    ref_prob = (
        reference_counts + epsilon
    ) / (
        reference_counts.sum()
    )


    cur_prob = (
        current_counts + epsilon
    ) / (
        current_counts.sum()
    )


    js = jensenshannon(
        ref_prob,
        cur_prob
    ) ** 2


    return float(js)



# ============================================================
# Run Prediction Drift
# ============================================================

def run_prediction_drift(
    reference_predictions,
    current_predictions
):


    psi = calculate_psi(
        reference_predictions,
        current_predictions
    )


    ks_stat, ks_pvalue = ks_2samp(
        reference_predictions,
        current_predictions
    )


    js = calculate_js(
        reference_predictions,
        current_predictions
    )


    wasserstein = wasserstein_distance(
        reference_predictions,
        current_predictions
    )


    # Graded PSI severity, same bands as the statistical detector,
    # so the scorer can map it (a binary "DRIFT" label scored 0)
    severity = get_psi_severity(psi)


    results = {
        "psi": round(psi,6),
        "ks_statistic": round(ks_stat,6),
        "ks_pvalue": ks_pvalue,
        "js_divergence": round(js,6),
        "wasserstein": round(wasserstein,6),
        "severity": severity
    }


    return results



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    print("="*60)
    print(" Sentinel AI — Prediction Drift Detection")
    print("="*60)


    model = load_model()


    reference, current = load_data()



    print("\n Generating reference predictions...")
    reference_predictions = generate_predictions(
        model,
        reference
    )


    print(" Generating current predictions...")
    current_predictions = generate_predictions(
        model,
        current
    )



    print("\n Running prediction drift analysis...")


    results = run_prediction_drift(
        reference_predictions,
        current_predictions
    )
    print("\nPrediction Statistics")

    print(
        "Reference mean:",
        np.mean(reference_predictions)
    )

    print(
        "Current mean:",
        np.mean(current_predictions)
    )

    print(
        "Reference std:",
        np.std(reference_predictions)
    )

    print(
        "Current std:",
        np.std(current_predictions)
    )


    print("\n" + "="*60)
    print(" Prediction Drift Results ")
    print("="*60)


    for key,value in results.items():
        print(
            f"{key}: {value}"
        )



    os.makedirs(
        "reports",
        exist_ok=True
    )


    pd.DataFrame(
        [results]
    ).to_csv(
        "reports/prediction_drift.csv",
        index=False
    )


    print(
        "\n Results saved → reports/prediction_drift.csv"
    )