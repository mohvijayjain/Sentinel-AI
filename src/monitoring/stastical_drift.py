# src/drift/statistical.py

import numpy as np
import pandas as pd
import os

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# ── Configuration ──────────────────────────────────────
REFERENCE_PATH = "data/reference/reference_data.parquet"
CURRENT_PATH   = "data/processed/yellow_tripdata_2026-04_clean.parquet"
TARGET_COLUMN  = "trip_duration"

FEATURES = [
    "trip_distance", "pickup_hour", "pickup_day_of_week",
    "pickup_month", "is_weekend", "is_rush_hour",
    "PULocationID", "DOLocationID", "payment_type",
    "VendorID", "RatecodeID",
]

CATEGORICAL_FEATURES = [
    "PULocationID", "DOLocationID", "payment_type",
    "VendorID", "RatecodeID",
]

IGNORED_FEATURES = ["pickup_month"]

# ── PSI Thresholds ─────────────────────────────────────
PSI_NO_DRIFT = 0.10
PSI_LOW      = 0.20
PSI_MEDIUM   = 0.25
PSI_HIGH     = 0.50

# ── Load Data ──────────────────────────────────────────
def load_data():
    reference = pd.read_parquet(REFERENCE_PATH)
    current   = pd.read_parquet(CURRENT_PATH)
    print(f"  Reference shape: {reference.shape}")
    print(f"  Current shape:   {current.shape}")
    return reference, current

# ── PSI Numerical ──────────────────────────────────────
def calculate_psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna().astype(float)
    current   = pd.Series(current).dropna().astype(float)

    if len(reference) == 0 or len(current) == 0:
        return np.nan
    if reference.nunique() <= 1:
        return 0.0 if current.nunique() <= 1 else np.inf

    quantiles  = np.linspace(0, 1, bins + 1)
    bin_edges  = np.quantile(reference, quantiles)
    bin_edges  = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0]  = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    epsilon   = 1e-10
    ref_pct   = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_pct   = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# ── PSI Categorical ────────────────────────────────────
def calculate_psi_categorical(reference, current):
    reference = pd.Series(reference).dropna().astype(str)
    current   = pd.Series(current).dropna().astype(str)

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    categories    = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts    = reference.value_counts().reindex(categories, fill_value=0).values
    cur_counts    = current.value_counts().reindex(categories, fill_value=0).values

    epsilon   = 1e-10
    ref_pct   = np.clip(ref_counts / len(reference), epsilon, None)
    cur_pct   = np.clip(cur_counts / len(current),   epsilon, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# ── JS Numerical ───────────────────────────────────────
def calculate_js_numeric(reference, current, bins=10):
    reference = pd.Series(reference).dropna().astype(float)
    current   = pd.Series(current).dropna().astype(float)

    if len(reference) == 0 or len(current) == 0:
        return np.nan
    if reference.nunique() <= 1:
        return 0.0

    quantiles  = np.linspace(0, 1, bins + 1)
    bin_edges  = np.quantile(reference, quantiles)
    bin_edges  = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0]  = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    epsilon   = 1e-10
    ref_prob  = np.clip(ref_counts / len(reference), epsilon, None)
    cur_prob  = np.clip(cur_counts / len(current),   epsilon, None)
    ref_prob /= ref_prob.sum()
    cur_prob /= cur_prob.sum()

    return float(jensenshannon(ref_prob, cur_prob) ** 2)

# ── JS Categorical ─────────────────────────────────────
def calculate_js_categorical(reference, current):
    reference = pd.Series(reference).dropna().astype(str)
    current   = pd.Series(current).dropna().astype(str)

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    categories    = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts    = reference.value_counts().reindex(categories, fill_value=0).values
    cur_counts    = current.value_counts().reindex(categories, fill_value=0).values

    epsilon   = 1e-10
    ref_prob  = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_prob  = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    ref_prob /= ref_prob.sum()
    cur_prob /= cur_prob.sum()

    return float(jensenshannon(ref_prob, cur_prob) ** 2)

# ── KS Test ────────────────────────────────────────────
def calculate_ks(reference, current):
    reference = pd.Series(reference).dropna().astype(float)
    current   = pd.Series(current).dropna().astype(float)

    if len(reference) == 0 or len(current) == 0:
        return np.nan, np.nan
    if reference.nunique() <= 1 and current.nunique() <= 1:
        return 0.0, 1.0

    stat, pvalue = ks_2samp(reference, current)
    return float(stat), float(pvalue)

# ── Wasserstein (Normalized) ───────────────────────────
def calculate_wasserstein(reference, current):
    reference = pd.Series(reference).dropna().astype(float)
    current   = pd.Series(current).dropna().astype(float)

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    distance = wasserstein_distance(reference, current)

    # Normalize by reference std
    ref_std = reference.std()
    if ref_std > 0:
        distance = distance / ref_std

    return float(distance)

# ── Severity ───────────────────────────────────────────
def get_psi_severity(psi: float) -> str:
    if pd.isna(psi):    return "UNKNOWN"
    if psi < PSI_NO_DRIFT: return "NO_DRIFT"
    if psi < PSI_LOW:   return "LOW"
    if psi < PSI_MEDIUM: return "MEDIUM"
    if psi < PSI_HIGH:  return "HIGH"
    return "CRITICAL"

# ── Analyze Single Feature ─────────────────────────────
def analyze_feature(reference, current, feature) -> dict:
    is_categorical = feature in CATEGORICAL_FEATURES
    ref_f = reference[feature]
    cur_f = current[feature]

    if is_categorical:
        psi = calculate_psi_categorical(ref_f, cur_f)
        js  = calculate_js_categorical(ref_f, cur_f)
        return {
            "feature":      feature,
            "type":         "categorical",
            "psi":          round(psi, 6) if not np.isnan(psi) else np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue":    np.nan,
            "js_divergence": round(js, 6) if not np.isnan(js) else np.nan,
            "wasserstein":  np.nan,
            "severity":     get_psi_severity(psi),
        }

    psi         = calculate_psi(ref_f, cur_f)
    ks_stat, ks_p = calculate_ks(ref_f, cur_f)
    js          = calculate_js_numeric(ref_f, cur_f)
    wasserstein = calculate_wasserstein(ref_f, cur_f)

    return {
        "feature":       feature,
        "type":          "numerical",
        "psi":           round(psi, 6) if not np.isnan(psi) else np.nan,
        "ks_statistic":  round(ks_stat, 6) if not np.isnan(ks_stat) else np.nan,
        "ks_pvalue":     round(ks_p, 6) if not np.isnan(ks_p) else np.nan,
        "js_divergence": round(js, 6) if not np.isnan(js) else np.nan,
        "wasserstein":   round(wasserstein, 6) if not np.isnan(wasserstein) else np.nan,
        "severity":      get_psi_severity(psi),
    }

# ── Run Full Statistical Drift ─────────────────────────
def run_statistical_drift(reference, current):
    results = []

    for feature in FEATURES:
        if feature in IGNORED_FEATURES:
            print(f"  ⏭️  Skipping: {feature}")
            continue
        if feature not in reference.columns:
            print(f"  ⚠️  Not in reference: {feature}")
            continue
        if feature not in current.columns:
            print(f"  ⚠️  Not in current: {feature}")
            continue

        print(f"  🔍 Analyzing: {feature}")
        result = analyze_feature(reference, current, feature)
        results.append(result)

    results_df = pd.DataFrame(results)

    # ── Retrain Decision ───────────────────────────────
    drifted = results_df[
        results_df["severity"].isin(["HIGH", "CRITICAL"])
    ]["feature"].tolist()

    should_retrain = len(drifted) > 0

    return results_df, should_retrain, drifted


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🛡️  Sentinel AI — Statistical Drift Detection")
    print("=" * 60)

    reference_data, current_data = load_data()

    print("\n📊 Running drift analysis...")
    results, should_retrain, drifted_features = run_statistical_drift(
        reference_data, current_data
    )

    # ── Print Results Table ────────────────────────────
    print("\n" + "=" * 60)
    print("  DRIFT RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    # ── Print Summary ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  DRIFT SUMMARY")
    print("=" * 60)

    emoji_map = {
        "CRITICAL": "🚨",
        "HIGH":     "🔴",
        "MEDIUM":   "🟠",
        "LOW":      "🟡",
        "NO_DRIFT": "✅",
        "UNKNOWN":  "❓"
    }

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NO_DRIFT"]:
        features = results[
            results["severity"] == severity
        ]["feature"].tolist()
        if features:
            print(f"  {emoji_map[severity]} {severity}: {', '.join(features)}")

    print(f"\n  Should Retrain: {'YES 🔴' if should_retrain else 'NO ✅'}")
    print("=" * 60)

    # ── Save Results ───────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/statistical_drift.csv"
    results.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved → {output_path}")