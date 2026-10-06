import numpy as np 
import pandas as pd 
import shap 
import pickle 
import warnings 
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple

# Configuration

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

from src.monitoring.constants import IGNORED_FEATURES

#Thresholds

RELATIVE_SHIFT_THRESHOLD = 0.25
RANK_SHIFT_THRESHOLD = 2

# Relative shift assigned when reference importance is 0 but current is
# not (e.g. a feature the reference model never split on). 100% lands in
# CRITICAL_SHIFT and exceeds RELATIVE_SHIFT_THRESHOLD, so it is drifted.
ZERO_REFERENCE_SHIFT = 1.0

def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f" Model loaded from {model_path}")
    return model

def load_data(
    reference_path : str = REFERENCE_PATH,
    current_path: str = CURRENT_PATH
):
    reference = pd.read_parquet(reference_path)
    current  = pd.read_parquet(current_path)
    print(f" Reference shape: {reference.shape}")
    print(f" Current shape: {current.shape}")
    return reference, current
    
def compute_shap_importance(
    model,
    data: pd.DataFrame,
    features: List[str],
    sample_size: int = 5000,
    label: str = ""
)-> Dict[str, float]:
    """
    Compute mean absolute SHAP value per feature = feature importance from model's prespective
    """
    
    #sample for efficiency
    sample = data[features].sample(
        min(sample_size, len(data)),
        random_state=42
    )
    print(f" Computing SHAP for {label} ({len(sample)} samples)...")
    
    #TreeExplainer is fastest fo LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    #Mean absolute SHAP = importance
    importance = np.abs(shap_values).mean(axis=0)
    
    return {
        feature: round(float(imp), 6)
        for feature, imp in zip(features, importance)
    }

def get_rankings(importance: Dict[str, float]) -> Dict[str, int]:
    """"
    Rank features by importance 
    Rank 1 is most importance
    """
    sorted_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return{
        feature: rank+1
        for rank, (feature, _) in enumerate(sorted_features)
    }
    
#Analyze Shap Drift per feature
def analyze_shap_drift(
    ref_importance: Dict[str, float],
    cur_importance: Dict[str, float],
    ref_rankings: Dict[str, int],
    cur_rankings: Dict[str, int],
) -> List[Dict]:
    """
    Compare SHAP importance before vs after Flag 
    features with significant shifts
    """
    results = []
    
    for feature in ref_importance:
        ref_imp = ref_importance[feature]
        cur_imp = cur_importance.get(feature, 0)
        
        # computing shifts
        absolute_shift = abs(cur_imp - ref_imp)
        if ref_imp != 0:
            # True ratio, no epsilon, so exact boundaries stay exact
            relative_shift = absolute_shift / abs(ref_imp)
        elif cur_imp == 0:
            # Unused in both windows: nothing shifted
            relative_shift = 0.0
        else:
            # Unused in reference, used now: treat as maximal shift
            relative_shift = ZERO_REFERENCE_SHIFT
        
        # Rank shift
        ref_rank = ref_rankings.get(feature, 0)
        cur_rank = cur_rankings.get(feature, 0)
        rank_shift = abs(cur_rank - ref_rank)
        
        direction = "INCREASED" if cur_imp > ref_imp else "DECREASED"
        
        is_drifted = (
            relative_shift > RELATIVE_SHIFT_THRESHOLD or 
            rank_shift >= RANK_SHIFT_THRESHOLD
        )
        
        #Severity
        if relative_shift < 0.10:
            severity = "STABLE"
        elif relative_shift < 0.25:
            severity = "LOW_SHIFT"
        elif relative_shift < 0.50:
            severity = "MEDIUM_SHIFT"
        elif relative_shift < 0.75:
            severity = "HIGH_SHIFT"
        else:
            severity = "CRITICAL_SHIFT"
            
        results.append({
            "feature": feature,
            "ref_importance": round(ref_imp, 6),
            "cur_importance": round(cur_imp, 6),
            "absolute_shift": round(absolute_shift, 6),
            "relative_shift_%": round(relative_shift * 100, 2),
            "direction": direction,
            "ref_rank": ref_rank,
            "cur_rank": cur_rank,
            "rank_shift": rank_shift,
            "is_drifted": is_drifted,
            "severity": severity,
        })
        
    results.sort(
        key=lambda x:x["relative_shift_%"],
        reverse = True
    )
    return results
def print_shap_comparison(
    ref_importance: Dict[str, float],
    cur_importance: Dict[str, float],
    ref_rankings: Dict[str, int],
    cur_rankings: Dict[str, int],
): 
    print("\n  Feature Importance Comparison:")
    print(f"  {'Feature':<25} {'Ref Imp':>10} {'Cur Imp':>10} "
        f"{'Shift%':>10} {'Ref Rank':>10} {'Cur Rank':>10}")
    print("  " + "-" * 75)
    
    for feature in sorted(
        ref_importance,
        key=lambda x: ref_importance[x],
        reverse=True
    ):
        ref_imp = ref_importance[feature]
        cur_imp = cur_importance.get(feature, 0)
        shift_pct = abs(cur_imp - ref_imp) / (ref_imp + 1e-10)*100
        ref_rank = ref_rankings[feature]
        cur_rank = cur_rankings.get(feature, 0)
        rank_diff = cur_rank - ref_rank
        
        arrow = "↑" if cur_imp > ref_imp else "↓"
        rank_arrow = (
            f"↑{abs(rank_diff)}" if rank_diff < 0
            else f"↓{abs(rank_diff)}" if rank_diff > 0
            else "→"
        )
        flag = "🔴" if shift_pct > 25 else "🟡" if shift_pct>10 else "✅"
        
        print(
            f"  {feature:<25} {ref_imp:>10.4f} {cur_imp:>10.4f} "
            f"{arrow}{shift_pct:>8.1f}% {ref_rank:>8} {cur_rank:>6} "
            f"{rank_arrow:>5} {flag}"
        )
        
def run_shap_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model, 
    features: List[str] = FEATURES
)-> Tuple[pd.DataFrame, bool, List[str]]:
    
    ref_importance = compute_shap_importance(
        model, reference, features, label="Reference"
    )
    cur_importance = compute_shap_importance(
        model, current, features, label="Current"
    )
    
    # Get Rankings
    ref_rankings = get_rankings(ref_importance)
    cur_rankings = get_rankings(cur_importance)
    
    #Printing comparison
    print_shap_comparison(
        ref_importance, cur_importance,
        ref_rankings, cur_rankings
    )
    
    #Analyze drift
    results = analyze_shap_drift(
        ref_importance, cur_importance,
        ref_rankings, cur_rankings
    )
    
    results_df = pd.DataFrame(results)
    
    #Retrain decesion
    drifted_features = results_df[
    (results_df["is_drifted"] == True) &
    (~results_df["feature"].isin(IGNORED_FEATURES))
]["feature"].tolist()
    
    should_retrain = len(drifted_features) > 0
    
    return results_df, should_retrain, drifted_features

if __name__ == "__main__":
    import os
    
    print("="*60)
    print(" 🧠 Sentinel AI — SHAP Drift Detection")
    print("="*60)
    
    model = load_model()
    reference, current = load_data()
    
    print("\n Running SHAP drift analysis...")
    results, should_retrain, drifted = run_shap_drift(
        reference, current, model
    )
    
    print("\n" + "="*60)
    print(" SHAP Drift Results ")
    print("\n" + "="*60)
    print(results.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  SHAP DRIFT SUMMARY")
    print("=" * 60)
    
    for severity in [
        "CRITICAL_SHIFT", "HIGH_SHIFT",
        "MEDIUM_SHIFT", "LOW_SHIFT", "STABLE"
    ]:
        features = results[
            results["severity"] == severity
        ]["feature"].tolist()
        if features:
            emoji = {
                "CRITICAL_SHIFT": "🚨",
                "HIGH_SHIFT":     "🔴",
                "MEDIUM_SHIFT":   "🟠",
                "LOW_SHIFT":      "🟡",
                "STABLE":         "✅"
            }
            print(
                f"  {emoji[severity]} "
                f"{severity}: {', '.join(features)}"
            )
            
    print(
        f"\n SHAP Drifted FEatures: "
        f"{drifted if drifted else 'None'}"
    )
    print(
        f"Should Retrain: "
        f"{'YES 🔴' if should_retrain else 'NO ✅'}"
    )
    print("="*60)
    
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/shap_drift.csv"
    results.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved → {output_path}")
    