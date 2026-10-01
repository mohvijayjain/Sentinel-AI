import pandas as pd
import json
import os

from src.database.drift_repository import (insert_monitoring_run, insert_drift_scores)
from src.rag.knowledge_updater import upsert_monitoring_run


# ============================================================
# Configuration
# ============================================================

STATISTICAL_PATH = "reports/statistical_drift.csv"
SHAP_PATH = "reports/shap_drift.csv"
PREDICTION_PATH = "reports/prediction_drift.csv"


WEIGHTS = {
    "statistical": 0.4,
    "shap": 0.3,
    "prediction": 0.3
}


# ============================================================
# Convert severity into score
# ============================================================

def severity_score(value):

    mapping = {
        "NO_DRIFT": 0,
        "STABLE": 0,

        "LOW": 0.25,
        "LOW_SHIFT": 0.25,

        "MEDIUM": 0.5,
        "MEDIUM_SHIFT": 0.5,

        "HIGH": 0.75,
        "HIGH_SHIFT": 0.75,

        "CRITICAL": 1,
        "CRITICAL_SHIFT": 1
    }

    return mapping.get(
        value,
        0
    )


# ============================================================
# Statistical Drift Score
# ============================================================

def calculate_statistical_score():

    df = pd.read_csv(
        STATISTICAL_PATH
    )

    scores = []

    for severity in df["severity"]:

        scores.append(
            severity_score(severity)
        )


    if len(scores)==0:
        return 0


    return max(scores)



# ============================================================
# SHAP Drift Score
# ============================================================

def calculate_shap_score():

    df = pd.read_csv(
        SHAP_PATH
    )

    drifted = df[
        df["is_drifted"] == True
    ]


    if len(drifted)==0:
        return 0


    scores = [
        severity_score(x)
        for x in drifted["severity"]
    ]


    return max(scores)



# ============================================================
# Prediction Drift Score
# ============================================================

def calculate_prediction_score():

    df = pd.read_csv(
        PREDICTION_PATH
    )


    severity = df.iloc[0]["severity"]


    return severity_score(
        severity
    )



# ============================================================
# Overall Decision
# ============================================================

def get_action(score):

    if score < 0.25:
        return "WAIT"

    elif score < 0.5:
        return "MONITOR"

    elif score < 0.75:
        return "ALERT"

    else:
        return "RETRAIN"



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    print("="*60)
    print(" Sentinel AI — Drift Severity Scorer")
    print("="*60)


    statistical_score = calculate_statistical_score()

    shap_score = calculate_shap_score()

    prediction_score = calculate_prediction_score()



    overall_score = (
        statistical_score * WEIGHTS["statistical"]
        +
        shap_score * WEIGHTS["shap"]
        +
        prediction_score * WEIGHTS["prediction"]
    )


    action = get_action(
        overall_score
    )


    result = {

        "statistical_score":
            statistical_score,

        "shap_score":
            shap_score,

        "prediction_score":
            prediction_score,

        "overall_score":
            round(
                overall_score,
                3
            ),

        "action":
            action
    }


    print("\nDrift Summary")

    for k,v in result.items():

        print(
            f"{k}: {v}"
        )


    os.makedirs(
        "reports",
        exist_ok=True
    )

    
    with open(
        "reports/drift_summary.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )
        
    statistical_df = pd.read_csv(STATISTICAL_PATH)
    insert_drift_scores(statistical_df)
    
    run_id = insert_monitoring_run(
    statistical_score=statistical_score,
    shap_score=shap_score,
    prediction_score=prediction_score,
    overall_score=overall_score,
    action=action,
    drifted_features=[],
    report_path="reports/drift_summary.json"
)
    upsert_monitoring_run(
    run_id=run_id,
    statistical_score=statistical_score,
    shap_score=shap_score,
    prediction_score=prediction_score,
    overall_score=overall_score,
    action=action,
)


    print(
        "\nSaved → reports/drift_summary.json"
    )
    
    
    
    