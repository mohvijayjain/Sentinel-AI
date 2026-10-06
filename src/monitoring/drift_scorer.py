import pandas as pd
import json
import logging
import os

from src.database.drift_repository import (
    insert_monitoring_run,
    insert_drift_scores
)

from src.rag.monitoring_updater import (
    upsert_monitoring_run
)

from src.training.orchestrator import (
    run_retraining_pipeline
)

from src.monitoring.shap_drift import (
    IGNORED_FEATURES as SHAP_IGNORED_FEATURES
)


# ============================================================
# Configuration
# ============================================================

logger = logging.getLogger(__name__)

STATISTICAL_PATH = "reports/statistical_drift.csv"
SHAP_PATH = "reports/shap_drift.csv"
PREDICTION_PATH = "reports/prediction_drift.csv"


WEIGHTS = {
    "statistical": 0.4,
    "shap": 0.3,
    "prediction": 0.3
}


# ============================================================
# Convert Severity into Score
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

    if value not in mapping:

        # An unknown label means a detector and this mapping have
        # drifted apart; warn loudly instead of silently scoring 0
        logger.warning(
            f"Unrecognized drift severity {value!r}; scoring as 0"
        )

        return 0

    return mapping[value]


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

    if len(scores) == 0:
        return 0

    return max(scores)


# ============================================================
# SHAP Drift Score
# ============================================================

def calculate_shap_score():

    df = pd.read_csv(
        SHAP_PATH
    )

    # The detector's is_drifted is authoritative; features the
    # detector ignores never contribute, matching its own retrain logic
    drifted = df[
        (df["is_drifted"] == True)
        & (~df["feature"].isin(SHAP_IGNORED_FEATURES))
    ]

    if len(drifted) == 0:
        return 0

    # A drifted row can still be STABLE in magnitude (rank-only drift);
    # floor it at LOW_SHIFT so detected drift never scores 0
    scores = [
        severity_score(
            "LOW_SHIFT" if severity == "STABLE" else severity
        )
        for severity in drifted["severity"]
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
# Drifted Feature Names
# ============================================================

def extract_drifted_features(statistical_df):
    """
    Names of statistically drifted features, i.e. any whose severity
    scores above zero. Reuses severity_score so this stays consistent
    with the overall scoring logic.
    """

    return [
        str(row["feature"])
        for _, row in statistical_df.iterrows()
        if severity_score(row["severity"]) > 0
    ]


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

    print("=" * 60)
    print(" Sentinel AI — Drift Severity Scorer")
    print("=" * 60)


    # --------------------------------------------------------
    # Calculate individual drift scores
    # --------------------------------------------------------

    statistical_score = (
        calculate_statistical_score()
    )

    shap_score = (
        calculate_shap_score()
    )

    prediction_score = (
        calculate_prediction_score()
    )


    # --------------------------------------------------------
    # Calculate weighted overall score
    # --------------------------------------------------------

    overall_score = (

        statistical_score
        * WEIGHTS["statistical"]

        +

        shap_score
        * WEIGHTS["shap"]

        +

        prediction_score
        * WEIGHTS["prediction"]

    )


    # --------------------------------------------------------
    # Determine action
    # --------------------------------------------------------

    action = get_action(
        overall_score
    )


    # --------------------------------------------------------
    # Create drift summary
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Print drift summary
    # --------------------------------------------------------

    print("\nDrift Summary")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Save drift summary
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save statistical drift scores
    # --------------------------------------------------------

    statistical_df = pd.read_csv(
        STATISTICAL_PATH
    )

    insert_drift_scores(
        statistical_df
    )


    # --------------------------------------------------------
    # Collect drifted feature names for the monitoring row
    # --------------------------------------------------------

    drifted_features = extract_drifted_features(
        statistical_df
    )


    # --------------------------------------------------------
    # Save monitoring run to PostgreSQL
    # --------------------------------------------------------

    run_id = insert_monitoring_run(

        statistical_score=statistical_score,

        shap_score=shap_score,

        prediction_score=prediction_score,

        overall_score=overall_score,

        action=action,

        drifted_features=drifted_features,

        report_path="reports/drift_summary.json"
    )


    print(
        f"\n✅ Monitoring run saved → run_id={run_id}"
    )


    # --------------------------------------------------------
    # Index monitoring result into ChromaDB
    # --------------------------------------------------------

    upsert_monitoring_run(

        run_id=run_id,

        statistical_score=statistical_score,

        shap_score=shap_score,

        prediction_score=prediction_score,

        overall_score=overall_score,

        action=action
    )


    print(
        "✅ Monitoring result indexed into ChromaDB"
    )


    # ========================================================
    # AUTOMATIC RETRAINING TRIGGER
    # ========================================================

    if action == "RETRAIN":

        print("\n" + "=" * 60)
        print(" 🚨 AUTOMATIC RETRAINING TRIGGERED")
        print("=" * 60)

        print(
            "\nReason: Drift severity reached RETRAIN threshold."
        )

        print(
            "Starting retraining pipeline..."
        )


        # ----------------------------------------------------
        # Run complete retraining + promotion pipeline
        #
        # Wrapped so a training/promotion failure does not crash
        # the script: the monitoring row and RAG index are already
        # saved above, and the Champion is left untouched on failure.
        # ----------------------------------------------------

        retraining_result = None

        try:

            retraining_result = run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

        except Exception as exc:

            print(
                f"\n❌ Retraining pipeline failed: {exc}"
            )

            print(
                "Champion remains unchanged. "
                "Monitoring record is already saved."
            )


        # ----------------------------------------------------
        # Display retraining result
        # ----------------------------------------------------

        if retraining_result is not None:

            print("\nAutomatic Retraining Result")
            print("-" * 60)


            for key, value in retraining_result.items():

                print(
                    f"{key}: {value}"
                )


            print("-" * 60)


            # ------------------------------------------------
            # Final promotion status
            # ------------------------------------------------

            if retraining_result["promoted"]:

                print(
                    "\n✅ Challenger model PROMOTED."
                )

                print(
                    f"New Champion Version: "
                    f"{retraining_result['new_version']}"
                )

            else:

                print(
                    "\n❌ Challenger model REJECTED."
                )

                print(
                    f"Champion remains Version: "
                    f"{retraining_result['champion_version']}"
                )


    else:

        print(
            f"\nℹ️ No retraining required."
        )

        print(
            f"Action: {action}"
        )


    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\nSaved → reports/drift_summary.json"
    )

    print("=" * 60)
    print(" Sentinel AI Drift Pipeline Complete")
    print("=" * 60)