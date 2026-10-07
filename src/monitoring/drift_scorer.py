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

from src.common.error_redaction import (
    describe_error
)

from src.monitoring.constants import (
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

# TODO(design, deferred): Should pure concept/prediction drift be able
# to trigger retraining?
#
# With these weights and the get_action() bands (RETRAIN >= 0.75), SHAP +
# prediction without statistical drift caps at 0.60 (ALERT), so RETRAIN
# effectively requires statistical drift of at least MEDIUM. This may be
# deliberate. Not changed here; points to review together:
#   - prediction drift alone as a trigger
#   - SHAP drift alone as a trigger
#   - combined SHAP + prediction as a trigger
#   - statistical drift as a mandatory gate (the current effect)
#   - false-positive retrain risk
#   - retrain cost (Optuna search, MLflow runs, promotion gates)
#   - seasonal reference strategy (cf. pickup_month handling)
#   - champion/challenger safeguards already limiting bad promotions
#   - whether a separate concept-drift action should exist


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
# Load Drift Report
# ============================================================

def read_drift_report(path):
    """
    Load a detector report. A missing, zero-byte or zero-row report
    means that detector produced no signal: warn and return None so
    the caller scores it as 0 instead of crashing the pipeline.
    """

    try:

        df = pd.read_csv(path)

    except FileNotFoundError:

        logger.warning(
            f"Missing drift report {path!r}; scoring as 0."
        )

        return None

    except pd.errors.EmptyDataError:

        df = pd.DataFrame()

    if len(df) == 0:

        logger.warning(
            f"Empty drift report {path!r}; scoring as 0."
        )

        return None

    return df


def check_psi_not_nan(df, path):
    """
    A NaN PSI is corrupt detector output, not "no drift". Unlike an
    empty report it must not be scored as 0, so raise instead.
    """

    if df["psi"].isna().any():

        raise ValueError(
            f"PSI is NaN; drift report {path!r} "
            f"contains invalid PSI data."
        )


# ============================================================
# Statistical Drift Score
# ============================================================

def calculate_statistical_score(path=None):

    path = path or STATISTICAL_PATH

    df = read_drift_report(
        path
    )

    if df is None:
        return 0

    check_psi_not_nan(
        df,
        path
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

def calculate_shap_score(path=None):

    df = read_drift_report(
        path or SHAP_PATH
    )

    if df is None:
        return 0

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

def calculate_prediction_score(path=None):

    path = path or PREDICTION_PATH

    df = read_drift_report(
        path
    )

    if df is None:
        return 0

    # Only row 0 is scored, so only its PSI is checked
    check_psi_not_nan(
        df.iloc[[0]],
        path
    )

    severity = df.iloc[0]["severity"]

    return severity_score(
        severity
    )


# ============================================================
# Weighted Score + Action (shared by __main__ and uploaded runs)
# ============================================================

def score_reports(
    statistical_path=None,
    shap_path=None,
    prediction_path=None
):
    """
    The three detector scores, their weighted overall score (unrounded)
    and the action, from the reports at the given paths (default: the
    reports/ paths above).
    """

    statistical_score = calculate_statistical_score(statistical_path)
    shap_score = calculate_shap_score(shap_path)
    prediction_score = calculate_prediction_score(prediction_path)

    overall_score = (
        statistical_score * WEIGHTS["statistical"]
        + shap_score * WEIGHTS["shap"]
        + prediction_score * WEIGHTS["prediction"]
    )

    return {
        "statistical_score": statistical_score,
        "shap_score": shap_score,
        "prediction_score": prediction_score,
        "overall_score": overall_score,
        "action": get_action(overall_score),
    }


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
# Console Output
# ============================================================

def report(message=""):
    """
    Print one line of scorer output as pure ASCII.

    __main__ prints between the monitoring-run write, the Chroma index and
    the retraining trigger. On a stdout that cannot encode a character
    (cp1252 console, C-locale container, cron/CI pipe) print() raises
    UnicodeEncodeError, which would stop the run before retraining. The
    literals passed here are ASCII; anything non-ASCII in dynamic text
    (e.g. an error message) is backslash-escaped, as Python already does
    for stderr, so the line can always be written.
    """

    print(
        message.encode("ascii", "backslashreplace").decode("ascii")
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    report("=" * 60)
    report(" Sentinel AI - Drift Severity Scorer")
    report("=" * 60)


    # --------------------------------------------------------
    # Detector scores, weighted overall score and action
    # --------------------------------------------------------

    scores = score_reports()

    statistical_score = scores["statistical_score"]
    shap_score = scores["shap_score"]
    prediction_score = scores["prediction_score"]
    overall_score = scores["overall_score"]
    action = scores["action"]


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

    report("\nDrift Summary")

    for key, value in result.items():

        report(
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

    statistical_df = read_drift_report(
        STATISTICAL_PATH
    )

    # No statistical report: nothing to persist, keep the run going
    if statistical_df is None:
        statistical_df = pd.DataFrame()

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


    report(
        f"\nMonitoring run saved: run_id={run_id}"
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


    report(
        "Monitoring result indexed into ChromaDB"
    )


    # ========================================================
    # AUTOMATIC RETRAINING TRIGGER
    # ========================================================

    if action == "RETRAIN":

        report("\n" + "=" * 60)
        report(" AUTOMATIC RETRAINING TRIGGERED")
        report("=" * 60)

        report(
            "\nReason: Drift severity reached RETRAIN threshold."
        )

        report(
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

            # Type + redacted message only; raw str(exc) may hold secrets
            report(
                f"\nRetraining pipeline failed: {describe_error(exc)}"
            )

            report(
                "Champion remains unchanged. "
                "Monitoring record is already saved."
            )


        # ----------------------------------------------------
        # Display retraining result
        # ----------------------------------------------------

        if retraining_result is not None:

            report("\nAutomatic Retraining Result")
            report("-" * 60)


            for key, value in retraining_result.items():

                report(
                    f"{key}: {value}"
                )


            report("-" * 60)


            # ------------------------------------------------
            # Final promotion status
            # ------------------------------------------------

            if retraining_result["promoted"]:

                report(
                    "\nChallenger model PROMOTED."
                )

                report(
                    f"New Champion Version: "
                    f"{retraining_result['new_version']}"
                )

            else:

                report(
                    "\nChallenger model REJECTED."
                )

                report(
                    f"Champion remains Version: "
                    f"{retraining_result['champion_version']}"
                )


    else:

        report(
            "\nNo retraining required."
        )

        report(
            f"Action: {action}"
        )


    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    report(
        "\nSaved: reports/drift_summary.json"
    )

    report("=" * 60)
    report(" Sentinel AI Drift Pipeline Complete")
    report("=" * 60)