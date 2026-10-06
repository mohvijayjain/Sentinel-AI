import logging
from datetime import datetime
from sqlalchemy import text

import pandas as pd

from src.database.postgres import engine

from src.common.error_redaction import safe_error_message


logger = logging.getLogger(__name__)



def insert_drift_scores(
    drift_df
):

    query = text("""
        INSERT INTO drift_scores
        (
            feature_name,
            psi_score,
            is_drifted
        )

        VALUES
        (
            :feature_name,
            :psi_score,
            :is_drifted
        )
    """)


    with engine.begin() as conn:

        for _, row in drift_df.iterrows():

            conn.execute(
                query,
                {
                    "feature_name": row["feature"],
                    "psi_score": row["psi"],
                    "is_drifted": row["severity"] != "NO_DRIFT"
                }
            )



def insert_drift_event(
    drifted_features,
    action,
    report_path
):

    query = text("""
        INSERT INTO drift_events
        (
            drifted_features,
            action_taken,
            report_path
        )

        VALUES
        (
            :features,
            :action,
            :report
        )
    """)


    with engine.begin() as conn:

        conn.execute(
            query,
            {
                "features": ",".join(drifted_features),
                "action": action,
                "report": report_path
            }
        )
        
def get_latest_events(
    limit=10
):

    query = text("""
        SELECT *
        FROM drift_events
        ORDER BY detected_at DESC
        LIMIT :limit
    """)


    with engine.connect() as conn:

        result = conn.execute(
            query,
            {
                "limit": limit
            }
        )

        return result.fetchall()
    

def insert_monitoring_run(
    statistical_score,
    shap_score,
    prediction_score,
    overall_score,
    action,
    drifted_features,
    report_path
):

    query = text("""
        INSERT INTO monitoring_runs
        (
            statistical_score,
            shap_score,
            prediction_score,
            overall_score,
            action,
            drifted_features,
            report_path
        )

        VALUES
        (
            :statistical_score,
            :shap_score,
            :prediction_score,
            :overall_score,
            :action,
            :drifted_features,
            :report_path
        )

        RETURNING id
    """)

    with engine.begin() as conn:

        result = conn.execute(
            query,
            {
                "statistical_score": statistical_score,
                "shap_score": shap_score,
                "prediction_score": prediction_score,
                "overall_score": overall_score,
                "action": action,
                "drifted_features": ",".join(drifted_features),
                "report_path": report_path
            }
        )

        run_id = result.scalar_one()

    # ASCII log line, not an emoji print: after the commit, success
    # reporting must never be able to raise (e.g. UnicodeEncodeError on a
    # cp1252 / piped stdout) and turn a saved row into a reported failure.
    logger.info("Monitoring run inserted successfully: run_id=%s", run_id)

    return run_id
    
    
def get_latest_monitoring_run():

    query = text("""
        SELECT *
        FROM monitoring_runs
        ORDER BY run_time DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query)

        return result.fetchone()



def get_monitoring_history(limit=20):

    query = text("""
        SELECT *
        FROM monitoring_runs
        ORDER BY run_time DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                "limit": limit
            }
        )

        return result.fetchall()



def get_drifted_features():

    query = text("""
        SELECT *
        FROM drift_scores
        WHERE is_drifted = true
        ORDER BY psi_score DESC
    """)


    with engine.connect() as conn:

        result = conn.execute(query)

        return result.fetchall()



def get_feature_drift_scores():

    query = text("""
        SELECT *
        FROM drift_scores
        ORDER BY psi_score DESC
    """)


    with engine.connect() as conn:

        result = conn.execute(query)

        return result.fetchall()
    
    
def insert_retraining_event(
    triggered_reason,
    new_model_rmse,
    champion_rmse,
    promoted,
    mlflow_run_id,
    *,
    triggered_at,
    status,
    error_message=None
):
    """
    Persist one retraining attempt.

    triggered_at is the caller's canonical ISO-8601 UTC timestamp for the
    attempt; it is stored as UTC wall-clock time in the TIMESTAMP column
    (independent of the session time zone), never replaced by NOW().
    status is 'promoted', 'rejected' or 'failed'.

    error_message is re-sanitized here (secrets redacted, bounded) as
    defense in depth; idempotent, so an already-safe message is stored
    unchanged.
    """

    if error_message is not None:
        error_message = safe_error_message(error_message)

    query = text("""
        INSERT INTO retraining_events
        (
            triggered_at,
            triggered_reason,
            new_model_rmse,
            champion_rmse,
            promoted,
            mlflow_run_id,
            status,
            error_message
        )

        VALUES
        (
            CAST(:triggered_at AS TIMESTAMPTZ) AT TIME ZONE 'UTC',
            :triggered_reason,
            :new_model_rmse,
            :champion_rmse,
            :promoted,
            :mlflow_run_id,
            :status,
            :error_message
        )

        RETURNING id
    """)

    with engine.begin() as conn:

        result = conn.execute(
            query,
            {
                "triggered_at": triggered_at,
                "triggered_reason": triggered_reason,
                "new_model_rmse": new_model_rmse,
                "champion_rmse": champion_rmse,
                "promoted": promoted,
                "mlflow_run_id": mlflow_run_id,
                "status": status,
                "error_message": error_message
            }
        )

        event_id = result.scalar_one()

    # See insert_monitoring_run: success reporting cannot fail post-commit
    logger.info(
        "Retraining event inserted successfully: event_id=%s",
        event_id,
    )

    return event_id