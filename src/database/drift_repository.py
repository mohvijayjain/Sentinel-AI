from datetime import datetime
from sqlalchemy import text

import pandas as pd

from src.database.postgres import engine



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
    """)


    with engine.begin() as conn:

        conn.execute(
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

    print("✅ Monitoring run saved")
    
    
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