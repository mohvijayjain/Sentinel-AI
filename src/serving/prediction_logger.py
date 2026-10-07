"""
Write each successful /predict into prediction_logs (best effort).

PostgreSQL logging is an audit side effect, never part of the answer:
log_prediction() is scheduled as a background task after the response
and never raises, so a slow or failing database can neither delay nor
fail a prediction. Failures are logged (redacted), not swallowed.
"""

import pandas as pd

from src.common.error_redaction import redacted_traceback
from src.database import drift_repository
from src.ingestion.preprocess import engineer_features

from .logger import logger


def build_prediction_log(data, response, predicted_at: str) -> dict:
    """
    One prediction_logs row: the request's model inputs, the engineered
    time features (same engineer_features() the model is served with) and
    the returned prediction. Nothing from headers or credentials.
    """

    values = response if isinstance(response, dict) else response.model_dump()

    pickup = engineer_features(pd.DataFrame([{
        "tpep_pickup_datetime": pd.to_datetime(data.pickup_datetime, errors="raise"),
    }])).iloc[0]

    return {
        "predicted_at": predicted_at,
        "trip_distance": float(data.trip_distance),
        "pickup_hour": int(pickup["pickup_hour"]),
        "pickup_day_of_week": int(pickup["pickup_day_of_week"]),
        "pickup_month": int(pickup["pickup_month"]),
        "is_weekend": int(pickup["is_weekend"]),
        "is_rush_hour": int(pickup["is_rush_hour"]),
        "pulocationid": int(data.pulocationid),
        "dolocationid": int(data.dolocationid),
        "payment_type": int(data.payment_type),
        "vendorid": int(data.vendorid),
        "ratecodeid": float(data.ratecodeid),      # FLOAT in Sentinel.sql
        "prediction_seconds": float(values["prediction_seconds"]),
        "prediction_minutes": float(values["prediction_minutes"]),
        "model_version": str(values["model_version"]),
        "latency_ms": float(values["latency_ms"]),
    }


def log_prediction(data, response, predicted_at: str) -> bool:
    """Insert one prediction log. True on success; never raises."""

    try:
        drift_repository.insert_prediction_log(
            **build_prediction_log(data, response, predicted_at)
        )
        return True

    except Exception as error:
        # The prediction was already served; only the audit row is lost.
        # Database errors can echo the connection URL: redacted traceback.
        logger.error(
            "Prediction log not written; the prediction itself was served.\n%s",
            redacted_traceback(error),
        )
        return False
