import time
from datetime import datetime

import pandas as pd
from fastapi import HTTPException

from .logger import logger
from .schemas import PredictResponse
from src.ingestion.preprocess import engineer_features


def predict_trip(data, model, features, model_version):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )

    start = time.time()

    try:
        # ─────────────────────────────────────────────
        # 1. Convert API request to DataFrame
        # ─────────────────────────────────────────────

        input_data = pd.DataFrame([{
            "trip_distance": data.trip_distance,
            "tpep_pickup_datetime": data.pickup_datetime,
            "pulocationid": data.pulocationid,
            "dolocationid": data.dolocationid,
            "payment_type": data.payment_type,
            "vendorid": data.vendorid,
            "ratecodeid": data.ratecodeid,
        }])


        # ─────────────────────────────────────────────
        # 2. Convert pickup datetime
        # ─────────────────────────────────────────────

        input_data["tpep_pickup_datetime"] = pd.to_datetime(
            input_data["tpep_pickup_datetime"],
            errors="raise"
        )


        # ─────────────────────────────────────────────
        # 3. Shared feature engineering
        # ─────────────────────────────────────────────
        #
        # Creates:
        # pickup_hour
        # pickup_day_of_week
        # pickup_month
        # is_weekend
        # is_rush_hour
        #

        input_data = engineer_features(input_data)


        # ─────────────────────────────────────────────
        # 4. Select exactly the model features
        # ─────────────────────────────────────────────

        missing_features = [
            feature
            for feature in features
            if feature not in input_data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        input_data = input_data[features]


        # ─────────────────────────────────────────────
        # 5. Make prediction
        # ─────────────────────────────────────────────

        prediction = float(
            model.predict(input_data)[0]
        )


        # ─────────────────────────────────────────────
        # 6. Keep prediction within valid range
        # ─────────────────────────────────────────────

        prediction = max(
            60,
            min(prediction, 7200)
        )


        # ─────────────────────────────────────────────
        # 7. Calculate latency
        # ─────────────────────────────────────────────

        latency = round(
            (time.time() - start) * 1000,
            2
        )


        # ─────────────────────────────────────────────
        # 8. Logging
        # ─────────────────────────────────────────────

        logger.info(
            f"Prediction: {prediction:.1f}s | "
            f"Distance: {data.trip_distance}mi | "
            f"Pickup: {data.pickup_datetime} | "
            f"Latency: {latency}ms"
        )


        # ─────────────────────────────────────────────
        # 9. Response
        # ─────────────────────────────────────────────

        return PredictResponse(
            prediction_seconds=round(
                prediction,
                2
            ),
            prediction_minutes=round(
                prediction / 60,
                2
            ),
            model_version=model_version,
            latency_ms=latency,
            timestamp=datetime.now().isoformat(),
        )


    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )