import time
from datetime import datetime

import pandas as pd
from fastapi import HTTPException

from .logger import logger
from .schemas import PredictResponse


def predict_trip(
    data,
    model,
    features,
    model_version,
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet",
        )

    start = time.time()

    try:
        input_data = pd.DataFrame([data.model_dump()])
        input_data = input_data[features]

        prediction = float(model.predict(input_data)[0])
        prediction = max(60, min(prediction, 7200))

        latency = round((time.time() - start) * 1000, 2)

        logger.info(
            f"Prediction: {prediction:.1f}s | "
            f"Distance: {data.trip_distance}mi | "
            f"Hour: {data.pickup_hour} | "
            f"Latency: {latency}ms"
        )

        return PredictResponse(
            prediction_seconds=round(prediction, 2),
            prediction_minutes=round(prediction / 60, 2),
            model_version=model_version,
            latency_ms=latency,
            timestamp=datetime.now().isoformat(),
        )

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )