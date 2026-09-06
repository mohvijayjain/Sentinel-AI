from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    trip_distance: float = Field(
        ..., gt=0, description="Trip distance in miles"
    )

    pickup_datetime: datetime = Field(
        ..., description="Pickup date and time"
    )

    pulocationid: int = Field(
        ..., ge=1, le=263, description="Pickup zone ID"
    )

    dolocationid: int = Field(
        ..., ge=1, le=263, description="Dropoff zone ID"
    )

    payment_type: int = Field(
        ..., ge=1, le=6, description="Payment type"
    )

    vendorid: int = Field(
        ..., ge=1, le=2, description="Vendor ID"
    )

    ratecodeid: int = Field(
        ..., ge=1, le=6, description="Rate code"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "trip_distance": 3.5,
                "pickup_datetime": "2026-03-02T08:30:00",
                "pulocationid": 230,
                "dolocationid": 161,
                "payment_type": 1,
                "vendorid": 2,
                "ratecodeid": 1
            }
        }


class PredictResponse(BaseModel):
    prediction_seconds: float
    prediction_minutes: float
    model_version: str
    latency_ms: float
    timestamp: str