from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup(0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    pickup_month: int = Field(..., ge=1, le=12, description="Month(1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Weekend")
    is_rush_hour: int = Field(..., ge=0, le=23, description="1 if rush hour")
    PULocationID: int = Field(..., ge=0, le=263, description="Pickup zone ID")
    DOLocationID: int = Field(..., ge=0, le=263, description="Dropoff zone ID")
    payment_type: int = Field(..., ge=0, le=23, description="Payment type(1-6)")
    VendorID: int = Field(..., ge=1, le=2, description="Vendor ID")
    RatecodeID: int = Field(..., ge=1, le=6, description="Rate Code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trip_distance": 3.5,
                "pickup_hour": 8,
                "pickup_day_of_week": 0,
                "pickup_month": 3,
                "is_weekend": 1,
                "is_rush_hour": 0,
                "PULocationID": 230,
                "DOLocationID": 161,
                "payment_type": 1,
                "VendorID": 2,
                "RatecodeID": 1.0,
            }
        }
        
class PredictResponse(BaseModel):
    prediction_seconds: float
    prediction_minutes: float
    model_version: str
    latency_ms: float
    timestamp: str