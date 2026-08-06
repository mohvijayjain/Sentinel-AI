CREATE TABLE IF NOT EXISTS drift_scores(
id Serial PRIMARY KEY,
run_date TIMESTAMP DEFAULT NOW(),
feature_name VARCHAR(100),
psi_score FLOAT,
is_drifted BOOLEAN
);

CREATE TABLE IF NOT EXISTS drift_events(
id Serial PRIMARY KEY,
detected_at TIMESTAMP DEFAULT NOW(),
drifted_features TEXT,
action_taken FLOAT,
report_path TEXT
);

CREATE TABLE IF NOT EXISTS retraining_events(
id Serial PRIMARY KEY,
triggered_at TIMESTAMP DEFAULT NOW(),
triggered_reason TEXT,
new_model_rmse FLOAT,
champion_rmse FLOAT,
promoted BOOLEAN,
mlflow_run_id TEXT
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id                  SERIAL PRIMARY KEY,
    predicted_at        TIMESTAMP DEFAULT NOW(),
    trip_distance       FLOAT,
    pickup_hour         INT,
    pickup_day_of_week  INT,
    pickup_month        INT,
    is_weekend          INT,
    is_rush_hour        INT,
    PULocationID        INT,
    DOLocationID        INT,
    payment_type        INT,
    VendorID            INT,
    RatecodeID          FLOAT,
    prediction_seconds  FLOAT,
    prediction_minutes  FLOAT,
    model_version       VARCHAR(20),
    latency_ms          FLOAT
);

