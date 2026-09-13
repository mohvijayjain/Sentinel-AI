-- ============================================================
-- Sentinel-AI schema
--
-- Notes:
--   * Column names are unchanged: the application (drift_repository.py)
--     depends on run_time, detected_at, psi_score, is_drifted, etc.
--   * CREATE TABLE IF NOT EXISTS will NOT alter a table that already
--     exists. On a database that already has these tables, the new
--     constraints below apply only to fresh creates; the CREATE INDEX
--     statements, however, do run regardless and take effect immediately.
-- ============================================================


-- ============================================================
-- Per-feature drift scores (one row per feature per run)
-- ============================================================

CREATE TABLE IF NOT EXISTS drift_scores (
    id            SERIAL PRIMARY KEY,
    run_date      TIMESTAMP   NOT NULL DEFAULT NOW(),
    feature_name  VARCHAR(100),
    psi_score     FLOAT,
    is_drifted    BOOLEAN     NOT NULL DEFAULT FALSE
);

-- get_drifted_features(): WHERE is_drifted ORDER BY psi_score DESC
-- get_feature_drift_scores(): ORDER BY psi_score DESC
CREATE INDEX IF NOT EXISTS idx_drift_scores_psi
    ON drift_scores (psi_score DESC);

CREATE INDEX IF NOT EXISTS idx_drift_scores_is_drifted
    ON drift_scores (is_drifted);


-- ============================================================
-- Drift events
--
-- NOTE: action_taken is TEXT here. In the original schema it was FLOAT,
-- but insert_drift_event() writes a string action (e.g. "RETRAIN"),
-- which does not fit a FLOAT column. If the live table already exists
-- with FLOAT, fix it with:
--     ALTER TABLE drift_events ALTER COLUMN action_taken TYPE TEXT;
-- ============================================================

CREATE TABLE IF NOT EXISTS drift_events (
    id                SERIAL PRIMARY KEY,
    detected_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    drifted_features  TEXT,
    action_taken      TEXT,
    report_path       TEXT
);

-- get_latest_events(): ORDER BY detected_at DESC
CREATE INDEX IF NOT EXISTS idx_drift_events_detected_at
    ON drift_events (detected_at DESC);


-- ============================================================
-- Retraining events
-- ============================================================

CREATE TABLE IF NOT EXISTS retraining_events (
    id                SERIAL PRIMARY KEY,
    triggered_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    triggered_reason  TEXT,
    new_model_rmse    FLOAT,
    champion_rmse     FLOAT,
    promoted          BOOLEAN,
    mlflow_run_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_retraining_events_triggered_at
    ON retraining_events (triggered_at DESC);


-- ============================================================
-- Prediction logs (one row per served prediction)
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction_logs (
    id                  SERIAL PRIMARY KEY,
    predicted_at        TIMESTAMP NOT NULL DEFAULT NOW(),
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

CREATE INDEX IF NOT EXISTS idx_prediction_logs_predicted_at
    ON prediction_logs (predicted_at DESC);

-- Useful for per-version latency / accuracy analysis.
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
    ON prediction_logs (model_version);


-- ============================================================
-- Monitoring runs (one row per monitoring cycle)
--
-- Source of truth for the RAG knowledge layer: each row is rendered
-- to text and upserted into ChromaDB as monitoring_run:<id>.
--
-- action mirrors drift_scorer.get_action():
--     WAIT (<0.25), MONITOR (<0.5), ALERT (<0.75), RETRAIN (>=0.75)
-- ============================================================

CREATE TABLE IF NOT EXISTS monitoring_runs (
    id                 SERIAL PRIMARY KEY,
    run_time           TIMESTAMP NOT NULL DEFAULT NOW(),
    statistical_score  FLOAT     NOT NULL,
    shap_score         FLOAT     NOT NULL,
    prediction_score   FLOAT     NOT NULL,
    overall_score      FLOAT     NOT NULL,
    action             TEXT      NOT NULL
                    CHECK (action IN ('WAIT', 'MONITOR', 'ALERT', 'RETRAIN')),
    drifted_features   TEXT,
    report_path        TEXT
);

-- get_latest_monitoring_run() / get_monitoring_history():
--     ORDER BY run_time DESC LIMIT :limit
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_run_time
    ON monitoring_runs (run_time DESC);

-- Filtering the RAG index or dashboards by decision, e.g. all RETRAINs.
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_action
    ON monitoring_runs (action);