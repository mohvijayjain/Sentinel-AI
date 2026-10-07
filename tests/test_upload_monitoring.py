"""
POST /monitoring/run: upload a production dataset and run one monitoring
cycle through the EXISTING pipeline.

Real code under test: ingest_data reader + clean_and_engineer, the three
detectors via drift_runner.run_drift_detection (SHAP included), the
drift_scorer scoring, the endpoint. Fakes: a small synthetic raw TLC
dataset, a temp reference built from it with the real preprocessing, a
tiny real LightGBM model, and conftest's PostgreSQL / RAG / orchestrator
stubs (recorded here). No NVIDIA, no database, no millions of rows.
"""

import hashlib
import io
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import SERVING_FAKE_KEY


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
AUTH = {"x-api-key": SERVING_FAKE_KEY}


def raw_trips(n, seed, shifted=False):
    """Raw NYC TLC-shaped rows (CamelCase, timestamps, extra columns)."""

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    seconds = rng.integers(0, 28 * 24 * 3600, n)
    distance = rng.gamma(2.0, 1.5, n) + 0.6
    zones = rng.integers(1, 264, n)

    if shifted:
        distance = distance * 4                       # far longer trips
        zones = rng.choice([132, 138], n)             # airport-only pickups
        seconds = (seconds // 86400) * 86400 + rng.integers(0, 4, n) * 3600

    pickup = start + pd.to_timedelta(seconds, unit="s")
    duration = np.clip(distance * 170 + rng.normal(300, 60, n), 70, 7000)

    return pd.DataFrame({
        "VendorID": rng.integers(1, 3, n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration, unit="s"),
        "passenger_count": rng.integers(1, 4, n),
        "trip_distance": distance,
        "RatecodeID": rng.choice([1.0, 2.0], n, p=[0.9, 0.1]),
        "PULocationID": zones,
        "DOLocationID": rng.integers(1, 264, n),
        "payment_type": rng.integers(1, 3, n),
        "fare_amount": distance * 3.0,
    })


def to_csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def to_parquet_bytes(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def reference_and_model(tmp_path_factory):
    """Reference built with the real preprocessing + a tiny real model."""

    import lightgbm as lgb
    from src.ingestion.preprocess import FEATURES, TARGET, clean_and_engineer

    reference = clean_and_engineer(raw_trips(4000, seed=1), "reference")
    path = tmp_path_factory.mktemp("reference") / "reference_data.parquet"
    reference.to_parquet(path)

    model = lgb.LGBMRegressor(n_estimators=30, verbose=-1, random_state=0)
    model.fit(reference[FEATURES], reference[TARGET])

    return str(path), model


@pytest.fixture
def pipeline(serving_app, serving_artifacts, reference_and_model, monkeypatch):
    """The real app, the temp reference, recorded persistence calls."""

    from src.monitoring import upload_monitoring

    reference_path, model = reference_and_model
    monkeypatch.setattr(upload_monitoring, "REFERENCE_PATH", reference_path)

    repo = sys.modules["src.database.drift_repository"]
    rag = sys.modules["src.rag.monitoring_updater"]
    orchestrator = sys.modules["src.training.orchestrator"]

    calls = {"runs": [], "scores": [], "indexed": [], "retrain": [], "dirs": []}

    def insert_run(**kwargs):
        calls["runs"].append(kwargs)
        return 100 + len(calls["runs"])

    monkeypatch.setattr(repo, "insert_monitoring_run", insert_run)
    monkeypatch.setattr(repo, "insert_drift_scores",
                        lambda df: calls["scores"].append(df.copy()))
    monkeypatch.setattr(rag, "upsert_monitoring_run",
                        lambda **kw: calls["indexed"].append(kw) or True)
    monkeypatch.setattr(orchestrator, "run_retraining_pipeline",
                        lambda **kw: calls["retrain"].append(kw) or {"promoted": False},
                        raising=False)

    real_new_dir = upload_monitoring.new_run_dir

    def recording_new_dir():
        path = real_new_dir()
        calls["dirs"].append(path)
        return path

    monkeypatch.setattr(upload_monitoring, "new_run_dir", recording_new_dir)

    with TestClient(serving_app.app) as client:
        serving_app.app.state.model = model       # detectors need a tree model
        client.calls = calls
        client.reference_path = reference_path
        yield client


def upload(client, content, name="june.csv", period="June 2026", headers=AUTH):
    data = {"period": period} if period is not None else {}
    return client.post("/monitoring/run", headers=headers,
                       files={"file": (name, content)}, data=data)


# ============================================================
# Successful runs
# ============================================================

def _assert_consistent_scores(body):
    from src.monitoring.drift_scorer import WEIGHTS, get_action

    overall = (body["statistical_score"] * WEIGHTS["statistical"]
               + body["shap_score"] * WEIGHTS["shap"]
               + body["prediction_score"] * WEIGHTS["prediction"])
    assert body["overall_score"] == round(overall, 3)
    assert body["action"] == get_action(overall)


def test_valid_csv_runs_the_real_pipeline(pipeline):

    response = upload(pipeline, to_csv_bytes(raw_trips(1500, seed=2)))

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["run_id"] == 101
    assert body["file_name"] == "june.csv"
    assert body["label"] == "June 2026"
    assert body["rows_uploaded"] == 1500
    assert 0 < body["rows_processed"] <= 1500
    assert body["reference"]["path"] == pipeline.reference_path
    assert body["reference"]["rows"] == len(pd.read_parquet(pipeline.reference_path))
    assert body["retraining"] == "not_required"
    assert body["indexed_for_assistant"] is True
    _assert_consistent_scores(body)

    # Same-distribution data: no drift in distance / time features, the
    # model's view (SHAP) or its predictions. (Zone IDs have 263 categories,
    # so a 1,500-row sample can show categorical PSI noise: real detector
    # behaviour on tiny data, not asserted either way.)
    for feature in ("trip_distance", "pickup_hour", "pickup_day_of_week"):
        assert feature not in body["drifted_features"]
    assert body["shap_score"] == 0
    assert body["prediction_score"] == 0


def test_valid_parquet_runs_the_real_pipeline(pipeline):

    response = upload(pipeline, to_parquet_bytes(raw_trips(1500, seed=3)),
                      name="june.parquet")

    assert response.status_code == 200, response.text
    assert response.json()["rows_uploaded"] == 1500
    _assert_consistent_scores(response.json())


def test_drifted_data_is_detected_and_scored(pipeline):

    response = upload(pipeline, to_csv_bytes(raw_trips(1500, seed=4, shifted=True)))

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["statistical_score"] > 0
    assert body["action"] != "WAIT"
    assert "trip_distance" in body["drifted_features"]
    assert "PULocationID" in body["drifted_features"]
    _assert_consistent_scores(body)


def test_run_is_persisted_and_indexed_like_the_scorer(pipeline):

    body = upload(pipeline, to_csv_bytes(raw_trips(1500, seed=4, shifted=True))).json()

    [run] = pipeline.calls["runs"]
    assert run["statistical_score"] == body["statistical_score"]
    assert run["shap_score"] == body["shap_score"]
    assert run["prediction_score"] == body["prediction_score"]
    assert round(run["overall_score"], 3) == body["overall_score"]
    assert run["action"] == body["action"]
    assert run["drifted_features"] == body["drifted_features"]
    assert run["report_path"] == "upload:june.csv (June 2026)"

    [scores] = pipeline.calls["scores"]                 # per-feature drift_scores
    assert set(scores["feature"]) >= {"trip_distance", "PULocationID"}

    [indexed] = pipeline.calls["indexed"]               # RAG write-through
    assert indexed["run_id"] == body["run_id"]
    assert indexed["action"] == body["action"]


def test_each_upload_is_its_own_run(pipeline):

    first = upload(pipeline, to_csv_bytes(raw_trips(1200, seed=5)), period="April").json()
    second = upload(pipeline, to_csv_bytes(raw_trips(1200, seed=6, shifted=True)),
                    period="May").json()

    assert (first["run_id"], second["run_id"]) == (101, 102)
    assert first["rows_uploaded"] == second["rows_uploaded"] == 1200
    assert second["overall_score"] > first["overall_score"]
    assert "trip_distance" in second["drifted_features"]
    assert "trip_distance" not in first["drifted_features"]


# ============================================================
# Reference and report isolation
# ============================================================

def test_reference_is_never_modified(pipeline):

    before = sha256(pipeline.reference_path)

    upload(pipeline, to_csv_bytes(raw_trips(1200, seed=7, shifted=True)))

    assert sha256(pipeline.reference_path) == before


def test_committed_reports_untouched_and_run_dirs_removed(pipeline):

    reports = os.path.join(REPO_ROOT, "reports")
    before = {name: sha256(os.path.join(reports, name)) for name in sorted(os.listdir(reports))}

    upload(pipeline, to_csv_bytes(raw_trips(1200, seed=8)))
    upload(pipeline, to_csv_bytes(raw_trips(1200, seed=9, shifted=True)))

    after = {name: sha256(os.path.join(reports, name)) for name in sorted(os.listdir(reports))}
    assert after == before

    first, second = pipeline.calls["dirs"]
    assert first != second
    assert not os.path.exists(first) and not os.path.exists(second)


def test_detection_writes_into_the_private_run_dir(pipeline, monkeypatch):

    from src.monitoring import drift_runner

    seen = []
    real = drift_runner.run_drift_detection

    def spy(**kwargs):
        seen.append((kwargs["reports_dir"], sorted(os.listdir(kwargs["reports_dir"]))))
        result = real(**kwargs)
        seen.append((kwargs["reports_dir"], sorted(os.listdir(kwargs["reports_dir"]))))
        return result

    monkeypatch.setattr(drift_runner, "run_drift_detection", spy)

    upload(pipeline, to_csv_bytes(raw_trips(1200, seed=10)))

    (run_dir, before), (_, after) = seen
    assert run_dir == pipeline.calls["dirs"][0]
    assert before == ["upload.csv"]
    assert after == ["prediction_drift.csv", "shap_drift.csv",
                     "statistical_drift.csv", "upload.csv"]


def test_concurrent_run_is_refused_not_shared(pipeline):

    from src.monitoring import upload_monitoring

    assert upload_monitoring._run_lock.acquire(blocking=False)
    try:
        response = upload(pipeline, to_csv_bytes(raw_trips(500, seed=11)))
    finally:
        upload_monitoring._run_lock.release()

    assert response.status_code == 409
    assert pipeline.calls["runs"] == []
    assert not os.path.exists(pipeline.calls["dirs"][0])


# ============================================================
# Validation errors
# ============================================================

@pytest.mark.parametrize("name", ["june.xlsx", "june.json", "june", "june.csv.exe"])
def test_unsupported_extension(pipeline, name):

    response = upload(pipeline, b"a,b\n1,2\n", name=name)

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported file type: upload a .csv or .parquet file."}
    assert pipeline.calls["runs"] == []


def test_missing_columns_are_listed(pipeline):

    frame = raw_trips(300, seed=12).drop(columns=["RatecodeID", "PULocationID"])

    response = upload(pipeline, to_csv_bytes(frame))

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Missing required columns: PULocationID, RatecodeID"}


def test_column_names_are_case_sensitive(pipeline):

    frame = raw_trips(300, seed=13).rename(columns={"PULocationID": "pulocationid"})

    response = upload(pipeline, to_csv_bytes(frame))

    assert response.status_code == 422
    assert "PULocationID" in response.json()["detail"]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_empty_dataset(pipeline, fmt):

    frame = raw_trips(10, seed=14).iloc[0:0]
    content = to_csv_bytes(frame) if fmt == "csv" else to_parquet_bytes(frame)

    response = upload(pipeline, content, name=f"empty.{fmt}")

    assert response.status_code == 422
    assert response.json() == {"detail": "The dataset is empty."}


def test_all_rows_filtered_by_preprocessing(pipeline):

    frame = raw_trips(200, seed=15)
    frame["trip_distance"] = 100.0                      # every row an outlier

    response = upload(pipeline, to_csv_bytes(frame))

    assert response.status_code == 422
    assert "No valid rows after preprocessing" in response.json()["detail"]


def test_unreadable_file(pipeline):

    response = upload(pipeline, b"\x00\x01 not parquet", name="broken.parquet")

    assert response.status_code == 422
    assert response.json() == {"detail": "The file could not be read as CSV / Parquet."}


def test_size_limit(pipeline, monkeypatch):

    from src.monitoring import upload_monitoring

    monkeypatch.setattr(upload_monitoring, "MAX_UPLOAD_BYTES", 1024)

    response = upload(pipeline, to_csv_bytes(raw_trips(500, seed=16)))

    assert response.status_code == 413
    assert pipeline.calls["runs"] == []


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}], ids=["missing", "wrong"])
def test_requires_api_key(pipeline, headers):

    response = upload(pipeline, to_csv_bytes(raw_trips(100, seed=17)), headers=headers)

    assert response.status_code == 401
    assert pipeline.calls["dirs"] == []


def test_processing_failure_is_generic_and_redacted(pipeline, monkeypatch, caplog):

    from src.monitoring import drift_runner

    def boom(**kwargs):
        raise RuntimeError("could not connect postgresql://u:S3cretFakePw@db/x")

    monkeypatch.setattr(drift_runner, "run_drift_detection", boom)

    with caplog.at_level(logging.ERROR):
        response = upload(pipeline, to_csv_bytes(raw_trips(500, seed=18)))

    assert response.status_code == 500
    assert response.json() == {"detail": "Monitoring run failed while processing the dataset."}
    assert "S3cretFakePw" not in response.text
    assert "S3cretFakePw" not in caplog.text
    assert "Uploaded monitoring run failed" in caplog.text
    assert not os.path.exists(pipeline.calls["dirs"][0])


# ============================================================
# RETRAIN: existing pipeline, after the response
# ============================================================

def _force_retrain(monkeypatch):
    """Scores as if every detector were CRITICAL (the trigger branch only)."""

    from src.monitoring import drift_scorer

    monkeypatch.setattr(drift_scorer, "score_reports", lambda **kw: {
        "statistical_score": 1.0, "shap_score": 1.0, "prediction_score": 1.0,
        "overall_score": 1.0, "action": "RETRAIN"})


def test_retrain_starts_existing_pipeline(pipeline, monkeypatch):

    _force_retrain(monkeypatch)

    response = upload(pipeline, to_csv_bytes(raw_trips(500, seed=19)))

    assert response.status_code == 200
    assert response.json()["action"] == "RETRAIN"
    assert response.json()["retraining"] == "started"
    assert pipeline.calls["retrain"] == [{"triggered_reason": "drift_detected_upload"}]


def test_retrain_not_started_twice(pipeline, monkeypatch):

    from src.monitoring import upload_monitoring

    _force_retrain(monkeypatch)
    assert upload_monitoring._retrain_lock.acquire(blocking=False)
    try:
        response = upload(pipeline, to_csv_bytes(raw_trips(500, seed=20)))
    finally:
        upload_monitoring._retrain_lock.release()

    assert response.json()["retraining"] == "already_running"
    assert pipeline.calls["retrain"] == []


def test_retrain_failure_never_raises(monkeypatch, caplog):

    from src.monitoring import upload_monitoring

    orchestrator = sys.modules["src.training.orchestrator"]

    def failing(**kwargs):
        raise RuntimeError("mlflow down https://u:Pw9FakeSecret@mlflow:5000")

    monkeypatch.setattr(orchestrator, "run_retraining_pipeline", failing, raising=False)

    with caplog.at_level(logging.ERROR):
        upload_monitoring.start_retraining()

    assert "Retraining pipeline failed" in caplog.text
    assert "Pw9FakeSecret" not in caplog.text
    assert not upload_monitoring.retraining_running()


# ============================================================
# Existing callers unchanged
# ============================================================

def test_ingest_file_still_appends_processed_rows(tmp_path, monkeypatch):

    import types
    from src.ingestion import ingest_data

    written = {}
    fake_postgres = types.ModuleType("src.database.postgres")
    fake_postgres.engine = object()
    monkeypatch.setitem(sys.modules, "src.database.postgres", fake_postgres)
    monkeypatch.setattr(pd.DataFrame, "to_sql",
                        lambda self, name, engine, **kw: written.update(name=name, rows=len(self)))

    path = tmp_path / "raw.parquet"
    raw_trips(300, seed=21).to_parquet(path)

    ingest_data.ingest_file(str(path), "June")

    assert written["name"] == "taxi_trips"
    assert 0 < written["rows"] <= 300
