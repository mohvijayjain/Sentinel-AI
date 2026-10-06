"""
RETRAIN reachability from genuine detector output.

Every report row here is produced by the real detector code from numeric
inputs, never by hand-writing a severity label:

  statistical  stastical_drift.analyze_feature on shifted normal data
  shap         shap_drift.analyze_shap_drift on crafted importances,
               with rankings from shap_drift.get_rankings
  prediction   prediction_drift.run_prediction_drift on shifted arrays

The rows are written as the real report CSVs and scored by running
drift_scorer as __main__ (the weighted combine and the RETRAIN branch
only exist there). DB / RAG / retraining calls hit fresh MagicMocks on
the conftest stub modules.
"""

import functools
import itertools
import json
import os
import runpy
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from conftest import PREDICTION_COLUMNS, SHAP_COLUMNS, STATISTICAL_COLUMNS
from src.monitoring import shap_drift
from src.monitoring.constants import IGNORED_FEATURES
from src.monitoring.prediction_drift import run_prediction_drift
from src.monitoring.stastical_drift import analyze_feature


SCORER_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "src", "monitoring", "drift_scorer.py",
)

# Score of each band, per the scorer's mapping
PSI_BAND_SCORE = {
    "NO_DRIFT": 0, "LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0,
}
PSI_BANDS = list(PSI_BAND_SCORE)

SHAP_BAND_SCORE = {
    "STABLE": 0, "LOW_SHIFT": 0.25, "MEDIUM_SHIFT": 0.5,
    "HIGH_SHIFT": 0.75, "CRITICAL_SHIFT": 1.0,
}
SHAP_BANDS = list(SHAP_BAND_SCORE)


# ============================================================
# Genuine detector inputs per band
# ============================================================

# Mean shift (in reference std units) of N(0, 1) data, calibrated so
# both PSI implementations land in each band with these seeds
PSI_MEAN_SHIFT = {
    "NO_DRIFT": 0.1,    # PSI ~0.009
    "LOW": 0.35,        # PSI ~0.11
    "MEDIUM": 0.5,      # PSI ~0.23
    "HIGH": 0.6,        # PSI ~0.33
    "CRITICAL": 0.9,    # PSI ~0.74
}

SAMPLES = 20000


@functools.lru_cache(maxsize=None)
def _reference_and_base():

    reference = np.random.default_rng(42).normal(0, 1, SAMPLES)
    base = np.random.default_rng(7).normal(0, 1, SAMPLES)

    return reference, base


@functools.lru_cache(maxsize=None)
def statistical_rows(band):
    """Real analyze_feature rows: one feature in `band`, one unshifted."""

    reference, base = _reference_and_base()
    shifted = base + PSI_MEAN_SHIFT[band]

    ref_df = pd.DataFrame({"trip_distance": reference, "pickup_hour": reference})
    cur_df = pd.DataFrame({"trip_distance": shifted, "pickup_hour": reference})

    rows = [
        analyze_feature(ref_df, cur_df, "trip_distance"),
        analyze_feature(ref_df, cur_df, "pickup_hour"),
    ]

    # Precondition: the numeric input really produced the intended band
    assert rows[0]["severity"] == band, rows[0]
    assert rows[1]["severity"] == "NO_DRIFT", rows[1]

    return tuple(tuple(row.items()) for row in rows)


@functools.lru_cache(maxsize=None)
def prediction_result(band):
    """Real run_prediction_drift output for predictions shifted into `band`."""

    reference, base = _reference_and_base()

    result = run_prediction_drift(reference, base + PSI_MEAN_SHIFT[band])

    assert result["severity"] == band, result

    return tuple(result.items())


# Relative importance drop of trip_distance, per SHAP band. LOW needs the
# rank trigger to count as drifted: 100 -> 85 falls from rank 1 to 3.
SHAP_RELATIVE_DROP = {
    "STABLE": 0.03,
    "LOW_SHIFT": 0.15,
    "MEDIUM_SHIFT": 0.30,
    "HIGH_SHIFT": 0.60,
    "CRITICAL_SHIFT": 0.90,
}

SHAP_REFERENCE = {
    "trip_distance": 100.0,
    "pickup_hour": 95.0,
    "DOLocationID": 90.0,
    "pickup_month": 50.0,
}


@functools.lru_cache(maxsize=None)
def shap_rows(band, ignored_drift=True):
    """
    Real analyze_shap_drift rows with trip_distance in `band`. By default
    pickup_month (an IGNORED_FEATURES member) also collapses to
    CRITICAL_SHIFT, to prove ignored features cannot push toward RETRAIN.
    """

    current = dict(SHAP_REFERENCE)
    current["trip_distance"] = 100.0 * (1 - SHAP_RELATIVE_DROP[band])

    if ignored_drift:
        current["pickup_month"] = 5.0

    rows = shap_drift.analyze_shap_drift(
        SHAP_REFERENCE,
        current,
        shap_drift.get_rankings(SHAP_REFERENCE),
        shap_drift.get_rankings(current),
    )

    by_feature = {row["feature"]: row for row in rows}

    assert by_feature["trip_distance"]["severity"] == band
    assert by_feature["trip_distance"]["is_drifted"] is (band != "STABLE")

    if ignored_drift:
        assert "pickup_month" in IGNORED_FEATURES
        assert by_feature["pickup_month"]["severity"] == "CRITICAL_SHIFT"
        assert by_feature["pickup_month"]["is_drifted"] is True

    return tuple(tuple(row.items()) for row in rows)


# ============================================================
# Run the real scorer __main__ against those reports
# ============================================================

@pytest.fixture
def pipeline_mocks(monkeypatch):
    """
    Fresh MagicMocks on the conftest stub modules, recording call order.
    run_retraining_pipeline returns a promote-shaped dict like the real one.
    """

    calls = []

    def recorder(name, return_value=None):
        def side_effect(*args, **kwargs):
            calls.append(name)
            return return_value
        return MagicMock(name=name, side_effect=side_effect)

    repository = sys.modules["src.database.drift_repository"]
    rag = sys.modules["src.rag.monitoring_updater"]
    orchestrator = sys.modules["src.training.orchestrator"]

    mocks = {
        "insert_drift_scores": recorder("insert_drift_scores"),
        "insert_monitoring_run": recorder("insert_monitoring_run", 42),
        "upsert_monitoring_run": recorder("upsert_monitoring_run", True),
        "run_retraining_pipeline": recorder("run_retraining_pipeline", {
            "promoted": True, "new_version": "8", "champion_version": "7",
        }),
    }

    for name in ("insert_drift_scores", "insert_monitoring_run"):
        monkeypatch.setattr(repository, name, mocks[name])

    monkeypatch.setattr(rag, "upsert_monitoring_run",
                        mocks["upsert_monitoring_run"])
    monkeypatch.setattr(orchestrator, "run_retraining_pipeline",
                        mocks["run_retraining_pipeline"])

    mocks["calls"] = calls

    return mocks


def run_scorer(tmp_path, monkeypatch, statistical, shap, prediction):
    """Write the three reports into tmp_path/reports and run __main__."""

    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)

    frames = {
        "statistical": pd.DataFrame([dict(row) for row in statistical]),
        "shap": pd.DataFrame([dict(row) for row in shap]),
        "prediction": pd.DataFrame([dict(prediction)]),
    }

    # The detectors' own output has exactly the headers the scorer reads
    assert list(frames["statistical"].columns) == STATISTICAL_COLUMNS
    assert list(frames["shap"].columns) == SHAP_COLUMNS
    assert list(frames["prediction"].columns) == PREDICTION_COLUMNS

    for kind, frame in frames.items():
        frame.to_csv(reports / f"{kind}_drift.csv", index=False)

    monkeypatch.chdir(tmp_path)

    runpy.run_path(SCORER_FILE, run_name="__main__")

    with open(reports / "drift_summary.json") as f:
        return json.load(f)


def run_bands(tmp_path, monkeypatch, stat_band, shap_band, pred_band,
              ignored_drift=True):

    return run_scorer(
        tmp_path, monkeypatch,
        statistical_rows(stat_band),
        shap_rows(shap_band, ignored_drift),
        prediction_result(pred_band),
    )


# ============================================================
# (A) Enumeration: derive which real combinations reach RETRAIN
# ============================================================

def test_enumerate_band_combinations(tmp_path, monkeypatch, pipeline_mocks):
    """
    Every statistical x SHAP x prediction band combination (125), from
    real detector output through the real scorer. The RETRAIN set is
    derived here, not assumed.
    """

    outcomes = {}

    for stat_band, shap_band, pred_band in itertools.product(
        PSI_BANDS, SHAP_BANDS, PSI_BANDS
    ):
        summary = run_bands(
            tmp_path, monkeypatch, stat_band, shap_band, pred_band
        )

        # Each layer scored what its detector produced
        assert summary["statistical_score"] == PSI_BAND_SCORE[stat_band]
        assert summary["shap_score"] == SHAP_BAND_SCORE[shap_band]
        assert summary["prediction_score"] == PSI_BAND_SCORE[pred_band]

        outcomes[(stat_band, shap_band, pred_band)] = summary

    retrain = {k: v for k, v in outcomes.items() if v["action"] == "RETRAIN"}
    others = {k: v for k, v in outcomes.items() if v["action"] != "RETRAIN"}

    # RETRAIN is reachable from genuine inputs
    assert retrain, "no band combination reaches RETRAIN"

    # The action edge sits exactly at 0.75 and is reached exactly
    assert all(v["overall_score"] >= 0.75 for v in retrain.values())
    assert all(v["overall_score"] < 0.75 for v in others.values())
    assert min(v["overall_score"] for v in retrain.values()) == 0.75

    # The pipeline fired once per RETRAIN combination and never otherwise
    assert pipeline_mocks["run_retraining_pipeline"].call_count == len(retrain)


# ============================================================
# (A) Concrete RETRAIN cases, documenting how they are reached
# ============================================================

def test_retrain_reached_from_real_reports(tmp_path, monkeypatch, pipeline_mocks):

    summary = run_bands(tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL")

    assert summary == {
        "statistical_score": 0.75,
        "shap_score": 0.5,
        "prediction_score": 1.0,
        "overall_score": 0.75,
        "action": "RETRAIN",
    }


def test_all_layers_critical_reaches_full_score(
    tmp_path, monkeypatch, pipeline_mocks
):
    summary = run_bands(
        tmp_path, monkeypatch, "CRITICAL", "CRITICAL_SHIFT", "CRITICAL"
    )

    assert summary["overall_score"] == 1.0
    assert summary["action"] == "RETRAIN"


@pytest.mark.parametrize(
    "bands, scores, overall, action",
    [
        # One band below the edge on the prediction layer
        (("HIGH", "MEDIUM_SHIFT", "HIGH"), (0.75, 0.5, 0.75), 0.675, "ALERT"),
        # The highest reachable score that is still below the edge
        (("MEDIUM", "CRITICAL_SHIFT", "HIGH"), (0.5, 1.0, 0.75), 0.725, "ALERT"),
        # One band up on prediction: exactly on the edge
        (("HIGH", "MEDIUM_SHIFT", "CRITICAL"), (0.75, 0.5, 1.0), 0.75, "RETRAIN"),
    ],
    ids=["below-0.675", "below-0.725", "at-0.75"],
)
def test_retrain_trigger_edge(
    tmp_path, monkeypatch, pipeline_mocks, bands, scores, overall, action
):
    summary = run_bands(tmp_path, monkeypatch, *bands)

    assert (
        summary["statistical_score"],
        summary["shap_score"],
        summary["prediction_score"],
    ) == scores
    assert summary["overall_score"] == overall
    assert summary["action"] == action


def test_ignored_feature_drift_cannot_reach_retrain(
    tmp_path, monkeypatch, pipeline_mocks
):
    """
    pickup_month at CRITICAL_SHIFT is drifted but ignored. Had it counted
    (SHAP 1.0) this would be 0.3 + 0.3 + 0.3 = 0.9 RETRAIN; instead SHAP
    contributes 0.
    """

    summary = run_bands(tmp_path, monkeypatch, "HIGH", "STABLE", "CRITICAL")

    assert summary["shap_score"] == 0
    assert summary["overall_score"] == 0.6
    assert summary["action"] == "ALERT"
    pipeline_mocks["run_retraining_pipeline"].assert_not_called()


# ============================================================
# (B) RETRAIN is the branch that drives the orchestrator
# ============================================================

def test_retrain_action_calls_pipeline_once(tmp_path, monkeypatch, pipeline_mocks):

    run_bands(tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL")

    pipeline_mocks["run_retraining_pipeline"].assert_called_once_with(
        triggered_reason="drift_detected"
    )


@pytest.mark.parametrize(
    "bands",
    [
        ("NO_DRIFT", "STABLE", "NO_DRIFT"),          # WAIT
        ("LOW", "LOW_SHIFT", "LOW"),                 # MONITOR
        ("HIGH", "MEDIUM_SHIFT", "HIGH"),            # ALERT, just below
    ],
    ids=["WAIT", "MONITOR", "ALERT"],
)
def test_non_retrain_actions_never_call_pipeline(
    tmp_path, monkeypatch, pipeline_mocks, bands
):
    summary = run_bands(tmp_path, monkeypatch, *bands)

    assert summary["action"] != "RETRAIN"
    pipeline_mocks["run_retraining_pipeline"].assert_not_called()


def test_monitoring_run_saved_and_indexed_before_retraining(
    tmp_path, monkeypatch, pipeline_mocks
):
    run_bands(tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL")

    assert pipeline_mocks["calls"] == [
        "insert_drift_scores",
        "insert_monitoring_run",
        "upsert_monitoring_run",
        "run_retraining_pipeline",
    ]


def test_retraining_failure_does_not_crash_scorer(
    tmp_path, monkeypatch, pipeline_mocks
):
    """__main__ catches a pipeline failure; the run and summary survive."""

    pipeline_mocks["run_retraining_pipeline"].side_effect = RuntimeError("boom")

    summary = run_bands(tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL")

    assert summary["action"] == "RETRAIN"
    pipeline_mocks["run_retraining_pipeline"].assert_called_once()
    pipeline_mocks["insert_monitoring_run"].assert_called_once()
