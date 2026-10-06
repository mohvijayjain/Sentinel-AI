"""
NaN PSI policy. Deliberately the opposite of the empty-report rule:

    empty report -> score 0 + warning   (missing signal, tolerated)
    NaN PSI      -> ValueError          (corrupt data, surfaced)

The detector still labels a NaN PSI "UNKNOWN" (get_psi_severity is
unchanged); the scorer refuses to turn that into a 0.
"""

import logging
import math

import pytest

from src.monitoring.stastical_drift import (
    PSI_HIGH,
    PSI_LOW,
    PSI_MEDIUM,
    PSI_NO_DRIFT,
    get_psi_severity,
)


NAN_MESSAGE = "PSI is NaN; drift report .* contains invalid PSI data."


# ============================================================
# NaN PSI raises, and never scores 0
# ============================================================

def test_statistical_nan_psi_raises(drift_scorer, write_report, monkeypatch):

    path = write_report("statistical", [
        {"feature": "a", "psi": 0.01, "severity": "NO_DRIFT"},
        {"feature": "b", "psi": math.nan, "severity": "UNKNOWN"},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    with pytest.raises(ValueError, match=NAN_MESSAGE):
        drift_scorer.calculate_statistical_score()


def test_prediction_nan_psi_raises(drift_scorer, write_report, monkeypatch):

    path = write_report("prediction", [{"psi": math.nan, "severity": "UNKNOWN"}])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    with pytest.raises(ValueError, match=NAN_MESSAGE):
        drift_scorer.calculate_prediction_score()


def test_nan_psi_error_names_the_report(drift_scorer, write_report, monkeypatch):

    path = write_report("prediction", [{"psi": math.nan, "severity": "UNKNOWN"}])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    with pytest.raises(ValueError) as excinfo:
        drift_scorer.calculate_prediction_score()

    assert repr(path) in str(excinfo.value)


@pytest.mark.parametrize("severity", ["UNKNOWN", "NO_DRIFT", "CRITICAL"])
def test_nan_psi_raises_whatever_the_severity_label(
    drift_scorer, write_report, monkeypatch, severity
):
    # The PSI value is authoritative; no label can launder a NaN
    path = write_report("statistical", [
        {"feature": "a", "psi": math.nan, "severity": severity},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    with pytest.raises(ValueError, match=NAN_MESSAGE):
        drift_scorer.calculate_statistical_score()


@pytest.mark.parametrize(
    "kind, path_attr, scorer",
    [
        ("statistical", "STATISTICAL_PATH", "calculate_statistical_score"),
        ("prediction", "PREDICTION_PATH", "calculate_prediction_score"),
    ],
)
def test_nan_psi_never_becomes_zero(
    drift_scorer, write_report, monkeypatch, kind, path_attr, scorer
):
    path = write_report(kind, [{"psi": math.nan, "severity": "UNKNOWN"}])
    monkeypatch.setattr(drift_scorer, path_attr, path)

    result = None

    with pytest.raises(ValueError):
        result = getattr(drift_scorer, scorer)()

    assert result is None


def test_prediction_only_row_zero_psi_is_checked(
    drift_scorer, write_report, monkeypatch
):
    # The scorer reads row 0 only; an unread later row does not matter
    path = write_report("prediction", [
        {"psi": 0.3, "severity": "HIGH"},
        {"psi": math.nan, "severity": "UNKNOWN"},
    ])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    assert drift_scorer.calculate_prediction_score() == 0.75


# ============================================================
# Valid PSI is unchanged
# ============================================================

def test_psi_thresholds_unchanged():

    assert (PSI_NO_DRIFT, PSI_LOW, PSI_MEDIUM, PSI_HIGH) == (
        0.10, 0.20, 0.25, 0.50
    )


@pytest.mark.parametrize(
    "psi, severity, score",
    [
        (0.0, "NO_DRIFT", 0),
        (0.099999, "NO_DRIFT", 0),
        (0.10, "LOW", 0.25),
        (0.199999, "LOW", 0.25),
        (0.20, "MEDIUM", 0.5),
        (0.249999, "MEDIUM", 0.5),
        (0.25, "HIGH", 0.75),
        (0.499999, "HIGH", 0.75),
        (0.50, "CRITICAL", 1.0),
        (math.inf, "CRITICAL", 1.0),   # constant reference -> inf, not NaN
    ],
)
def test_valid_psi_scores_through_existing_bands(
    drift_scorer, write_report, monkeypatch, caplog, psi, severity, score
):
    assert get_psi_severity(psi) == severity

    statistical = write_report("statistical", [
        {"feature": "a", "psi": psi, "severity": severity},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", statistical)

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.calculate_statistical_score() == score

    prediction = write_report("prediction", [{"psi": psi, "severity": severity}])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", prediction)

    assert drift_scorer.calculate_prediction_score() == score
    assert caplog.records == []


# ============================================================
# Empty and NaN coexist without cancelling each other
# ============================================================

@pytest.mark.parametrize(
    "kind, path_attr, scorer",
    [
        ("statistical", "STATISTICAL_PATH", "calculate_statistical_score"),
        ("prediction", "PREDICTION_PATH", "calculate_prediction_score"),
    ],
)
def test_empty_report_scores_zero_but_nan_report_raises(
    drift_scorer, write_report, monkeypatch, caplog, kind, path_attr, scorer
):
    # Empty first: tolerated, warned, 0
    monkeypatch.setattr(drift_scorer, path_attr, write_report(kind, []))

    with caplog.at_level(logging.WARNING):
        assert getattr(drift_scorer, scorer)() == 0

    assert any("Empty drift report" in r.getMessage() for r in caplog.records)

    # Same detector, one NaN row: raised, not swallowed by the empty guard
    monkeypatch.setattr(drift_scorer, path_attr, write_report(
        kind, [{"psi": math.nan, "severity": "UNKNOWN"}]
    ))

    with pytest.raises(ValueError, match=NAN_MESSAGE):
        getattr(drift_scorer, scorer)()
