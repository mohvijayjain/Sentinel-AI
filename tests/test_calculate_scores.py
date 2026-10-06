"""
calculate_statistical_score / calculate_shap_score /
calculate_prediction_score, run against temp CSVs that carry the real
report headers. The module-level *_PATH constants are monkeypatched per
test, so the real reports/ directory is never read.
"""

import pytest

from src.monitoring.shap_drift import (
    IGNORED_FEATURES as SHAP_IGNORED_FEATURES
)


BANDS = [
    ("NO_DRIFT", 0),
    ("LOW", 0.25),
    ("MEDIUM", 0.5),
    ("HIGH", 0.75),
    ("CRITICAL", 1.0),
]

SHAP_BANDS = [
    ("STABLE", 0),
    ("LOW_SHIFT", 0.25),
    ("MEDIUM_SHIFT", 0.5),
    ("HIGH_SHIFT", 0.75),
    ("CRITICAL_SHIFT", 1.0),
]


# ============================================================
# Statistical
# ============================================================

@pytest.mark.parametrize("label, expected", BANDS)
def test_statistical_max_reaches_each_band(
    drift_scorer, write_report, monkeypatch, label, expected
):
    # Target band sits among NO_DRIFT rows, not first, so max() matters
    path = write_report("statistical", [
        {"feature": "a", "severity": "NO_DRIFT"},
        {"feature": "b", "severity": label},
        {"feature": "c", "severity": "NO_DRIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    assert drift_scorer.calculate_statistical_score() == expected


def test_statistical_takes_max_over_mixed_rows(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("statistical", [
        {"feature": "a", "severity": "LOW"},
        {"feature": "b", "severity": "HIGH"},
        {"feature": "c", "severity": "MEDIUM"},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    assert drift_scorer.calculate_statistical_score() == 0.75


def test_statistical_empty_file_scores_zero(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("statistical", [])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    assert drift_scorer.calculate_statistical_score() == 0


# ============================================================
# SHAP
# ============================================================

def test_shap_drifted_medium_shift_scores_half(
    drift_scorer, write_report, monkeypatch
):
    """Regression for the MEDIUM_shift casing bug (used to score 0)."""

    # pickup_month is in IGNORED_FEATURES, so use a feature that counts
    path = write_report("shap", [
        {"feature": "pickup_hour", "is_drifted": True,
         "severity": "MEDIUM_SHIFT"},
        {"feature": "trip_distance", "is_drifted": False,
         "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0.5


# Score of a single is_drifted == True row: as SHAP_BANDS, except a
# drifted STABLE row (rank-only drift) is floored at LOW_SHIFT
DRIFTED_SHAP_SCORES = [
    (label, 0.25 if label == "STABLE" else score)
    for label, score in SHAP_BANDS
]


@pytest.mark.parametrize("label, expected", DRIFTED_SHAP_SCORES)
def test_shap_drifted_row_reaches_each_band(
    drift_scorer, write_report, monkeypatch, label, expected
):
    path = write_report("shap", [
        {"feature": "a", "is_drifted": True, "severity": label},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == expected


def test_shap_ignores_non_drifted_rows(
    drift_scorer, write_report, monkeypatch
):
    # The CRITICAL row is not drifted, so only the LOW row counts
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "severity": "CRITICAL_SHIFT"},
        {"feature": "b", "is_drifted": True, "severity": "LOW_SHIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0.25


def test_shap_none_drifted_scores_zero(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "severity": "HIGH_SHIFT"},
        {"feature": "b", "is_drifted": False, "severity": "MEDIUM_SHIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


def test_shap_all_stable_scores_zero(
    drift_scorer, write_report, monkeypatch
):
    # Stable and not drifted everywhere. A drifted STABLE row is floored
    # at LOW_SHIFT instead, see test_shap_rank_only_drift_floors_to_low
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "severity": "STABLE"},
        {"feature": "b", "is_drifted": False, "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


def test_shap_non_drifted_row_never_contributes(
    drift_scorer, write_report, monkeypatch
):
    # Detector said not drifted: ignored even with the top severity
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "severity": "CRITICAL_SHIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


@pytest.mark.parametrize("feature", SHAP_IGNORED_FEATURES)
def test_shap_ignored_feature_never_contributes(
    drift_scorer, write_report, monkeypatch, feature
):
    path = write_report("shap", [
        {"feature": feature, "is_drifted": True, "severity": "CRITICAL_SHIFT"},
        {"feature": "pickup_hour", "is_drifted": True, "severity": "LOW_SHIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0.25


def test_shap_only_ignored_feature_drifted_scores_zero(
    drift_scorer, write_report, monkeypatch
):
    """The live pickup_month MEDIUM_SHIFT case: no longer scores 0.5."""

    path = write_report("shap", [
        {"feature": "pickup_month", "is_drifted": True,
         "severity": "MEDIUM_SHIFT"},
        {"feature": "trip_distance", "is_drifted": False,
         "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


def test_shap_ignored_features_come_from_detector(drift_scorer):
    """Scorer must use the detector's list, not its own copy."""

    from src.monitoring import shap_drift

    assert drift_scorer.SHAP_IGNORED_FEATURES is shap_drift.IGNORED_FEATURES


def test_shap_rank_only_drift_floors_to_low(
    drift_scorer, write_report, monkeypatch
):
    """Detector flagged drift (rank) but magnitude is STABLE -> 0.25."""

    path = write_report("shap", [
        {"feature": "a", "is_drifted": True, "rank_shift": 3,
         "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0.25


def test_shap_stable_floor_never_applies_to_non_drifted(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "rank_shift": 1,
         "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


def test_shap_stable_floor_never_applies_to_ignored_feature(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("shap", [
        {"feature": "pickup_month", "is_drifted": True, "rank_shift": 2,
         "severity": "STABLE"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


@pytest.mark.parametrize("label, expected", SHAP_BANDS[1:])
def test_shap_floor_does_not_change_real_severities(
    drift_scorer, write_report, monkeypatch, label, expected
):
    # A rank-only STABLE row alongside must not lift or lower these
    path = write_report("shap", [
        {"feature": "a", "is_drifted": True, "severity": "STABLE"},
        {"feature": "b", "is_drifted": True, "severity": label},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == expected


def test_shap_empty_file_scores_zero(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("shap", [])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    assert drift_scorer.calculate_shap_score() == 0


# ============================================================
# Prediction
# ============================================================

@pytest.mark.parametrize("label, expected", BANDS)
def test_prediction_reaches_each_band(
    drift_scorer, write_report, monkeypatch, label, expected
):
    path = write_report("prediction", [{"psi": 0.3, "severity": label}])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    assert drift_scorer.calculate_prediction_score() == expected


def test_prediction_reads_only_first_row(
    drift_scorer, write_report, monkeypatch
):
    path = write_report("prediction", [
        {"severity": "LOW"},
        {"severity": "CRITICAL"},
    ])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    assert drift_scorer.calculate_prediction_score() == 0.25
