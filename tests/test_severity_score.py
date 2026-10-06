"""
drift_scorer.severity_score: label -> numeric score, and the warning
path for labels the mapping does not know.
"""

import logging
import sys

import pytest


@pytest.mark.parametrize(
    "label, expected",
    [
        ("NO_DRIFT", 0),
        ("STABLE", 0),
        ("LOW", 0.25),
        ("LOW_SHIFT", 0.25),
        ("MEDIUM", 0.5),
        ("MEDIUM_SHIFT", 0.5),
        ("HIGH", 0.75),
        ("HIGH_SHIFT", 0.75),
        ("CRITICAL", 1.0),
        ("CRITICAL_SHIFT", 1.0),
    ],
)
def test_known_labels_map_exactly(drift_scorer, caplog, label, expected):

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.severity_score(label) == expected

    # A recognized label must never take the warning path
    assert caplog.records == []


@pytest.mark.parametrize(
    "label",
    [
        # Regression: the old binary prediction label
        "DRIFT",
        # Regression: the old SHAP casing typo
        "MEDIUM_shift",
        # Other plausible vocabulary drift
        "UNKNOWN",
        "medium",
        "",
    ],
)
def test_unknown_label_scores_zero_and_warns(drift_scorer, caplog, label):

    with caplog.at_level(logging.WARNING):
        score = drift_scorer.severity_score(label)

    assert score == 0

    warnings = [
        record for record in caplog.records
        if record.levelno == logging.WARNING
    ]

    assert len(warnings) == 1
    assert repr(label) in warnings[0].getMessage()


def test_non_string_label_warns(drift_scorer, caplog):
    """A NaN read from a CSV with an empty severity cell is also unknown."""

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.severity_score(float("nan")) == 0

    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_infra_modules_are_stubbed(drift_scorer):
    """Guard: importing drift_scorer must not load real infra modules."""

    for name in (
        "src.database.drift_repository",
        "src.rag.monitoring_updater",
        "src.training.orchestrator",
    ):
        assert sys.modules[name].__file__.startswith("<stub ")

    assert "mlflow" not in sys.modules
