"""
Empty drift reports: a missing, zero-byte or zero-row report means the
detector produced no signal. Each scorer warns, scores 0 and the
pipeline carries on. A report that has rows (even all-zero ones) is not
empty and must not take this path.
"""

import json
import logging
import os
import runpy
import shutil
import sys

import pytest


SCORERS = [
    ("statistical", "STATISTICAL_PATH", "calculate_statistical_score"),
    ("shap", "SHAP_PATH", "calculate_shap_score"),
    ("prediction", "PREDICTION_PATH", "calculate_prediction_score"),
]

SCORER_IDS = [kind for kind, _, _ in SCORERS]


def _warnings(caplog):

    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


# ============================================================
# Each scorer: empty -> 0 + warning naming the path
# ============================================================

@pytest.mark.parametrize("kind, path_attr, scorer", SCORERS, ids=SCORER_IDS)
def test_header_only_report_scores_zero_and_warns(
    drift_scorer, write_report, monkeypatch, caplog, kind, path_attr, scorer
):
    path = write_report(kind, [])
    monkeypatch.setattr(drift_scorer, path_attr, path)

    with caplog.at_level(logging.WARNING):
        assert getattr(drift_scorer, scorer)() == 0

    assert _warnings(caplog) == [
        f"Empty drift report {path!r}; scoring as 0."
    ]


@pytest.mark.parametrize("kind, path_attr, scorer", SCORERS, ids=SCORER_IDS)
def test_zero_byte_report_scores_zero_and_warns(
    drift_scorer, tmp_path, monkeypatch, caplog, kind, path_attr, scorer
):
    path = tmp_path / f"{kind}_drift.csv"
    path.write_text("")
    monkeypatch.setattr(drift_scorer, path_attr, str(path))

    with caplog.at_level(logging.WARNING):
        assert getattr(drift_scorer, scorer)() == 0

    assert _warnings(caplog) == [
        f"Empty drift report {str(path)!r}; scoring as 0."
    ]


@pytest.mark.parametrize("kind, path_attr, scorer", SCORERS, ids=SCORER_IDS)
def test_missing_report_scores_zero_and_warns(
    drift_scorer, tmp_path, monkeypatch, caplog, kind, path_attr, scorer
):
    path = str(tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(drift_scorer, path_attr, path)

    with caplog.at_level(logging.WARNING):
        assert getattr(drift_scorer, scorer)() == 0

    assert _warnings(caplog) == [
        f"Missing drift report {path!r}; scoring as 0."
    ]


# ============================================================
# Non-empty reports are not "empty", even when they score 0
# ============================================================

def test_all_no_drift_statistical_report_is_not_empty(
    drift_scorer, write_report, monkeypatch, caplog
):
    path = write_report("statistical", [
        {"feature": "a", "severity": "NO_DRIFT"},
        {"feature": "b", "severity": "NO_DRIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", path)

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.calculate_statistical_score() == 0

    assert _warnings(caplog) == []


def test_shap_report_with_no_drifted_rows_is_not_empty(
    drift_scorer, write_report, monkeypatch, caplog
):
    path = write_report("shap", [
        {"feature": "a", "is_drifted": False, "severity": "HIGH_SHIFT"},
    ])
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", path)

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.calculate_shap_score() == 0

    assert _warnings(caplog) == []


def test_no_drift_prediction_report_is_not_empty(
    drift_scorer, write_report, monkeypatch, caplog
):
    path = write_report("prediction", [{"psi": 0.01, "severity": "NO_DRIFT"}])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    with caplog.at_level(logging.WARNING):
        assert drift_scorer.calculate_prediction_score() == 0

    assert _warnings(caplog) == []


# ============================================================
# The full scorer pipeline continues past an empty report
# ============================================================

def _run_scorer_main(tmp_path, monkeypatch, write_report, reports):
    """
    Lay out reports/ in a temp cwd and run drift_scorer as __main__.
    `reports` maps kind -> rows, or None for a zero-byte file.
    DB / RAG / retraining calls hit the conftest stubs.
    """

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    for kind, rows in reports.items():

        target = reports_dir / f"{kind}_drift.csv"

        if rows is None:
            target.write_text("")
        else:
            shutil.move(write_report(kind, rows), target)

    monkeypatch.chdir(tmp_path)

    repository = sys.modules["src.database.drift_repository"]
    repository.insert_drift_scores.reset_mock()

    scorer_file = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "src", "monitoring", "drift_scorer.py",
    )

    runpy.run_path(scorer_file, run_name="__main__")

    with open(reports_dir / "drift_summary.json") as f:
        summary = json.load(f)

    return summary, repository.insert_drift_scores


@pytest.mark.parametrize("empty_statistical", [[], None],
                         ids=["header-only", "zero-byte"])
def test_pipeline_continues_with_empty_statistical_report(
    tmp_path, monkeypatch, write_report, caplog, empty_statistical
):
    with caplog.at_level(logging.WARNING):
        summary, insert_drift_scores = _run_scorer_main(
            tmp_path, monkeypatch, write_report, {
                "statistical": empty_statistical,
                "shap": [{"feature": "pickup_hour", "is_drifted": True,
                          "severity": "HIGH_SHIFT"}],
                "prediction": [{"psi": 0.22, "severity": "MEDIUM"}],
            },
        )

    # 0.4 * 0 + 0.3 * 0.75 + 0.3 * 0.5
    assert summary == {
        "statistical_score": 0,
        "shap_score": 0.75,
        "prediction_score": 0.5,
        "overall_score": 0.375,
        "action": "MONITOR",
    }

    # The DB insert still ran, with nothing to persist
    insert_drift_scores.assert_called_once()
    assert len(insert_drift_scores.call_args.args[0]) == 0

    assert any("Empty drift report" in m for m in _warnings(caplog))


def test_pipeline_continues_with_all_reports_empty(
    tmp_path, monkeypatch, write_report, caplog
):
    with caplog.at_level(logging.WARNING):
        summary, _ = _run_scorer_main(
            tmp_path, monkeypatch, write_report, {
                "statistical": [],
                "shap": [],
                "prediction": None,
            },
        )

    assert summary["overall_score"] == 0.0
    assert summary["action"] == "WAIT"
