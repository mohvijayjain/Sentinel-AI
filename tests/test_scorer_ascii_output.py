"""
Console output can no longer stop the self-healing path.

drift_scorer __main__ printed emoji (and an em dash, arrows) between the
monitoring-run write, the Chroma index and the retraining trigger:

    insert_monitoring_run -> print("... saved -> ...")  <- non-ASCII
    upsert_monitoring_run -> print("... indexed ...")   <- non-ASCII
    if RETRAIN: print("... RETRAINING TRIGGERED")      <- non-ASCII
                run_retraining_pipeline(...)

On a stdout that cannot encode them (cp1252 console, C-locale container,
cron / CI pipe) print raised UnicodeEncodeError and the run died before
retraining. All output now goes through drift_scorer.report(), which
writes pure ASCII. The drift runner that produces the scorer's reports
had the same problem on its live path.

Every check forces a strict-ASCII stream itself, so nothing depends on
the host OS or its console encoding.
"""

import ast
import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest

from src.monitoring import shap_drift, stastical_drift

# Real drift_scorer __main__ over genuine detector reports, with the DB,
# Chroma and retraining calls on fresh MagicMocks
from test_retrain_reachability import pipeline_mocks, run_bands  # noqa: F401


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MONITORING = os.path.join(REPO_ROOT, "src", "monitoring")

RETRAIN_BANDS = ("HIGH", "MEDIUM_SHIFT", "CRITICAL")      # overall 0.75

EXPECTED_CALLS = [
    "insert_drift_scores",
    "insert_monitoring_run",
    "upsert_monitoring_run",
    "run_retraining_pipeline",
]


class StrictAsciiOutput:
    """stdout and stderr that raise on any non-ASCII character."""

    def __init__(self):
        self.out = self._stream()
        self.err = self._stream()

    @staticmethod
    def _stream():
        return io.TextIOWrapper(
            io.BytesIO(), encoding="ascii", errors="strict", write_through=True
        )

    @contextlib.contextmanager
    def active(self):
        # Entered in the test body: pytest re-installs its capture stdout
        # at the start of the call phase, so a fixture swap would not apply
        with contextlib.redirect_stdout(self.out), \
                contextlib.redirect_stderr(self.err):
            yield self

    @property
    def text(self):
        return self.out.buffer.getvalue().decode("ascii")


@pytest.fixture
def ascii_only():
    return StrictAsciiOutput()


# ============================================================
# The harness really is strict
# ============================================================

def test_strict_stream_rejects_the_old_output(ascii_only):
    """The previous scorer line would have raised here."""

    with ascii_only.active():
        with pytest.raises(UnicodeEncodeError):
            print("✅ Monitoring result indexed into ChromaDB")


# ============================================================
# RETRAIN still runs on an ASCII-only console
# ============================================================

def test_retrain_runs_with_ascii_only_output(
    tmp_path, monkeypatch, pipeline_mocks, ascii_only
):
    with ascii_only.active():
        summary = run_bands(tmp_path, monkeypatch, *RETRAIN_BANDS)

    assert summary["action"] == "RETRAIN"

    # Postgres write, Chroma index, then retraining, exactly once
    assert pipeline_mocks["calls"] == EXPECTED_CALLS
    pipeline_mocks["run_retraining_pipeline"].assert_called_once_with(
        triggered_reason="drift_detected"
    )

    # The reporting happened, in ASCII
    out = ascii_only.text
    assert "Monitoring run saved: run_id=42" in out
    assert "Monitoring result indexed into ChromaDB" in out
    assert "AUTOMATIC RETRAINING TRIGGERED" in out
    assert "Challenger model PROMOTED." in out
    assert "Sentinel AI Drift Pipeline Complete" in out


def test_rejection_reported_with_ascii_only_output(
    tmp_path, monkeypatch, pipeline_mocks, ascii_only
):
    pipeline_mocks["run_retraining_pipeline"].side_effect = None
    pipeline_mocks["run_retraining_pipeline"].return_value = {
        "promoted": False, "new_version": None, "champion_version": "7",
    }

    with ascii_only.active():
        run_bands(tmp_path, monkeypatch, *RETRAIN_BANDS)

    pipeline_mocks["run_retraining_pipeline"].assert_called_once()
    assert "Challenger model REJECTED." in ascii_only.text
    assert "Champion remains Version: 7" in ascii_only.text


def test_non_ascii_error_text_cannot_crash_the_scorer(
    tmp_path, monkeypatch, pipeline_mocks, ascii_only
):
    """
    Dynamic text is not under our control: a failure message with
    non-ASCII characters is escaped, not allowed to raise.
    """

    pipeline_mocks["run_retraining_pipeline"].side_effect = RuntimeError(
        "connexion refusée ✗"
    )

    with ascii_only.active():
        summary = run_bands(tmp_path, monkeypatch, *RETRAIN_BANDS)

    assert summary["action"] == "RETRAIN"
    # (the raising side_effect replaces the call recorder for the pipeline)
    assert pipeline_mocks["calls"] == EXPECTED_CALLS[:3]
    pipeline_mocks["run_retraining_pipeline"].assert_called_once_with(
        triggered_reason="drift_detected"
    )

    out = ascii_only.text
    assert "Retraining pipeline failed: RuntimeError: connexion refus" in out
    assert "\\xe9" in out and "\\u2717" in out
    assert "Champion remains unchanged" in out
    assert "Sentinel AI Drift Pipeline Complete" in out


def test_non_ascii_result_values_cannot_crash_the_scorer(
    tmp_path, monkeypatch, pipeline_mocks, ascii_only
):
    pipeline_mocks["run_retraining_pipeline"].side_effect = None
    pipeline_mocks["run_retraining_pipeline"].return_value = {
        "promoted": True, "new_version": "8", "champion_version": "7",
        "note": "déployé",
    }

    with ascii_only.active():
        run_bands(tmp_path, monkeypatch, *RETRAIN_BANDS)

    assert "note: d\\xe9ploy\\xe9" in ascii_only.text
    assert "Challenger model PROMOTED." in ascii_only.text


# ============================================================
# Action boundaries unchanged (under ASCII-only output too)
# ============================================================

@pytest.mark.parametrize(
    "bands, action, overall, retrain_calls",
    [
        (("NO_DRIFT", "STABLE", "NO_DRIFT"), "WAIT", 0.0, 0),
        (("LOW", "LOW_SHIFT", "LOW"), "MONITOR", 0.25, 0),
        (("HIGH", "MEDIUM_SHIFT", "HIGH"), "ALERT", 0.675, 0),
        (RETRAIN_BANDS, "RETRAIN", 0.75, 1),
    ],
    ids=["WAIT", "MONITOR", "ALERT", "RETRAIN"],
)
def test_only_retrain_triggers_retraining(
    tmp_path, monkeypatch, pipeline_mocks, ascii_only,
    bands, action, overall, retrain_calls,
):
    with ascii_only.active():
        summary = run_bands(tmp_path, monkeypatch, *bands)

    assert summary["action"] == action
    assert summary["overall_score"] == overall
    assert (
        pipeline_mocks["run_retraining_pipeline"].call_count == retrain_calls
    )

    # The monitoring run is saved and indexed for every action
    pipeline_mocks["insert_monitoring_run"].assert_called_once()
    pipeline_mocks["upsert_monitoring_run"].assert_called_once()

    if retrain_calls == 0:
        assert "No retraining required." in ascii_only.text
        assert f"Action: {action}" in ascii_only.text


# ============================================================
# report() itself
# ============================================================

def test_report_escapes_instead_of_raising(drift_scorer, ascii_only):

    with ascii_only.active():
        drift_scorer.report("✅ saved → run_id=1")
        drift_scorer.report()

    assert ascii_only.text.splitlines() == ["\\u2705 saved \\u2192 run_id=1", ""]


def test_report_leaves_ascii_untouched(drift_scorer, ascii_only):

    with ascii_only.active():
        drift_scorer.report("Monitoring result indexed into ChromaDB")

    assert ascii_only.text.strip() == "Monitoring result indexed into ChromaDB"


# ============================================================
# Drift runner live path (produces the scorer's reports)
# ============================================================

def test_statistical_run_prints_ascii_only(ascii_only):

    rng = np.random.default_rng(0)
    features = [f for f in stastical_drift.FEATURES if f != "pickup_month"]

    def frame(shift):
        return pd.DataFrame({
            f: (rng.integers(1, 5, 500) if f in stastical_drift.CATEGORICAL_FEATURES
                else rng.normal(shift, 1, 500))
            for f in features
        })

    with ascii_only.active():
        results, _, _ = stastical_drift.run_statistical_drift(frame(0), frame(1))

    assert len(results) == len(features)
    assert "Skipping: pickup_month" in ascii_only.text


def test_shap_comparison_prints_ascii_only(ascii_only):

    ref = {"trip_distance": 1.0, "pickup_hour": 0.5, "payment_type": 0.1}
    # shifts: 50%, 20%, 0%
    cur = {"trip_distance": 0.5, "pickup_hour": 0.6, "payment_type": 0.1}

    with ascii_only.active():
        shap_drift.print_shap_comparison(
            ref, cur, shap_drift.get_rankings(ref), shap_drift.get_rankings(cur)
        )

    lines = ascii_only.text.splitlines()
    by_feature = {line.split()[0]: line for line in lines[3:] if line.strip()}

    # Same thresholds as before (>25% HIGH, >10% WARN), ASCII glyphs
    assert by_feature["trip_distance"].endswith("HIGH")
    assert by_feature["pickup_hour"].endswith("WARN")
    assert by_feature["payment_type"].endswith("OK")


# ============================================================
# Static guards
# ============================================================

@pytest.mark.parametrize("filename", ["drift_scorer.py", "drift_runner.py"])
def test_live_path_modules_are_pure_ascii(filename):

    with open(os.path.join(MONITORING, filename), "rb") as f:
        data = f.read()

    offenders = [
        number for number, line in enumerate(data.splitlines(), 1)
        if any(byte > 127 for byte in line)
    ]

    assert offenders == []


@pytest.mark.parametrize(
    "filename, functions",
    [
        ("stastical_drift.py", ["run_statistical_drift"]),
        ("shap_drift.py", ["run_shap_drift", "print_shap_comparison",
                           "compute_shap_importance", "analyze_shap_drift"]),
        ("prediction_drift.py", ["run_prediction_drift", "generate_predictions"]),
    ],
)
def test_runner_called_detector_functions_are_pure_ascii(filename, functions):
    """
    String literals (anything that can be printed) only; comments cannot
    reach output. Standalone __main__ blocks are CLI-only, not checked.
    """

    with open(os.path.join(MONITORING, filename), encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found = {
        node.name: node
        for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    for name in functions:
        assert name in found, name
        non_ascii = [
            node.value for node in ast.walk(found[name])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and not node.value.isascii()
        ]
        assert non_ascii == [], (name, non_ascii)


def test_scorer_main_writes_only_through_report():

    with open(os.path.join(MONITORING, "drift_scorer.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())

    [main] = [
        node for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    ]

    bare_prints = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]

    assert bare_prints == []
