"""
No raw secret reaches logs or stdout when retraining fails.

The orchestrator used logger.exception(...) and drift_scorer printed
f"... {exc}", both emitting the raw exception text. They now log a
redacted traceback / print a redacted one-line description. The
exception object itself still propagates unchanged.
"""

import logging

import pytest

from src.common.error_redaction import (
    REDACTED,
    describe_error,
    redacted_traceback,
)

# Real orchestrator with fake collaborators
from test_retraining_events import fakes, orchestrator  # noqa: F401

# Real drift_scorer __main__ over genuine detector reports
from test_retrain_reachability import pipeline_mocks, run_bands  # noqa: F401


API_KEY = "nvapi-ABCDEFGHIJKLMNOP123456"
BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl123"
DB_PASSWORD = "Sup3rS3cretPw"

SECRETS = (API_KEY, BEARER, DB_PASSWORD)


def _secret_error():
    """An exception whose message AND chained cause carry secrets."""

    try:
        raise ConnectionError(
            f"postgresql://sentinel:{DB_PASSWORD}@postgres:5432/sentinel_db"
        )
    except ConnectionError as cause:
        try:
            raise RuntimeError(
                f"embedding call failed: NVIDIA_API_KEY={API_KEY} "
                f"Authorization: Bearer {BEARER}"
            ) from cause
        except RuntimeError as error:
            return error


def _assert_no_secrets(text):
    for secret in SECRETS:
        assert secret not in text, secret


# ============================================================
# Helpers
# ============================================================

def test_redacted_traceback_keeps_structure_but_not_secrets():

    text = redacted_traceback(_secret_error())

    _assert_no_secrets(text)
    assert text.startswith("Traceback (most recent call last):")
    assert "ConnectionError: postgresql://sentinel:" in text
    assert "The above exception was the direct cause" in text
    assert "RuntimeError: embedding call failed" in text
    assert REDACTED in text


def test_describe_error_is_one_redacted_line():

    text = describe_error(_secret_error())

    _assert_no_secrets(text)
    assert text.startswith("RuntimeError: embedding call failed")
    assert "\n" not in text


def test_helpers_do_not_modify_the_exception():

    error = _secret_error()
    raw = str(error)

    redacted_traceback(error)
    describe_error(error)

    assert str(error) == raw
    assert API_KEY in str(error)


# ============================================================
# Orchestrator logs
# ============================================================

def test_pipeline_failure_log_is_redacted(orchestrator, fakes, caplog, capsys):

    original = _secret_error()
    fakes.retrain_main.side_effect = original

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError) as excinfo:
            orchestrator.run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

    # Same object, raw message untouched
    assert excinfo.value is original
    assert API_KEY in str(excinfo.value)

    _assert_no_secrets(caplog.text)
    captured = capsys.readouterr()
    _assert_no_secrets(captured.out + captured.err)

    # Still useful for debugging
    assert "Automated retraining pipeline failed" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "RuntimeError: embedding call failed" in caplog.text

    # No log record carries the raw exception for a handler to format
    assert all(record.exc_info is None for record in caplog.records)


def test_failed_record_insert_log_is_redacted(orchestrator, fakes, caplog):

    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")
    fakes.insert_event.side_effect = ConnectionError(
        f"postgresql://sentinel:{DB_PASSWORD}@postgres:5432/db refused"
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError, match="optuna blew up"):
            orchestrator.run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

    _assert_no_secrets(caplog.text)
    assert "Could not record failed retraining attempt" in caplog.text
    assert "postgres:5432/db refused" in caplog.text


def test_failed_index_log_is_redacted(orchestrator, fakes, caplog):

    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")
    fakes.upsert_event.side_effect = RuntimeError(
        f"chroma auth failed: Authorization: Bearer {BEARER}"
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError, match="optuna blew up"):
            orchestrator.run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

    _assert_no_secrets(caplog.text)
    assert "Could not index failed retraining event" in caplog.text


def test_success_path_logs_no_errors(orchestrator, fakes, caplog):

    with caplog.at_level(logging.DEBUG):
        result = orchestrator.run_retraining_pipeline(
            triggered_reason="drift_detected"
        )

    assert result["promoted"] is True
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "AUTOMATED RETRAINING RESULT: PROMOTED" in caplog.text


# ============================================================
# drift_scorer __main__ print
# ============================================================

def test_scorer_print_is_redacted(
    tmp_path, monkeypatch, pipeline_mocks, capsys, caplog
):
    pipeline_mocks["run_retraining_pipeline"].side_effect = _secret_error()

    with caplog.at_level(logging.DEBUG):
        summary = run_bands(
            tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL"
        )

    captured = capsys.readouterr()

    assert summary["action"] == "RETRAIN"
    _assert_no_secrets(captured.out + captured.err + caplog.text)

    assert "Retraining pipeline failed: RuntimeError: embedding call failed" \
        in captured.out
    assert REDACTED in captured.out
    assert "Champion remains unchanged" in captured.out


def test_scorer_success_output_unchanged(
    tmp_path, monkeypatch, pipeline_mocks, capsys
):
    run_bands(tmp_path, monkeypatch, "HIGH", "MEDIUM_SHIFT", "CRITICAL")

    out = capsys.readouterr().out

    assert "Retraining pipeline failed" not in out
    assert "Challenger model PROMOTED." in out
