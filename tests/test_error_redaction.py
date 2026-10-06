"""
Secret redaction for persisted error text (src/common/error_redaction).

A failed retraining attempt's error_message goes to PostgreSQL and is
embedded into ChromaDB, where RAG can surface it. These tests pin:
  * what is redacted and what is deliberately kept readable,
  * redact-then-bound ordering and idempotence,
  * the original exception is untouched,
  * no raw secret reaches the Postgres params or the Chroma payload,
    through the orchestrator AND when calling either store directly.
"""

import pytest

from src.common.error_redaction import (
    ERROR_MESSAGE_MAX_LENGTH,
    REDACTED,
    redact_secrets,
    safe_error_message,
)

# Real orchestrator / repository / RAG fixtures (fake infra underneath)
from test_retraining_events import (  # noqa: F401
    EVENT_ID,
    _chroma_record,
    _postgres_row,
    engine,
    fakes,
    orchestrator,
    rag,
    repository,
    store,
    wired,
)


SECRET = "s3cr3tP4ss"


# ============================================================
# Ordinary error text stays readable
# ============================================================

@pytest.mark.parametrize(
    "message",
    [
        "KeyError: 'trip_distance' not in index; got 11 features",
        "ValueError: rmse=4.1 > 3.9 for pickup_hour at row 1234",
        'FATAL: password authentication failed for user "sentinel"',
        "connection to server at \"postgres\" (172.18.0.2), port 5432 failed",
        "SyntaxError: unexpected token ')' at line 3",
        "max_tokens=5 exceeded; token_count=12",
        "Bearer authentication failed for MLflow at http://mlflow:5000/api/2.0",
        "File /app/src/training/retrain.py, line 42, in main",
        "Retraining did not return a valid MLflow run_id.",
    ],
)
def test_normal_errors_unchanged(message):

    assert redact_secrets(message) == message
    assert safe_error_message(message) == message


# ============================================================
# Secrets are redacted, surrounding context kept
# ============================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        # API keys
        ("NVIDIA_API_KEY=nvapi-abcdefghijklmnop1234 missing",
         f"NVIDIA_API_KEY={REDACTED} missing"),
        ('{"api_key": "abc123xyz", "model": "m"}',
         f'{{"api_key": {REDACTED}, "model": "m"}}'),
        ("invalid key sk-proj1234567890abcdefgh",
         f"invalid key {REDACTED}"),
        ("AWS key AKIAABCDEFGHIJKLMNOP denied",
         f"AWS key {REDACTED} denied"),
        # Bearer tokens / Authorization headers
        ("401: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig123 expired",
         f"401: Bearer {REDACTED} expired"),
        ("Authorization: Bearer abc.def123",
         f"Authorization: {REDACTED}"),
        ("headers={'Authorization': 'Basic dXNlcjpwYXNz'}",
         f"headers={{'Authorization': '{REDACTED}'}}"),
        # Connection strings
        ("postgresql://sentinel:sentinel123@postgres:5432/sentinel_db",
         f"postgresql://sentinel:{REDACTED}@postgres:5432/sentinel_db"),
        ("host=db port=5432 user=sentinel password=hunter2 dbname=x",
         f"host=db port=5432 user=sentinel password={REDACTED} dbname=x"),
        ("Server=x;Uid=sa;Pwd=hunter2;",
         f"Server=x;Uid=sa;Pwd={REDACTED};"),
        # Token / secret env values
        ("GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaa",
         f"GITHUB_TOKEN={REDACTED}"),
        ("client_secret: 'very secret value' rejected",
         f"client_secret: {REDACTED} rejected"),
    ],
)
def test_secrets_redacted(raw, expected):

    assert safe_error_message(raw) == expected


def test_connection_string_keeps_host_and_user():

    safe = safe_error_message(
        f"could not connect: postgresql://admin:{SECRET}@db.internal:5432/x"
    )

    assert SECRET not in safe
    assert "admin" in safe and "db.internal:5432" in safe


# ============================================================
# Redact, then bound
# ============================================================

@pytest.mark.parametrize("offset", range(470, 500, 3))
def test_long_error_bounded_and_redacted_at_any_cut(offset):
    """A secret straddling the cut is redacted first, never sliced."""

    raw = "x" * offset + f" password={SECRET} trailing context"

    safe = safe_error_message(raw)

    assert len(safe) <= ERROR_MESSAGE_MAX_LENGTH
    assert SECRET not in safe
    for size in range(4, len(SECRET) + 1):
        assert SECRET[:size] not in safe


def test_multiline_secret_is_redacted():

    safe = safe_error_message("Authorization:\n    Bearer abc123def456\nnext")

    assert "abc123def456" not in safe
    assert "\n" not in safe


@pytest.mark.parametrize(
    "raw",
    [
        f"password={SECRET}",
        "Authorization: Bearer abc123def456",
        f"postgresql://u:{SECRET}@h/db",
        "x" * 480 + f" password={SECRET} tail",
        "x" * 488 + " Bearer abc123def456ghi",
        "plain error text",
    ],
)
def test_idempotent(raw):
    """Re-applying (repository / RAG layers) never changes safe text."""

    once = safe_error_message(raw)

    assert safe_error_message(once) == once


# ============================================================
# Orchestrator: sanitized copy persisted, exception untouched
# ============================================================

def _secret_failure():
    return RuntimeError(
        f"could not reach postgresql://sentinel:{SECRET}@postgres:5432/db"
    )


def test_original_exception_object_and_message_unchanged(orchestrator, fakes):

    original = _secret_failure()
    raw_message = str(original)
    fakes.retrain_main.side_effect = original

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert excinfo.value is original
    assert str(excinfo.value) == raw_message
    assert SECRET in str(excinfo.value)


def test_orchestrator_passes_sanitized_message_to_both_stores(
    orchestrator, fakes
):
    fakes.retrain_main.side_effect = _secret_failure()

    with pytest.raises(RuntimeError):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    inserted = fakes.insert_event.call_args.kwargs["error_message"]
    indexed = fakes.upsert_event.call_args.kwargs["error_message"]

    assert inserted == indexed
    assert SECRET not in inserted
    assert inserted.startswith("RuntimeError: could not reach postgresql://")
    assert REDACTED in inserted


def test_successful_events_unaffected(orchestrator, fakes):

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert fakes.insert_event.call_args.kwargs.get("error_message") is None
    assert fakes.upsert_event.call_args.kwargs.get("error_message") is None


# ============================================================
# End to end: no raw-secret path to either store
# ============================================================

def test_no_raw_secret_reaches_postgres_or_chroma(wired):

    wired.retrain_main.side_effect = _secret_failure()

    with pytest.raises(RuntimeError):
        wired.run(triggered_reason="drift_detected")

    _, params = _postgres_row(wired.engine)
    record = _chroma_record(wired.store)

    assert SECRET not in str(params)
    assert SECRET not in record["document"]
    assert SECRET not in str(record["metadata"])

    assert params["error_message"] == record["metadata"]["error_message"]
    assert REDACTED in record["document"]


def test_repository_sanitizes_direct_callers(repository, engine):
    """Defense in depth: a caller passing a raw secret is still safe."""

    repository.insert_retraining_event(
        "manual_retraining", None, None, None, None,
        triggered_at="2026-10-06T12:00:00+00:00",
        status="failed",
        error_message=f"boom password={SECRET}\n" + "y" * 600,
    )

    _, params = _postgres_row(engine)

    assert SECRET not in params["error_message"]
    assert len(params["error_message"]) <= ERROR_MESSAGE_MAX_LENGTH


def test_rag_sanitizes_direct_callers(rag, store):

    rag.upsert_retraining_event(
        EVENT_ID, "2026-10-06T12:00:00+00:00", "manual_retraining",
        None, None, None, None,
        status="failed",
        error_message=f"Authorization: Bearer {SECRET}123",
    )

    record = store.get(f"retraining_event:{EVENT_ID}")

    assert SECRET not in record["document"]
    assert SECRET not in str(record["metadata"])


def test_repository_keeps_none_for_completed_events(repository, engine):

    repository.insert_retraining_event(
        "drift_detected", 4.1, 4.6, True, "run-1",
        triggered_at="2026-10-06T12:00:00+00:00", status="promoted",
    )

    _, params = _postgres_row(engine)

    assert params["error_message"] is None
