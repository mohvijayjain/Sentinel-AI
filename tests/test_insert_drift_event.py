"""
insert_drift_event writes one drift_events row in a single transaction.

Replaces the root scratch script test_repository.py, which called
insert_drift_event(["pickup_month"], "RETRAIN", "reports/drift_summary.json")
against the configured database as soon as pytest imported it, and
asserted nothing. Same call here, against a fake engine.
"""

import pytest

from test_repository_success_reporting import (  # noqa: F401
    TransactionalEngine,
    load_repository,
)


class RecordingEngine(TransactionalEngine):
    """Also records each statement and its bound parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statements = []

    def execute(self, query, params=None):
        self.statements.append((str(query), params))
        return super().execute(query, params)


def test_inserts_one_row_and_commits(load_repository):

    engine = RecordingEngine()
    repository = load_repository(engine)

    returned = repository.insert_drift_event(
        drifted_features=["pickup_month"],
        action="RETRAIN",
        report_path="reports/drift_summary.json",
    )

    assert returned is None
    assert engine.events == ["begin", "commit"]

    [(sql, params)] = engine.statements
    assert "INSERT INTO drift_events" in sql
    assert params == {
        "features": "pickup_month",
        "action": "RETRAIN",
        "report": "reports/drift_summary.json",
    }


def test_features_are_comma_joined(load_repository):

    engine = RecordingEngine()

    load_repository(engine).insert_drift_event(
        drifted_features=["trip_distance", "pickup_hour"],
        action="ALERT",
        report_path="reports/drift_summary.json",
    )

    assert engine.statements[0][1]["features"] == "trip_distance,pickup_hour"


def test_failed_insert_rolls_back_and_raises(load_repository):

    engine = RecordingEngine(fail_on_execute=RuntimeError("insert failed"))

    with pytest.raises(RuntimeError, match="insert failed"):
        load_repository(engine).insert_drift_event(
            drifted_features=["pickup_month"],
            action="RETRAIN",
            report_path="reports/drift_summary.json",
        )

    assert engine.events == ["begin", "rollback"]
