"""
Real src/training/orchestrator.py run_retraining_pipeline with the
expensive work faked: no Optuna, MLflow, Postgres, Chroma or NVIDIA.

conftest replaces src.training.orchestrator in sys.modules with a stub
(drift_scorer imports it), and the real module imports retrain/promote,
which load mlflow, lightgbm and dotenv. So the fixture:

  1. registers fake src.training.retrain / src.training.promote modules,
  2. adds insert_retraining_event / upsert_retraining_event to the
     conftest DB / RAG stub modules,
  3. executes the real orchestrator.py source under a private name,
     leaving the conftest stub in place for everything else,
  4. monkeypatches the four collaborators on that module per test.
"""

import importlib.util
import logging
import os
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock

import pytest


ORCHESTRATOR_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir,
    "src", "training", "orchestrator.py",
))

RUN_ID = "challenger-run-abc123"
EVENT_ID = 17

RETRAIN_RESULT = {
    "run_id": RUN_ID,
    "metrics": {"rmse": 4.10, "mae": 2.05, "r2": 0.912},
    "model": object(),
}

PROMOTED = {
    "promoted": True,
    "mlflow_run_id": RUN_ID,
    "new_model_rmse": 4.10,
    "new_model_mae": 2.05,
    "new_model_r2": 0.912,
    "champion_rmse": 4.60,
    "champion_mae": 2.31,
    "champion_r2": 0.894,
    "champion_version": "7",
    "new_version": "8",
}

REJECTED = {
    **PROMOTED,
    "promoted": False,
    "new_model_rmse": 4.90,
    "new_model_mae": 2.48,
    "new_model_r2": 0.881,
    "new_version": None,
}


# ============================================================
# Load the real orchestrator against fakes
# ============================================================

@pytest.fixture
def orchestrator(monkeypatch):

    fake_retrain = types.ModuleType("src.training.retrain")
    fake_retrain.main = MagicMock(name="retrain.main")

    fake_promote = types.ModuleType("src.training.promote")
    fake_promote.main = MagicMock(name="promote.main")

    monkeypatch.setitem(sys.modules, "src.training.retrain", fake_retrain)
    monkeypatch.setitem(sys.modules, "src.training.promote", fake_promote)

    monkeypatch.setattr(
        sys.modules["src.database.drift_repository"],
        "insert_retraining_event", MagicMock(), raising=False,
    )
    monkeypatch.setattr(
        sys.modules["src.rag.monitoring_updater"],
        "upsert_retraining_event", MagicMock(), raising=False,
    )

    spec = importlib.util.spec_from_file_location(
        "sentinel_real_orchestrator", ORCHESTRATOR_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def fakes(orchestrator, monkeypatch):
    """
    Fakes for the four collaborators, all appending to one call log so
    order can be asserted. Defaults: retrain succeeds, promote promotes.
    """

    calls = []

    def recorder(name, return_value):
        def side_effect(*args, **kwargs):
            calls.append(name)
            return return_value
        return MagicMock(name=name, side_effect=side_effect)

    retrain_main = recorder("retrain.main", RETRAIN_RESULT)
    promote_model = recorder("promote_model", PROMOTED)
    insert_event = recorder("insert_retraining_event", EVENT_ID)
    upsert_event = recorder("upsert_retraining_event", True)

    monkeypatch.setattr(orchestrator.retrain, "main", retrain_main)
    monkeypatch.setattr(orchestrator, "promote_model", promote_model)
    monkeypatch.setattr(orchestrator, "insert_retraining_event", insert_event)
    monkeypatch.setattr(orchestrator, "upsert_retraining_event", upsert_event)

    return types.SimpleNamespace(
        calls=calls,
        retrain_main=retrain_main,
        promote_model=promote_model,
        insert_event=insert_event,
        upsert_event=upsert_event,
    )


def test_real_orchestrator_loaded_without_heavy_deps(orchestrator):

    assert os.path.samefile(orchestrator.__file__, ORCHESTRATOR_FILE)
    assert callable(orchestrator._require_promotion_dict)

    # The conftest stub is still what drift_scorer sees
    assert sys.modules["src.training.orchestrator"].__file__.startswith("<stub ")

    for heavy in ("mlflow", "optuna"):
        assert heavy not in sys.modules


# ============================================================
# Step 3: call chain and data flow
# ============================================================

def test_call_order(orchestrator, fakes):

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    assert fakes.calls == [
        "retrain.main",
        "promote_model",
        "insert_retraining_event",
        "upsert_retraining_event",
    ]


def test_each_collaborator_called_once(orchestrator, fakes):

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    fakes.retrain_main.assert_called_once_with()
    fakes.promote_model.assert_called_once_with(RUN_ID)
    fakes.insert_event.assert_called_once()
    fakes.upsert_event.assert_called_once()


@pytest.mark.parametrize("promotion", [PROMOTED, REJECTED],
                         ids=["promoted", "rejected"])
def test_promote_result_flows_into_events(
    orchestrator, fakes, promotion
):
    fakes.promote_model.side_effect = None
    fakes.promote_model.return_value = promotion

    orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    expected_event = {
        "triggered_reason": "drift_detected",
        "new_model_rmse": promotion["new_model_rmse"],
        "champion_rmse": promotion["champion_rmse"],
        "promoted": promotion["promoted"],
        "mlflow_run_id": promotion["mlflow_run_id"],
        # Issue #2: outcome recorded explicitly
        "status": "promoted" if promotion["promoted"] else "rejected",
    }

    fakes.insert_event.assert_called_once()

    insert_kwargs = dict(fakes.insert_event.call_args.kwargs)
    upsert_kwargs = dict(fakes.upsert_event.call_args.kwargs)
    triggered_at = upsert_kwargs.pop("triggered_at")

    # Issue #3: Postgres gets the same triggered_at as Chroma
    assert insert_kwargs.pop("triggered_at") == triggered_at

    assert insert_kwargs == expected_event
    assert upsert_kwargs == {"event_id": EVENT_ID, **expected_event}

    # One UTC ISO timestamp shared by the event
    assert datetime.fromisoformat(triggered_at).utcoffset().total_seconds() == 0


def test_default_triggered_reason_is_manual(orchestrator, fakes):

    orchestrator.run_retraining_pipeline()

    assert (
        fakes.insert_event.call_args.kwargs["triggered_reason"]
        == "manual_retraining"
    )


# ============================================================
# Step 3: _require_promotion_dict guard
# ============================================================

def test_promotion_guard_passes_valid_dict_through(orchestrator):

    assert orchestrator._require_promotion_dict(PROMOTED) is PROMOTED


@pytest.mark.parametrize("bad", [1, 0, None, "promoted", [PROMOTED]],
                         ids=["int-1", "int-0", "None", "str", "list"])
def test_promotion_guard_rejects_non_dict(orchestrator, bad):

    with pytest.raises(TypeError, match="must return a dict"):
        orchestrator._require_promotion_dict(bad)


@pytest.mark.parametrize("missing", ["promoted", "mlflow_run_id",
                                     "champion_version", "new_version"])
def test_promotion_guard_rejects_missing_key(orchestrator, missing):

    incomplete = {k: v for k, v in PROMOTED.items() if k != missing}

    with pytest.raises(KeyError, match=missing):
        orchestrator._require_promotion_dict(incomplete)


@pytest.mark.parametrize(
    "bad, error",
    [(1, TypeError), ({"promoted": True}, KeyError)],
    ids=["legacy-int", "partial-dict"],
)
def test_bad_promote_return_aborts_before_events(
    orchestrator, fakes, bad, error
):
    fakes.promote_model.side_effect = None
    fakes.promote_model.return_value = bad

    with pytest.raises(error):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    # Issue #2: no completed event, one failed event instead
    _assert_one_failed_event(fakes, mlflow_run_id=RUN_ID)


def _assert_one_failed_event(fakes, mlflow_run_id):
    """Exactly one failed event, with nothing invented, in both stores."""

    fakes.insert_event.assert_called_once()
    fakes.upsert_event.assert_called_once()

    insert_kwargs = fakes.insert_event.call_args.kwargs
    upsert_kwargs = fakes.upsert_event.call_args.kwargs

    assert insert_kwargs["status"] == upsert_kwargs["status"] == "failed"
    assert insert_kwargs["mlflow_run_id"] == mlflow_run_id
    assert upsert_kwargs["event_id"] == EVENT_ID

    for key in ("promoted", "new_model_rmse", "champion_rmse"):
        assert insert_kwargs[key] is None


# ============================================================
# Step 4: outcomes
# ============================================================

def test_promotion_outcome(orchestrator, fakes):

    result = orchestrator.run_retraining_pipeline(
        triggered_reason="drift_detected"
    )

    assert result["promoted"] is True
    assert result["new_version"] == "8"
    assert result["champion_version"] == "7"
    assert result["run_id"] == RUN_ID
    assert result["event_id"] == EVENT_ID
    assert result["retrain_metrics"] == RETRAIN_RESULT["metrics"]

    assert fakes.insert_event.call_args.kwargs["promoted"] is True
    assert fakes.upsert_event.call_args.kwargs["promoted"] is True


def test_rejection_outcome(orchestrator, fakes, caplog):

    fakes.promote_model.side_effect = None
    fakes.promote_model.return_value = REJECTED

    with caplog.at_level(logging.WARNING):
        result = orchestrator.run_retraining_pipeline(
            triggered_reason="drift_detected"
        )

    # Promotion absent: no new version, champion reported unchanged
    assert result["promoted"] is False
    assert result["new_version"] is None
    assert result["champion_version"] == "7"

    # A rejected run is still a recorded retraining event
    assert fakes.insert_event.call_args.kwargs["promoted"] is False
    assert fakes.upsert_event.call_args.kwargs["promoted"] is False

    assert "AUTOMATED RETRAINING RESULT: REJECTED" in caplog.text


def test_result_carries_every_promotion_field(orchestrator, fakes):

    result = orchestrator.run_retraining_pipeline(
        triggered_reason="drift_detected"
    )

    for key in orchestrator._REQUIRED_PROMOTION_KEYS:
        assert result[key] == PROMOTED[key]


def test_retrain_failure_reraises_and_records_nothing(
    orchestrator, fakes, caplog
):
    fakes.retrain_main.side_effect = RuntimeError("optuna blew up")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="optuna blew up"):
            orchestrator.run_retraining_pipeline(
                triggered_reason="drift_detected"
            )

    fakes.promote_model.assert_not_called()

    # Issue #2: the failed attempt is recorded (no run was produced)
    _assert_one_failed_event(fakes, mlflow_run_id=None)

    assert "Automated retraining pipeline failed" in caplog.text


def test_promote_failure_reraises_and_records_nothing(orchestrator, fakes):

    fakes.promote_model.side_effect = RuntimeError("mlflow unreachable")

    with pytest.raises(RuntimeError, match="mlflow unreachable"):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    fakes.retrain_main.assert_called_once()

    # Issue #2: recorded as failed, keeping the real challenger run id
    _assert_one_failed_event(fakes, mlflow_run_id=RUN_ID)


@pytest.mark.parametrize(
    "retrain_result",
    [{"metrics": {}}, {"run_id": None, "metrics": {}}, {"run_id": ""}],
    ids=["missing", "None", "empty"],
)
def test_missing_run_id_aborts_before_promotion(
    orchestrator, fakes, retrain_result
):
    fakes.retrain_main.side_effect = None
    fakes.retrain_main.return_value = retrain_result

    with pytest.raises(ValueError, match="valid MLflow run_id"):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    fakes.promote_model.assert_not_called()

    # Issue #2: recorded as failed; no run id invented
    _assert_one_failed_event(fakes, mlflow_run_id=None)


def test_insert_failure_skips_rag_index(orchestrator, fakes):

    fakes.insert_event.side_effect = RuntimeError("postgres down")

    with pytest.raises(RuntimeError, match="postgres down"):
        orchestrator.run_retraining_pipeline(triggered_reason="drift_detected")

    fakes.upsert_event.assert_not_called()
