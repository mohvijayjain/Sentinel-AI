"""
Locks the 0.4 / 0.3 / 0.3 weighting contract:

    overall = 0.4 * statistical + 0.3 * shap + 0.3 * prediction

asserted through get_action. drift_scorer computes the overall score
inline under `if __name__ == "__main__"`, so the formula is rebuilt
here from the module's own WEIGHTS.
"""

import pytest


# (statistical, shap, prediction, overall, action)
CONTRACT = [
    (1.0, 0, 0, 0.40, "MONITOR"),
    (0, 1.0, 0, 0.30, "MONITOR"),
    (0, 0, 1.0, 0.30, "MONITOR"),
    (1.0, 1.0, 0, 0.70, "ALERT"),
    (1.0, 0, 1.0, 0.70, "ALERT"),
    (0, 1.0, 1.0, 0.60, "ALERT"),        # pure concept drift
    (0.75, 0.5, 1.0, 0.75, "RETRAIN"),   # minimal trigger
    (0.75, 0.75, 0.75, 0.75, "RETRAIN"), # minimal trigger
    (0.75, 1.0, 0.5, 0.75, "RETRAIN"),   # minimal trigger
    (1.0, 1.0, 1.0, 1.00, "RETRAIN"),
]

IDS = [f"stat{s}-shap{h}-pred{p}" for s, h, p, _, _ in CONTRACT]


def _overall(drift_scorer, statistical, shap, prediction):

    weights = drift_scorer.WEIGHTS

    return round(
        statistical * weights["statistical"]
        + shap * weights["shap"]
        + prediction * weights["prediction"],
        4,
    )


def test_weights_are_locked(drift_scorer):

    assert drift_scorer.WEIGHTS == {
        "statistical": 0.4,
        "shap": 0.3,
        "prediction": 0.3,
    }


@pytest.mark.parametrize(
    "statistical, shap, prediction, overall, action", CONTRACT, ids=IDS
)
def test_overall_band_contract(
    drift_scorer, statistical, shap, prediction, overall, action
):
    score = _overall(drift_scorer, statistical, shap, prediction)

    assert score == overall
    assert drift_scorer.get_action(score) == action


def test_all_zero_is_wait(drift_scorer):

    assert drift_scorer.get_action(_overall(drift_scorer, 0, 0, 0)) == "WAIT"


# ============================================================
# Same contract, end to end from report CSVs
# ============================================================

STAT_LABEL = {0: "NO_DRIFT", 0.25: "LOW", 0.5: "MEDIUM",
              0.75: "HIGH", 1.0: "CRITICAL"}
SHAP_LABEL = {0: "STABLE", 0.25: "LOW_SHIFT", 0.5: "MEDIUM_SHIFT",
              0.75: "HIGH_SHIFT", 1.0: "CRITICAL_SHIFT"}


@pytest.mark.parametrize(
    "statistical, shap, prediction, overall, action", CONTRACT, ids=IDS
)
def test_overall_band_contract_from_reports(
    drift_scorer, write_report, monkeypatch, caplog,
    statistical, shap, prediction, overall, action,
):
    monkeypatch.setattr(drift_scorer, "STATISTICAL_PATH", write_report(
        "statistical", [{"severity": STAT_LABEL[statistical]}]
    ))
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", write_report(
        # shap == 0 means the detector found no drift; a drifted STABLE
        # row would be floored to 0.25 by calculate_shap_score
        "shap", [{"is_drifted": shap > 0, "severity": SHAP_LABEL[shap]}]
    ))
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", write_report(
        "prediction", [{"severity": STAT_LABEL[prediction]}]
    ))

    score = _overall(
        drift_scorer,
        drift_scorer.calculate_statistical_score(),
        drift_scorer.calculate_shap_score(),
        drift_scorer.calculate_prediction_score(),
    )

    assert score == overall
    assert drift_scorer.get_action(score) == action

    # Every label in the pipeline was recognized, none silently zeroed
    assert caplog.records == []
