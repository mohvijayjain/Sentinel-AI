"""
Detector-side severity contracts:

  - stastical_drift.get_psi_severity, which now backs both statistical
    and prediction severity.
  - shap_drift.analyze_shap_drift severity banding and is_drifted.

Every `<` cutoff is tested on both sides.
"""

import math

import pandas as pd
import pytest

from src.monitoring import prediction_drift
from src.monitoring import shap_drift
from src.monitoring.stastical_drift import get_psi_severity


# ============================================================
# get_psi_severity
# ============================================================

@pytest.mark.parametrize(
    "psi, expected",
    [
        (0.0, "NO_DRIFT"),
        (0.099999, "NO_DRIFT"),
        (0.10, "LOW"),
        (0.199999, "LOW"),
        (0.20, "MEDIUM"),
        (0.249999, "MEDIUM"),
        (0.25, "HIGH"),
        (0.499999, "HIGH"),
        (0.50, "CRITICAL"),
        (5.0, "CRITICAL"),
    ],
)
def test_psi_severity_bands(psi, expected):

    assert get_psi_severity(psi) == expected


def test_psi_severity_nan_is_unknown():

    assert get_psi_severity(math.nan) == "UNKNOWN"


@pytest.mark.parametrize("psi", [0.0, 0.10, 0.20, 0.25, 0.50])
def test_psi_labels_are_known_to_scorer(drift_scorer, caplog, psi):
    """Every finite PSI band must be a key the scorer recognizes."""

    drift_scorer.severity_score(get_psi_severity(psi))

    assert caplog.records == []


def test_prediction_drift_uses_shared_psi_bands():

    assert prediction_drift.get_psi_severity is get_psi_severity


@pytest.mark.parametrize("shift", [0.0, 0.5, 5.0])
def test_prediction_drift_severity_matches_its_psi(shift):
    """run_prediction_drift labels its own PSI with the shared bands."""

    reference = [float(i % 100) for i in range(2000)]
    current = [value + shift * 10 for value in reference]

    results = prediction_drift.run_prediction_drift(reference, current)

    assert results["severity"] == get_psi_severity(results["psi"])
    assert results["severity"] not in ("DRIFT",)


def test_prediction_drift_identical_distributions_no_drift():

    values = [float(i % 50) for i in range(1000)]

    results = prediction_drift.run_prediction_drift(values, values)

    assert results["severity"] == "NO_DRIFT"


# ============================================================
# analyze_shap_drift
# ============================================================

# 1e8 + 1e-10 == 1e8 in float64, so the epsilon guard in the relative
# shift does not skew results and every cutoff below is hit exactly.
REF_IMPORTANCE = 1e8


def _analyze_single(cur_importance, ref_rank=1, cur_rank=1):

    results = shap_drift.analyze_shap_drift(
        ref_importance={"f": REF_IMPORTANCE},
        cur_importance={"f": cur_importance},
        ref_rankings={"f": ref_rank},
        cur_rankings={"f": cur_rank},
    )

    assert len(results) == 1

    return results[0]


@pytest.mark.parametrize(
    "relative_shift, expected",
    [
        (0.0, "STABLE"),
        (0.099999, "STABLE"),
        (0.10, "LOW_SHIFT"),
        (0.249999, "LOW_SHIFT"),
        (0.25, "MEDIUM_SHIFT"),
        (0.499999, "MEDIUM_SHIFT"),
        (0.50, "HIGH_SHIFT"),
        (0.749999, "HIGH_SHIFT"),
        (0.75, "CRITICAL_SHIFT"),
        (2.0, "CRITICAL_SHIFT"),
    ],
)
def test_shap_severity_bands(relative_shift, expected):

    cur = REF_IMPORTANCE * (1 + relative_shift)

    assert _analyze_single(cur)["severity"] == expected


def test_shap_medium_band_is_exact_uppercase():
    """Regression for the MEDIUM_shift casing typo."""

    severity = _analyze_single(REF_IMPORTANCE * 1.3)["severity"]

    assert severity == "MEDIUM_SHIFT"
    assert severity.isupper()


def test_shap_banding_is_symmetric_for_decreases():

    result = _analyze_single(REF_IMPORTANCE * (1 - 0.3))

    assert result["direction"] == "DECREASED"
    assert result["severity"] == "MEDIUM_SHIFT"


@pytest.mark.parametrize(
    "relative_shift",
    [0.0, 0.10, 0.25, 0.50, 0.75],
)
def test_shap_labels_are_known_to_scorer(drift_scorer, caplog, relative_shift):

    severity = _analyze_single(
        REF_IMPORTANCE * (1 + relative_shift)
    )["severity"]

    drift_scorer.severity_score(severity)

    assert caplog.records == []


# --- is_drifted: relative-shift trigger (rank held constant) ---------

@pytest.mark.parametrize(
    "relative_shift, expected",
    [
        (0.0, False),
        (0.25, False),       # strict `>`: exactly 0.25 is not drifted
        (0.250001, True),
        (0.9, True),
    ],
)
def test_is_drifted_by_relative_shift_only(relative_shift, expected):

    result = _analyze_single(
        REF_IMPORTANCE * (1 + relative_shift),
        ref_rank=3,
        cur_rank=3,
    )

    assert result["rank_shift"] == 0
    assert result["is_drifted"] is expected


# --- is_drifted: rank-shift trigger (importance held constant) -------

@pytest.mark.parametrize(
    "ref_rank, cur_rank, expected",
    [
        (3, 3, False),
        (3, 4, False),
        (3, 5, True),        # rank_shift == 2 meets `>= 2`
        (5, 3, True),        # moving up counts the same as moving down
        (1, 9, True),
    ],
)
def test_is_drifted_by_rank_shift_only(ref_rank, cur_rank, expected):

    result = _analyze_single(
        REF_IMPORTANCE,
        ref_rank=ref_rank,
        cur_rank=cur_rank,
    )

    assert result["relative_shift_%"] == 0
    assert result["is_drifted"] is expected


def test_results_sorted_by_relative_shift_descending():

    results = shap_drift.analyze_shap_drift(
        ref_importance={"a": 1e8, "b": 1e8, "c": 1e8},
        cur_importance={"a": 1.05e8, "b": 1.6e8, "c": 1.3e8},
        ref_rankings={"a": 1, "b": 2, "c": 3},
        cur_rankings={"a": 1, "b": 2, "c": 3},
    )

    assert [r["feature"] for r in results] == ["b", "c", "a"]


# ============================================================
# Detector -> CSV -> scorer: the 25% boundary end to end
# ============================================================

def _score_detector_output(
    drift_scorer, tmp_path, monkeypatch, cur_importance,
    ref_rank=3, cur_rank=3,
):
    """Run the real detector, write its rows as the report, score it."""

    results = shap_drift.analyze_shap_drift(
        ref_importance={"f": REF_IMPORTANCE},
        cur_importance={"f": cur_importance},
        ref_rankings={"f": ref_rank},
        cur_rankings={"f": cur_rank},
    )

    path = tmp_path / "shap_drift.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    monkeypatch.setattr(drift_scorer, "SHAP_PATH", str(path))

    return results[0], drift_scorer.calculate_shap_score()


@pytest.mark.parametrize(
    "relative_shift, severity, is_drifted, score",
    [
        # Just below: LOW_SHIFT band, not drifted -> ignored
        (0.249999, "LOW_SHIFT", False, 0),
        # Exactly 0.25: MEDIUM_SHIFT band but `>` says not drifted -> ignored
        (0.25, "MEDIUM_SHIFT", False, 0),
        # Just above: drifted -> contributes its existing severity
        (0.250001, "MEDIUM_SHIFT", True, 0.5),
    ],
)
def test_quarter_boundary_detector_to_scorer(
    drift_scorer, tmp_path, monkeypatch,
    relative_shift, severity, is_drifted, score,
):
    row, shap_score = _score_detector_output(
        drift_scorer, tmp_path, monkeypatch,
        REF_IMPORTANCE * (1 + relative_shift),
    )

    assert row["severity"] == severity
    assert row["is_drifted"] is is_drifted
    assert shap_score == score


def test_rank_only_drift_detector_to_scorer(
    drift_scorer, tmp_path, monkeypatch,
):
    # No magnitude shift, rank moves by 2: drifted STABLE -> floored 0.25
    row, shap_score = _score_detector_output(
        drift_scorer, tmp_path, monkeypatch,
        REF_IMPORTANCE, ref_rank=3, cur_rank=5,
    )

    assert row["severity"] == "STABLE"
    assert row["is_drifted"] is True
    assert shap_score == 0.25


def test_below_quarter_with_rank_trigger_detector_to_scorer(
    drift_scorer, tmp_path, monkeypatch,
):
    # Under the magnitude cutoff but drifted by rank: scores its own band
    row, shap_score = _score_detector_output(
        drift_scorer, tmp_path, monkeypatch,
        REF_IMPORTANCE * 1.249999, ref_rank=3, cur_rank=5,
    )

    assert row["severity"] == "LOW_SHIFT"
    assert row["is_drifted"] is True
    assert shap_score == 0.25


# ============================================================
# Relative shift is the true ratio (no epsilon skew)
# ============================================================

def _analyze_pair(ref_importance, cur_importance, ref_rank=1, cur_rank=1):

    return shap_drift.analyze_shap_drift(
        ref_importance={"f": ref_importance},
        cur_importance={"f": cur_importance},
        ref_rankings={"f": ref_rank},
        cur_rankings={"f": cur_rank},
    )[0]


@pytest.mark.parametrize(
    "cur_importance, expected",
    [
        (1.10, "LOW_SHIFT"),       # exactly 0.10 -> not < 0.10
        (1.20, "LOW_SHIFT"),       # 0.20 is not a SHAP cutoff
        (1.25, "MEDIUM_SHIFT"),    # exactly 0.25 -> not < 0.25
        (1.50, "HIGH_SHIFT"),      # exactly 0.50 -> not < 0.50
        (1.75, "CRITICAL_SHIFT"),  # exactly 0.75 -> not < 0.75
        (0.75, "MEDIUM_SHIFT"),    # exact 0.25 decrease, same band
    ],
)
def test_unit_reference_boundaries_are_exact(cur_importance, expected):

    assert _analyze_pair(1.0, cur_importance)["severity"] == expected


def test_unit_reference_quarter_is_not_drifted():
    """Exact 0.25 is MEDIUM_SHIFT but `> 0.25` keeps it not drifted."""

    result = _analyze_pair(1.0, 1.25)

    assert result["relative_shift_%"] == 25.0
    assert result["is_drifted"] is False


def test_small_reference_no_longer_misclassified():
    """ref=0.001: the old `ref + 1e-10` put an exact 25% into LOW_SHIFT."""

    ref, cur = 0.001, 0.00125

    old_relative_shift = abs(cur - ref) / (ref + 1e-10)
    assert old_relative_shift < 0.25     # old formula: LOW_SHIFT

    result = _analyze_pair(ref, cur)

    assert result["relative_shift_%"] == 25.0
    assert result["severity"] == "MEDIUM_SHIFT"


def test_non_boundary_row_unchanged():

    result = _analyze_pair(2.0, 2.6)

    assert result["relative_shift_%"] == 30.0
    assert result["severity"] == "MEDIUM_SHIFT"
    assert result["direction"] == "INCREASED"
    assert result["is_drifted"] is True


# ============================================================
# Zero reference importance
# ============================================================

@pytest.mark.parametrize("ref_importance", [0.0, -0.0, 0])
def test_zero_reference_zero_current_is_stable(ref_importance):

    result = _analyze_pair(ref_importance, 0.0)

    assert result["relative_shift_%"] == 0.0
    assert result["severity"] == "STABLE"
    assert result["is_drifted"] is False


def test_zero_reference_missing_current_is_stable():
    """A feature absent from cur_importance defaults to 0."""

    result = shap_drift.analyze_shap_drift(
        ref_importance={"f": 0.0},
        cur_importance={},
        ref_rankings={"f": 1},
        cur_rankings={"f": 1},
    )[0]

    assert result["severity"] == "STABLE"
    assert result["is_drifted"] is False


@pytest.mark.parametrize("cur_importance", [1e-12, 0.5, 1e6])
def test_zero_reference_nonzero_current_is_critical(cur_importance):

    result = _analyze_pair(0.0, cur_importance)

    assert result["severity"] == "CRITICAL_SHIFT"
    assert result["is_drifted"] is True
    assert result["relative_shift_%"] == shap_drift.ZERO_REFERENCE_SHIFT * 100


@pytest.mark.parametrize("ref_importance", [0.0, -0.0, 5e-324, 1e-12, 1.0, 1e300])
@pytest.mark.parametrize("cur_importance", [0.0, 5e-324, 1e-12, 1.0, 1e300])
def test_no_divide_by_zero_for_any_input(ref_importance, cur_importance):

    result = _analyze_pair(ref_importance, cur_importance)

    assert not math.isnan(result["relative_shift_%"])
    assert result["relative_shift_%"] >= 0
