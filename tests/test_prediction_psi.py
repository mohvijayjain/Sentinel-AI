"""
prediction_drift.calculate_psi: every current value must be binned.

The old code used finite equal-width edges over the reference range, so
np.histogram silently dropped current values outside it: half the
predictions at 10000 scored PSI 0 (NO_DRIFT), and an all-outside batch
scored inf. The outer edges are now -inf / +inf.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.monitoring.prediction_drift import calculate_psi, run_prediction_drift
from src.monitoring.stastical_drift import get_psi_severity


# Reference predictions spread evenly over 0..99
REFERENCE = np.arange(1000) % 100 * 1.0


def _legacy_psi(reference, current, bins=10):
    """The pre-fix algorithm, kept here only to prove what changed."""

    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # The legacy 0/0 when every value was dropped is the point here
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_pct = (ref_counts + 1e-10) / ref_counts.sum()
        cur_pct = (cur_counts + 1e-10) / cur_counts.sum()
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _half_replaced(value):
    return np.concatenate([REFERENCE[:500], np.full(500, value)])


# ============================================================
# 1. Current fully inside the reference range
# ============================================================

@pytest.mark.parametrize(
    "current",
    [
        REFERENCE[::-1],                       # same distribution
        REFERENCE % 50,                        # squeezed into lower half
        np.clip(REFERENCE + 20, 0, 99),        # piled up at the top
    ],
    ids=["same", "lower-half", "clipped-shift"],
)
def test_inside_range_results_unchanged(current):
    """In-range values bin identically, so PSI is bit-for-bit the same."""

    assert calculate_psi(REFERENCE, current) == _legacy_psi(REFERENCE, current)


def test_inside_range_concentration_is_detected():

    psi = calculate_psi(REFERENCE, REFERENCE % 50)

    assert math.isfinite(psi)
    assert get_psi_severity(psi) == "CRITICAL"


# ============================================================
# 2-5. Partially / fully outside the reference range
# ============================================================

@pytest.mark.parametrize(
    "current",
    [_half_replaced(-10000.0), _half_replaced(10000.0)],
    ids=["half-below", "half-above"],
)
def test_partially_outside_range_is_severe_drift(current):

    # The bug: these values were dropped and the batch looked identical
    assert _legacy_psi(REFERENCE, current) == pytest.approx(0, abs=1e-12)

    psi = calculate_psi(REFERENCE, current)

    assert psi > 0
    assert math.isfinite(psi)
    assert get_psi_severity(psi) == "CRITICAL"


@pytest.mark.parametrize(
    "current",
    [REFERENCE - 200, REFERENCE + 200],
    ids=["all-below", "all-above"],
)
def test_fully_outside_range_is_finite_critical(current):

    # The bug: every value dropped, 0/0 counts, PSI inf
    assert math.isinf(_legacy_psi(REFERENCE, current))

    psi = calculate_psi(REFERENCE, current)

    assert math.isfinite(psi)
    assert psi > 0
    assert get_psi_severity(psi) == "CRITICAL"


@pytest.mark.parametrize("outlier", [-500.0, 500.0], ids=["below", "above"])
def test_every_out_of_range_value_counts(outlier):
    """More mass outside the range means strictly more PSI."""

    # Appended, not swapped in, so the in-range part is unchanged
    psis = [
        calculate_psi(
            REFERENCE,
            np.concatenate([REFERENCE, np.full(n, outlier)]),
        )
        for n in (0, 10, 50, 200)
    ]

    assert psis[0] == pytest.approx(0, abs=1e-12)
    assert psis == sorted(psis)
    assert len(set(psis)) == len(psis)


def test_out_of_range_value_lands_in_edge_bin():
    """Just past the max is binned with the top values, not ignored."""

    at_max = np.concatenate([REFERENCE[:900], np.full(100, 99.0)])
    past_max = np.concatenate([REFERENCE[:900], np.full(100, 1e6)])

    assert calculate_psi(REFERENCE, past_max) == calculate_psi(REFERENCE, at_max)


def test_run_prediction_drift_flags_out_of_range_shift():

    results = run_prediction_drift(REFERENCE, _half_replaced(10000.0))

    assert results["psi"] > 0
    assert results["severity"] != "NO_DRIFT"


# ============================================================
# 6. Identical distributions
# ============================================================

@pytest.mark.parametrize(
    "values",
    [REFERENCE, np.random.default_rng(0).normal(50, 10, 5000)],
    ids=["uniform", "normal"],
)
def test_identical_inputs_score_zero(values):

    psi = calculate_psi(values, values.copy())

    assert psi == pytest.approx(0, abs=1e-12)
    assert get_psi_severity(psi) == "NO_DRIFT"


# ============================================================
# 7. Severity bands unchanged on genuine predictions
# ============================================================

@pytest.mark.parametrize(
    "mean_shift, band",
    [(0.1, "NO_DRIFT"), (0.35, "LOW"), (0.5, "MEDIUM"),
     (0.6, "HIGH"), (0.9, "CRITICAL")],
)
def test_bands_on_shifted_normal_predictions(mean_shift, band):

    reference = np.random.default_rng(42).normal(0, 1, 20000)
    current = np.random.default_rng(7).normal(0, 1, 20000) + mean_shift

    results = run_prediction_drift(reference, current)

    assert results["severity"] == band
    assert results["severity"] == get_psi_severity(results["psi"])


# ============================================================
# 8. NaN / empty input is not silently accepted
# ============================================================

@pytest.mark.parametrize(
    "reference, current",
    [
        (REFERENCE, []),
        ([], REFERENCE),
        (REFERENCE, [np.nan, np.nan]),
        ([np.nan], REFERENCE),
    ],
    ids=["empty-current", "empty-reference", "all-nan-current",
         "all-nan-reference"],
)
def test_no_usable_values_gives_nan(reference, current):

    psi = calculate_psi(reference, current)

    assert math.isnan(psi)
    assert get_psi_severity(psi) == "UNKNOWN"


def test_scattered_nan_predictions_are_ignored():

    with_nans = np.concatenate([REFERENCE, [np.nan] * 10])

    assert calculate_psi(REFERENCE, with_nans) == calculate_psi(
        REFERENCE, REFERENCE
    )


def test_nan_prediction_psi_is_refused_by_scorer(
    drift_scorer, write_report, monkeypatch
):
    psi = calculate_psi(REFERENCE, [])

    path = write_report("prediction", [
        {"psi": psi, "severity": get_psi_severity(psi)},
    ])
    monkeypatch.setattr(drift_scorer, "PREDICTION_PATH", path)

    with pytest.raises(ValueError, match="PSI is NaN"):
        drift_scorer.calculate_prediction_score()
