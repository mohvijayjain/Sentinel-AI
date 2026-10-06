"""
prediction_drift.calculate_js: same out-of-range fix as calculate_psi.

JS used finite equal-width edges over the reference range, so
np.histogram dropped current values outside it (half the predictions at
10000 -> JS 0; all outside -> NaN from 0/0). It now shares
_binned_counts with calculate_psi: identical edges, outer bins -inf/+inf.

js_divergence is reported in the prediction drift CSV only; the scorer
does not read it, so this does not move any drift score or action.
"""

import math

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from src.monitoring import prediction_drift
from src.monitoring.prediction_drift import calculate_js, calculate_psi


REFERENCE = np.arange(1000) % 100 * 1.0

# Squared JS distance with natural log is bounded by ln 2
JS_MAX = math.log(2)


def _legacy_js(reference, current, bins=10):
    """The pre-fix algorithm, kept here only to prove what changed."""

    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        ref_prob = (ref_counts + 1e-10) / ref_counts.sum()
        cur_prob = (cur_counts + 1e-10) / cur_counts.sum()
        return float(jensenshannon(ref_prob, cur_prob) ** 2)


def _half_replaced(value):
    return np.concatenate([REFERENCE[:500], np.full(500, value)])


# ============================================================
# Identical and in-range
# ============================================================

@pytest.mark.parametrize(
    "values",
    [REFERENCE, np.random.default_rng(0).normal(50, 10, 5000)],
    ids=["uniform", "normal"],
)
def test_identical_distributions_score_zero(values):

    assert calculate_js(values, values.copy()) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "current",
    [REFERENCE[::-1], REFERENCE % 50, np.clip(REFERENCE + 20, 0, 99)],
    ids=["same", "lower-half", "clipped-shift"],
)
def test_in_range_results_unchanged(current):
    """In-range values bin identically, so JS is bit-for-bit the same."""

    assert calculate_js(REFERENCE, current) == _legacy_js(REFERENCE, current)


# ============================================================
# Out of range
# ============================================================

@pytest.mark.parametrize(
    "current",
    [_half_replaced(-10000.0), _half_replaced(10000.0)],
    ids=["half-below", "half-above"],
)
def test_partially_outside_range_is_meaningful_divergence(current):

    # The bug: out-of-range values were dropped, batch looked identical
    assert _legacy_js(REFERENCE, current) == pytest.approx(0, abs=1e-12)

    js = calculate_js(REFERENCE, current)

    assert 0.1 < js < JS_MAX


@pytest.mark.parametrize(
    "current",
    [REFERENCE - 200, REFERENCE + 200],
    ids=["all-below", "all-above"],
)
def test_fully_outside_range_is_finite_and_large(current):

    # The bug: every value dropped, 0/0 counts, NaN
    assert math.isnan(_legacy_js(REFERENCE, current))

    js = calculate_js(REFERENCE, current)

    assert math.isfinite(js)
    assert 0.4 < js <= JS_MAX


@pytest.mark.parametrize("outlier", [-500.0, 500.0], ids=["below", "above"])
def test_no_out_of_range_value_is_silently_dropped(outlier):
    """More appended out-of-range mass means strictly more divergence."""

    values = [
        calculate_js(REFERENCE, np.concatenate([REFERENCE, np.full(n, outlier)]))
        for n in (0, 10, 50, 200)
    ]

    assert values[0] == pytest.approx(0, abs=1e-12)
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_out_of_range_value_lands_in_edge_bin():

    at_max = np.concatenate([REFERENCE[:900], np.full(100, 99.0)])
    past_max = np.concatenate([REFERENCE[:900], np.full(100, 1e6)])

    assert calculate_js(REFERENCE, past_max) == calculate_js(REFERENCE, at_max)


# ============================================================
# PSI and JS cannot drift apart again
# ============================================================

def test_psi_and_js_share_one_binning(monkeypatch):

    seen = []
    real = prediction_drift._binned_counts

    def spy(reference, current, bins=10):
        seen.append(bins)
        return real(reference, current, bins)

    monkeypatch.setattr(prediction_drift, "_binned_counts", spy)

    calculate_psi(REFERENCE, REFERENCE + 5)
    calculate_js(REFERENCE, REFERENCE + 5)

    assert seen == [10, 10]


def test_run_prediction_drift_reports_out_of_range_js():

    results = prediction_drift.run_prediction_drift(
        REFERENCE, _half_replaced(10000.0)
    )

    assert results["js_divergence"] > 0.1


# ============================================================
# NaN / empty follow the PSI convention
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

    assert math.isnan(calculate_js(reference, current))
    assert math.isnan(calculate_psi(reference, current))


def test_scattered_nan_values_are_ignored():

    with_nans = np.concatenate([REFERENCE, [np.nan] * 10])

    assert calculate_js(REFERENCE, with_nans) == calculate_js(
        REFERENCE, REFERENCE
    )
