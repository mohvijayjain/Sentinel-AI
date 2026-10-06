"""
stastical_drift.calculate_js_numeric / calculate_js_categorical: audit
for the out-of-range drop fixed in prediction_drift.

Result: already safe, NOT modified. calculate_js_numeric opens its outer
quantile edges to -inf / +inf before np.histogram, and the categorical
version bins over the union of reference and current categories, so
every current value is counted. These tests pin that.

Neither feeds scoring: they fill the statistical report's js_divergence
column, which drift_scorer never reads.
"""

import math
import os

import numpy as np
import pytest

from src.monitoring.stastical_drift import (
    calculate_js_categorical,
    calculate_js_numeric,
)


REFERENCE = np.random.default_rng(1).normal(50, 10, 5000)

JS_MAX = math.log(2)


def _half_replaced(value):
    return np.concatenate([REFERENCE[:2500], np.full(2500, value)])


def test_identical_distributions_score_zero():

    assert calculate_js_numeric(REFERENCE, REFERENCE.copy()) == pytest.approx(
        0, abs=1e-12
    )


@pytest.mark.parametrize(
    "current",
    [_half_replaced(-1e6), _half_replaced(1e6)],
    ids=["half-below", "half-above"],
)
def test_partially_outside_range_counts(current):

    assert calculate_js_numeric(REFERENCE, current) > 0.1


@pytest.mark.parametrize(
    "current",
    [REFERENCE - 1e6, REFERENCE + 1e6],
    ids=["all-below", "all-above"],
)
def test_fully_outside_range_is_near_maximal(current):

    js = calculate_js_numeric(REFERENCE, current)

    assert math.isfinite(js)
    assert 0.5 < js <= JS_MAX


@pytest.mark.parametrize("outlier", [-1e6, 1e6], ids=["below", "above"])
def test_no_out_of_range_value_is_silently_dropped(outlier):

    values = [
        calculate_js_numeric(
            REFERENCE, np.concatenate([REFERENCE, np.full(n, outlier)])
        )
        for n in (0, 25, 100, 400)
    ]

    assert values[0] == pytest.approx(0, abs=1e-12)
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_out_of_range_lands_in_edge_bin():

    top = REFERENCE.max()

    at_max = np.concatenate([REFERENCE, np.full(200, top)])
    past_max = np.concatenate([REFERENCE, np.full(200, top + 1e6)])

    assert calculate_js_numeric(REFERENCE, past_max) == calculate_js_numeric(
        REFERENCE, at_max
    )


@pytest.mark.parametrize(
    "reference, current",
    [(REFERENCE, []), ([], REFERENCE), (REFERENCE, [np.nan, np.nan])],
    ids=["empty-current", "empty-reference", "all-nan-current"],
)
def test_no_usable_values_gives_nan(reference, current):

    assert math.isnan(calculate_js_numeric(reference, current))


def test_unseen_category_is_counted():

    reference = ["1", "2"] * 500
    current = ["1", "2"] * 250 + ["99"] * 500

    assert calculate_js_categorical(reference, current) > 0.1


def test_scorer_does_not_consume_js_divergence():

    path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "src", "monitoring", "drift_scorer.py",
    )

    with open(path, encoding="utf-8") as f:
        assert "js_divergence" not in f.read()
