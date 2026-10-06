"""
src/evaluation/segment.evaluate_segments: pickup-zone column case.

promote passes the frozen test frame, now spelled like preprocess.FEATURES
(PULocationID). The segment gate hard-coded "pulocationid", so every real
promotion failed with "Missing segment columns: ['pulocationid']". The
column is now resolved case-insensitively; results are identical for
either spelling.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.segment import evaluate_segments


def _inputs(zone_column):
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({
        "is_rush_hour": rng.integers(0, 2, n),
        "is_weekend": rng.integers(0, 2, n),
        zone_column: rng.choice([132, 138, 161, 230, 237], n),
    })
    y = rng.normal(900, 200, n)
    champion = y + rng.normal(0, 50, n)
    challenger = y + rng.normal(0, 40, n)
    return frame, y, champion, challenger


@pytest.mark.parametrize("zone_column", ["PULocationID", "pulocationid"])
def test_either_spelling_is_accepted(zone_column):

    frame, y, champion, challenger = _inputs(zone_column)

    result = evaluate_segments(frame, y, champion, challenger)

    assert "passed" in result


def test_results_identical_for_both_spellings():

    camel = evaluate_segments(*_inputs("PULocationID"))
    lower = evaluate_segments(*_inputs("pulocationid"))

    assert camel == lower


def test_missing_zone_column_still_reported():

    frame, y, champion, challenger = _inputs("PULocationID")
    frame = frame.drop(columns=["PULocationID"])

    with pytest.raises(ValueError, match=r"Missing segment columns: \['pulocationid'\]"):
        evaluate_segments(frame, y, champion, challenger)
