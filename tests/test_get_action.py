"""
drift_scorer.get_action: overall score -> action band.

Every `<` cutoff is tested on both sides: the exact value lands in the
upper band, a value just below it lands in the lower band.
"""

import pytest


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "WAIT"),
        (0.249999, "WAIT"),
        (0.25, "MONITOR"),
        (0.499999, "MONITOR"),
        (0.5, "ALERT"),
        (0.749999, "ALERT"),
        (0.75, "RETRAIN"),
        (1.0, "RETRAIN"),
    ],
)
def test_action_bands(drift_scorer, score, expected):

    assert drift_scorer.get_action(score) == expected
