"""
Recent Test Gate

Purpose:
    Validate that the Challenger remains healthy on recent/current data.

Important:
    This is NOT a Champion-vs-Challenger comparison gate.

    It compares:
        Challenger Frozen performance
                    vs
        Challenger Recent performance

    Champion recent-vs-frozen degradation is recorded only as
    contextual information and does NOT affect the gate decision.

Current Sentinel-AI interpretation:
    The April 2026 completed-trip dataset is the current
    drift-month validation set.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_MIN_ROWS = 5_000
DEFAULT_MAX_RMSE_DEGRADATION = 0.10  # 10%
DEFAULT_MAX_MAE_DEGRADATION = 0.10   # 10%


def _calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """Calculate RMSE and MAE."""

    rmse = float(
        np.sqrt(mean_squared_error(y_true, predictions))
    )

    mae = float(
        mean_absolute_error(y_true, predictions)
    )

    return {
        "rmse": rmse,
        "mae": mae,
    }


def _validate_array(
    name: str,
    array: np.ndarray,
) -> None:
    """Validate that an array contains only finite values."""

    if not np.all(np.isfinite(array)):
        bad = int((~np.isfinite(array)).sum())

        raise ValueError(
            f"{name} contains {bad} non-finite value(s) "
            "(NaN or inf). Recent test gate requires "
            "clean values."
        )


def _calculate_degradation(
    frozen_value: float,
    recent_value: float,
) -> Dict[str, float]:
    """
    Calculate absolute and relative degradation.

    Positive relative degradation means recent performance
    became worse.

    Negative relative degradation means recent performance
    improved.
    """

    absolute_change = recent_value - frozen_value

    if frozen_value > 0:
        relative_degradation = (
            absolute_change / frozen_value
        )
    else:
        if recent_value > 0:
            relative_degradation = float("inf")
        else:
            relative_degradation = 0.0

    return {
        "absolute": float(absolute_change),
        "relative": float(relative_degradation),
        "relative_pct": float(relative_degradation * 100),
    }


def evaluate_recent_test(
    recent_df: pd.DataFrame,
    y_recent: np.ndarray,
    challenger_recent_pred: np.ndarray,
    challenger_frozen_metrics: Dict[str, float],
    champion_frozen_metrics: Dict[str, float],
    champion_recent_pred: np.ndarray,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION,
    max_mae_degradation: float = DEFAULT_MAX_MAE_DEGRADATION,
) -> Dict[str, Any]:
    """
    Evaluate Challenger freshness on recent labeled data.

    Parameters
    ----------
    recent_df:
        Recent labeled test dataframe.

    y_recent:
        Ground-truth target values for recent data.

    challenger_recent_pred:
        Challenger predictions on recent data.

    challenger_frozen_metrics:
        Already-calculated Challenger metrics on the frozen
        test set. Must contain:
            {"rmse": ..., "mae": ...}

    champion_frozen_metrics:
        Already-calculated Champion metrics on the frozen
        test set. Used only for contextual comparison.

    champion_recent_pred:
        Champion predictions on recent data. Used only to
        calculate Champion recent-vs-frozen degradation.

    min_rows:
        Minimum recent rows required.

    max_rmse_degradation:
        Maximum allowed Challenger RMSE degradation relative
        to its frozen performance.

    max_mae_degradation:
        Maximum allowed Challenger MAE degradation relative
        to its frozen performance.

    Returns
    -------
    Dict[str, Any]
        Detailed Recent Test Gate result.
    """

    # ---------------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------------

    if not isinstance(recent_df, pd.DataFrame):
        raise TypeError(
            "recent_df must be a pandas DataFrame."
        )

    if min_rows < 1:
        raise ValueError(
            "min_rows must be >= 1."
        )

    if not 0 <= max_rmse_degradation < 1:
        raise ValueError(
            "max_rmse_degradation must be between 0 and 1."
        )

    if not 0 <= max_mae_degradation < 1:
        raise ValueError(
            "max_mae_degradation must be between 0 and 1."
        )

    required_metrics = {"rmse", "mae"}

    if not required_metrics.issubset(
        challenger_frozen_metrics
    ):
        raise ValueError(
            "challenger_frozen_metrics must contain "
            "'rmse' and 'mae'."
        )

    if not required_metrics.issubset(
        champion_frozen_metrics
    ):
        raise ValueError(
            "champion_frozen_metrics must contain "
            "'rmse' and 'mae'."
        )

    # ---------------------------------------------------------
    # Convert arrays
    # ---------------------------------------------------------

    y_recent = np.asarray(
        y_recent,
        dtype=np.float64,
    )

    challenger_recent_pred = np.asarray(
        challenger_recent_pred,
        dtype=np.float64,
    )

    champion_recent_pred = np.asarray(
        champion_recent_pred,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Validate lengths
    # ---------------------------------------------------------

    n_rows = len(y_recent)

    if n_rows == 0:
        return {
            "passed": False,
            "evaluated": False,
            "reason": "INSUFFICIENT_RECENT_DATA",
            "reason_detail": "Recent test set is empty.",
            "rows": 0,
            "min_rows": min_rows,
        }

    if len(recent_df) != n_rows:
        raise ValueError(
            "recent_df length does not match y_recent."
        )

    if len(challenger_recent_pred) != n_rows:
        raise ValueError(
            "challenger_recent_pred length does not "
            "match y_recent."
        )

    if len(champion_recent_pred) != n_rows:
        raise ValueError(
            "champion_recent_pred length does not "
            "match y_recent."
        )

    # ---------------------------------------------------------
    # Validate finite values
    # ---------------------------------------------------------

    _validate_array("y_recent", y_recent)
    _validate_array(
        "challenger_recent_pred",
        challenger_recent_pred,
    )
    _validate_array(
        "champion_recent_pred",
        champion_recent_pred,
    )

    # ---------------------------------------------------------
    # Minimum data gate
    # ---------------------------------------------------------

    if n_rows < min_rows:
        return {
            "passed": False,
            "evaluated": False,
            "reason": "INSUFFICIENT_RECENT_DATA",
            "reason_detail": (
                f"{n_rows:,} recent rows available; "
                f"{min_rows:,} required."
            ),
            "rows": n_rows,
            "min_rows": min_rows,
            "thresholds": {
                "max_rmse_degradation": max_rmse_degradation,
                "max_mae_degradation": max_mae_degradation,
            },
        }

    # ---------------------------------------------------------
    # Calculate recent metrics
    # ---------------------------------------------------------

    challenger_recent_metrics = _calculate_metrics(
        y_recent,
        challenger_recent_pred,
    )

    champion_recent_metrics = _calculate_metrics(
        y_recent,
        champion_recent_pred,
    )

    # ---------------------------------------------------------
    # Challenger frozen → recent degradation
    # ---------------------------------------------------------

    challenger_rmse_degradation = _calculate_degradation(
        challenger_frozen_metrics["rmse"],
        challenger_recent_metrics["rmse"],
    )

    challenger_mae_degradation = _calculate_degradation(
        challenger_frozen_metrics["mae"],
        challenger_recent_metrics["mae"],
    )

    # ---------------------------------------------------------
    # Champion frozen → recent degradation
    #
    # IMPORTANT:
    # This is context only.
    # It does NOT affect the Challenger gate.
    # ---------------------------------------------------------

    champion_rmse_degradation = _calculate_degradation(
        champion_frozen_metrics["rmse"],
        champion_recent_metrics["rmse"],
    )

    champion_mae_degradation = _calculate_degradation(
        champion_frozen_metrics["mae"],
        champion_recent_metrics["mae"],
    )

    # ---------------------------------------------------------
    # Challenger freshness checks
    # ---------------------------------------------------------

    rmse_passed = (
        challenger_rmse_degradation["relative"]
        <= max_rmse_degradation
    )

    mae_passed = (
        challenger_mae_degradation["relative"]
        <= max_mae_degradation
    )

    passed = rmse_passed and mae_passed

    # ---------------------------------------------------------
    # Build reason
    # ---------------------------------------------------------

    if passed:
        reason = (
            "PASS: Challenger remains within the allowed "
            "recent-data degradation limits."
        )
    else:
        failures = []

        if not rmse_passed:
            failures.append(
                f"RMSE degradation "
                f"{challenger_rmse_degradation['relative_pct']:.3f}% "
                f"exceeds "
                f"{max_rmse_degradation * 100:.1f}%"
            )

        if not mae_passed:
            failures.append(
                f"MAE degradation "
                f"{challenger_mae_degradation['relative_pct']:.3f}% "
                f"exceeds "
                f"{max_mae_degradation * 100:.1f}%"
            )

        reason = (
            "FAIL: Challenger degraded beyond the allowed "
            "recent-data limits — "
            + "; ".join(failures)
            + "."
        )

    # ---------------------------------------------------------
    # Return complete evaluation
    # ---------------------------------------------------------

    return {
        "passed": passed,
        "evaluated": True,
        "reason": reason,
        "rows": n_rows,
        "min_rows": min_rows,

        "thresholds": {
            "max_rmse_degradation": max_rmse_degradation,
            "max_mae_degradation": max_mae_degradation,
        },

        "challenger": {
            "frozen": {
                "rmse": float(
                    challenger_frozen_metrics["rmse"]
                ),
                "mae": float(
                    challenger_frozen_metrics["mae"]
                ),
            },
            "recent": challenger_recent_metrics,
            "degradation": {
                "rmse": challenger_rmse_degradation,
                "mae": challenger_mae_degradation,
            },
            "checks": {
                "rmse_passed": rmse_passed,
                "mae_passed": mae_passed,
            },
        },

        "champion_context": {
            "frozen": {
                "rmse": float(
                    champion_frozen_metrics["rmse"]
                ),
                "mae": float(
                    champion_frozen_metrics["mae"]
                ),
            },
            "recent": champion_recent_metrics,
            "degradation": {
                "rmse": champion_rmse_degradation,
                "mae": champion_mae_degradation,
            },
        },
    }