from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuration
# ============================================================

DEFAULT_DEGRADATION_THRESHOLD = 0.10       # 10% relative
DEFAULT_MIN_ABS_DEGRADATION = 20.0         # seconds — absolute floor
DEFAULT_MIN_ROWS = 5_000
DEFAULT_TOP_ZONES = 10

# Optional: named high-value zones to always check regardless of volume.
# NYC TLC zone IDs — JFK=132, LaGuardia=138, Newark=1. Adjust to your data.
DEFAULT_NAMED_ZONES = {132: "jfk", 138: "laguardia", 1: "newark"}


def _calculate_metrics(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


# ============================================================
# Single segment
# ============================================================

def evaluate_segment(
    name: str,
    mask,
    y_true: np.ndarray,
    champion_pred: np.ndarray,
    challenger_pred: np.ndarray,
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
    min_abs_degradation: float = DEFAULT_MIN_ABS_DEGRADATION,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> dict:

    mask = np.asarray(mask, dtype=bool)
    rows = int(mask.sum())

    if rows < min_rows:
        return {
            "segment": name,
            "rows": rows,
            "evaluated": False,
            "passed": True,
            "reason": f"Skipped: only {rows:,} rows; minimum required is {min_rows:,}.",
        }

    y = y_true[mask]
    champ = champion_pred[mask]
    chall = challenger_pred[mask]

    cm = _calculate_metrics(y, champ)
    hm = _calculate_metrics(y, chall)

    # Absolute degradation (challenger - champion). Positive = worse.
    rmse_abs = hm["rmse"] - cm["rmse"]
    mae_abs = hm["mae"] - cm["mae"]

    # Relative degradation, guarded against a ~zero champion error.
    rmse_rel = rmse_abs / cm["rmse"] if cm["rmse"] > 0 else (float("inf") if rmse_abs > 0 else 0.0)
    mae_rel = mae_abs / cm["mae"] if cm["mae"] > 0 else (float("inf") if mae_abs > 0 else 0.0)

    # A segment FAILS only if it degraded BOTH beyond the relative threshold
    # AND beyond the absolute floor. This stops a trivial few-second wobble on
    # an easy (low-RMSE) segment from blocking an otherwise good model, while
    # still catching real regressions on hard segments.
    rmse_fail = (rmse_rel > degradation_threshold) and (rmse_abs > min_abs_degradation)
    mae_fail = (mae_rel > degradation_threshold) and (mae_abs > min_abs_degradation)

    passed = not (rmse_fail or mae_fail)

    if passed:
        reason = "PASS: challenger did not degrade beyond the allowed threshold."
    else:
        failures = []
        if rmse_fail:
            failures.append(f"RMSE +{rmse_abs:.1f}s ({rmse_rel * 100:.1f}%)")
        if mae_fail:
            failures.append(f"MAE +{mae_abs:.1f}s ({mae_rel * 100:.1f}%)")
        reason = "FAIL: " + "; ".join(failures) + "."

    return {
        "segment": name,
        "rows": rows,
        "evaluated": True,
        "passed": passed,
        "reason": reason,
        "champion": cm,
        "challenger": hm,
        "rmse_degradation_abs": float(rmse_abs),
        "rmse_degradation_rel": float(rmse_rel),
        "mae_degradation_abs": float(mae_abs),
        "mae_degradation_rel": float(mae_rel),
        "degradation_threshold": degradation_threshold,
        "min_abs_degradation": min_abs_degradation,
    }


# ============================================================
# All segments
# ============================================================

def evaluate_segments(
    test_df: pd.DataFrame,
    y_true,
    champion_pred,
    challenger_pred,
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
    min_abs_degradation: float = DEFAULT_MIN_ABS_DEGRADATION,
    min_rows: int = DEFAULT_MIN_ROWS,
    top_zones: int = DEFAULT_TOP_ZONES,
    named_zones: dict | None = None,
) -> dict:

    named_zones = DEFAULT_NAMED_ZONES if named_zones is None else named_zones

    y_true = np.asarray(y_true, dtype=np.float64)
    champion_pred = np.asarray(champion_pred, dtype=np.float64)
    challenger_pred = np.asarray(challenger_pred, dtype=np.float64)

    if not (len(test_df) == len(y_true) == len(champion_pred) == len(challenger_pred)):
        raise ValueError(
            "test_df, y_true, champion_pred and challenger_pred must have the same length."
        )

    # Alignment guard: masks are taken positionally, so test_df must share the
    # same row order as the prediction arrays. Enforce a clean 0..n-1 index.
    test_df = test_df.reset_index(drop=True)

    # The pickup-zone column is PULocationID in frames spelled like
    # preprocess.FEATURES and pulocationid in older ones; resolve it
    # case-insensitively, as promote._prepare_features does.
    zone_col = next(
        (c for c in test_df.columns if str(c).lower() == "pulocationid"), None
    )

    required = ["is_rush_hour", "is_weekend"]
    missing = [c for c in required if c not in test_df.columns]
    if zone_col is None:
        missing.append("pulocationid")
    if missing:
        raise ValueError(f"Missing segment columns: {missing}")

    results = []

    def run(name, mask):
        results.append(
            evaluate_segment(
                name, mask, y_true, champion_pred, challenger_pred,
                degradation_threshold, min_abs_degradation, min_rows,
            )
        )

    # Time-of-day
    run("rush_hour", test_df["is_rush_hour"] == 1)
    run("non_rush_hour", test_df["is_rush_hour"] == 0)

    # Day-of-week
    run("weekend", test_df["is_weekend"] == 1)
    run("weekday", test_df["is_weekend"] == 0)

    # Named high-value zones (always checked, regardless of volume)
    checked_zone_ids = set()
    for zone_id, label in named_zones.items():
        if (test_df[zone_col] == zone_id).any():
            run(f"zone_{label}_{zone_id}", test_df[zone_col] == zone_id)
            checked_zone_ids.add(zone_id)

    # High-volume pickup zones (skip any already covered as named)
    top_pickup_zones = test_df[zone_col].value_counts().head(top_zones).index
    for zone in top_pickup_zones:
        if zone in checked_zone_ids:
            continue
        run(f"pickup_zone_{zone}", test_df[zone_col] == zone)

    # Gate
    evaluated = [r for r in results if r["evaluated"]]
    skipped = [r for r in results if not r["evaluated"]]
    failed = [r for r in evaluated if not r["passed"]]
    passed = len(failed) == 0

    if passed:
        reason = (
            f"PASS: none of {len(evaluated)} evaluated segments exceeded "
            f"{degradation_threshold * 100:.0f}% AND {min_abs_degradation:.0f}s degradation."
        )
    else:
        reason = "FAIL: segment degradation in: " + ", ".join(r["segment"] for r in failed)

    return {
        "passed": passed,
        "reason": reason,
        "degradation_threshold": degradation_threshold,
        "min_abs_degradation": min_abs_degradation,
        "min_rows": min_rows,
        "top_zones": top_zones,
        "segments": results,
        "evaluated_segments": len(evaluated),
        "skipped_segments": len(skipped),
        "failed_segments": len(failed),
    }