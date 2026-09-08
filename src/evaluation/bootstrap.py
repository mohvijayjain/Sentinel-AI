from __future__ import annotations

import numpy as np


def bootstrap_gate(
    y_true,
    champion_pred,
    challenger_pred,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    gate_on: tuple[str, ...] = ("rmse",),
) -> dict:
    """Paired bootstrap comparison of champion vs challenger.

    Same resample indices are used for both models each iteration (paired),
    so the comparison isolates model difference, not sampling difference.

    gate_on: which metrics must be statistically significant for the gate to
    pass. Default ("rmse",) gates on RMSE only; MAE/R2 are still reported and
    can be gated by the plain multi-metric check in promote.py. Pass
    ("rmse", "mae", "r2") to require all three.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    champion_pred = np.asarray(champion_pred, dtype=np.float64)
    challenger_pred = np.asarray(challenger_pred, dtype=np.float64)

    if not (len(y_true) == len(champion_pred) == len(challenger_pred)):
        raise ValueError(
            "y_true, champion_pred and challenger_pred must have the same length."
        )
    if len(y_true) == 0:
        raise ValueError("Bootstrap evaluation received no rows.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    valid = {"rmse", "mae", "r2"}
    if not set(gate_on) <= valid:
        raise ValueError(f"gate_on must be a subset of {valid}, got {gate_on}.")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    # ----------------------------------------------------------------
    # Precompute per-row errors ONCE. The loop then only indexes and
    # averages these — no sklearn calls, no re-subtraction per iteration.
    # ----------------------------------------------------------------
    champ_sq = (y_true - champion_pred) ** 2
    chall_sq = (y_true - challenger_pred) ** 2
    champ_abs = np.abs(y_true - champion_pred)
    chall_abs = np.abs(y_true - challenger_pred)

    def _metrics(idx):
        y_s = y_true[idx]
        # R2 denominator depends on the resample's own variance, so it must
        # be recomputed each iteration — it is NOT a simple mean of per-row values.
        ss_tot = np.sum((y_s - y_s.mean()) ** 2)

        champ_rmse = np.sqrt(champ_sq[idx].mean())
        chall_rmse = np.sqrt(chall_sq[idx].mean())
        champ_mae = champ_abs[idx].mean()
        chall_mae = chall_abs[idx].mean()

        # Guard the degenerate case where a resample has ~zero target variance.
        if ss_tot <= 0:
            champ_r2 = chall_r2 = 0.0
        else:
            champ_r2 = 1.0 - (champ_sq[idx].sum() / ss_tot)
            chall_r2 = 1.0 - (chall_sq[idx].sum() / ss_tot)

        return champ_rmse, chall_rmse, champ_mae, chall_mae, champ_r2, chall_r2

    # ----------------------------------------------------------------
    # Point estimates on the full set (idx = all rows, in order)
    # ----------------------------------------------------------------
    full = np.arange(n)
    (c_rmse, h_rmse, c_mae, h_mae, c_r2, h_r2) = _metrics(full)

    champion_metrics = {"rmse": c_rmse, "mae": c_mae, "r2": c_r2}
    challenger_metrics = {"rmse": h_rmse, "mae": h_mae, "r2": h_r2}

    # ----------------------------------------------------------------
    # Bootstrap distributions of (challenger - champion)
    #   RMSE/MAE: negative => challenger better
    #   R2:       positive => challenger better
    # ----------------------------------------------------------------
    rmse_diff = np.empty(n_iterations)
    mae_diff = np.empty(n_iterations)
    r2_diff = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        cr, hr, cm, hm, cr2, hr2 = _metrics(idx)
        rmse_diff[i] = hr - cr
        mae_diff[i] = hm - cm
        r2_diff[i] = hr2 - cr2

    alpha = 1.0 - confidence_level
    lo_p = 100 * (alpha / 2)
    hi_p = 100 * (1 - alpha / 2)

    rmse_ci = np.percentile(rmse_diff, [lo_p, hi_p])
    mae_ci = np.percentile(mae_diff, [lo_p, hi_p])
    r2_ci = np.percentile(r2_diff, [lo_p, hi_p])

    # For RMSE/MAE, significant improvement => entire CI below 0 (upper < 0).
    # For R2,        significant improvement => entire CI above 0 (lower > 0).
    rmse_sig = bool(rmse_ci[1] < 0)
    mae_sig = bool(mae_ci[1] < 0)
    r2_sig = bool(r2_ci[0] > 0)

    significant = {"rmse": rmse_sig, "mae": mae_sig, "r2": r2_sig}
    passed = all(significant[m] for m in gate_on)

    # Human-readable reason (also feeds the later LLM explanation layer).
    failing = [m for m in gate_on if not significant[m]]
    if passed:
        reason = (
            f"PASS: {', '.join(gate_on)} improvement is statistically "
            f"significant at {int(confidence_level * 100)}% confidence."
        )
    else:
        parts = []
        for m in failing:
            ci = {"rmse": rmse_ci, "mae": mae_ci, "r2": r2_ci}[m]
            parts.append(f"{m.upper()} CI [{ci[0]:.3f}, {ci[1]:.3f}] does not clear zero")
        reason = "REJECT: improvement not significant — " + "; ".join(parts) + "."

    def _block(champ, chall, ci, sig):
        return {
            "champion": float(champ),
            "challenger": float(chall),
            "difference": float(chall - champ),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "significant": sig,
        }

    return {
        "passed": passed,
        "reason": reason,
        "gate_on": list(gate_on),
        "rmse": _block(champion_metrics["rmse"], challenger_metrics["rmse"], rmse_ci, rmse_sig),
        "mae": _block(champion_metrics["mae"], challenger_metrics["mae"], mae_ci, mae_sig),
        "r2": _block(champion_metrics["r2"], challenger_metrics["r2"], r2_ci, r2_sig),
        "iterations": n_iterations,
        "confidence_level": confidence_level,
        "seed": seed,
    }