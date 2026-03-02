"""Structural break and parameter-stability tests for the Lucas Critique.

Functions
---------
chow_test
    Classical Chow (1960) F-test for a known break point.
recursive_cusum
    CUSUM of recursive residuals for unknown break detection.
compute_rolling_performance
    Rolling-window RMSE to track degradation over time.
compare_pre_post_performance
    Aggregate pre/post comparison across multiple models.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import forecast_rmse


# ---------------------------------------------------------------------------
# Chow test
# ---------------------------------------------------------------------------


def chow_test(
    y: np.ndarray,
    X: np.ndarray,
    break_index: int,
) -> dict[str, float]:
    """Chow (1960) F-test for a structural break at a known point.

    Tests whether the regression coefficients are stable across the two
    sub-samples defined by ``break_index``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Dependent variable.
    X : array-like, shape (n, p)
        Regressor matrix (should include a constant column).
    break_index : int
        Index at which the break occurs (splits into [0:break_index] and
        [break_index:]).

    Returns
    -------
    dict with keys:
        ``F_stat``   : float — Chow F-statistic.
        ``p_value``  : float — p-value under F(p, n - 2p) distribution.
        ``df1``      : int   — numerator degrees of freedom.
        ``df2``      : int   — denominator degrees of freedom.
        ``reject_H0``: bool  — True if break is detected at 5% level.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if break_index <= p or break_index >= n - p:
        raise ValueError(
            f"break_index={break_index} too close to boundary for {p} regressors."
        )

    def ssr(X_sub: np.ndarray, y_sub: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
        resid = y_sub - X_sub @ beta
        return float(resid @ resid)

    ssr_full = ssr(X, y)
    ssr1 = ssr(X[:break_index], y[:break_index])
    ssr2 = ssr(X[break_index:], y[break_index:])
    ssr_restricted = ssr1 + ssr2

    df1 = p
    df2 = n - 2 * p
    if df2 <= 0:
        raise ValueError("Too few observations relative to number of regressors.")

    F = ((ssr_full - ssr_restricted) / df1) / (ssr_restricted / df2)
    p_value = float(1 - stats.f.cdf(F, df1, df2))

    return {
        "F_stat": float(F),
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "reject_H0": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# CUSUM of recursive residuals
# ---------------------------------------------------------------------------


def recursive_cusum(
    y: np.ndarray,
    X: np.ndarray,
    significance: float = 0.05,
) -> dict[str, object]:
    """CUSUM of recursive residuals (Brown, Durbin & Evans 1975).

    Tracks cumulative recursive residuals over time.  A structural break
    causes the CUSUM path to drift outside the ±5% significance bounds.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
    significance : float
        Significance level for the critical band (default 0.05 → ±5% bounds).

    Returns
    -------
    dict with keys:
        ``cusum``        : np.ndarray  — CUSUM path.
        ``upper_bound``  : np.ndarray  — upper critical boundary.
        ``lower_bound``  : np.ndarray  — lower critical boundary.
        ``break_detected``: bool       — True if CUSUM exits band.
        ``break_index``  : int or None — first index outside band (or None).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Compute recursive residuals
    recursive_resids = []
    for t in range(p, n):
        X_t = X[:t]
        y_t = y[:t]
        beta_t, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
        x_new = X[t]
        y_hat = float(x_new @ beta_t)
        sigma_t = float(np.sqrt(1 + x_new @ np.linalg.pinv(X_t.T @ X_t) @ x_new))
        recursive_resids.append((y[t] - y_hat) / sigma_t)

    w = np.array(recursive_resids)
    sigma_w = float(np.std(w, ddof=1)) if len(w) > 1 else 1.0
    cusum = np.cumsum(w) / (sigma_w + 1e-9)

    # Critical band: ±c * sqrt(n - p) where c from significance level
    c = stats.norm.ppf(1 - significance / 2) * np.sqrt(n - p)
    t_idx = np.arange(len(cusum))
    upper = c * np.ones_like(cusum)
    lower = -upper

    outside = np.where(np.abs(cusum) > c)[0]
    break_detected = len(outside) > 0
    break_index = int(outside[0]) if break_detected else None

    return {
        "cusum": cusum,
        "upper_bound": upper,
        "lower_bound": lower,
        "break_detected": bool(break_detected),
        "break_index": break_index,
    }


# ---------------------------------------------------------------------------
# Rolling performance
# ---------------------------------------------------------------------------


def compute_rolling_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 30,
) -> pd.DataFrame:
    """Rolling-window RMSE over time.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame with columns ``t``, ``rolling_rmse``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rmse_vals = np.full(n, np.nan)

    for i in range(window - 1, n):
        rmse_vals[i] = forecast_rmse(
            y_true[i - window + 1 : i + 1],
            y_pred[i - window + 1 : i + 1],
        )

    return pd.DataFrame({"t": np.arange(n), "rolling_rmse": rmse_vals})


# ---------------------------------------------------------------------------
# Pre/post comparison
# ---------------------------------------------------------------------------


def compare_pre_post_performance(
    predictions: dict[str, np.ndarray],
    y_full: np.ndarray,
    break_index: int,
) -> pd.DataFrame:
    """Compare each model's RMSE in the pre-break and post-break periods.

    Parameters
    ----------
    predictions : dict
        ``{model_name: y_pred_full}`` — predictions for the concatenated series.
    y_full : array-like, shape (n,)
        Ground-truth y for the full concatenated period.
    break_index : int
        Index separating pre-break (0:break_index) from post-break (break_index:).

    Returns
    -------
    pd.DataFrame
        Columns: model, pre_rmse, post_rmse, LSR (Lucas Sensitivity Ratio).
    """
    from .metrics import lucas_sensitivity_ratio

    y_full = np.asarray(y_full, dtype=float)
    rows = []
    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=float)
        pre_rmse = forecast_rmse(y_full[:break_index], y_pred[:break_index])
        post_rmse = forecast_rmse(y_full[break_index:], y_pred[break_index:])
        lsr = lucas_sensitivity_ratio(pre_rmse, post_rmse)
        rows.append({"model": name, "pre_rmse": pre_rmse, "post_rmse": post_rmse, "LSR": lsr})

    return pd.DataFrame(rows).sort_values("LSR").reset_index(drop=True)
