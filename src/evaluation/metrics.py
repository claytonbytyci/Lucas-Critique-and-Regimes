"""Performance metrics for regime-switching model evaluation.

Metrics fall into two groups:

    Forecast accuracy:
        - forecast_rmse        Root mean squared forecast error
        - forecast_mae         Mean absolute error
        - directional_accuracy Fraction of correctly signed changes

    Regime detection:
        - regime_accuracy           Accuracy of regime classification vs truth
        - regime_conditional_rmse   Per-regime RMSE breakdown

    Lucas critique:
        - lucas_sensitivity_ratio   Performance degradation ratio pre→post break
        - summarise_model_performance  Aggregate summary DataFrame
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Forecast accuracy

def forecast_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between true and predicted values.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def forecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of observations where the predicted sign of change matches truth.

    Uses first-differences: sign(Δy_true) == sign(Δy_pred).

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred)))

# Regime detection

def regime_accuracy( # I don't actually like this test because it counts the number of errors but a model that guesses frequently of a switch is penalised
    true_regimes: np.ndarray,
    pred_regimes: np.ndarray,
    allow_permutation: bool = True,
) -> float:
    """Classification accuracy of predicted regime labels vs ground truth.

    Because regime labels are arbitrary (regime 0 in the model may correspond
    to regime 1 in the DGP), when ``allow_permutation=True`` we try both
    label permutations and return the maximum accuracy.

    Parameters
    ----------
    true_regimes : array-like, shape (n,)
        Ground-truth regime integers from the DGP.
    pred_regimes : array-like, shape (n,)
        Predicted regime integers.
    allow_permutation : bool
        If True, evaluate all label permutations (valid for binary case).

    Returns
    -------
    float in [0, 1]
    """
    true = np.asarray(true_regimes, dtype=int)
    pred = np.asarray(pred_regimes, dtype=int)

    if not allow_permutation:
        return float(np.mean(true == pred))

    unique_true = np.unique(true)
    unique_pred = np.unique(pred)

    if len(unique_pred) == 1:
        # Degenerate: model assigned all to one regime
        return float(np.mean(true == pred))

    best = 0.0
    # For binary case, try identity and swap
    for flip in [False, True]:
        pred_mapped = (1 - pred) if flip and set(unique_pred) == {0, 1} else pred
        acc = float(np.mean(true == pred_mapped))
        if acc > best:
            best = acc
    return best


def adjusted_rand_regime( # this is allegedly a good metric for clustering quality, invariant to label permutations, meaning that it doesn't matter what the regime is labelled, as long as the clustering vaguely matches the true clustering, it will give a high score. It can be negative if the clustering is worse than random, and 1 if it's perfect.
    true_regimes: np.ndarray, pred_regimes: np.ndarray
) -> float:
    """Adjusted Rand Index for clustering quality (label-permutation invariant).

    Parameters
    ----------
    true_regimes : array-like, shape (n,)
    pred_regimes : array-like, shape (n,)

    Returns
    -------
    float in [-1, 1]  (1 = perfect, 0 = random)
    """
    return float(adjusted_rand_score(true_regimes, pred_regimes))


def regime_conditional_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regimes: np.ndarray,
) -> dict[int, float]:
    """RMSE broken down by true regime.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    regimes : array-like, shape (n,)  — true regime integers

    Returns
    -------
    dict mapping regime_int → RMSE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regimes = np.asarray(regimes, dtype=int)
    result: dict[int, float] = {}
    for reg in np.unique(regimes):
        mask = regimes == reg
        result[int(reg)] = forecast_rmse(y_true[mask], y_pred[mask])
    return result

# Lucas critique summary

def lucas_sensitivity_ratio(
    pre_rmse: float, post_rmse: float, epsilon: float = 1e-9
) -> float:
    """Ratio of post-break RMSE to pre-break RMSE.

    LSR = post_rmse / pre_rmse

    LSR = 1  → no degradation (perfectly Lucas-stable).
    LSR > 1  → performance worsens post-break (Lucas-critique vulnerable).
    LSR < 1  → performance improves (unusual; may indicate regularisation).

    Parameters
    ----------
    pre_rmse : float
    post_rmse : float
    epsilon : float
        Small value to avoid division by zero, if the pre-break RMSE is very small, only really an issue for MSNN.

    Returns
    -------
    float
    """
    return float(post_rmse / (pre_rmse + epsilon))


def summarise_model_performance(
    results: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Build a tidy comparison DataFrame from a dict of model result dicts.

    Parameters
    ----------
    results : dict
        ``{model_name: {"pre_rmse": ..., "post_rmse": ..., "regime_acc": ..., ...}}``

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metric names, plus a computed ``LSR`` column.
    """
    df = pd.DataFrame(results).T
    if "pre_rmse" in df.columns and "post_rmse" in df.columns:
        df["LSR"] = df.apply(
            lambda row: lucas_sensitivity_ratio(row["pre_rmse"], row["post_rmse"]),
            axis=1,
        )
    df.index.name = "model"
    return df.reset_index()
