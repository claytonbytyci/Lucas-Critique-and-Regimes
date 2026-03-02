"""Visualisation utilities for regime-switching and Lucas-critique analyses.

All plotting functions return a ``matplotlib.axes.Axes`` (or ``Figure``)
object so callers can save, compose, or further customise them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

_REGIME_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]  # red, green, blue, orange
_FIGURE_DIR = Path(__file__).resolve().parents[2] / "analyses" / "figures"

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


# ---------------------------------------------------------------------------
# DGP / series visualisation
# ---------------------------------------------------------------------------


def plot_simulated_series(
    df: pd.DataFrame,
    title: str = "Simulated Markov-Switching AR(1) Series",
    figsize: tuple[int, int] = (14, 5),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ``y`` over time with regime shading.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``y`` and ``regime`` columns.
    title : str
    figsize : tuple
    ax : Axes or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(df.index, df["y"], lw=0.9, color="steelblue", label="y", zorder=3)

    # Shade regime regions
    regimes = df["regime"].to_numpy()
    n = len(regimes)
    unique = np.unique(regimes)
    colors = _REGIME_COLORS[: len(unique)]

    start = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[i - 1]:
            reg = regimes[start]
            color = colors[reg % len(colors)]
            ax.axvspan(start, i - 1, alpha=0.18, color=color, zorder=1)
            start = i

    patches = [
        mpatches.Patch(color=colors[r % len(colors)], alpha=0.4, label=f"Regime {r}")
        for r in unique
    ]
    ax.legend(handles=[ax.lines[0]] + patches, loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    return ax


# ---------------------------------------------------------------------------
# Regime comparison
# ---------------------------------------------------------------------------


def plot_regime_comparison(
    true_regimes: np.ndarray,
    pred_regimes_dict: dict[str, np.ndarray],
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Compare true regimes against multiple models' predicted regimes.

    Parameters
    ----------
    true_regimes : array-like, shape (n,)
        Ground-truth regime labels.
    pred_regimes_dict : dict
        ``{model_name: predicted_regimes_array}``
    figsize : tuple or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_models = len(pred_regimes_dict)
    n_rows = n_models + 1
    if figsize is None:
        figsize = (14, 2 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    t = np.arange(len(true_regimes))

    def _plot_strip(ax: plt.Axes, regimes: np.ndarray, label: str) -> None:
        for j, reg in enumerate(np.unique(regimes)):
            mask = regimes == reg
            ax.fill_between(
                t, 0, 1,
                where=mask,
                color=_REGIME_COLORS[reg % len(_REGIME_COLORS)],
                alpha=0.7,
                label=f"Regime {reg}",
            )
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va="center")
        ax.legend(loc="upper right", fontsize=8)

    _plot_strip(axes[0], np.asarray(true_regimes, dtype=int), "True")

    for ax, (name, pred) in zip(axes[1:], pred_regimes_dict.items()):
        _plot_strip(ax, np.asarray(pred, dtype=int), name)

    axes[-1].set_xlabel("t")
    fig.suptitle("Regime Detection: True vs Model Predictions", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rolling RMSE
# ---------------------------------------------------------------------------


def plot_rolling_rmse(
    rolling_dict: dict[str, pd.DataFrame],
    break_index: int | None = None,
    window: int = 30,
    figsize: tuple[int, int] = (13, 5),
) -> plt.Axes:
    """Plot rolling RMSE over time for multiple models.

    Parameters
    ----------
    rolling_dict : dict
        ``{model_name: DataFrame with columns [t, rolling_rmse]}``
    break_index : int or None
        If given, draw a vertical dashed line at the structural break.
    window : int
        Window size (for axis labelling only).
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    _, ax = plt.subplots(figsize=figsize)
    for name, df in rolling_dict.items():
        ax.plot(df["t"], df["rolling_rmse"], label=name, lw=1.5)

    if break_index is not None:
        ax.axvline(break_index, color="black", linestyle="--", lw=1.5, label="Break point")

    ax.set_xlabel("t")
    ax.set_ylabel(f"Rolling RMSE (window={window})")
    ax.set_title("Rolling Forecast RMSE: Pre- and Post-Break Performance")
    ax.legend(fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Lucas critique summary
# ---------------------------------------------------------------------------


def plot_lucas_critique_results(
    comparison_df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Grouped bar chart showing pre/post RMSE and LSR per model.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`evaluation.lucas_critique.compare_pre_post_performance`.
        Columns: model, pre_rmse, post_rmse, LSR.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: pre vs post RMSE
    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width / 2, comparison_df["pre_rmse"], width, label="Pre-break", color="#1f77b4")
    ax.bar(x + width / 2, comparison_df["post_rmse"], width, label="Post-break", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title("Pre vs Post-Break RMSE")
    ax.legend(fontsize=9)

    # Right: Lucas Sensitivity Ratio
    ax2 = axes[1]
    colors = ["#d62728" if lsr > 1.5 else "#ff7f0e" if lsr > 1.1 else "#2ca02c"
              for lsr in comparison_df["LSR"]]
    ax2.bar(x, comparison_df["LSR"], color=colors)
    ax2.axhline(1.0, color="black", linestyle="--", lw=1.2, label="No degradation (LSR=1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df["model"], rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Lucas Sensitivity Ratio (LSR = post/pre RMSE)")
    ax2.set_title("Lucas Sensitivity Ratio by Model")
    ax2.legend(fontsize=9)

    fig.suptitle("Lucas Critique Analysis: Parameter Stability Post-Break", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar
# ---------------------------------------------------------------------------


def plot_model_comparison_bar(
    summary_df: pd.DataFrame,
    metric: str = "post_rmse",
    figsize: tuple[int, int] = (9, 5),
) -> plt.Axes:
    """Horizontal bar chart comparing models on a single metric.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must have ``model`` and ``metric`` columns.
    metric : str
        Column to plot.
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    df_sorted = summary_df.sort_values(metric)
    _, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("Blues_r", len(df_sorted))
    ax.barh(df_sorted["model"], df_sorted[metric], color=colors)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()
    return ax


# ---------------------------------------------------------------------------
# Transition probability heatmap
# ---------------------------------------------------------------------------


def plot_regime_transition_heatmap(
    transition: np.ndarray,
    regime_labels: list[str] | None = None,
    title: str = "Regime Transition Matrix",
    figsize: tuple[int, int] = (5, 4),
) -> plt.Axes:
    """Heatmap of a Markov transition matrix.

    Parameters
    ----------
    transition : np.ndarray, shape (K, K)
    regime_labels : list[str] or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    K = transition.shape[0]
    labels = regime_labels or [f"Regime {k}" for k in range(K)]
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        transition,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("To regime")
    ax.set_ylabel("From regime")
    return ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Save figure to the analyses/figures directory.

    Parameters
    ----------
    fig : Figure
    name : str
        Filename without extension (saves as PNG).
    subdir : str
        Optional subdirectory under figures/.

    Returns
    -------
    Path to saved file.
    """
    save_dir = _FIGURE_DIR / subdir if subdir else _FIGURE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
