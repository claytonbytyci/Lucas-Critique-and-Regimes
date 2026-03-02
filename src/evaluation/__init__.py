"""Evaluation helpers: metrics, Lucas-critique tests, and visualisation."""

from __future__ import annotations

from .lucas_critique import chow_test, compute_rolling_performance, compare_pre_post_performance
from .metrics import (
    regime_accuracy,
    forecast_rmse,
    forecast_mae,
    directional_accuracy,
    regime_conditional_rmse,
    lucas_sensitivity_ratio,
    summarise_model_performance,
)
from .visualization import (
    plot_simulated_series,
    plot_regime_comparison,
    plot_rolling_rmse,
    plot_lucas_critique_results,
    plot_model_comparison_bar,
    plot_regime_transition_heatmap,
)

__all__ = [
    # metrics
    "regime_accuracy",
    "forecast_rmse",
    "forecast_mae",
    "directional_accuracy",
    "regime_conditional_rmse",
    "lucas_sensitivity_ratio",
    "summarise_model_performance",
    # lucas critique
    "chow_test",
    "compute_rolling_performance",
    "compare_pre_post_performance",
    # visualisation
    "plot_simulated_series",
    "plot_regime_comparison",
    "plot_rolling_rmse",
    "plot_lucas_critique_results",
    "plot_model_comparison_bar",
    "plot_regime_transition_heatmap",
]
