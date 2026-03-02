"""Tests for evaluation.metrics and evaluation.lucas_critique."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from evaluation.metrics import (
    forecast_rmse,
    forecast_mae,
    directional_accuracy,
    regime_accuracy,
    adjusted_rand_regime,
    regime_conditional_rmse,
    lucas_sensitivity_ratio,
    summarise_model_performance,
)
from evaluation.lucas_critique import (
    chow_test,
    recursive_cusum,
    compute_rolling_performance,
    compare_pre_post_performance,
)


# ---------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------


class TestForecastMetrics:
    def test_rmse_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert forecast_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert forecast_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_perfect_prediction(self) -> None:
        y = np.array([5.0, -3.0, 2.0])
        assert forecast_mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([2.0, 4.0])
        assert forecast_mae(y_true, y_pred) == pytest.approx(3.0)

    def test_rmse_greater_equal_mae(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        assert forecast_rmse(y_true, y_pred) >= forecast_mae(y_true, y_pred)

    def test_directional_accuracy_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        assert directional_accuracy(y, y) == pytest.approx(1.0)

    def test_directional_accuracy_opposite(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_directional_accuracy_range(self) -> None:
        rng = np.random.default_rng(1)
        y_true = rng.normal(0, 1, 50)
        y_pred = rng.normal(0, 1, 50)
        da = directional_accuracy(y_true, y_pred)
        assert 0.0 <= da <= 1.0

    def test_directional_accuracy_too_short(self) -> None:
        assert np.isnan(directional_accuracy(np.array([1.0]), np.array([1.0])))


# ---------------------------------------------------------------------------
# Regime metrics
# ---------------------------------------------------------------------------


class TestRegimeMetrics:
    def test_regime_accuracy_perfect(self) -> None:
        true = np.array([0, 1, 0, 0, 1])
        assert regime_accuracy(true, true) == pytest.approx(1.0)

    def test_regime_accuracy_with_permutation(self) -> None:
        true = np.array([0, 0, 1, 1, 0])
        pred = np.array([1, 1, 0, 0, 1])  # exact inverse
        # With permutation allowed, should return 1.0
        assert regime_accuracy(true, pred, allow_permutation=True) == pytest.approx(1.0)

    def test_regime_accuracy_range(self) -> None:
        rng = np.random.default_rng(2)
        true = rng.integers(0, 2, 100)
        pred = rng.integers(0, 2, 100)
        acc = regime_accuracy(true, pred)
        assert 0.0 <= acc <= 1.0

    def test_adjusted_rand_perfect(self) -> None:
        labels = np.array([0, 0, 1, 1, 0, 1])
        assert adjusted_rand_regime(labels, labels) == pytest.approx(1.0)

    def test_adjusted_rand_range(self) -> None:
        rng = np.random.default_rng(3)
        true = rng.integers(0, 2, 80)
        pred = rng.integers(0, 2, 80)
        ari = adjusted_rand_regime(true, pred)
        assert -1.0 <= ari <= 1.0

    def test_regime_conditional_rmse_keys(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        regimes = np.array([0, 0, 1, 1])
        result = regime_conditional_rmse(y_true, y_pred, regimes)
        assert set(result.keys()) == {0, 1}
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lucas critique metrics
# ---------------------------------------------------------------------------


class TestLucasMetrics:
    def test_lsr_no_change(self) -> None:
        assert lucas_sensitivity_ratio(1.0, 1.0) == pytest.approx(1.0)

    def test_lsr_degradation(self) -> None:
        assert lucas_sensitivity_ratio(1.0, 2.0) == pytest.approx(2.0)

    def test_lsr_improvement(self) -> None:
        assert lucas_sensitivity_ratio(2.0, 1.0) == pytest.approx(0.5)

    def test_summarise_model_performance(self) -> None:
        results = {
            "ModelA": {"pre_rmse": 1.0, "post_rmse": 2.0},
            "ModelB": {"pre_rmse": 1.0, "post_rmse": 1.2},
        }
        df = summarise_model_performance(results)
        assert "model" in df.columns
        assert "LSR" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Structural break tests
# ---------------------------------------------------------------------------


class TestChowTest:
    def _make_data(self, n: int = 200, break_at: int = 100) -> tuple:
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = np.zeros(n)
        # Regime 1: intercept 0
        y[:break_at] = X[:break_at] @ np.array([0.0, 1.0]) + rng.normal(0, 0.5, break_at)
        # Regime 2: intercept 5 (big shift)
        y[break_at:] = X[break_at:] @ np.array([5.0, 1.0]) + rng.normal(0, 0.5, n - break_at)
        return y, X

    def test_chow_detects_large_break(self) -> None:
        y, X = self._make_data()
        result = chow_test(y, X, break_index=100)
        assert result["reject_H0"] is True

    def test_chow_no_break_data(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 1.0, n)
        result = chow_test(y, X, break_index=100)
        # May or may not reject — just check it runs
        assert "F_stat" in result
        assert "p_value" in result

    def test_chow_keys(self) -> None:
        y, X = self._make_data()
        result = chow_test(y, X, break_index=100)
        for key in ("F_stat", "p_value", "df1", "df2", "reject_H0"):
            assert key in result

    def test_chow_boundary_raises(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = rng.normal(0, 1, n)
        with pytest.raises(ValueError):
            chow_test(y, X, break_index=1)  # too close to start


class TestRollingPerformance:
    def test_rolling_rmse_shape(self) -> None:
        y_true = np.random.default_rng(0).normal(0, 1, 100)
        y_pred = np.random.default_rng(1).normal(0, 1, 100)
        df = compute_rolling_performance(y_true, y_pred, window=20)
        assert len(df) == 100
        assert "rolling_rmse" in df.columns

    def test_rolling_rmse_perfect_prediction(self) -> None:
        y = np.ones(50)
        df = compute_rolling_performance(y, y, window=10)
        # After warm-up, RMSE should be 0
        assert df["rolling_rmse"].dropna().max() == pytest.approx(0.0, abs=1e-10)

    def test_rolling_rmse_non_negative(self) -> None:
        rng = np.random.default_rng(5)
        y_true = rng.normal(0, 1, 80)
        y_pred = rng.normal(0, 1, 80)
        df = compute_rolling_performance(y_true, y_pred, window=15)
        assert (df["rolling_rmse"].dropna() >= 0).all()


class TestComparePrePost:
    def test_compare_returns_dataframe(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        y_full = rng.normal(0, 1, n)
        predictions = {
            "ModelA": rng.normal(0, 1, n),
            "ModelB": y_full + rng.normal(0, 0.1, n),
        }
        df = compare_pre_post_performance(predictions, y_full, break_index=60)
        assert "model" in df.columns
        assert "LSR" in df.columns
        assert len(df) == 2

    def test_compare_lsr_positive(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        y_full = rng.normal(0, 1, n)
        predictions = {"M": rng.normal(0, 1, n)}
        df = compare_pre_post_performance(predictions, y_full, break_index=60)
        assert (df["LSR"] > 0).all()
