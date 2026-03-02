"""Tests for simulation.dgp and simulation.lucas_shift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the src packages importable without installation
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from simulation.dgp import MarkovSwitchingDGP, RegimeParams
from simulation.lucas_shift import (
    LucasShift,
    MILD_SHIFT,
    SEVERE_SHIFT,
    apply_lucas_shift,
    simulate_pre_post_break,
    concatenate_periods,
)


# ---------------------------------------------------------------------------
# DGP tests
# ---------------------------------------------------------------------------


class TestMarkovSwitchingDGP:
    def test_simulate_returns_dataframe(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        df = dgp.simulate(n_obs=200)
        assert isinstance(df, pd.DataFrame)
        assert "y" in df.columns
        assert "regime" in df.columns

    def test_simulate_correct_regime_range(self) -> None:
        dgp = MarkovSwitchingDGP(seed=1)
        df = dgp.simulate(n_obs=300)
        assert df["regime"].min() == 0
        assert df["regime"].max() == 1

    def test_simulate_length_approximately_correct(self) -> None:
        # After dropping NaN from rolling windows, length is slightly reduced
        dgp = MarkovSwitchingDGP(seed=2)
        df = dgp.simulate(n_obs=500)
        assert len(df) >= 480  # at most 20-row warm-up dropped

    def test_both_regimes_present(self) -> None:
        dgp = MarkovSwitchingDGP(seed=3)
        df = dgp.simulate(n_obs=500)
        assert df["regime"].nunique() == 2, "Both regimes should appear with n=500"

    def test_feature_columns_present(self) -> None:
        dgp = MarkovSwitchingDGP(seed=4)
        df = dgp.simulate(n_obs=100)
        for col in ["y_lag1", "roll_mean_5", "roll_std_5", "exog_0"]:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_nan_in_output(self) -> None:
        dgp = MarkovSwitchingDGP(seed=5)
        df = dgp.simulate(n_obs=200)
        assert not df.isnull().any().any(), "Output should contain no NaN"

    def test_stationary_distribution_sums_to_one(self) -> None:
        dgp = MarkovSwitchingDGP()
        pi = dgp.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_reproducibility(self) -> None:
        dgp1 = MarkovSwitchingDGP(seed=99)
        dgp2 = MarkovSwitchingDGP(seed=99)
        df1 = dgp1.simulate(300)
        df2 = dgp2.simulate(300)
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_transition_raises(self) -> None:
        bad_transition = np.array([[0.5, 0.5], [0.5, 0.6]])  # rows don't sum to 1
        with pytest.raises(ValueError, match="sum to 1"):
            MarkovSwitchingDGP(transition=bad_transition)

    def test_invalid_phi_raises(self) -> None:
        bad_regimes = [
            RegimeParams(mu=0.0, phi=1.1, sigma=1.0, label="explosive"),
            RegimeParams(mu=0.5, phi=0.3, sigma=0.5, label="stable"),
        ]
        transition = np.array([[0.9, 0.1], [0.1, 0.9]])
        with pytest.raises(ValueError, match="non-stationary"):
            MarkovSwitchingDGP(regimes=bad_regimes, transition=transition)

    def test_custom_three_regime_dgp(self) -> None:
        regimes = [
            RegimeParams(mu=-1.0, phi=0.6, sigma=2.0, label="recession"),
            RegimeParams(mu=0.2, phi=0.2, sigma=0.8, label="neutral"),
            RegimeParams(mu=1.5, phi=0.3, sigma=0.5, label="boom"),
        ]
        transition = np.array([
            [0.85, 0.10, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.10, 0.85],
        ])
        dgp = MarkovSwitchingDGP(regimes=regimes, transition=transition, seed=7)
        df = dgp.simulate(n_obs=600)
        assert df["regime"].max() <= 2


# ---------------------------------------------------------------------------
# Lucas shift tests
# ---------------------------------------------------------------------------


class TestLucasShift:
    def test_apply_lucas_shift_returns_new_dgp(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        post_dgp = apply_lucas_shift(dgp, MILD_SHIFT)
        assert post_dgp is not dgp

    def test_shift_changes_parameters(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        post_dgp = apply_lucas_shift(dgp, MILD_SHIFT)
        # Recession (regime 0) mu should have shifted
        assert post_dgp.regimes[0].mu != dgp.regimes[0].mu

    def test_shift_preserves_unshifted_regimes(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        shift = LucasShift(delta_mu={0: -2.0})  # only shift regime 0
        post_dgp = apply_lucas_shift(dgp, shift)
        assert post_dgp.regimes[1].mu == dgp.regimes[1].mu
        assert post_dgp.regimes[1].phi == dgp.regimes[1].phi

    def test_simulate_pre_post_break_shapes(self) -> None:
        dgp = MarkovSwitchingDGP(seed=42)
        df_pre, df_post, _ = simulate_pre_post_break(dgp, MILD_SHIFT, n_pre=300, n_post=150)
        assert len(df_pre) >= 280  # after NaN drop from rolling windows
        assert len(df_post) >= 130

    def test_pre_post_period_labels(self) -> None:
        dgp = MarkovSwitchingDGP(seed=42)
        df_pre, df_post, _ = simulate_pre_post_break(dgp, MILD_SHIFT, n_pre=200, n_post=100)
        assert (df_pre["period"] == "pre").all()
        assert (df_post["period"] == "post").all()

    def test_concatenate_periods(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        df_pre, df_post, _ = simulate_pre_post_break(dgp, MILD_SHIFT, n_pre=200, n_post=100)
        df_full = concatenate_periods(df_pre, df_post)
        assert "t" in df_full.columns
        assert "is_post" in df_full.columns
        assert df_full["is_post"].sum() == len(df_post)

    def test_severe_shift_is_more_extreme(self) -> None:
        dgp = MarkovSwitchingDGP(seed=0)
        post_mild = apply_lucas_shift(dgp, MILD_SHIFT)
        post_severe = apply_lucas_shift(dgp, SEVERE_SHIFT)
        delta_mild = abs(post_mild.regimes[0].mu - dgp.regimes[0].mu)
        delta_severe = abs(post_severe.regimes[0].mu - dgp.regimes[0].mu)
        assert delta_severe > delta_mild
