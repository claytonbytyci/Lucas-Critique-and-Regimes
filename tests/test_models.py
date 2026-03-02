"""Tests for all regime-switching model implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from simulation.dgp import MarkovSwitchingDGP

hmmlearn = pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
xgboost = pytest.importorskip("xgboost", reason="xgboost not installed")

from models.hmm import HMMRegimeModel
from models.threshold import ThresholdModel
from models.ml_regime import MLRegimeModel
from models.mixture_experts import MixtureOfExpertsModel


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    """A consistent 300-row training DataFrame."""
    dgp = MarkovSwitchingDGP(seed=0)
    return dgp.simulate(n_obs=300)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    """A consistent 100-row test DataFrame (different seed)."""
    dgp = MarkovSwitchingDGP(seed=7)
    return dgp.simulate(n_obs=100)


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------


class TestHMMRegimeModel:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        result = model.fit(train_df)
        assert result is model

    def test_predict_shape(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        model.fit(train_df)
        preds = model.predict(test_df)
        assert preds.shape == (len(test_df),)

    def test_predict_regimes_shape(self, train_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert regimes.shape == (len(train_df),)

    def test_predict_regimes_binary(self, train_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert set(regimes).issubset({0, 1})

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2)
        with pytest.raises(RuntimeError):
            model.predict(test_df)

    def test_regime_probabilities_shape(self, train_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        model.fit(train_df)
        probs = model.regime_probabilities(train_df)
        assert probs.shape == (len(train_df), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predictions_are_finite(self, train_df: pd.DataFrame) -> None:
        model = HMMRegimeModel(n_components=2, random_state=0)
        model.fit(train_df)
        preds = model.predict(train_df)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Threshold (TAR)
# ---------------------------------------------------------------------------


class TestThresholdModel:
    def test_fit_sets_threshold(self, train_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        assert model.threshold is not None
        assert isinstance(model.threshold, float)

    def test_threshold_in_data_range(self, train_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        z = train_df["y_lag1"]
        assert z.min() <= model.threshold <= z.max()

    def test_predict_shape(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        preds = model.predict(test_df)
        assert preds.shape == (len(test_df),)

    def test_predict_regimes_binary(self, train_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert set(regimes).issubset({0, 1})

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        with pytest.raises(RuntimeError):
            model.predict(test_df)

    def test_predictions_finite(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        preds = model.predict(test_df)
        assert np.all(np.isfinite(preds))

    def test_both_regimes_used(self, train_df: pd.DataFrame) -> None:
        model = ThresholdModel()
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        # With 300 obs, both regimes should be active
        assert 0 in regimes and 1 in regimes


# ---------------------------------------------------------------------------
# ML Regime (XGBoost)
# ---------------------------------------------------------------------------


class TestMLRegimeModel:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = MLRegimeModel(n_regimes=2)
        result = model.fit(train_df)
        assert result is model

    def test_predict_shape(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = MLRegimeModel(n_regimes=2)
        model.fit(train_df)
        preds = model.predict(test_df)
        assert preds.shape == (len(test_df),)

    def test_predict_regimes_binary(self, train_df: pd.DataFrame) -> None:
        model = MLRegimeModel(n_regimes=2)
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert set(regimes).issubset({0, 1})

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        model = MLRegimeModel(n_regimes=2)
        with pytest.raises(RuntimeError):
            model.predict(test_df)

    def test_regime_probabilities_sum_to_one(
        self, train_df: pd.DataFrame
    ) -> None:
        model = MLRegimeModel(n_regimes=2)
        model.fit(train_df)
        probs = model.regime_probabilities(train_df)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predictions_finite(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = MLRegimeModel(n_regimes=2)
        model.fit(train_df)
        preds = model.predict(test_df)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------


class TestMixtureOfExpertsModel:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel(n_experts=2, n_iter=20, random_state=0)
        result = model.fit(train_df)
        assert result is model

    def test_predict_shape(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel(n_experts=2, n_iter=20, random_state=0)
        model.fit(train_df)
        preds = model.predict(test_df)
        assert preds.shape == (len(test_df),)

    def test_predict_regimes_binary(self, train_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel(n_experts=2, n_iter=20, random_state=0)
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert set(regimes).issubset({0, 1})

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel()
        with pytest.raises(RuntimeError):
            model.predict(test_df)

    def test_predictions_finite(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel(n_experts=2, n_iter=20, random_state=0)
        model.fit(train_df)
        preds = model.predict(test_df)
        assert np.all(np.isfinite(preds))

    def test_soft_probabilities_sum_to_one(self, train_df: pd.DataFrame) -> None:
        model = MixtureOfExpertsModel(n_experts=2, n_iter=20, random_state=0)
        model.fit(train_df)
        probs = model.regime_probabilities(train_df)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
