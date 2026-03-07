"""Tests for all regime-switching model implementations."""

from __future__ import annotations

import importlib.util
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
from models.markov_switching_nn import MarkovSwitchingNeuralNetwork
from models.linear_baselines import ARModel, ARMAModel, ModelAverageEnsemble

_statsmodels_available = importlib.util.find_spec("statsmodels") is not None


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


# ---------------------------------------------------------------------------
# Markov Switching Neural Network (MSNN)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_msnn(train_df: pd.DataFrame) -> MarkovSwitchingNeuralNetwork:
    """Pre-fitted MSNN with minimal iterations for speed."""
    model = MarkovSwitchingNeuralNetwork(
        k_regimes=2, n_iter=2, mlp_epochs=5, random_state=0
    )
    model.fit(train_df)
    return model


class TestMarkovSwitchingNeuralNetwork:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = MarkovSwitchingNeuralNetwork(
            k_regimes=2, n_iter=2, mlp_epochs=5, random_state=0
        )
        assert model.fit(train_df) is model

    def test_predict_shape(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork, test_df: pd.DataFrame
    ) -> None:
        preds = fitted_msnn.predict(test_df)
        assert preds.shape == (len(test_df),)

    def test_predict_finite(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork, test_df: pd.DataFrame
    ) -> None:
        preds = fitted_msnn.predict(test_df)
        assert np.all(np.isfinite(preds))

    def test_predict_regimes_shape(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork, test_df: pd.DataFrame
    ) -> None:
        regimes = fitted_msnn.predict_regimes(test_df)
        assert regimes.shape == (len(test_df),)

    def test_predict_regimes_valid_values(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork, test_df: pd.DataFrame
    ) -> None:
        regimes = fitted_msnn.predict_regimes(test_df)
        assert set(regimes).issubset({0, 1})

    def test_regime_probabilities_shape_and_sum(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork, train_df: pd.DataFrame
    ) -> None:
        probs = fitted_msnn.regime_probabilities(train_df)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape == (len(train_df), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_transition_matrix_shape_and_rows(
        self, fitted_msnn: MarkovSwitchingNeuralNetwork
    ) -> None:
        A = fitted_msnn.transition_matrix
        assert isinstance(A, pd.DataFrame)
        assert A.shape == (2, 2)
        assert np.allclose(A.values.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        model = MarkovSwitchingNeuralNetwork()
        with pytest.raises(Exception):
            model.predict(test_df)

    def test_three_regime_fit(self, train_df: pd.DataFrame) -> None:
        model = MarkovSwitchingNeuralNetwork(
            k_regimes=3, n_iter=2, mlp_epochs=5, random_state=0
        )
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert set(regimes).issubset({0, 1, 2})

    def test_reproducibility(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        m1 = MarkovSwitchingNeuralNetwork(
            k_regimes=2, n_iter=2, mlp_epochs=5, random_state=99
        )
        m2 = MarkovSwitchingNeuralNetwork(
            k_regimes=2, n_iter=2, mlp_epochs=5, random_state=99
        )
        m1.fit(train_df)
        m2.fit(train_df)
        np.testing.assert_array_equal(m1.predict(test_df), m2.predict(test_df))


# ---------------------------------------------------------------------------
# AR Model
# ---------------------------------------------------------------------------


class TestARModel:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = ARModel(order=2)
        assert model.fit(train_df) is model

    def test_predict_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARModel(order=2)
        model.fit(train_df)
        assert model.predict(test_df).shape == (len(test_df),)

    def test_predict_finite(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARModel(order=2)
        model.fit(train_df)
        assert np.all(np.isfinite(model.predict(test_df)))

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            ARModel(order=2).predict(test_df)

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError):
            ARModel(order=3)

    def test_predict_regimes_all_zeros(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARModel(order=2)
        model.fit(train_df)
        regimes = model.predict_regimes(test_df)
        assert np.all(regimes == 0)
        assert regimes.shape == (len(test_df),)

    def test_ridge_vs_ols_differ(self, train_df: pd.DataFrame) -> None:
        ols = ARModel(order=2, alpha_ridge=0.0)
        ridge = ARModel(order=2, alpha_ridge=100.0)
        ols.fit(train_df)
        ridge.fit(train_df)
        assert not np.allclose(ols._coef, ridge._coef)

    def test_order_one(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARModel(order=1, include_exog=False)
        model.fit(train_df)
        assert model.predict(test_df).shape == (len(test_df),)

    def test_summary_not_fitted(self) -> None:
        assert "not fitted" in ARModel().summary()

    def test_summary_fitted(self, train_df: pd.DataFrame) -> None:
        model = ARModel()
        model.fit(train_df)
        assert "ARModel" in model.summary()


# ---------------------------------------------------------------------------
# ARMA Model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _statsmodels_available, reason="statsmodels not installed")
class TestARMAModel:
    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = ARMAModel(p=1, q=0)
        assert model.fit(train_df) is model

    def test_predict_in_sample_shape(self, train_df: pd.DataFrame) -> None:
        model = ARMAModel(p=1, q=0)
        model.fit(train_df)
        assert model.predict(train_df).shape == (len(train_df),)

    def test_predict_out_of_sample_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARMAModel(p=1, q=0)
        model.fit(train_df)
        assert model.predict(test_df).shape == (len(test_df),)

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            ARMAModel().predict(test_df)

    def test_predict_regimes_all_zeros(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARMAModel(p=1, q=0)
        model.fit(train_df)
        regimes = model.predict_regimes(test_df)
        assert np.all(regimes == 0)
        assert regimes.shape == (len(test_df),)

    def test_predictions_finite(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = ARMAModel(p=1, q=0)
        model.fit(train_df)
        assert np.all(np.isfinite(model.predict(test_df)))

    def test_summary_fitted(self, train_df: pd.DataFrame) -> None:
        model = ARMAModel(p=1, q=0)
        model.fit(train_df)
        assert len(model.summary()) > 0

    def test_summary_not_fitted(self) -> None:
        assert "not fitted" in ARMAModel().summary()


# ---------------------------------------------------------------------------
# Model Average Ensemble
# ---------------------------------------------------------------------------


class TestModelAverageEnsemble:
    def _fitted_pair(self, train_df: pd.DataFrame) -> dict:
        m1 = ARModel(order=1, include_exog=False)
        m2 = ARModel(order=2, include_exog=False)
        m1.fit(train_df)
        m2.fit(train_df)
        return {"AR1": m1, "AR2": m2}

    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        m1 = ARModel(order=1, include_exog=False)
        m2 = ARModel(order=2, include_exog=False)
        ensemble = ModelAverageEnsemble({"AR1": m1, "AR2": m2})
        assert ensemble.fit(train_df) is ensemble

    def test_predict_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        ensemble = ModelAverageEnsemble(self._fitted_pair(train_df))
        assert ensemble.predict(test_df).shape == (len(test_df),)

    def test_predict_finite(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        ensemble = ModelAverageEnsemble(self._fitted_pair(train_df))
        assert np.all(np.isfinite(ensemble.predict(test_df)))

    def test_predict_equals_mean_of_members(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        models = self._fitted_pair(train_df)
        expected = np.mean(
            [m.predict(test_df) for m in models.values()], axis=0
        )
        ensemble = ModelAverageEnsemble(models)
        np.testing.assert_allclose(ensemble.predict(test_df), expected, rtol=1e-5)

    def test_weighted_predict_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        ensemble = ModelAverageEnsemble(
            self._fitted_pair(train_df), weights=[0.3, 0.7]
        )
        assert ensemble.predict(test_df).shape == (len(test_df),)

    def test_predict_regimes_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        ensemble = ModelAverageEnsemble(self._fitted_pair(train_df))
        assert ensemble.predict_regimes(test_df).shape == (len(test_df),)

    def test_empty_models_returns_zeros(self, test_df: pd.DataFrame) -> None:
        ensemble = ModelAverageEnsemble({})
        preds = ensemble.predict(test_df)
        assert np.all(preds == 0)
        assert preds.shape == (len(test_df),)

    def test_summary_contains_model_names(self, train_df: pd.DataFrame) -> None:
        ensemble = ModelAverageEnsemble(self._fitted_pair(train_df))
        summary = ensemble.summary()
        assert "AR1" in summary and "AR2" in summary


# ---------------------------------------------------------------------------
# Markov Switching Model (statsmodels)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _statsmodels_available, reason="statsmodels not installed")
class TestMarkovSwitchingModel:
    def _model(self) -> object:
        from models.markov_switching import MarkovSwitchingModel
        return MarkovSwitchingModel(k_regimes=2, max_iter=30)

    def test_fit_returns_self(self, train_df: pd.DataFrame) -> None:
        model = self._model()
        assert model.fit(train_df) is model

    def test_predict_in_sample_shape(self, train_df: pd.DataFrame) -> None:
        model = self._model()
        model.fit(train_df)
        assert model.predict(train_df).shape == (len(train_df),)

    def test_predict_out_of_sample_shape(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        model = self._model()
        model.fit(train_df)
        assert model.predict(test_df).shape == (len(test_df),)

    def test_predict_before_fit_raises(self, test_df: pd.DataFrame) -> None:
        from models.markov_switching import MarkovSwitchingModel
        model = MarkovSwitchingModel(k_regimes=2)
        with pytest.raises(RuntimeError):
            model.predict(test_df)

    def test_predict_regimes_in_sample_shape(self, train_df: pd.DataFrame) -> None:
        model = self._model()
        model.fit(train_df)
        regimes = model.predict_regimes(train_df)
        assert regimes.shape == (len(train_df),)
        assert set(regimes).issubset({0, 1})

    def test_regime_probabilities_shape_and_sum(
        self, train_df: pd.DataFrame
    ) -> None:
        model = self._model()
        model.fit(train_df)
        probs = model.regime_probabilities()
        assert probs.shape == (len(train_df), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_regime_probabilities_before_fit_raises(self) -> None:
        from models.markov_switching import MarkovSwitchingModel
        model = MarkovSwitchingModel(k_regimes=2)
        with pytest.raises(RuntimeError):
            model.regime_probabilities()

    def test_summary_fitted(self, train_df: pd.DataFrame) -> None:
        model = self._model()
        model.fit(train_df)
        assert len(model.summary()) > 0

    def test_summary_not_fitted(self) -> None:
        from models.markov_switching import MarkovSwitchingModel
        model = MarkovSwitchingModel(k_regimes=2)
        assert "not fitted" in model.summary()
