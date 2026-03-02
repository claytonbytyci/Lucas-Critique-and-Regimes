"""Regime-switching model implementations.

All models expose a common sklearn-style interface:

    model.fit(df_train)          -> self
    model.predict(df)            -> np.ndarray  (y predictions)
    model.predict_regimes(df)    -> np.ndarray  (integer regime labels)

The ``df`` argument is always a pandas DataFrame with at least the columns
produced by :class:`simulation.dgp.MarkovSwitchingDGP`.
"""

from __future__ import annotations

from .hmm import HMMRegimeModel
from .markov_switching import MarkovSwitchingModel
from .ml_regime import MLRegimeModel
from .mixture_experts import MixtureOfExpertsModel
from .threshold import ThresholdModel

__all__ = [
    "MarkovSwitchingModel",
    "HMMRegimeModel",
    "ThresholdModel",
    "MLRegimeModel",
    "MixtureOfExpertsModel",
]
