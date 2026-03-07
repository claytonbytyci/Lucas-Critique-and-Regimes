"""Lucas-critique structural-break utilities.

The Lucas Critique (1976) holds that once agents perceive a policy change,
the behavioural parameters of any model estimated on pre-change data are
invalidated.  I operationalise this by:

    1. Estimating a regime model on data from a *pre-break* DGP.
    2. Replacing the DGP with a *post-break* DGP that has shifted parameters.
    3. Evaluating models on the post-break sample.

Performance degradation measures the model's "Lucas sensitivity".
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .dgp import MarkovSwitchingDGP, RegimeParams # importing the dgp and regime parameters for the simulation of the data with shifts

# Lucas shift specification

@dataclass
class LucasShift:
    """Specification of a parameter shift applied to each regime.

    Each field is a *dict* mapping regime index (0-based int) to a new value.
    Regimes not listed keep their original parameter.  Pass ``None`` to skip
    shifting a given parameter.

    Example
    -------
    Shift regime 0's mean from -0.4 to -1.2 and increase noise in both regimes:

    >>> shift = LucasShift(
    ...     delta_mu={0: -1.2},
    ...     delta_sigma={0: 2.5, 1: 1.0},
    ...     delta_phi={},
    ...     delta_transition=None,
    ... )
    """

    delta_mu: dict[int, float] = None           # new absolute mu per regime
    delta_phi: dict[int, float] = None          # new absolute phi per regime
    delta_sigma: dict[int, float] = None        # new absolute sigma per regime
    delta_transition: np.ndarray | None = None  # entire new transition matrix

    def __post_init__(self) -> None: # this is to ensure that if any of the dicts are left unfilled, they just return an empty set
        if self.delta_mu is None:
            self.delta_mu = {}
        if self.delta_phi is None:
            self.delta_phi = {}
        if self.delta_sigma is None:
            self.delta_sigma = {}


# Canonical "mild" Lucas shift: recession deepens, recovery weakens so regimes are more distinct (can play around with the parameters here)
MILD_SHIFT = LucasShift(
    delta_mu={0: -1.2, 1: 0.4},    # recession deeper, expansion weaker
    delta_phi={0: 0.80},            # recession becomes more persistent
    delta_sigma={0: 2.5},           # recession more volatile
    delta_transition=np.array(
        [[0.93, 0.07],
         [0.20, 0.80]]
    ),
)

# Severe shift: regimes almost swap character
SEVERE_SHIFT = LucasShift(
    delta_mu={0: -2.0, 1: 0.1},
    delta_phi={0: 0.85, 1: 0.60},
    delta_sigma={0: 3.0, 1: 1.5},
    delta_transition=np.array(
        [[0.95, 0.05],
         [0.30, 0.70]]
    ),
)

# Core functions

def apply_lucas_shift(dgp: MarkovSwitchingDGP, shift: LucasShift) -> MarkovSwitchingDGP:
    """Return a new DGP with shifted parameters, leaving the original unchanged.

    Parameters:
    dgp : MarkovSwitchingDGP
        The original (pre-break) DGP.
    shift : LucasShift
        Parameter changes to apply.

    Returns:
    MarkovSwitchingDGP
        A new DGP instance with updated parameters.
    """
    new_regimes: list[RegimeParams] = [] # creating the new regimes with shifing parameters
    for i, r in enumerate(dgp.regimes):
        new_regimes.append(
            RegimeParams(
                mu=shift.delta_mu.get(i, r.mu),
                phi=shift.delta_phi.get(i, r.phi),
                sigma=shift.delta_sigma.get(i, r.sigma),
                label=r.label,
            )
        )

    new_transition = (
        shift.delta_transition if shift.delta_transition is not None
        else dgp.transition.copy()
    )

    return MarkovSwitchingDGP(
        regimes=new_regimes,
        transition=new_transition,
        n_exog=dgp.n_exog,
        seed=dgp.seed,  # deterministic post-break sample given same seed
    )


def simulate_pre_post_break(
    dgp: MarkovSwitchingDGP,
    shift: LucasShift,
    n_pre: int = 400,
    n_post: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, MarkovSwitchingDGP]:
    """Simulate pre-break and post-break samples from two different DGPs.

    The post-break DGP is obtained by applying ``shift`` to the pre-break DGP,
    mimicking the Lucas critique scenario: a policy change that alters the
    structural parameters agents optimise against.

    Parameters
    ----------
    dgp : MarkovSwitchingDGP
        Pre-break DGP.
    shift : LucasShift
        Lucas-critique parameter shift.
    n_pre : int
        Number of observations in the pre-break period.
    n_post : int
        Number of observations in the post-break period.

    Returns
    -------
    df_pre : pd.DataFrame
        Simulated pre-break data (with ``period="pre"`` column).
    df_post : pd.DataFrame
        Simulated post-break data (with ``period="post"`` column).
    post_dgp : MarkovSwitchingDGP
        The shifted DGP used to generate ``df_post``.
    """
    # Pre-break: standard seed
    df_pre = dgp.simulate(n_obs=n_pre)
    df_pre["period"] = "pre"

    # Post-break: shifted DGP, different seed to avoid correlating samples
    post_dgp = apply_lucas_shift(dgp, shift)
    # Offset the seed so the noise realisations differ
    post_dgp.seed = (dgp.seed or 0) + 9999
    df_post = post_dgp.simulate(n_obs=n_post)
    df_post["period"] = "post"

    return df_pre, df_post, post_dgp


def concatenate_periods(df_pre: pd.DataFrame, df_post: pd.DataFrame) -> pd.DataFrame:
    """Stack pre and post DataFrames with a continuous integer time index."""
    df = pd.concat([df_pre, df_post], ignore_index=True)
    df["t"] = np.arange(len(df))
    df["is_post"] = (df["period"] == "post").astype(int)
    return df
