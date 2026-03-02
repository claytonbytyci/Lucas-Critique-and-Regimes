"""Simulation utilities: data-generating processes and Lucas-critique structural breaks."""

from __future__ import annotations

from .dgp import MarkovSwitchingDGP, RegimeParams
from .lucas_shift import (
    LucasShift,
    MILD_SHIFT,
    SEVERE_SHIFT,
    apply_lucas_shift,
    simulate_pre_post_break,
    concatenate_periods,
)

__all__ = [
    "MarkovSwitchingDGP",
    "RegimeParams",
    "LucasShift",
    "MILD_SHIFT",
    "SEVERE_SHIFT",
    "apply_lucas_shift",
    "simulate_pre_post_break",
    "concatenate_periods",
]
