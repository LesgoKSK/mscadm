from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibratedDistribution:
    """A full weighted distribution and its reduced SUC representation."""

    full_scenarios: np.ndarray
    probabilities: np.ndarray
    suc_scenarios: np.ndarray
    suc_probabilities: np.ndarray
    ess: float
    entropy: float
    transport_cost: float
    used_fallback: bool = False
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        full = np.asarray(self.full_scenarios)
        probability = np.asarray(self.probabilities)
        reduced = np.asarray(self.suc_scenarios)
        reduced_probability = np.asarray(self.suc_probabilities)
        if full.ndim != 3 or full.shape[1:] != (10, 24):
            raise ValueError("full_scenarios must have shape [member,10,24]")
        if probability.shape != (len(full),):
            raise ValueError("probabilities do not align with full_scenarios")
        if reduced.ndim != 3 or reduced.shape[1:] != (6, 24):
            raise ValueError("suc_scenarios must have shape [cluster,6,24]")
        if reduced_probability.shape != (len(reduced),):
            raise ValueError("suc_probabilities do not align with suc_scenarios")
        for name, value in (
            ("full_scenarios", full),
            ("probabilities", probability),
            ("suc_scenarios", reduced),
            ("suc_probabilities", reduced_probability),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if full.min() < 0.0 or full.max() > 1.0:
            raise ValueError("full_scenarios must lie in [0,1]")
        if np.any(reduced < 0.0):
            raise ValueError("suc_scenarios cannot contain negative power")
        if np.any(probability < 0.0) or not np.isclose(probability.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")
        if np.any(reduced_probability < 0.0) or not np.isclose(
            reduced_probability.sum(), 1.0
        ):
            raise ValueError("suc_probabilities must be non-negative and sum to one")

