"""Hour-wise empirical PIT comparator with the shared RAHC tail rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .v2_calibration import apply_probability_map


def midrank_pit(scenarios: np.ndarray, observations: np.ndarray) -> np.ndarray:
    values = np.asarray(scenarios)
    truth = np.asarray(observations)
    if values.ndim != 3 or truth.shape != (len(values), values.shape[2]):
        raise ValueError("scenario and observation shapes do not align")
    less = np.sum(values < truth[:, None, :], axis=1)
    equal = np.sum(values == truth[:, None, :], axis=1)
    return (less + 0.5 * equal + 0.5) / (values.shape[1] + 1.0)


@dataclass
class HourEmpiricalCalibrator:
    sorted_pits: np.ndarray
    grid_size: int = 4097

    @classmethod
    def fit(cls, scenarios: np.ndarray, observations: np.ndarray) -> "HourEmpiricalCalibrator":
        return cls(np.sort(midrank_pit(scenarios, observations), axis=0))

    def inverse_probabilities(self, members: int, strength: float) -> np.ndarray:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0,1]")
        grid = np.linspace(0.0, 1.0, self.grid_size)
        desired = (np.arange(members, dtype=np.float64) + 0.5) / members
        empirical_level = (np.arange(len(self.sorted_pits), dtype=np.float64) + 0.5) / len(
            self.sorted_pits
        )
        result = np.empty((members, self.sorted_pits.shape[1]), dtype=np.float64)
        for hour in range(self.sorted_pits.shape[1]):
            empirical_cdf = np.interp(
                grid,
                self.sorted_pits[:, hour],
                empirical_level,
                left=0.0,
                right=1.0,
            )
            blended = (1.0 - strength) * grid + strength * empirical_cdf
            result[:, hour] = np.interp(desired, np.maximum.accumulate(blended), grid)
        return np.clip(result, 0.0, 1.0)

    def transform(
        self,
        scenarios: np.ndarray,
        *,
        strength: float,
        tail_rule: str = "bounded",
    ) -> np.ndarray:
        probabilities = self.inverse_probabilities(scenarios.shape[1], strength)
        broadcast = np.broadcast_to(
            probabilities[None, :, :],
            (len(scenarios), scenarios.shape[1], scenarios.shape[2]),
        ).copy()
        return apply_probability_map(scenarios, broadcast, tail_rule=tail_rule)


__all__ = ["HourEmpiricalCalibrator", "midrank_pit"]
