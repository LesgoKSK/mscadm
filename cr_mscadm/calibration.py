from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate(scenarios: np.ndarray, observations: np.ndarray | None = None) -> None:
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape [days, members, hours]")
    if observations is not None and observations.shape != (len(scenarios), scenarios.shape[2]):
        raise ValueError("observations must have shape [days, hours]")


def pit_values(scenarios: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Mid-rank finite-ensemble PIT values with boundary smoothing."""

    _validate(scenarios, observations)
    less = np.sum(scenarios < observations[:, None, :], axis=1)
    equal = np.sum(scenarios == observations[:, None, :], axis=1)
    return (less + 0.5 * equal + 0.5) / (scenarios.shape[1] + 1.0)


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    """Stable zero-based ranks along the ensemble axis."""

    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    days = np.arange(values.shape[0])[:, None, None]
    hours = np.arange(values.shape[2])[None, None, :]
    ranks[days, order, hours] = np.arange(values.shape[1])[None, :, None]
    return ranks


@dataclass
class CopulaPITCalibrator:
    """Validation-only marginal PIT calibration preserving ensemble ordering.

    The calibrated CDF is ``G_h(y)=H_h(F_h(y))``. Sampling from ``G`` uses
    raw probability ``H_h^{-1}(q)``. Applying the same increasing rank map at
    every hour introduces no pairwise rank inversions, so each conditional
    ensemble's temporal empirical copula is retained.
    """

    sorted_pits: np.ndarray
    strength: float = 1.0
    grid_size: int = 4097

    @classmethod
    def fit(
        cls,
        scenarios: np.ndarray,
        observations: np.ndarray,
        *,
        strength: float = 1.0,
        grid_size: int = 4097,
    ) -> "CopulaPITCalibrator":
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        pits = pit_values(scenarios, observations)
        return cls(np.sort(pits, axis=0), float(strength), int(grid_size))

    @property
    def hours(self) -> int:
        return self.sorted_pits.shape[1]

    def _inverse_probabilities(self, members: int) -> np.ndarray:
        grid = np.linspace(0.0, 1.0, self.grid_size)
        desired = (np.arange(members, dtype=np.float64) + 0.5) / members
        result = np.empty((members, self.hours), dtype=np.float64)
        empirical_levels = (np.arange(len(self.sorted_pits), dtype=np.float64) + 0.5) / len(
            self.sorted_pits
        )
        for hour in range(self.hours):
            empirical_cdf = np.interp(
                grid,
                self.sorted_pits[:, hour],
                empirical_levels,
                left=0.0,
                right=1.0,
            )
            blended = (1.0 - self.strength) * grid + self.strength * empirical_cdf
            blended = np.maximum.accumulate(blended)
            result[:, hour] = np.interp(desired, blended, grid)
        return np.clip(result, 0.0, 1.0)

    def transform(self, scenarios: np.ndarray) -> np.ndarray:
        _validate(scenarios)
        if scenarios.shape[2] != self.hours:
            raise ValueError("scenario hour count does not match calibrator")
        days, members, hours = scenarios.shape
        raw_probabilities = self._inverse_probabilities(members)
        sorted_values = np.sort(scenarios.astype(np.float64), axis=1, kind="stable")
        source_probability = (np.arange(members, dtype=np.float64) + 0.5) / members
        calibrated_sorted = np.empty_like(sorted_values)
        for day in range(days):
            for hour in range(hours):
                calibrated_sorted[day, :, hour] = np.interp(
                    raw_probabilities[:, hour],
                    source_probability,
                    sorted_values[day, :, hour],
                    left=sorted_values[day, 0, hour],
                    right=sorted_values[day, -1, hour],
                )
        ranks = _ordinal_ranks(scenarios)
        calibrated = np.take_along_axis(calibrated_sorted, ranks, axis=1)
        return calibrated.astype(scenarios.dtype, copy=False)

    def to_dict(self) -> dict:
        return {
            "sorted_pits": self.sorted_pits.tolist(),
            "strength": self.strength,
            "grid_size": self.grid_size,
            "definition": "G_h(y)=H_h(F_h(y)); sample with F_h^-1(H_h^-1(q))",
        }


def count_rank_inversions(before: np.ndarray, after: np.ndarray) -> int:
    """Count strict pairwise ordering reversals (ties are not reversals)."""

    _validate(before)
    _validate(after)
    if before.shape != after.shape:
        raise ValueError("arrays must have identical shapes")
    inversions = 0
    members = before.shape[1]
    for left in range(members):
        for right in range(left + 1, members):
            sign_before = np.sign(before[:, left, :] - before[:, right, :])
            sign_after = np.sign(after[:, left, :] - after[:, right, :])
            inversions += int(np.sum(sign_before * sign_after < 0))
    return inversions


__all__ = ["CopulaPITCalibrator", "pit_values", "count_rank_inversions"]
