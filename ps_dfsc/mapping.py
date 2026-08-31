from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class WindFarmMapping:
    """Deterministic mapping from ten GEFCom zones to six RTS wind farms."""

    groups: tuple[tuple[int, ...], ...]
    wind_buses: tuple[int, ...] = (3, 5, 7, 16, 21, 23)
    zone_capacity_mw: float = 120.0

    def __post_init__(self) -> None:
        if len(self.groups) != 6:
            raise ValueError("mapping must contain six wind-farm groups")
        flattened = tuple(zone for group in self.groups for zone in group)
        if sorted(flattened) != list(range(10)):
            raise ValueError("mapping must contain every zero-based zone exactly once")
        if sorted(map(len, self.groups)) != [1, 1, 2, 2, 2, 2]:
            raise ValueError("mapping must contain four pairs and two singletons")
        if len(self.wind_buses) != 6 or len(set(self.wind_buses)) != 6:
            raise ValueError("wind_buses must contain six distinct buses")
        if self.zone_capacity_mw <= 0.0:
            raise ValueError("zone_capacity_mw must be positive")

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map normalized [...,10,24] trajectories to MW [...,6,24]."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or array.shape[-2:] != (10, 24):
            raise ValueError("values must end with [10,24]")
        if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise ValueError("normalized zone power must be finite and in [0,1]")
        farms = [
            array[..., list(group), :].sum(axis=-2) * self.zone_capacity_mw
            for group in self.groups
        ]
        return np.stack(farms, axis=-2)

    def cyclic(self, shift: int) -> "WindFarmMapping":
        shift = int(shift) % 6
        return WindFarmMapping(
            groups=self.groups,
            wind_buses=self.wind_buses[shift:] + self.wind_buses[:shift],
            zone_capacity_mw=self.zone_capacity_mw,
        )


def _best_four_pairs(correlation: np.ndarray) -> tuple[tuple[int, int], ...]:
    best_score = -np.inf
    best: tuple[tuple[int, int], ...] | None = None

    def search(
        remaining: tuple[int, ...],
        pairs: tuple[tuple[int, int], ...],
        singles_left: int,
        score: float,
    ) -> None:
        nonlocal best_score, best
        pairs_needed = 4 - len(pairs)
        if pairs_needed == 0:
            if len(remaining) != singles_left:
                return
            canonical = tuple(sorted(tuple(sorted(pair)) for pair in pairs))
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and (best is None or canonical < best)
            ):
                best_score, best = score, canonical
            return
        if len(remaining) < 2 * pairs_needed + singles_left:
            return
        first = remaining[0]
        if singles_left:
            search(remaining[1:], pairs, singles_left - 1, score)
        for position in range(1, len(remaining)):
            second = remaining[position]
            rest = remaining[1:position] + remaining[position + 1 :]
            search(
                rest,
                pairs + ((first, second),),
                singles_left,
                score + float(correlation[first, second]),
            )

    search(tuple(range(10)), (), 2, 0.0)
    if best is None:
        raise RuntimeError("failed to construct constrained wind-farm groups")
    return best


def fit_wind_farm_mapping(
    training_power: np.ndarray,
    *,
    wind_buses: tuple[int, ...] = (3, 5, 7, 16, 21, 23),
    zone_capacity_mw: float = 120.0,
) -> WindFarmMapping:
    """Fit four correlated pairs and two singletons using training targets only."""

    values = np.asarray(training_power, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (10, 24):
        raise ValueError("training_power must have shape [day,10,24]")
    if not np.isfinite(values).all():
        raise ValueError("training_power contains non-finite values")
    flattened = values.transpose(1, 0, 2).reshape(10, -1)
    correlation = np.corrcoef(flattened)
    correlation = np.nan_to_num(correlation, nan=-1.0)
    pairs = _best_four_pairs(correlation)
    paired = {zone for pair in pairs for zone in pair}
    singles = tuple((zone,) for zone in range(10) if zone not in paired)
    groups = tuple(sorted((*pairs, *singles), key=lambda item: min(item)))
    return WindFarmMapping(
        groups=groups,
        wind_buses=wind_buses,
        zone_capacity_mw=zone_capacity_mw,
    )


def all_cyclic_mappings(mapping: WindFarmMapping) -> tuple[WindFarmMapping, ...]:
    return tuple(mapping.cyclic(shift) for shift in range(6))


__all__ = ["WindFarmMapping", "all_cyclic_mappings", "fit_wind_farm_mapping"]
