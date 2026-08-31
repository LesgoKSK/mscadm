from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from repro.metrics import interval_scores


@dataclass(frozen=True)
class GroupingProtocol:
    wind_edges: np.ndarray
    uncertainty_edges: np.ndarray
    levels: tuple[float, ...] = (0.5, 0.8, 0.9)

    @classmethod
    def fit(cls, scenarios: np.ndarray) -> "GroupingProtocol":
        probabilities = np.linspace(0.0, 1.0, 6)
        return cls(
            np.quantile(scenarios.mean(axis=1).ravel(), probabilities),
            np.quantile(scenarios.std(axis=1).ravel(), probabilities),
        )

    @staticmethod
    def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(values, edges[1:-1]), 0, 4)

    def wind_bin(self, scenarios: np.ndarray) -> np.ndarray:
        return self._bin(scenarios.mean(axis=1), self.wind_edges)

    def uncertainty_bin(self, scenarios: np.ndarray) -> np.ndarray:
        return self._bin(scenarios.std(axis=1), self.uncertainty_edges)

    def to_dict(self) -> dict:
        return {
            "wind_edges": self.wind_edges.tolist(),
            "uncertainty_edges": self.uncertainty_edges.tolist(),
            "levels": list(self.levels),
            "fit_split": "validation raw forecast features only",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GroupingProtocol":
        return cls(
            np.asarray(payload["wind_edges"], dtype=np.float64),
            np.asarray(payload["uncertainty_edges"], dtype=np.float64),
            tuple(float(value) for value in payload["levels"]),
        )


def conditional_interval_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    zone: np.ndarray,
    protocol: GroupingProtocol,
) -> pd.DataFrame:
    if scenarios.ndim != 3 or observations.shape != (len(scenarios), 24):
        raise ValueError("scenario and observation shapes are invalid")
    interval_cache = {
        float(nominal): (
            np.quantile(scenarios, (1.0 - nominal) / 2.0, axis=1),
            np.quantile(scenarios, (1.0 + nominal) / 2.0, axis=1),
        )
        for nominal in protocol.levels
    }
    zone_cells = np.repeat(zone[:, None], 24, axis=1) - 1
    hour_cells = np.repeat(np.arange(24)[None, :], len(scenarios), axis=0)
    wind = protocol.wind_bin(scenarios)
    uncertainty = protocol.uncertainty_bin(scenarios)
    definitions = [
        ("zone", zone_cells, [f"zone_{index}" for index in range(1, 11)]),
        ("hour", hour_cells, [f"hour_{index:02d}" for index in range(1, 25)]),
        ("wind_quintile", wind, [f"wind_q{index}" for index in range(1, 6)]),
        ("uncertainty_quintile", uncertainty, [f"uncertainty_q{index}" for index in range(1, 6)]),
        (
            "zone_x_uncertainty",
            zone_cells * 5 + uncertainty,
            [f"zone_{zone_index}_uncertainty_q{quintile}" for zone_index in range(1, 11) for quintile in range(1, 6)],
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    for group_type, group_values, labels in definitions:
        for index, label in enumerate(labels):
            mask = group_values == index
            count = int(mask.sum())
            if count == 0:
                continue
            for nominal in protocol.levels:
                lower, upper = interval_cache[float(nominal)]
                coverage = float(np.mean((observations[mask] >= lower[mask]) & (observations[mask] <= upper[mask])))
                width = float(np.mean(upper[mask] - lower[mask]))
                rows.append(
                    {
                        "group_type": group_type,
                        "group": label,
                        "nominal": float(nominal),
                        "coverage": coverage,
                        "width": width,
                        "absolute_error": abs(coverage - nominal),
                        "signed_error": coverage - nominal,
                        "n": count,
                    }
                )
    return pd.DataFrame(rows)


def conditional_summary(frame: pd.DataFrame) -> dict[str, float]:
    selected = frame[frame["nominal"] == 0.9]
    primary = selected[selected["group_type"].isin(["zone", "hour", "wind_quintile", "uncertainty_quintile"])]
    intersection = selected[selected["group_type"] == "zone_x_uncertainty"]
    output = {
        "conditional_ACE90": float(primary["absolute_error"].mean()),
        "conditional_worst_coverage90": float(primary["coverage"].min()),
        "conditional_worst_abs_error90": float(primary["absolute_error"].max()),
        "intersection_ACE90": float(intersection["absolute_error"].mean()),
        "intersection_worst_coverage90": float(intersection["coverage"].min()),
        "intersection_worst_abs_error90": float(intersection["absolute_error"].max()),
    }
    for group_type, prefix in [
        ("zone", "zone"),
        ("hour", "hour"),
        ("wind_quintile", "wind"),
        ("uncertainty_quintile", "uncertainty"),
    ]:
        group = selected[selected["group_type"] == group_type]
        output[f"{prefix}_ACE90"] = float(group["absolute_error"].mean())
        output[f"{prefix}_coverage_range90"] = float(group["coverage"].max() - group["coverage"].min())
        output[f"{prefix}_worst_coverage90"] = float(group["coverage"].min())
    return output


def selection_objective(
    scenarios: np.ndarray,
    observations: np.ndarray,
    zone: np.ndarray,
    protocol: GroupingProtocol,
    *,
    crps_value: float,
    aggregate_weight: float = 0.1,
    conditional_weight: float = 0.05,
) -> tuple[float, dict[str, float]]:
    aggregate = interval_scores(scenarios, observations, np.asarray([0.5, 0.8, 0.9]))
    aggregate_error = float(np.mean(np.abs(aggregate["ace"])))
    conditional = conditional_summary(
        conditional_interval_metrics(scenarios, observations, zone, protocol)
    )["conditional_ACE90"]
    objective = crps_value + aggregate_weight * aggregate_error + conditional_weight * conditional
    return objective, {
        "CRPS": float(crps_value),
        "aggregate_calibration_error": aggregate_error,
        "conditional_ACE90": conditional,
        "objective": float(objective),
    }


__all__ = ["GroupingProtocol", "conditional_interval_metrics", "conditional_summary", "selection_objective"]
