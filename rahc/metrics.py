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
        mean = scenarios.mean(axis=1).ravel()
        uncertainty = scenarios.std(axis=1).ravel()
        probabilities = np.linspace(0.0, 1.0, 6)
        wind_edges = np.quantile(mean, probabilities)
        uncertainty_edges = np.quantile(uncertainty, probabilities)
        return cls(wind_edges.astype(np.float64), uncertainty_edges.astype(np.float64))

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


def _group_rows(
    scenarios: np.ndarray,
    observations: np.ndarray,
    group_type: str,
    group_values: np.ndarray,
    labels: list[str],
    levels: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(labels):
        mask = group_values == index
        count = int(mask.sum())
        if count == 0:
            continue
        for nominal in levels:
            lower = np.quantile(scenarios, (1.0 - nominal) / 2.0, axis=1)
            upper = np.quantile(scenarios, (1.0 + nominal) / 2.0, axis=1)
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
    return rows


def conditional_interval_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    zone: np.ndarray,
    protocol: GroupingProtocol,
) -> pd.DataFrame:
    if scenarios.ndim != 3 or observations.shape != (len(scenarios), 24):
        raise ValueError("scenario and observation shapes are invalid")
    cells = len(scenarios) * 24
    zone_cells = np.repeat(zone[:, None], 24, axis=1) - 1
    hour_cells = np.repeat(np.arange(24)[None, :], len(scenarios), axis=0)
    wind = protocol.wind_bin(scenarios)
    uncertainty = protocol.uncertainty_bin(scenarios)
    intersection = zone_cells * 5 + uncertainty
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        _group_rows(
            scenarios,
            observations,
            "zone",
            zone_cells,
            [f"zone_{index}" for index in range(1, 11)],
            protocol.levels,
        )
    )
    rows.extend(
        _group_rows(
            scenarios,
            observations,
            "hour",
            hour_cells,
            [f"hour_{index:02d}" for index in range(1, 25)],
            protocol.levels,
        )
    )
    rows.extend(
        _group_rows(
            scenarios,
            observations,
            "wind_quintile",
            wind,
            [f"wind_q{index}" for index in range(1, 6)],
            protocol.levels,
        )
    )
    rows.extend(
        _group_rows(
            scenarios,
            observations,
            "uncertainty_quintile",
            uncertainty,
            [f"uncertainty_q{index}" for index in range(1, 6)],
            protocol.levels,
        )
    )
    rows.extend(
        _group_rows(
            scenarios,
            observations,
            "zone_x_uncertainty",
            intersection,
            [f"zone_{zone_index}_uncertainty_q{quintile}" for zone_index in range(1, 11) for quintile in range(1, 6)],
            protocol.levels,
        )
    )
    frame = pd.DataFrame(rows)
    expected_minimum = cells // 100
    if frame.empty or frame["n"].min() < expected_minimum:
        # This is diagnostic only: sparse cells remain in the table, but signal
        # an unexpected grouping protocol rather than silently dropping them.
        frame.attrs["sparse_group_warning"] = True
    return frame


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
        "zone_ACE90": float(selected[selected["group_type"] == "zone"]["absolute_error"].mean()),
        "hour_ACE90": float(selected[selected["group_type"] == "hour"]["absolute_error"].mean()),
        "wind_ACE90": float(selected[selected["group_type"] == "wind_quintile"]["absolute_error"].mean()),
        "uncertainty_ACE90": float(selected[selected["group_type"] == "uncertainty_quintile"]["absolute_error"].mean()),
    }
    for group_type in ("zone", "hour", "wind_quintile", "uncertainty_quintile"):
        group = selected[selected["group_type"] == group_type]
        output[f"{group_type}_coverage_range90"] = float(group["coverage"].max() - group["coverage"].min())
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


__all__ = [
    "GroupingProtocol",
    "conditional_interval_metrics",
    "conditional_summary",
    "selection_objective",
]
