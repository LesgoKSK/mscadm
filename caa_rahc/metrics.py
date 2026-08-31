"""Date-clustered forecast and atom diagnostics for CAA selection/evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from rahc.evaluation import crps_cells, family_equal_conditional_ace, interval_cell_metrics


def _variogram_per_case(scenarios: np.ndarray, observations: np.ndarray) -> np.ndarray:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    observed = np.abs(truth[:, :, None] - truth[:, None, :]) ** 0.5
    expected = np.mean(
        np.abs(values[:, :, :, None] - values[:, :, None, :]) ** 0.5,
        axis=1,
    )
    return np.sum((observed - expected) ** 2, axis=(1, 2))


def _aggregate_cases_by_day(values: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates = np.asarray(day).astype("datetime64[D]")
    unique, inverse = np.unique(dates, return_inverse=True)
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] != len(dates):
        raise ValueError("metric cases do not align with day labels")
    flat = array.reshape(len(array), -1).mean(axis=1)
    sums = np.bincount(inverse, weights=flat, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    return unique, sums / counts


def per_date_metrics(
    scenarios: np.ndarray, observations: np.ndarray, day: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the six registered non-inferiority metrics for every date."""

    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    if values.ndim != 3 or truth.shape != (len(values), values.shape[2]):
        raise ValueError("scenarios/truth must align as [case,member,hour]/[case,hour]")
    covered, width, winkler = interval_cell_metrics(values, truth, 0.90)
    del covered
    cell_crps = crps_cells(values, truth)
    mean_absolute = np.abs(values.mean(axis=1) - truth)
    ramp_crps = crps_cells(np.diff(values, axis=2), np.diff(truth, axis=1))
    vs = _variogram_per_case(values, truth)
    sources = {
        "CRPS": cell_crps,
        "MAE": mean_absolute,
        "VS": vs[:, None],
        "ramp_CRPS": ramp_crps,
        "winkler_90": winkler,
        "width_90": width,
    }
    result: dict[str, np.ndarray] = {}
    expected_dates: np.ndarray | None = None
    for name, source in sources.items():
        dates, daily = _aggregate_cases_by_day(source, day)
        if expected_dates is None:
            expected_dates = dates
        elif not np.array_equal(dates, expected_dates):
            raise AssertionError("date aggregation changed across metrics")
        result[name] = daily
    result["dates"] = expected_dates
    return result


def stack_per_date_metrics(
    scenarios_by_seed: Sequence[np.ndarray], observations: np.ndarray, day: np.ndarray
) -> dict[str, np.ndarray]:
    records = [per_date_metrics(values, observations, day) for values in scenarios_by_seed]
    dates = records[0]["dates"]
    if any(not np.array_equal(item["dates"], dates) for item in records[1:]):
        raise ValueError("model seeds do not share date ordering")
    result = {
        name: np.stack([item[name] for item in records])
        for name in ("CRPS", "MAE", "VS", "ramp_CRPS", "winkler_90", "width_90")
    }
    result["dates"] = dates
    return result


def conditional_ace_by_seed(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    assignments: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            family_equal_conditional_ace(values, observations, dict(assignments), nominal=0.9)
            for values in scenarios_by_seed
        ],
        dtype=np.float64,
    )


def analytic_atom_scores(
    zero_probability: np.ndarray,
    upper_probability: np.ndarray | float,
    observations: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Brier/log loss for analytic atom probabilities, independent of M."""

    truth = np.asarray(observations)
    p0 = np.broadcast_to(np.asarray(zero_probability, dtype=np.float64), truth.shape)
    p1 = np.broadcast_to(np.asarray(upper_probability, dtype=np.float64), truth.shape)
    if np.any(p0 < 0.0) or np.any(p1 < 0.0) or np.any(p0 + p1 > 1.0 + 1e-12):
        raise ValueError("invalid atom probabilities")
    y0 = truth == 0.0
    y1 = truth == 1.0

    def binary(probability: np.ndarray, event: np.ndarray, prefix: str) -> dict[str, float]:
        p = np.clip(probability, epsilon, 1.0 - epsilon)
        return {
            f"{prefix}_Brier": float(np.mean((probability - event) ** 2)),
            f"{prefix}_log_loss": float(
                -np.mean(event * np.log(p) + (~event) * np.log1p(-p))
            ),
            f"{prefix}_predicted_rate": float(probability.mean()),
            f"{prefix}_observed_rate": float(event.mean()),
        }

    return {**binary(p0, y0, "zero"), **binary(p1, y1, "one")}


__all__ = [
    "analytic_atom_scores",
    "conditional_ace_by_seed",
    "per_date_metrics",
    "stack_per_date_metrics",
]
