from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _validate(
    scenarios: np.ndarray, probabilities: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scenarios, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    if values.ndim != 4 or values.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [day,member,10,24]")
    if weights.shape != values.shape[:2]:
        raise ValueError("probabilities must have shape [day,member]")
    if truth.shape != (len(values), 10, 24):
        raise ValueError("observations must have shape [day,10,24]")
    if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("daily probabilities must be non-negative and sum to one")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("metrics received non-finite values")
    return values, weights, truth


def weighted_quantile(
    values: np.ndarray, probabilities: np.ndarray, quantile: float
) -> np.ndarray:
    """Weighted quantile over axis one of [day,member,...]."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    if array.ndim < 2 or weights.shape != array.shape[:2]:
        raise ValueError("weights must align with the first two value axes")
    result = np.empty((len(array), *array.shape[2:]), dtype=np.float64)
    for day in range(len(array)):
        flat = array[day].reshape(array.shape[1], -1)
        order = np.argsort(flat, axis=0)
        sorted_values = np.take_along_axis(flat, order, axis=0)
        sorted_weights = np.take_along_axis(
            np.broadcast_to(weights[day, :, None], flat.shape), order, axis=0
        )
        cumulative = np.cumsum(sorted_weights, axis=0)
        index = np.argmax(cumulative >= quantile, axis=0)
        result[day] = sorted_values[index, np.arange(flat.shape[1])].reshape(
            array.shape[2:]
        )
    return result


def _weighted_crps_day(
    values: np.ndarray, weights: np.ndarray, truth: np.ndarray
) -> float:
    flat = values.reshape(len(values), -1)
    observed = truth.reshape(-1)
    first = np.sum(weights[:, None] * np.abs(flat - observed[None]), axis=0)
    pairwise = np.abs(flat[:, None, :] - flat[None, :, :])
    second = np.einsum("i,j,ijc->c", weights, weights, pairwise, optimize=True)
    return float(np.mean(first - 0.5 * second))


def _energy_day(values: np.ndarray, weights: np.ndarray, truth: np.ndarray) -> float:
    flat = values.reshape(len(values), -1)
    observed = truth.reshape(-1)
    first = np.sum(weights * np.linalg.norm(flat - observed[None], axis=1))
    pairwise = np.linalg.norm(flat[:, None] - flat[None, :], axis=2)
    second = np.einsum("i,j,ij->", weights, weights, pairwise, optimize=True)
    return float(first - 0.5 * second)


def _variogram_day(
    values: np.ndarray, weights: np.ndarray, truth: np.ndarray, order: float = 0.5
) -> float:
    temporal_truth = np.abs(truth[:, 1:] - truth[:, :-1]) ** order
    temporal_sample = np.abs(values[:, :, 1:] - values[:, :, :-1]) ** order
    temporal_expected = np.tensordot(weights, temporal_sample, axes=(0, 0))
    spatial_truth = np.abs(truth[1:] - truth[:-1]) ** order
    spatial_sample = np.abs(values[:, 1:] - values[:, :-1]) ** order
    spatial_expected = np.tensordot(weights, spatial_sample, axes=(0, 0))
    return float(
        np.mean((temporal_truth - temporal_expected) ** 2)
        + np.mean((spatial_truth - spatial_expected) ** 2)
    )


def weighted_per_date_metrics(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    *,
    assignments: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    values, weights, truth = _validate(scenarios, probabilities, observations)
    days = len(values)
    crps = np.empty(days)
    energy = np.empty(days)
    variogram = np.empty(days)
    ramp_crps = np.empty(days)
    for day in range(days):
        crps[day] = _weighted_crps_day(values[day], weights[day], truth[day])
        energy[day] = _energy_day(values[day], weights[day], truth[day])
        variogram[day] = _variogram_day(values[day], weights[day], truth[day])
        aggregate = values[day].mean(axis=1)
        aggregate_truth = truth[day].mean(axis=0)
        ramp_crps[day] = _weighted_crps_day(
            np.diff(aggregate, axis=1)[:, None, :],
            weights[day],
            np.diff(aggregate_truth)[None, :],
        )
    lower = weighted_quantile(values, weights, 0.05)
    upper = weighted_quantile(values, weights, 0.95)
    covered = (truth >= lower) & (truth <= upper)
    width = upper - lower
    alpha = 0.10
    winkler = width.copy()
    winkler += (2.0 / alpha) * (lower - truth) * (truth < lower)
    winkler += (2.0 / alpha) * (truth - upper) * (truth > upper)
    zero_probability = np.sum(weights[:, :, None, None] * (values == 0.0), axis=1)
    zero_brier = ((zero_probability - (truth == 0.0)) ** 2).mean(axis=(1, 2))
    result = {
        "CRPS": crps,
        "ES": energy,
        "VS": variogram,
        "ramp_CRPS": ramp_crps,
        "coverage_90": covered.mean(axis=(1, 2)),
        "winkler_90": winkler.mean(axis=(1, 2)),
        "zero_Brier": zero_brier,
    }
    if assignments is not None:
        result["conditional_ACE"] = conditional_ace_per_date(
            covered, assignments, nominal=0.90
        )
    return result


def conditional_ace_per_date(
    covered: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    *,
    nominal: float = 0.90,
) -> np.ndarray:
    coverage = np.asarray(covered, dtype=bool)
    if coverage.ndim != 3 or coverage.shape[1:] != (10, 24):
        raise ValueError("covered must have shape [day,10,24]")
    family_scores: list[np.ndarray] = []
    cells = coverage.reshape(len(coverage), -1)
    for name, raw in assignments.items():
        labels = np.asarray(raw)
        if labels.shape == coverage.shape:
            labels = labels.reshape(len(coverage), -1)
        elif labels.shape == coverage.shape[1:]:
            labels = np.broadcast_to(labels.reshape(1, -1), cells.shape)
        elif labels.shape == (len(coverage),):
            labels = np.broadcast_to(labels[:, None], cells.shape)
        else:
            raise ValueError(f"assignment family {name!r} has incompatible shape")
        day_score = np.empty(len(coverage))
        for day in range(len(coverage)):
            group_errors = [
                abs(float(cells[day, labels[day] == group].mean()) - nominal)
                for group in np.unique(labels[day])
                if np.any(labels[day] == group)
            ]
            day_score[day] = float(np.mean(group_errors))
        family_scores.append(day_score)
    if not family_scores:
        raise ValueError("at least one conditional assignment family is required")
    return np.stack(family_scores).mean(axis=0)


def make_conditional_assignments(
    observations: np.ndarray,
    *,
    days: np.ndarray,
    wind_thresholds: tuple[float, float],
    ramp_thresholds: tuple[float, float],
) -> dict[str, np.ndarray]:
    truth = np.asarray(observations, dtype=np.float64)
    date = np.asarray(days).astype("datetime64[D]")
    if truth.shape != (len(date), 10, 24):
        raise ValueError("observations/days do not align")
    month = date.astype("datetime64[M]").astype(int) % 12 + 1
    season = ((month % 12) // 3).astype(np.int64)
    hour_block = np.broadcast_to(np.arange(24) // 6, (10, 24))
    aggregate = truth.mean(axis=1)
    wind_level = np.digitize(aggregate.mean(axis=1), wind_thresholds)
    ramp_level = np.digitize(np.abs(np.diff(aggregate, axis=1)).mean(axis=1), ramp_thresholds)
    return {
        "season": season,
        "hour_block": hour_block,
        "wind_level": wind_level,
        "ramp_level": ramp_level,
    }


__all__ = [
    "conditional_ace_per_date",
    "make_conditional_assignments",
    "weighted_per_date_metrics",
    "weighted_quantile",
]
