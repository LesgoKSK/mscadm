from __future__ import annotations

import numpy as np


def _as64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def validate_shapes(scenarios: np.ndarray, observations: np.ndarray) -> None:
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape [days, scenarios, periods]")
    if observations.shape != (scenarios.shape[0], scenarios.shape[2]):
        raise ValueError("observations must have shape [days, periods]")


def point_scores(scenarios: np.ndarray, observations: np.ndarray) -> tuple[float, float]:
    prediction = _as64(scenarios).mean(axis=1)
    error = prediction - _as64(observations)
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def crps(scenarios: np.ndarray, observations: np.ndarray) -> float:
    samples, truth = _as64(scenarios), _as64(observations)
    first = np.abs(samples - truth[:, None]).mean(axis=1)
    pairwise = np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    return float(np.mean(first - pairwise / 2))


def quantile_score(
    scenarios: np.ndarray,
    observations: np.ndarray,
    levels: np.ndarray | None = None,
) -> float:
    levels = np.arange(1, 100, dtype=np.float64) / 100 if levels is None else _as64(levels)
    forecasts = np.quantile(_as64(scenarios), levels, axis=1).transpose(1, 0, 2)
    error = _as64(observations)[:, None] - forecasts
    loss = np.maximum(levels[None, :, None] * error, (levels[None, :, None] - 1) * error)
    return float(np.mean(loss))


def energy_score(scenarios: np.ndarray, observations: np.ndarray) -> float:
    samples, truth = _as64(scenarios), _as64(observations)
    first = np.linalg.norm(samples - truth[:, None], axis=-1).mean(axis=1)
    second = np.linalg.norm(samples[:, :, None] - samples[:, None, :], axis=-1).mean(axis=(1, 2))
    return float(np.mean(first - second / 2))


def variogram_score(scenarios: np.ndarray, observations: np.ndarray, order: float = 0.5) -> float:
    samples, truth_values = _as64(scenarios), _as64(observations)
    truth = np.abs(truth_values[:, :, None] - truth_values[:, None, :]) ** order
    sampled = np.abs(samples[:, :, :, None] - samples[:, :, None, :]) ** order
    expected = sampled.mean(axis=1)
    return float(np.mean(np.sum((truth - expected) ** 2, axis=(1, 2))))


def interval_scores(
    scenarios: np.ndarray,
    observations: np.ndarray,
    confidence_levels: np.ndarray | None = None,
) -> dict[str, list[float]]:
    levels = (
        np.arange(0.1, 1.0, 0.1, dtype=np.float64)
        if confidence_levels is None
        else _as64(confidence_levels)
    )
    samples, truth = _as64(scenarios), _as64(observations)
    coverage_errors: list[float] = []
    widths: list[float] = []
    coverages: list[float] = []
    for level in levels:
        lower = np.quantile(samples, (1 - level) / 2, axis=1)
        upper = np.quantile(samples, (1 + level) / 2, axis=1)
        coverage = np.mean((truth >= lower) & (truth <= upper))
        coverage_errors.append(float(coverage - level))
        coverages.append(float(coverage))
        widths.append(float(np.mean(upper - lower)))
    return {
        "confidence": levels.tolist(),
        "ace": coverage_errors,
        "coverage": coverages,
        "piaw": widths,
    }


def all_scores(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    validate_shapes(scenarios, observations)
    mae, rmse = point_scores(scenarios, observations)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "CRPS": crps(scenarios, observations),
        "QS": quantile_score(scenarios, observations),
        "ES": energy_score(scenarios, observations),
        "VS": variogram_score(scenarios, observations),
    }


__all__ = [
    "validate_shapes",
    "point_scores",
    "crps",
    "quantile_score",
    "energy_score",
    "variogram_score",
    "interval_scores",
    "all_scores",
]
