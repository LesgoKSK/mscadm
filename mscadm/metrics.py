from __future__ import annotations

import numpy as np


def point_metrics(scenarios: np.ndarray, observations: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = scenarios.mean(axis=1)
    valid = mask.astype(bool)
    error = prediction[valid] - observations[valid]
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


def crps_ensemble(scenarios: np.ndarray, observations: np.ndarray, mask: np.ndarray) -> float:
    first = np.mean(np.abs(scenarios - observations[:, None, :]), axis=1)
    pairwise = np.abs(scenarios[:, :, None, :] - scenarios[:, None, :, :]).mean(axis=(1, 2))
    return float(np.mean((first - 0.5 * pairwise)[mask.astype(bool)]))


def quantile_score(
    scenarios: np.ndarray, observations: np.ndarray, quantiles: list[float], mask: np.ndarray
) -> float:
    valid = mask.astype(bool)
    scores = []
    for level in quantiles:
        forecast = np.quantile(scenarios, level, axis=1)
        error = observations - forecast
        scores.append(np.mean(np.maximum(level * error, (level - 1.0) * error)[valid]))
    return float(np.mean(scores))


def energy_score(scenarios: np.ndarray, observations: np.ndarray) -> float:
    first = np.linalg.norm(scenarios - observations[:, None, :], axis=-1).mean(axis=1)
    pairwise = np.linalg.norm(
        scenarios[:, :, None, :] - scenarios[:, None, :, :], axis=-1
    ).mean(axis=(1, 2))
    return float(np.mean(first - 0.5 * pairwise))


def variogram_score(scenarios: np.ndarray, observations: np.ndarray, exponent: float = 0.5) -> float:
    observed_difference = np.abs(observations[:, :, None] - observations[:, None, :]) ** exponent
    scenario_difference = np.abs(
        scenarios[:, :, :, None] - scenarios[:, :, None, :]
    ) ** exponent
    expected_difference = scenario_difference.mean(axis=1)
    return float(np.mean(np.sum((observed_difference - expected_difference) ** 2, axis=(1, 2))))


def evaluate_all(
    scenarios: np.ndarray, observations: np.ndarray, mask: np.ndarray, quantiles: list[float]
) -> dict[str, float | int]:
    results: dict[str, float | int] = point_metrics(scenarios, observations, mask)
    results["crps"] = crps_ensemble(scenarios, observations, mask)
    results["qs"] = quantile_score(scenarios, observations, quantiles, mask)
    complete = mask.all(axis=1)
    results["complete_sequences"] = int(complete.sum())
    results["es"] = energy_score(scenarios[complete], observations[complete]) if complete.any() else float("nan")
    results["vs"] = variogram_score(scenarios[complete], observations[complete]) if complete.any() else float("nan")
    return results

