"""Validation-only quality metrics for the formal architecture-v1 stages.

This module has no data loader and cannot open calibration or selection.  It
operates only on arrays supplied by the stage-scoped fit bundle.  Calendar day
is the reporting unit; ensemble members are never inferential replicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evaluation import masked_per_day_metrics


def _daily_masked_mean(values: np.ndarray, mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if array.shape != valid.shape:
        raise ValueError(f"{name} values and mask do not align")
    count = valid.reshape(len(valid), -1).sum(axis=1)
    if np.any(count == 0):
        raise ValueError(f"a validation day has no observed {name} cells")
    return np.where(valid, array, 0.0).reshape(len(valid), -1).sum(axis=1) / count


def normalized_joint_energy_score(
    scenarios: np.ndarray,
    observations: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Per-day empirical V-stat ES normalized by ``sqrt(valid dimension)``."""

    sample = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if sample.ndim != 4 or sample.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [D,M,10,24]")
    if truth.shape != (len(sample), 10, 24) or mask.shape != truth.shape:
        raise ValueError("truth/mask do not align with scenarios")
    if sample.shape[1] < 2:
        raise ValueError("energy score requires at least two members")
    result = np.empty(len(sample), dtype=np.float64)
    for day_index in range(len(sample)):
        valid = mask[day_index].reshape(-1)
        dimension = int(valid.sum())
        if dimension == 0:
            raise ValueError("a validation day has no observed joint dimensions")
        members = sample[day_index].reshape(sample.shape[1], -1)[:, valid]
        target = truth[day_index].reshape(-1)[valid]
        scale = np.sqrt(float(dimension))
        first = np.linalg.norm(members - target[None], axis=1).mean() / scale
        differences = members[:, None, :] - members[None, :, :]
        second = 0.5 * np.linalg.norm(differences, axis=-1).mean() / scale
        result[day_index] = first - second
    if not np.isfinite(result).all() or np.any(result < -1e-10):
        raise FloatingPointError("normalized joint ES is invalid")
    return np.maximum(result, 0.0)


def lagged_increment_variogram_score(
    scenarios: np.ndarray,
    observations: np.ndarray,
    observed_mask: np.ndarray,
    *,
    lag: int = 1,
) -> np.ndarray:
    """Masked per-day variogram score for same-zone increment dependence.

    This is the proper-score endpoint paired with ramp CRPS.  It compares the
    observed absolute difference between two power increments ``lag`` positions
    apart with the ensemble mean of the same quantity.  A pair is supervised
    only when all target cells required by both increments were raw-observed.
    """

    sample = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if sample.ndim != 4 or sample.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [D,M,10,24]")
    if truth.shape != (len(sample), 10, 24) or mask.shape != truth.shape:
        raise ValueError("truth/mask do not align with scenarios")
    ramp_periods = sample.shape[-1] - 1
    if lag < 1 or lag >= ramp_periods:
        raise ValueError("increment lag must lie in [1,22]")
    forecast_ramp = np.diff(sample, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)
    ramp_mask = mask[..., :-1] & mask[..., 1:]
    forecast_difference = (
        forecast_ramp[..., :-lag] - forecast_ramp[..., lag:]
    )
    truth_difference = truth_ramp[..., :-lag] - truth_ramp[..., lag:]
    pair_mask = ramp_mask[..., :-lag] & ramp_mask[..., lag:]
    forecast_variogram = np.mean(np.abs(forecast_difference), axis=1)
    observed_variogram = np.abs(truth_difference)
    squared = (forecast_variogram - observed_variogram) ** 2
    return _daily_masked_mean(
        squared,
        pair_mask,
        name=f"lag-{lag} increment variogram",
    )


def validation_per_day_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    observed_mask: np.ndarray,
    *,
    zero_probability: np.ndarray,
    one_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the pre-registered G1 validation metrics for one scenario set."""

    sample = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    base = masked_per_day_metrics(
        sample,
        truth,
        mask,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )
    lower = np.quantile(sample, 0.05, axis=1)
    upper = np.quantile(sample, 0.95, axis=1)
    coverage = _daily_masked_mean(
        ((truth >= lower) & (truth <= upper)).astype(np.float64),
        mask,
        name="coverage",
    )
    width = _daily_masked_mean(upper - lower, mask, name="interval width")
    result = {
        key: np.asarray(value)
        for key, value in base.items()
        if not key.startswith("valid_")
    }
    result.update(
        {
            "normalized_joint_ES": normalized_joint_energy_score(
                sample, truth, mask
            ),
            "lagged_increment_variogram_score": lagged_increment_variogram_score(
                sample, truth, mask, lag=1
            ),
            "coverage90": coverage,
            "width90": width,
        }
    )
    return result


def aggregate_sampling_replicates(
    replicates: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Average sampling replicas within each calendar day, then summarize."""

    if not replicates:
        raise ValueError("at least one sampling replicate is required")
    keys = tuple(sorted(replicates[0]))
    if any(tuple(sorted(item)) != keys for item in replicates):
        raise ValueError("sampling replicate metric schemas differ")
    per_day: dict[str, np.ndarray] = {}
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [np.asarray(item[key], dtype=np.float64) for item in replicates]
        if any(value.ndim != 1 for value in values):
            raise ValueError(f"metric {key} must be a per-day vector")
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"metric {key} replicate lengths differ")
        averaged = np.mean(np.stack(values, axis=0), axis=0)
        if not np.isfinite(averaged).all():
            raise FloatingPointError(f"metric {key} contains non-finite values")
        per_day[key] = averaged
        aggregate[key] = float(averaged.mean())
    return {"per_day": per_day, "aggregate": aggregate}


__all__ = [
    "aggregate_sampling_replicates",
    "lagged_increment_variogram_score",
    "normalized_joint_energy_score",
    "validation_per_day_metrics",
]
