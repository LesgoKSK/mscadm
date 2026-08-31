"""Paired validation gates for the T0/T1 temporal-mechanism experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite per-day vector")
    return array


def paired_mean_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    transform: str = "difference",
) -> dict[str, Any]:
    """Bootstrap a paired daily mean contrast.

    ``difference`` estimates ``mean(left-right)``.  ``relative_difference``
    estimates that contrast divided by the resampled mean of ``right``.
    Calendar day, never ensemble member, is the resampling unit.
    """

    left_values = _vector(left, name="left")
    right_values = _vector(right, name="right")
    if left_values.shape != right_values.shape:
        raise ValueError("paired vectors do not align")
    if repetitions < 100:
        raise ValueError("paired bootstrap requires at least 100 repetitions")
    if transform not in ("difference", "relative_difference"):
        raise ValueError("unknown paired contrast transform")

    def statistic(index: np.ndarray) -> float:
        numerator = float((left_values[index] - right_values[index]).mean())
        if transform == "difference":
            return numerator
        denominator = float(right_values[index].mean())
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError("relative contrast denominator is zero")
        return numerator / denominator

    all_index = np.arange(len(left_values))
    point = statistic(all_index)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    for draw in range(int(repetitions)):
        draws[draw] = statistic(rng.integers(0, len(all_index), len(all_index)))
    return {
        "unit_of_inference": "calendar_day",
        "n_days": int(len(all_index)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "transform": transform,
        "estimate": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_below_zero": float(np.mean(draws < 0.0)),
        "absolute_draw_q975": float(np.quantile(np.abs(draws), 0.975)),
    }


def average_training_seeds(
    per_seed: Mapping[int | str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Average model-seed estimates within day without creating replicates."""

    if not per_seed:
        raise ValueError("at least one training seed is required")
    ordered = [per_seed[key] for key in sorted(per_seed, key=lambda item: int(item))]
    keys = tuple(sorted(ordered[0]))
    if any(tuple(sorted(item)) != keys for item in ordered):
        raise ValueError("training-seed metric schemas differ")
    result: dict[str, np.ndarray] = {}
    for key in keys:
        vectors = [_vector(item[key], name=key) for item in ordered]
        if any(value.shape != vectors[0].shape for value in vectors[1:]):
            raise ValueError(f"training-seed day vectors differ for {key}")
        result[key] = np.mean(np.stack(vectors, axis=0), axis=0)
    return result


def chronological_vs_control_gate(
    t0: Mapping[str, np.ndarray],
    chronological: Mapping[str, np.ndarray],
    shuffled: Mapping[str, np.ndarray],
    *,
    rules: Mapping[str, Any],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Apply superiority, non-inferiority, and matched-shuffle gates."""

    primary = rules["T1_vs_T0_primary"]
    noninferiority = rules["noninferiority"]
    negative = rules["negative_control"]

    ramp = paired_mean_bootstrap(
        _vector(t0["ramp_CRPS"], name="T0 ramp"),
        _vector(chronological["ramp_CRPS"], name="T1 ramp"),
        repetitions=repetitions,
        seed=seed + 1,
    )
    lagged = paired_mean_bootstrap(
        _vector(t0["lagged_increment_variogram_score"], name="T0 lagged"),
        _vector(chronological["lagged_increment_variogram_score"], name="T1 lagged"),
        repetitions=repetitions,
        seed=seed + 2,
        transform="relative_difference",
    )
    level_ni = paired_mean_bootstrap(
        _vector(chronological["level_CRPS"], name="T1 level"),
        _vector(t0["level_CRPS"], name="T0 level"),
        repetitions=repetitions,
        seed=seed + 3,
    )
    joint_ni = paired_mean_bootstrap(
        _vector(chronological["normalized_joint_ES"], name="T1 joint"),
        _vector(t0["normalized_joint_ES"], name="T0 joint"),
        repetitions=repetitions,
        seed=seed + 4,
        transform="relative_difference",
    )
    coverage = paired_mean_bootstrap(
        _vector(chronological["coverage90"], name="T1 coverage"),
        _vector(t0["coverage90"], name="T0 coverage"),
        repetitions=repetitions,
        seed=seed + 5,
    )
    width = paired_mean_bootstrap(
        _vector(chronological["width90"], name="T1 width"),
        _vector(t0["width90"], name="T0 width"),
        repetitions=repetitions,
        seed=seed + 6,
        transform="relative_difference",
    )
    shuffle_ramp = paired_mean_bootstrap(
        _vector(shuffled["ramp_CRPS"], name="shuffle ramp"),
        _vector(chronological["ramp_CRPS"], name="chronological ramp"),
        repetitions=repetitions,
        seed=seed + 7,
    )
    shuffle_lagged = paired_mean_bootstrap(
        _vector(shuffled["lagged_increment_variogram_score"], name="shuffle lagged"),
        _vector(chronological["lagged_increment_variogram_score"], name="chronological lagged"),
        repetitions=repetitions,
        seed=seed + 8,
    )
    checks = {
        "ramp_practical_superiority": ramp["estimate"]
        >= float(primary["ramp_CRPS_practical_improvement_min"]),
        "ramp_statistical_superiority": ramp["ci_low"]
        > float(primary["ramp_CRPS_bootstrap_CI_lower_must_exceed"]),
        "lagged_practical_superiority": lagged["estimate"]
        >= float(primary["lagged_increment_variogram_relative_improvement_min"]),
        "lagged_statistical_superiority": lagged["ci_low"]
        > float(primary["lagged_increment_variogram_bootstrap_CI_lower_must_exceed"]),
        "level_noninferiority": level_ni["ci_high"]
        <= float(noninferiority["level_CRPS_T1_minus_T0_upper_margin"]),
        "joint_noninferiority": joint_ni["ci_high"]
        <= float(noninferiority["normalized_joint_ES_relative_T1_minus_T0_upper_margin"]),
        "coverage_noninferiority": coverage["absolute_draw_q975"]
        <= float(noninferiority["coverage90_absolute_difference_max"]),
        "width_noninferiority": width["ci_high"]
        <= float(noninferiority["width90_relative_increase_max"]),
        "shuffle_ramp_practical": shuffle_ramp["estimate"]
        >= float(negative["ramp_CRPS_practical_improvement_min"]),
        "shuffle_ramp_statistical": shuffle_ramp["ci_low"]
        > float(negative["bootstrap_CI_lower_must_exceed"]),
        "shuffle_lagged_statistical": shuffle_lagged["ci_low"]
        > float(negative["bootstrap_CI_lower_must_exceed"]),
    }
    return {
        "schema": "architecture_v1_temporal_mechanism_gate_v1",
        "contrasts": {
            "T0_minus_T1_ramp": ramp,
            "T0_minus_T1_lagged_relative": lagged,
            "T1_minus_T0_level": level_ni,
            "T1_minus_T0_joint_relative": joint_ni,
            "T1_minus_T0_coverage": coverage,
            "T1_minus_T0_width_relative": width,
            "shuffle_minus_chronological_ramp": shuffle_ramp,
            "shuffle_minus_chronological_lagged": shuffle_lagged,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


__all__ = [
    "average_training_seeds",
    "chronological_vs_control_gate",
    "paired_mean_bootstrap",
]
