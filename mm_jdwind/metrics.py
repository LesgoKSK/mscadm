from __future__ import annotations

import numpy as np

from caa_rahc.metrics import analytic_atom_scores
from cr_mscadm.experiment import extended_scores
from repro.metrics import crps

from .data import JointSplitData, flatten_joint_cases


def joint_energy_score(
    scenarios: np.ndarray, observations: np.ndarray, *, day_chunk: int = 8
) -> float:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("joint energy inputs do not align")
    scores: list[np.ndarray] = []
    for start in range(0, len(values), day_chunk):
        sample = values[start : start + day_chunk].reshape(
            -1, values.shape[1], values.shape[2] * values.shape[3]
        )
        observed = truth[start : start + day_chunk].reshape(-1, sample.shape[-1])
        first = np.linalg.norm(sample - observed[:, None], axis=-1).mean(axis=1)
        second = np.linalg.norm(
            sample[:, :, None] - sample[:, None, :], axis=-1
        ).mean(axis=(1, 2))
        scores.append(first - 0.5 * second)
    return float(np.concatenate(scores).mean())


def adjacency_variogram_score(
    scenarios: np.ndarray, observations: np.ndarray, *, order: float = 0.5
) -> float:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    temporal_truth = np.abs(truth[:, :, 1:] - truth[:, :, :-1]) ** order
    temporal_sample = np.abs(values[:, :, :, 1:] - values[:, :, :, :-1]) ** order
    temporal = (
        temporal_truth - temporal_sample.mean(axis=1)
    ) ** 2
    zone_truth = np.abs(truth[:, 1:] - truth[:, :-1]) ** order
    zone_sample = np.abs(values[:, :, 1:] - values[:, :, :-1]) ** order
    spatial = (zone_truth - zone_sample.mean(axis=1)) ** 2
    return float(temporal.mean() + spatial.mean())


def _longest_zero_run(values: np.ndarray) -> np.ndarray:
    binary = np.asarray(values) == 0.0
    flat = binary.reshape(-1, binary.shape[-1])
    result = np.zeros(len(flat), dtype=np.int64)
    for row, sequence in enumerate(flat):
        best = current = 0
        for item in sequence:
            current = current + 1 if item else 0
            best = max(best, current)
        result[row] = best
    return result.reshape(binary.shape[:-1])


def state_structure_metrics(
    scenarios: np.ndarray, observations: np.ndarray
) -> dict[str, float]:
    sample_zero = scenarios == 0.0
    truth_zero = observations == 0.0
    sample_any = sample_zero.any(axis=-1).mean(axis=1)
    truth_any = truth_zero.any(axis=-1)
    sample_runs = _longest_zero_run(scenarios).mean(axis=1)
    truth_runs = _longest_zero_run(observations)
    return {
        "finite_zero_rate": float(sample_zero.mean()),
        "observed_zero_rate": float(truth_zero.mean()),
        "daily_zone_any_zero_Brier": float(np.mean((sample_any - truth_any) ** 2)),
        "zero_run_length_MAE": float(np.mean(np.abs(sample_runs - truth_runs))),
    }


def evaluate_joint(
    scenarios: np.ndarray,
    split: JointSplitData,
    *,
    zero_probability: np.ndarray,
    one_probability: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(scenarios, dtype=np.float32)
    if values.shape[0] != len(split) or values.shape[2:] != (10, 24):
        raise ValueError("scenario archive does not align with joint split")
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("scenarios are non-finite or outside [0,1]")
    flattened, observations = flatten_joint_cases(values, split.target)
    scores = extended_scores(flattened, observations)
    scores["joint_ES_240"] = joint_energy_score(values, split.target)
    scores["adjacency_VS"] = adjacency_variogram_score(values, split.target)
    aggregate = values.mean(axis=2)
    aggregate_truth = split.target.mean(axis=1)
    scores["aggregate_CRPS"] = crps(aggregate, aggregate_truth)
    scores["aggregate_ramp_CRPS"] = crps(
        np.diff(aggregate, axis=2), np.diff(aggregate_truth, axis=1)
    )
    scores.update(
        analytic_atom_scores(
            zero_probability, one_probability, split.target
        )
    )
    scores.update(state_structure_metrics(values, split.target))
    return {key: float(value) for key, value in scores.items()}


def per_date_joint_metrics(
    scenarios: np.ndarray, observations: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    first = np.abs(values - truth[:, None]).mean(axis=1)
    pairwise = np.abs(values[:, :, None] - values[:, None, :]).mean(axis=(1, 2))
    crps_day = (first - pairwise / 2).mean(axis=(1, 2))
    mean = values.mean(axis=1)
    mae_day = np.abs(mean - truth).mean(axis=(1, 2))
    lower = np.quantile(values, 0.05, axis=1)
    upper = np.quantile(values, 0.95, axis=1)
    coverage = ((truth >= lower) & (truth <= upper)).mean(axis=(1, 2))
    aggregate = values.mean(axis=2)
    aggregate_truth = truth.mean(axis=1)
    aggregate_first = np.abs(aggregate - aggregate_truth[:, None]).mean(axis=1)
    aggregate_pairwise = np.abs(
        aggregate[:, :, None] - aggregate[:, None, :]
    ).mean(axis=(1, 2))
    aggregate_crps = (aggregate_first - aggregate_pairwise / 2).mean(axis=1)
    return {
        "CRPS": crps_day,
        "MAE": mae_day,
        "coverage_90": coverage,
        "aggregate_CRPS": aggregate_crps,
    }


__all__ = [
    "adjacency_variogram_score",
    "evaluate_joint",
    "joint_energy_score",
    "per_date_joint_metrics",
    "state_structure_metrics",
]
