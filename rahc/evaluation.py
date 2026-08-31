"""Memory-conscious scoring and calendar-day clustered inference for RAHC."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import rankdata


def crps_cells(scenarios: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Empirical CRPS for every case/hour in O(M log M), shape ``[N,H]``."""

    samples = np.sort(np.asarray(scenarios, dtype=np.float64), axis=1)
    truth = np.asarray(observations, dtype=np.float64)
    members = samples.shape[1]
    first = np.abs(samples - truth[:, None, :]).mean(axis=1)
    coefficient = 2.0 * np.arange(1, members + 1) - members - 1.0
    half_pairwise = np.sum(samples * coefficient[None, :, None], axis=1) / (members**2)
    return first - half_pairwise


def interval_cell_metrics(
    scenarios: np.ndarray, observations: np.ndarray, nominal: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return covered, width, and Winkler score arrays for a central interval."""

    alpha = 1.0 - nominal
    lower = np.quantile(scenarios, alpha / 2.0, axis=1)
    upper = np.quantile(scenarios, 1.0 - alpha / 2.0, axis=1)
    truth = np.asarray(observations)
    covered = (truth >= lower) & (truth <= upper)
    width = upper - lower
    winkler = width.copy()
    below = truth < lower
    above = truth > upper
    winkler[below] += 2.0 / alpha * (lower[below] - truth[below])
    winkler[above] += 2.0 / alpha * (truth[above] - upper[above])
    return covered, width, winkler


def _energy_score(scenarios: np.ndarray, observations: np.ndarray) -> float:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    total = 0.0
    for case in range(len(values)):
        sample = values[case]
        first = np.linalg.norm(sample - truth[case][None, :], axis=1).mean()
        second = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2).mean()
        total += first - 0.5 * second
    return float(total / len(values))


def _variogram_score(scenarios: np.ndarray, observations: np.ndarray) -> float:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    total = 0.0
    for case in range(len(values)):
        observed = np.abs(truth[case, :, None] - truth[case, None, :]) ** 0.5
        expected = np.mean(
            np.abs(values[case, :, :, None] - values[case, :, None, :]) ** 0.5,
            axis=0,
        )
        total += np.sum((observed - expected) ** 2)
    return float(total / len(values))


def _temporal_rank_rmse(scenarios: np.ndarray, observations: np.ndarray) -> float:
    observed_rank = np.apply_along_axis(rankdata, 0, observations)
    generated_rank = np.apply_along_axis(rankdata, 0, scenarios.reshape(-1, scenarios.shape[-1]))
    observed = np.corrcoef(observed_rank, rowvar=False)
    generated = np.corrcoef(generated_rank, rowvar=False)
    mask = ~np.eye(observed.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((observed[mask] - generated[mask]) ** 2)))


def overall_scores(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    """Full forecast-quality panel, including boundary-event diagnostics."""

    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    prediction = values.mean(axis=1)
    result: dict[str, float] = {
        "MAE": float(np.mean(np.abs(prediction - truth))),
        "RMSE": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "CRPS": float(crps_cells(values, truth).mean()),
    }
    levels = np.arange(1, 100, dtype=np.float64) / 100.0
    quantiles = np.quantile(values, levels, axis=1).transpose(1, 0, 2)
    error = truth[:, None, :] - quantiles
    pinball = np.maximum(levels[None, :, None] * error, (levels[None, :, None] - 1.0) * error)
    result["QS"] = float(pinball.mean())
    result["ES"] = _energy_score(values, truth)
    result["VS"] = _variogram_score(values, truth)
    ace_values: list[float] = []
    for nominal in (0.5, 0.8, 0.9):
        covered, width, winkler = interval_cell_metrics(values, truth, nominal)
        suffix = int(round(100 * nominal))
        coverage = float(covered.mean())
        result[f"coverage_{suffix}"] = coverage
        result[f"width_{suffix}"] = float(width.mean())
        result[f"winkler_{suffix}"] = float(winkler.mean())
        ace_values.append(abs(coverage - nominal))
    result["global_ACE_50_80_90"] = float(np.mean(ace_values))
    ramps = np.diff(values, axis=2)
    truth_ramps = np.diff(truth, axis=1)
    result["ramp_CRPS"] = float(crps_cells(ramps, truth_ramps).mean())
    result["temporal_rank_RMSE"] = _temporal_rank_rmse(values, truth)
    probability_zero = np.mean(values == 0.0, axis=1)
    probability_one = np.mean(values == 1.0, axis=1)
    result["zero_event_Brier"] = float(np.mean((probability_zero - (truth == 0.0)) ** 2))
    result["one_event_Brier"] = float(np.mean((probability_one - (truth == 1.0)) ** 2))
    result["scenario_zero_fraction"] = float(np.mean(values == 0.0))
    result["scenario_one_fraction"] = float(np.mean(values == 1.0))
    return result


def family_equal_conditional_ace(
    scenarios: np.ndarray,
    observations: np.ndarray,
    assignments: dict[str, np.ndarray],
    *,
    nominal: float = 0.9,
) -> float:
    covered, _, _ = interval_cell_metrics(scenarios, observations, nominal)
    family_values: list[float] = []
    for codes in assignments.values():
        errors: list[float] = []
        for code in np.unique(codes):
            selected = codes == code
            if np.any(selected):
                errors.append(abs(float(covered[selected].mean()) - nominal))
        family_values.append(float(np.mean(errors)))
    return float(np.mean(family_values))


def selection_record(
    scenarios_by_seed: Iterable[np.ndarray],
    observations: np.ndarray,
    assignments: dict[str, np.ndarray],
) -> dict[str, float]:
    per_seed: list[dict[str, float]] = []
    for values in scenarios_by_seed:
        intervals = []
        for nominal in (0.5, 0.8, 0.9):
            covered, _, _ = interval_cell_metrics(values, observations, nominal)
            intervals.append(abs(float(covered.mean()) - nominal))
        per_seed.append(
            {
                "CRPS": float(crps_cells(values, observations).mean()),
                "global_ACE": float(np.mean(intervals)),
                "conditional_ACE90": family_equal_conditional_ace(
                    values, observations, assignments, nominal=0.9
                ),
            }
        )
    crps = float(np.mean([item["CRPS"] for item in per_seed]))
    global_ace = float(np.mean([item["global_ACE"] for item in per_seed]))
    conditional_ace = float(np.mean([item["conditional_ACE90"] for item in per_seed]))
    return {
        "mean_seed_CRPS": crps,
        "mean_seed_global_ACE_50_80_90": global_ace,
        "mean_seed_family_equal_conditional_ACE90": conditional_ace,
        "objective": crps + 0.10 * global_ace + 0.05 * conditional_ace,
        **{
            f"seed{seed}_{metric}": value
            for seed, item in enumerate(per_seed)
            for metric, value in item.items()
        },
    }


__all__ = [
    "crps_cells",
    "family_equal_conditional_ace",
    "interval_cell_metrics",
    "overall_scores",
    "selection_record",
]
