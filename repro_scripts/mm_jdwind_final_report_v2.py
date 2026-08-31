from __future__ import annotations

import numpy as np

from repro_scripts import mm_jdwind_final_report as report


def adjacency_vs_per_day(
    samples: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    temporal_truth = np.abs(truth[:, :, 1:] - truth[:, :, :-1]) ** 0.5
    temporal_sample = np.abs(
        samples[:, :, :, 1:] - samples[:, :, :, :-1]
    ) ** 0.5
    spatial_truth = np.abs(truth[:, 1:] - truth[:, :-1]) ** 0.5
    spatial_sample = np.abs(samples[:, :, 1:] - samples[:, :, :-1]) ** 0.5
    temporal_error = (
        temporal_truth - temporal_sample.mean(axis=1)
    ).reshape(len(samples), -1)
    spatial_error = (
        spatial_truth - spatial_sample.mean(axis=1)
    ).reshape(len(samples), -1)
    return np.square(temporal_error).mean(axis=1) + np.square(
        spatial_error
    ).mean(axis=1)


def per_day(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    zero_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    mean = values.mean(axis=1)
    lower = np.quantile(values, 0.05, axis=1)
    upper = np.quantile(values, 0.95, axis=1)
    aggregate = values.mean(axis=2)
    aggregate_truth = truth.mean(axis=1)
    ramps = np.diff(values, axis=-1)
    truth_ramps = np.diff(truth, axis=-1)
    sample_any_zero = (values == 0.0).any(axis=-1).mean(axis=1)
    truth_any_zero = (truth == 0.0).any(axis=-1)
    return {
        "CRPS": report.coordinate_crps(values, truth),
        "MAE": np.abs(mean - truth).reshape(len(values), -1).mean(axis=1),
        "coverage_90": (
            (truth >= lower) & (truth <= upper)
        ).reshape(len(values), -1).mean(axis=1),
        "width_90": (upper - lower).reshape(len(values), -1).mean(axis=1),
        "joint_ES_240": report.energy_per_day(values, truth),
        "adjacency_VS": adjacency_vs_per_day(values, truth),
        "aggregate_CRPS": report.coordinate_crps(aggregate, aggregate_truth),
        "ramp_CRPS": report.coordinate_crps(ramps, truth_ramps),
        "zero_Brier": np.square(
            zero_probability - (truth == 0.0)
        ).reshape(len(values), -1).mean(axis=1),
        "daily_zone_any_zero_Brier": np.square(
            sample_any_zero - truth_any_zero
        ).mean(axis=1),
    }


if __name__ == "__main__":
    report.adjacency_vs_per_day = adjacency_vs_per_day
    report.per_day = per_day
    report.main()
