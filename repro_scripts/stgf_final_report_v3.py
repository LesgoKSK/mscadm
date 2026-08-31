from __future__ import annotations

import numpy as np

from repro_scripts import stgf_final_report as report
from repro_scripts.stgf_final_report_v2 import adjacency_vs_per_day


def per_day(
    scenarios: np.ndarray, observations: np.ndarray
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
    zero_probability = (values == 0.0).mean(axis=1)
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
        "aggregate_CRPS": report.coordinate_crps(
            aggregate, aggregate_truth
        ),
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
