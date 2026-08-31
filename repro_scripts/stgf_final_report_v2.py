from __future__ import annotations

import numpy as np

from repro_scripts import stgf_final_report as report


def adjacency_vs_per_day(
    samples: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    temporal_truth = np.abs(truth[:, :, 1:] - truth[:, :, :-1]) ** 0.5
    temporal_sample = np.abs(
        samples[:, :, :, 1:] - samples[:, :, :, :-1]
    ) ** 0.5
    spatial_truth = np.abs(truth[:, 1:] - truth[:, :-1]) ** 0.5
    spatial_sample = np.abs(
        samples[:, :, 1:] - samples[:, :, :-1]
    ) ** 0.5
    temporal_error = (
        temporal_truth - temporal_sample.mean(axis=1)
    ).reshape(len(samples), -1)
    spatial_error = (
        spatial_truth - spatial_sample.mean(axis=1)
    ).reshape(len(samples), -1)
    return np.square(temporal_error).mean(axis=1) + np.square(
        spatial_error
    ).mean(axis=1)


if __name__ == "__main__":
    report.adjacency_vs_per_day = adjacency_vs_per_day
    report.main()
