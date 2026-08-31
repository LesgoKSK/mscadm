from __future__ import annotations

import numpy as np

from mm_jdwind.data import JointSplitData
from mm_jdwind.metrics import evaluate_joint

from .graph import SpectralArtifacts


def _spectral_coefficients(
    values: np.ndarray,
    graph_basis: np.ndarray,
    temporal_basis: np.ndarray,
) -> np.ndarray:
    if values.ndim == 3:
        return np.einsum(
            "zg,dzh,kh->dgk",
            graph_basis,
            values,
            temporal_basis,
            optimize=True,
        )
    if values.ndim == 4:
        return np.einsum(
            "zg,dmzh,kh->dmgk",
            graph_basis,
            values,
            temporal_basis,
            optimize=True,
        )
    raise ValueError("values must be day x zone x hour or day x member x zone x hour")


def spectral_structure_scores(
    scenarios: np.ndarray,
    observations: np.ndarray,
    artifacts: SpectralArtifacts,
) -> dict[str, float]:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    sample_coefficient = _spectral_coefficients(
        values, artifacts.graph_basis, artifacts.temporal_basis
    )
    truth_coefficient = _spectral_coefficients(
        truth, artifacts.graph_basis, artifacts.temporal_basis
    )
    sample_energy = np.square(sample_coefficient).mean(axis=1)
    truth_energy = np.square(truth_coefficient)
    energy_error = np.abs(sample_energy - truth_energy)
    scenario_flat = values.transpose(2, 0, 1, 3).reshape(10, -1)
    truth_flat = truth.transpose(1, 0, 2).reshape(10, -1)
    sample_correlation = np.corrcoef(scenario_flat)
    truth_correlation = np.corrcoef(truth_flat)
    correlation_error = np.linalg.norm(
        np.nan_to_num(sample_correlation) - np.nan_to_num(truth_correlation),
        ord="fro",
    ) / 10.0
    graph_cut = 3
    frequency_cut = 6
    return {
        "spectral_energy_MAE": float(energy_error.mean()),
        "low_graph_low_time_energy_MAE": float(
            energy_error[:, :graph_cut, :frequency_cut].mean()
        ),
        "high_graph_energy_MAE": float(
            energy_error[:, graph_cut:, :].mean()
        ),
        "high_time_frequency_energy_MAE": float(
            energy_error[:, :, frequency_cut:].mean()
        ),
        "cross_zone_correlation_Frobenius": float(correlation_error),
    }


def evaluate_stgf(
    scenarios: np.ndarray,
    split: JointSplitData,
    artifacts: SpectralArtifacts,
) -> dict[str, float]:
    values = np.asarray(scenarios, dtype=np.float32)
    zero_probability = (values == 0.0).mean(axis=1)
    one_probability = (values == 1.0).mean(axis=1)
    scores = evaluate_joint(
        values,
        split,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )
    scores.update(spectral_structure_scores(values, split.target, artifacts))
    return scores


__all__ = ["evaluate_stgf", "spectral_structure_scores"]
