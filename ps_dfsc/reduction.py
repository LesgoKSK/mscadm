from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_fixed_assignments(
    base_scenarios_mw: np.ndarray, clusters: int = 20, *, seed: int = 0
) -> np.ndarray:
    values = np.asarray(base_scenarios_mw, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (6, 24):
        raise ValueError("base_scenarios_mw must have shape [member,6,24]")
    if not 1 <= clusters <= len(values):
        raise ValueError("clusters must be between one and the member count")
    fitted = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit(
        values.reshape(len(values), -1)
    )
    return fitted.labels_.astype(np.int64)


def weighted_cluster_reduction(
    scenarios_mw: np.ndarray,
    probabilities: np.ndarray,
    assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scenarios_mw, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(assignments, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (6, 24):
        raise ValueError("scenarios_mw must have shape [member,6,24]")
    if weights.shape != (len(values),) or labels.shape != (len(values),):
        raise ValueError("weights/assignments must align with scenarios")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("assignments must be contiguous labels starting at zero")
    reduced: list[np.ndarray] = []
    mass: list[float] = []
    for label in unique:
        selected = labels == label
        cluster_mass = float(weights[selected].sum())
        if cluster_mass <= 0.0:
            raise ValueError(f"cluster {label} has zero probability")
        reduced.append(
            np.tensordot(weights[selected] / cluster_mass, values[selected], axes=(0, 0))
        )
        mass.append(cluster_mass)
    probability = np.asarray(mass, dtype=np.float64)
    probability /= probability.sum()
    return np.stack(reduced), probability


__all__ = ["fit_fixed_assignments", "weighted_cluster_reduction"]
