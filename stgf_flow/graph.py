from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mm_jdwind.data import JointSplitData


@dataclass(frozen=True)
class SpectralArtifacts:
    adjacency: np.ndarray
    graph_basis: np.ndarray
    graph_eigenvalues: np.ndarray
    temporal_basis: np.ndarray
    temporal_frequencies: np.ndarray
    logit_mean: np.ndarray
    logit_std: np.ndarray
    transform_mode: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.adjacency.shape != (10, 10):
            raise ValueError("adjacency must be 10 x 10")
        if self.graph_basis.shape != (10, 10):
            raise ValueError("graph basis must be 10 x 10")
        if self.graph_eigenvalues.shape != (10,):
            raise ValueError("graph eigenvalues must have length 10")
        if self.temporal_basis.shape != (24, 24):
            raise ValueError("temporal basis must be 24 x 24")
        if self.temporal_frequencies.shape != (24,):
            raise ValueError("temporal frequencies must have length 24")
        if self.logit_mean.shape != (10, 24) or self.logit_std.shape != (10, 24):
            raise ValueError("logit statistics must have shape 10 x 24")
        if self.transform_mode not in {
            "time_domain",
            "graph_only",
            "time_frequency",
            "stgf",
        }:
            raise ValueError(f"unknown transform mode: {self.transform_mode}")

    def torch_payload(self) -> dict[str, Any]:
        return {
            "adjacency": torch.from_numpy(self.adjacency.copy()),
            "graph_basis": torch.from_numpy(self.graph_basis.copy()),
            "graph_eigenvalues": torch.from_numpy(
                self.graph_eigenvalues.copy()
            ),
            "temporal_basis": torch.from_numpy(self.temporal_basis.copy()),
            "temporal_frequencies": torch.from_numpy(
                self.temporal_frequencies.copy()
            ),
            "logit_mean": torch.from_numpy(self.logit_mean.copy()),
            "logit_std": torch.from_numpy(self.logit_std.copy()),
            "transform_mode": self.transform_mode,
            "metadata": self.metadata,
        }

    @classmethod
    def from_torch_payload(cls, payload: dict[str, Any]) -> "SpectralArtifacts":
        def array(name: str) -> np.ndarray:
            value = payload[name]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        return cls(
            adjacency=array("adjacency"),
            graph_basis=array("graph_basis"),
            graph_eigenvalues=array("graph_eigenvalues"),
            temporal_basis=array("temporal_basis"),
            temporal_frequencies=array("temporal_frequencies"),
            logit_mean=array("logit_mean"),
            logit_std=array("logit_std"),
            transform_mode=str(payload["transform_mode"]),
            metadata=dict(payload.get("metadata", {})),
        )


def dct_basis(length: int = 24) -> np.ndarray:
    positions = np.arange(length, dtype=np.float64)
    modes = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (positions[None] + 0.5) * modes / length)
    basis[0] *= np.sqrt(1.0 / length)
    basis[1:] *= np.sqrt(2.0 / length)
    return basis.astype(np.float32)


def correlation_graph(
    target: np.ndarray,
    *,
    neighbors: int = 3,
    power: float = 2.0,
    minimum_weight: float = 1e-4,
) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (10, 24):
        raise ValueError("training target must have shape day x 10 x 24")
    if not 1 <= neighbors <= 9:
        raise ValueError("neighbors must be between 1 and 9")
    series = values.transpose(1, 0, 2).reshape(10, -1)
    correlation = np.corrcoef(series)
    correlation = np.nan_to_num(correlation, nan=0.0)
    similarity = np.maximum(correlation, 0.0) ** power
    np.fill_diagonal(similarity, 0.0)
    directed = np.zeros_like(similarity)
    for zone in range(10):
        indices = np.argsort(similarity[zone])[-neighbors:]
        directed[zone, indices] = similarity[zone, indices]
    adjacency = np.maximum(directed, directed.T)
    adjacency[adjacency < minimum_weight] = 0.0
    degree = adjacency.sum(axis=1)
    if np.any(degree <= 0):
        raise RuntimeError("training-only correlation graph contains an isolated zone")
    return adjacency.astype(np.float32)


def graph_fourier(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(adjacency, dtype=np.float64)
    if weights.shape != (10, 10) or not np.allclose(weights, weights.T):
        raise ValueError("adjacency must be a symmetric 10 x 10 matrix")
    degree = weights.sum(axis=1)
    inverse_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    laplacian = np.eye(10) - inverse_sqrt @ weights @ inverse_sqrt
    eigenvalues, basis = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = basis[:, order]
    for mode in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0:
            basis[:, mode] *= -1
    return basis.astype(np.float32), eigenvalues.astype(np.float32)


def target_logit_statistics(
    target: np.ndarray, *, epsilon: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(np.asarray(target, dtype=np.float64), epsilon, 1.0 - epsilon)
    logit = np.log(clipped) - np.log1p(-clipped)
    mean = logit.mean(axis=0)
    std = logit.std(axis=0)
    std = np.maximum(std, 0.25)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_spectral_artifacts(
    split: JointSplitData,
    *,
    transform_mode: str,
    neighbors: int = 3,
    correlation_power: float = 2.0,
    epsilon: float = 1e-4,
) -> SpectralArtifacts:
    adjacency = correlation_graph(
        split.target, neighbors=neighbors, power=correlation_power
    )
    graph_basis, eigenvalues = graph_fourier(adjacency)
    temporal = dct_basis(24)
    temporal_frequencies = np.arange(24, dtype=np.float32) / 23.0
    if transform_mode in {"time_domain", "time_frequency"}:
        used_graph_basis = np.eye(10, dtype=np.float32)
        used_eigenvalues = np.arange(10, dtype=np.float32) / 9.0
    else:
        used_graph_basis = graph_basis
        used_eigenvalues = eigenvalues
    if transform_mode in {"time_domain", "graph_only"}:
        used_temporal = np.eye(24, dtype=np.float32)
    else:
        used_temporal = temporal
    mean, std = target_logit_statistics(split.target, epsilon=epsilon)
    edge_count = int(np.count_nonzero(np.triu(adjacency, 1)))
    return SpectralArtifacts(
        adjacency=adjacency,
        graph_basis=used_graph_basis,
        graph_eigenvalues=used_eigenvalues,
        temporal_basis=used_temporal,
        temporal_frequencies=temporal_frequencies,
        logit_mean=mean,
        logit_std=std,
        transform_mode=transform_mode,
        metadata={
            "fit_split": "train only",
            "graph_rule": "positive target-correlation symmetric kNN",
            "neighbors": int(neighbors),
            "correlation_power": float(correlation_power),
            "edge_count": edge_count,
            "target_logit_epsilon": float(epsilon),
            "spatial_transform": (
                "identity"
                if transform_mode in {"time_domain", "time_frequency"}
                else "normalized_laplacian_graph_fourier"
            ),
            "temporal_transform": (
                "identity"
                if transform_mode in {"time_domain", "graph_only"}
                else "orthonormal_dct_ii"
            ),
        },
    )


def encode_field(
    values: torch.Tensor,
    graph_basis: torch.Tensor,
    temporal_basis: torch.Tensor,
) -> torch.Tensor:
    if values.ndim == 3:
        return torch.einsum(
            "zg,bzh,kh->bgk", graph_basis, values, temporal_basis
        )
    if values.ndim == 4:
        return torch.einsum(
            "zg,bzhf,kh->bgkf", graph_basis, values, temporal_basis
        )
    raise ValueError("field must have shape BxZxH or BxZxHxF")


def decode_field(
    coefficients: torch.Tensor,
    graph_basis: torch.Tensor,
    temporal_basis: torch.Tensor,
) -> torch.Tensor:
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape BxGxK")
    return torch.einsum(
        "zg,bgk,kh->bzh", graph_basis, coefficients, temporal_basis
    )


__all__ = [
    "SpectralArtifacts",
    "correlation_graph",
    "dct_basis",
    "decode_field",
    "encode_field",
    "fit_spectral_artifacts",
    "graph_fourier",
    "target_logit_statistics",
]
