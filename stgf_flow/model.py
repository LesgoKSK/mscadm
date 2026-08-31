from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .graph import (
    SpectralArtifacts,
    decode_field,
    encode_field,
)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 8:
            raise ValueError("time embedding dimension must be at least 8")
        half = dimension // 2
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * half, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time[:, None] * self.frequencies[None] * 2.0 * math.pi
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


class AxialAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        *,
        heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(dimension)
        self.time_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.zone_norm = nn.LayerNorm(dimension)
        self.zone_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, ff_multiplier * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, zones, hours, dimension = values.shape
        time_values = self.time_norm(values).reshape(
            batch * zones, hours, dimension
        )
        attended, _ = self.time_attention(
            time_values, time_values, time_values, need_weights=False
        )
        values = values + attended.reshape(batch, zones, hours, dimension)
        zone_values = (
            self.zone_norm(values)
            .permute(0, 2, 1, 3)
            .reshape(batch * hours, zones, dimension)
        )
        attended, _ = self.zone_attention(
            zone_values, zone_values, zone_values, need_weights=False
        )
        values = values + attended.reshape(
            batch, hours, zones, dimension
        ).permute(0, 2, 1, 3)
        return values + self.feed_forward(self.ff_norm(values))


class PhysicalCenter(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(condition_dim, model_dim)
        self.zone_position = nn.Parameter(torch.randn(1, 10, 1, model_dim) * 0.02)
        self.hour_position = nn.Parameter(torch.randn(1, 1, 24, model_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(
                    model_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        values = (
            self.condition(condition) + self.zone_position + self.hour_position
        )
        for block in self.blocks:
            values = block(values)
        return self.output(values).squeeze(-1)


class SpectralVelocity(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value = nn.Linear(1, model_dim)
        self.condition = nn.Linear(condition_dim, model_dim)
        self.time = FourierTimeEmbedding(model_dim)
        self.graph_coordinate = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.temporal_coordinate = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.graph_position = nn.Parameter(
            torch.randn(1, 10, 1, model_dim) * 0.02
        )
        self.frequency_position = nn.Parameter(
            torch.randn(1, 1, 24, model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(
                    model_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        spectral_condition: torch.Tensor,
        graph_eigenvalues: torch.Tensor,
        temporal_frequencies: torch.Tensor,
    ) -> torch.Tensor:
        graph_coordinate = self.graph_coordinate(
            graph_eigenvalues[:, None]
        )[None, :, None]
        temporal_coordinate = self.temporal_coordinate(
            temporal_frequencies[:, None]
        )[None, None]
        values = (
            self.value(state[..., None])
            + self.condition(spectral_condition)
            + self.time(time)[:, None, None]
            + graph_coordinate
            + temporal_coordinate
            + self.graph_position
            + self.frequency_position
        )
        for block in self.blocks:
            values = block(values)
        return self.output(values).squeeze(-1)


class STGFFlow(nn.Module):
    def __init__(
        self,
        artifacts: SpectralArtifacts,
        *,
        condition_dim: int = 20,
        center_dim: int = 64,
        center_depth: int = 2,
        flow_dim: int = 96,
        flow_depth: int = 4,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        logit_epsilon: float = 1e-4,
        low_frequency_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.logit_epsilon = float(logit_epsilon)
        self.low_frequency_weight = float(low_frequency_weight)
        self.transform_mode = artifacts.transform_mode
        self.artifact_metadata = dict(artifacts.metadata)
        self.register_buffer(
            "adjacency", torch.from_numpy(artifacts.adjacency.copy())
        )
        self.register_buffer(
            "graph_basis", torch.from_numpy(artifacts.graph_basis.copy())
        )
        self.register_buffer(
            "graph_eigenvalues",
            torch.from_numpy(artifacts.graph_eigenvalues.copy()),
        )
        self.register_buffer(
            "temporal_basis",
            torch.from_numpy(artifacts.temporal_basis.copy()),
        )
        self.register_buffer(
            "temporal_frequencies",
            torch.from_numpy(artifacts.temporal_frequencies.copy()),
        )
        self.register_buffer(
            "logit_mean", torch.from_numpy(artifacts.logit_mean.copy())
        )
        self.register_buffer(
            "logit_std", torch.from_numpy(artifacts.logit_std.copy())
        )
        self.center = PhysicalCenter(
            condition_dim,
            model_dim=center_dim,
            depth=center_depth,
            heads=heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )
        self.flow = SpectralVelocity(
            condition_dim,
            model_dim=flow_dim,
            depth=flow_depth,
            heads=heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )

    def standardized_logit(self, observation: torch.Tensor) -> torch.Tensor:
        clipped = observation.clamp(
            self.logit_epsilon, 1.0 - self.logit_epsilon
        )
        logit = torch.logit(clipped)
        return (logit - self.logit_mean) / self.logit_std

    def physical_from_standardized(self, value: torch.Tensor) -> torch.Tensor:
        logit = value * self.logit_std + self.logit_mean
        return torch.sigmoid(logit)

    def spectral_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return encode_field(
            condition, self.graph_basis, self.temporal_basis
        )

    def spectral_residual(
        self, observation: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        centered = self.standardized_logit(observation) - self.center(condition)
        return encode_field(centered, self.graph_basis, self.temporal_basis)

    def velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        spectral_condition = self.spectral_condition(condition)
        return self.flow(
            state,
            time,
            spectral_condition,
            self.graph_eigenvalues,
            self.temporal_frequencies,
        )

    def reconstruct(
        self, residual: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        physical_residual = decode_field(
            residual, self.graph_basis, self.temporal_basis
        )
        standardized = self.center(condition) + physical_residual
        return self.physical_from_standardized(standardized).clamp(0.0, 1.0)

    def center_loss(
        self, condition: torch.Tensor, observation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        target = self.standardized_logit(observation)
        prediction = self.center(condition)
        loss = (prediction - target).square().mean()
        physical = self.physical_from_standardized(prediction)
        return {
            "loss": loss,
            "standardized_mse": loss,
            "physical_mae": (physical - observation).abs().mean(),
        }

    def flow_loss(
        self,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            target = self.spectral_residual(observation, condition)
        noise = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
        time = torch.rand(
            len(target),
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
        state = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
        desired = target - noise
        prediction = self.velocity(state, time, condition)
        normalized_graph = self.graph_eigenvalues / self.graph_eigenvalues.max().clamp_min(
            1e-6
        )
        band = normalized_graph[:, None] + self.temporal_frequencies[None]
        weight = 1.0 + self.low_frequency_weight / (1.0 + 4.0 * band)
        weight = weight / weight.mean()
        error = (prediction - desired).square()
        return {
            "loss": (error * weight[None]).mean(),
            "velocity_mse": error.mean(),
            "low_band_mse": error[:, :3, :6].mean(),
            "high_band_mse": error[:, 3:, 6:].mean(),
        }

    def model_spec(self) -> dict[str, Any]:
        return {
            "transform_mode": self.transform_mode,
            "artifact_metadata": self.artifact_metadata,
            "logit_epsilon": self.logit_epsilon,
            "low_frequency_weight": self.low_frequency_weight,
        }


__all__ = [
    "AxialAttentionBlock",
    "FourierTimeEmbedding",
    "PhysicalCenter",
    "STGFFlow",
    "SpectralVelocity",
]
