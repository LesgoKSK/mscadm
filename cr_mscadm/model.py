from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from repro.models.mscadm import MSCADM, MultiScaleEmbedding


class ConditionalLocationScale(nn.Module):
    """Predict a 24-hour conditional location and strictly positive scale."""

    def __init__(
        self,
        *,
        condition_dim: int = 20,
        model_dim: int = 96,
        bottleneck: int = 48,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        minimum_scale: float = 0.05,
        maximum_scale: float = 2.5,
    ) -> None:
        super().__init__()
        if not 0 < minimum_scale < maximum_scale:
            raise ValueError("scales must satisfy 0 < minimum < maximum")
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.embedding = MultiScaleEmbedding(condition_dim, model_dim, bottleneck)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.embedding(condition))
        raw = self.output(self.norm(hidden))
        location = raw[..., :1]
        fraction = torch.sigmoid(raw[..., 1:2])
        scale = self.minimum_scale + (self.maximum_scale - self.minimum_scale) * fraction
        return location, scale


class ResidualMSCADM(nn.Module):
    """MS-CADM operating on a standardized conditional residual.

    ``mode='fixed'`` retains a condition-dependent location but replaces the
    scale head by a training-set, hour-wise constant. ``mode='hetero'`` uses
    the predicted conditional scale.
    """

    def __init__(
        self,
        *,
        head_config: dict | None = None,
        denoiser_config: dict | None = None,
        mode: str = "hetero",
    ) -> None:
        super().__init__()
        if mode not in {"fixed", "hetero"}:
            raise ValueError("mode must be 'fixed' or 'hetero'")
        self.mode = mode
        self.statistics = ConditionalLocationScale(**(head_config or {}))
        self.denoiser = MSCADM(**(denoiser_config or {}))
        sequence_length = int((denoiser_config or {}).get("sequence_length", 24))
        self.register_buffer("fixed_scale", torch.ones(1, sequence_length, 1))

    def predict_statistics(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, conditional_scale = self.statistics(condition)
        if self.mode == "fixed":
            scale = self.fixed_scale.expand(len(condition), -1, -1)
        else:
            scale = conditional_scale
        return location, scale

    def standardize(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        location, scale = self.predict_statistics(condition)
        return (target - location) / scale

    def reconstruct(self, residual: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        location, scale = self.predict_statistics(condition)
        return location + scale * residual

    def forward(
        self, noisy_residual: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.denoiser(noisy_residual, timestep, condition)


def gaussian_crps(
    observation: torch.Tensor, location: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Closed-form CRPS of a Gaussian forecast (lower is better)."""

    z = (observation - location) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return scale * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def statistics_loss(
    observation: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    crps_weight: float,
    mean_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    error = observation - location
    laplace = torch.log(scale) + error.abs() / scale
    crps = gaussian_crps(observation, location, scale)
    mean = F.smooth_l1_loss(location, observation, reduction="none")
    total = laplace + crps_weight * crps + mean_weight * mean
    return {
        "loss": total.mean(),
        "laplace": laplace.mean(),
        "crps_head": crps.mean(),
        "mean_loss": mean.mean(),
    }


__all__ = [
    "ConditionalLocationScale",
    "ResidualMSCADM",
    "gaussian_crps",
    "statistics_loss",
]
