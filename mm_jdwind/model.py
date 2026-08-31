from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from repro.models.mscadm import SinusoidalEmbedding

from .data import INTERIOR_STATE, MASK_STATE, ONE_STATE, ZERO_STATE


def _logit(value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    bounded = value.clamp(epsilon, 1.0 - epsilon)
    return torch.log(bounded) - torch.log1p(-bounded)


class AxialBlock(nn.Module):
    """Temporal then cross-zone attention for a [B,Z,T,D] field."""

    def __init__(self, dim: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.zone_norm = nn.LayerNorm(dim)
        self.zone_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_multiplier * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, zones, hours, dim = value.shape
        temporal = self.time_norm(value).reshape(batch * zones, hours, dim)
        attended, _ = self.time_attention(
            temporal, temporal, temporal, need_weights=False
        )
        value = value + attended.reshape(batch, zones, hours, dim)
        spatial = self.zone_norm(value).transpose(1, 2).reshape(
            batch * hours, zones, dim
        )
        attended, _ = self.zone_attention(spatial, spatial, spatial, need_weights=False)
        value = value + attended.reshape(batch, hours, zones, dim).transpose(1, 2)
        return value + self.ff(self.ff_norm(value))


class JointConditionEncoder(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
        zones: int = 10,
        hours: int = 24,
    ) -> None:
        super().__init__()
        self.zones = int(zones)
        self.hours = int(hours)
        self.input = nn.Linear(condition_dim, model_dim)
        self.zone_position = nn.Parameter(torch.randn(1, zones, 1, model_dim) * 0.02)
        self.hour_position = nn.Parameter(torch.randn(1, 1, hours, model_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AxialBlock(model_dim, heads, ff_multiplier, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 4 or condition.shape[1:3] != (
            self.zones,
            self.hours,
        ):
            raise ValueError(
                f"condition must be [B,{self.zones},{self.hours},F], "
                f"got {tuple(condition.shape)}"
            )
        hidden = self.input(condition) + self.zone_position + self.hour_position
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


@dataclass
class MixedMeasureOutput:
    location: torch.Tensor
    scale: torch.Tensor
    zero_logit: torch.Tensor
    upper_conditional_logit: torch.Tensor

    @property
    def zero_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.zero_logit)

    @property
    def upper_conditional_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.upper_conditional_logit)

    @property
    def one_probability(self) -> torch.Tensor:
        return (1.0 - self.zero_probability) * self.upper_conditional_probability

    @property
    def interior_probability(self) -> torch.Tensor:
        return (1.0 - self.zero_probability) * (
            1.0 - self.upper_conditional_probability
        )

    @property
    def state_probabilities(self) -> torch.Tensor:
        return torch.stack(
            (self.zero_probability, self.interior_probability, self.one_probability),
            dim=-1,
        )


class MixedMeasureHead(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int = 20,
        model_dim: int = 96,
        depth: int = 2,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        minimum_scale: float = 0.05,
        maximum_scale: float = 3.0,
        upper_logit_prior: float = -10.0,
        upper_logit_deviation: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_scale < maximum_scale:
            raise ValueError("invalid scale bounds")
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.upper_logit_deviation = float(upper_logit_deviation)
        self.encoder = JointConditionEncoder(
            condition_dim=condition_dim,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )
        self.output = nn.Linear(model_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer(
            "upper_logit_prior", torch.tensor(float(upper_logit_prior))
        )

    def set_upper_probability_prior(self, probability: float) -> None:
        bounded = min(max(float(probability), 1e-8), 1.0 - 1e-8)
        self.upper_logit_prior.fill_(math.log(bounded / (1.0 - bounded)))

    def forward(self, condition: torch.Tensor) -> MixedMeasureOutput:
        raw = self.output(self.encoder(condition))
        fraction = torch.sigmoid(raw[..., 1])
        scale = self.minimum_scale + (
            self.maximum_scale - self.minimum_scale
        ) * fraction
        upper = self.upper_logit_prior + self.upper_logit_deviation * torch.tanh(
            raw[..., 3]
        )
        return MixedMeasureOutput(
            location=raw[..., 0],
            scale=scale,
            zero_logit=raw[..., 2],
            upper_conditional_logit=upper,
        )


def mixed_measure_head_loss(
    observation: torch.Tensor,
    state: torch.Tensor,
    output: MixedMeasureOutput,
    *,
    mean_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    zero = state == ZERO_STATE
    nonzero = ~zero
    one = state == ONE_STATE
    interior = state == INTERIOR_STATE
    zero_loss = F.binary_cross_entropy_with_logits(
        output.zero_logit, zero.float(), reduction="mean"
    )
    if bool(nonzero.any()):
        upper_loss = F.binary_cross_entropy_with_logits(
            output.upper_conditional_logit[nonzero],
            one[nonzero].float(),
            reduction="mean",
        )
    else:
        upper_loss = output.upper_conditional_logit.sum() * 0.0
    if bool(interior.any()):
        transformed = _logit(observation[interior])
        location = output.location[interior]
        scale = output.scale[interior]
        standardized = (transformed - location) / scale
        continuous_nll = (
            torch.log(scale) + 0.5 * standardized.square()
        ).mean()
        mean_loss = F.smooth_l1_loss(location, transformed)
    else:
        continuous_nll = output.location.sum() * 0.0
        mean_loss = output.location.sum() * 0.0
    loss = zero_loss + upper_loss + continuous_nll + mean_weight * mean_loss
    return {
        "loss": loss,
        "zero_bce": zero_loss,
        "upper_bce": upper_loss,
        "continuous_nll": continuous_nll,
        "mean_loss": mean_loss,
    }


class CategoricalJumpGenerator(nn.Module):
    """Absorbing-mask discrete jump generator for correlated boundary states."""

    def __init__(
        self,
        *,
        condition_dim: int = 20,
        model_dim: int = 96,
        depth: int = 3,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(condition_dim, model_dim)
        self.state = nn.Embedding(4, model_dim)
        self.zone_position = nn.Parameter(torch.randn(1, 10, 1, model_dim) * 0.02)
        self.hour_position = nn.Parameter(torch.randn(1, 1, 24, model_dim) * 0.02)
        self.time = nn.Sequential(
            SinusoidalEmbedding(model_dim),
            nn.Linear(model_dim, 2 * model_dim),
            nn.SiLU(),
            nn.Linear(2 * model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                AxialBlock(model_dim, heads, ff_multiplier, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.correction = nn.Linear(model_dim, 3)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(
        self,
        condition: torch.Tensor,
        partial_state: torch.Tensor,
        time: torch.Tensor,
        base_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if partial_state.shape != condition.shape[:3]:
            raise ValueError("partial state does not align with condition")
        hidden = (
            self.condition(condition)
            + self.state(partial_state)
            + self.zone_position
            + self.hour_position
            + self.time(time)[:, None, None]
        )
        for block in self.blocks:
            hidden = block(hidden)
        base_logits = torch.log(base_probabilities.clamp_min(1e-8))
        return base_logits + self.correction(self.norm(hidden))

    def loss(
        self,
        condition: torch.Tensor,
        target_state: torch.Tensor,
        base_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = len(condition)
        time = torch.rand(batch, device=condition.device)
        masked = torch.rand_like(target_state, dtype=torch.float32) > time[:, None, None]
        # Guarantee at least one supervised token for every sample.
        none = ~masked.flatten(1).any(dim=1)
        if bool(none.any()):
            masked[none, 0, 0] = True
        partial = target_state.masked_fill(masked, MASK_STATE)
        logits = self(condition, partial, time, base_probabilities)
        loss = F.cross_entropy(logits[masked], target_state[masked])
        accuracy = (logits[masked].argmax(-1) == target_state[masked]).float().mean()
        return {"loss": loss, "masked_accuracy": accuracy, "masked_fraction": masked.float().mean()}


class ResidualRectifiedFlow(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int = 20,
        model_dim: int = 112,
        depth: int = 4,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(condition_dim, model_dim)
        self.value = nn.Linear(1, model_dim)
        self.state = nn.Embedding(3, model_dim)
        self.zone_position = nn.Parameter(torch.randn(1, 10, 1, model_dim) * 0.02)
        self.hour_position = nn.Parameter(torch.randn(1, 1, 24, model_dim) * 0.02)
        self.time = nn.Sequential(
            SinusoidalEmbedding(model_dim),
            nn.Linear(model_dim, 4 * model_dim),
            nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                AxialBlock(model_dim, heads, ff_multiplier, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.value(value[..., None])
            + self.condition(condition)
            + self.state(state)
            + self.zone_position
            + self.hour_position
            + self.time(time)[:, None, None]
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))[..., 0]

    def loss(
        self,
        target_residual: torch.Tensor,
        condition: torch.Tensor,
        state: torch.Tensor,
        *,
        atom_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        noise = torch.randn_like(target_residual)
        time = torch.rand(len(target_residual), device=target_residual.device)
        path = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target_residual
        velocity = self(path, time, condition, state)
        target_velocity = target_residual - noise
        weights = torch.where(
            state == INTERIOR_STATE,
            torch.ones_like(target_residual),
            torch.full_like(target_residual, float(atom_weight)),
        )
        squared = (velocity - target_velocity).square()
        loss = (weights * squared).sum() / weights.sum().clamp_min(1.0)
        interior = state == INTERIOR_STATE
        interior_mse = squared[interior].mean() if bool(interior.any()) else loss * 0.0
        return {"loss": loss, "interior_mse": interior_mse}


class MMJDWind(nn.Module):
    def __init__(
        self,
        *,
        head: dict | None = None,
        jump: dict | None = None,
        flow: dict | None = None,
    ) -> None:
        super().__init__()
        self.head = MixedMeasureHead(**(head or {}))
        self.jump = CategoricalJumpGenerator(**(jump or {}))
        self.flow = ResidualRectifiedFlow(**(flow or {}))

    def statistics(self, condition: torch.Tensor) -> MixedMeasureOutput:
        return self.head(condition)

    def residual(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        statistics: MixedMeasureOutput,
    ) -> torch.Tensor:
        transformed = _logit(observation)
        residual = (transformed - statistics.location) / statistics.scale
        return torch.where(
            state == INTERIOR_STATE,
            residual.clamp(-8.0, 8.0),
            torch.zeros_like(residual),
        )

    def reconstruct(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
        statistics: MixedMeasureOutput,
    ) -> torch.Tensor:
        continuous = torch.sigmoid(
            statistics.location + statistics.scale * residual
        )
        result = torch.where(state == ZERO_STATE, torch.zeros_like(continuous), continuous)
        return torch.where(state == ONE_STATE, torch.ones_like(result), result)


__all__ = [
    "AxialBlock",
    "CategoricalJumpGenerator",
    "JointConditionEncoder",
    "MMJDWind",
    "MixedMeasureHead",
    "MixedMeasureOutput",
    "ResidualRectifiedFlow",
    "mixed_measure_head_loss",
]
