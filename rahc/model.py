from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("hour", "hour_zone", "hour_regime", "full", "full_no_shrink")


def beta_binomial_log_probability(
    rank: torch.Tensor,
    members: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    value = rank.to(alpha.dtype)
    count = torch.as_tensor(float(members), dtype=alpha.dtype, device=alpha.device)
    combination = torch.lgamma(count + 1.0) - torch.lgamma(value + 1.0) - torch.lgamma(count - value + 1.0)
    numerator = (
        torch.lgamma(value + alpha)
        + torch.lgamma(count - value + beta)
        - torch.lgamma(count + alpha + beta)
    )
    denominator = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return combination + numerator - denominator


def beta_binomial_nll(
    rank: torch.Tensor,
    members: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return -beta_binomial_log_probability(rank, members, alpha, beta).mean()


class HierarchicalBetaBinomial(nn.Module):
    """Conditional Beta-Binomial rank model with additive partial pooling."""

    def __init__(
        self,
        *,
        variant: str,
        regime_dim: int,
        hidden_dim: int = 24,
        hours: int = 24,
        zones: int = 10,
        uncertainty_feature: int = 1,
        maximum_log_shape: float = 3.5,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown RAHC variant: {variant}")
        self.variant = variant
        self.uncertainty_feature = int(uncertainty_feature)
        self.maximum_log_shape = float(maximum_log_shape)
        self.global_effect = nn.Parameter(torch.zeros(2))
        self.hour_effect = nn.Embedding(hours, 2)
        nn.init.zeros_(self.hour_effect.weight)
        self.use_zone = variant in {"hour_zone", "full", "full_no_shrink"}
        self.use_regime = variant in {"hour_regime", "full", "full_no_shrink"}
        self.zone_effect = nn.Embedding(zones, 2) if self.use_zone else None
        if self.zone_effect is not None:
            nn.init.zeros_(self.zone_effect.weight)
        self.regime_network = (
            nn.Sequential(
                nn.Linear(regime_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
            if self.use_regime
            else None
        )
        if self.regime_network is not None:
            nn.init.zeros_(self.regime_network[-1].weight)
            nn.init.zeros_(self.regime_network[-1].bias)
        self.zone_uncertainty = (
            nn.Embedding(zones, 2) if variant in {"full", "full_no_shrink"} else None
        )
        if self.zone_uncertainty is not None:
            nn.init.zeros_(self.zone_uncertainty.weight)

    def log_shapes(
        self,
        regime: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        value = self.global_effect + self.hour_effect(hour)
        if self.zone_effect is not None:
            value = value + self.zone_effect(zone)
        if self.regime_network is not None:
            value = value + self.regime_network(regime)
        if self.zone_uncertainty is not None:
            slope = self.zone_uncertainty(zone)
            value = value + slope * regime[:, self.uncertainty_feature : self.uncertainty_feature + 1]
        value = value * float(strength)
        return value.clamp(-self.maximum_log_shape, self.maximum_log_shape)

    def forward(
        self,
        regime: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_shapes = self.log_shapes(regime, hour, zone, strength=strength)
        shapes = torch.exp(log_shapes)
        return shapes[:, 0], shapes[:, 1], log_shapes

    def regularization(self, log_shapes: torch.Tensor) -> torch.Tensor:
        parameter_terms = [parameter.square().mean() for parameter in self.parameters()]
        parameter_penalty = torch.stack(parameter_terms).mean()
        identity_penalty = log_shapes.square().mean()
        return identity_penalty + parameter_penalty


__all__ = [
    "VARIANTS",
    "HierarchicalBetaBinomial",
    "beta_binomial_log_probability",
    "beta_binomial_nll",
]
