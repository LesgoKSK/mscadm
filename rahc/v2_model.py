"""Identifiable hierarchical Beta--Binomial model used by RAHC."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = (
    "global",
    "hour",
    "hour_zone",
    "hour_regime",
    "additive",
    "full",
    "full_no_shrink",
)


def beta_binomial_log_pmf(
    rank: torch.Tensor,
    members: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Element-wise Beta--Binomial log probability."""

    value = rank.to(alpha.dtype)
    total = torch.as_tensor(float(members), dtype=alpha.dtype, device=alpha.device)
    log_choose = (
        torch.lgamma(total + 1.0)
        - torch.lgamma(value + 1.0)
        - torch.lgamma(total - value + 1.0)
    )
    log_beta_num = (
        torch.lgamma(value + alpha)
        + torch.lgamma(total - value + beta)
        - torch.lgamma(total + alpha + beta)
    )
    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return log_choose + log_beta_num - log_beta_den


def censored_beta_binomial_nll(
    lower: torch.Tensor,
    upper: torch.Tensor,
    members: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood for a tie-censored rank interval.

    A continuous observation would reveal one pooled rank.  When clipping
    creates a tie block, only ``lower <= R <= upper`` is observed.  Summing
    the Beta--Binomial probabilities across that block avoids choosing the
    systematically biased ``<``, ``<=`` or deterministic mid-rank.
    """

    grid = torch.arange(members + 1, device=alpha.device, dtype=alpha.dtype)
    log_probability = beta_binomial_log_pmf(
        grid[None, :], members, alpha[:, None], beta[:, None]
    )
    valid = (grid[None, :] >= lower[:, None]) & (grid[None, :] <= upper[:, None])
    interval_log_probability = torch.logsumexp(
        log_probability.masked_fill(~valid, -torch.inf), dim=1
    )
    return -interval_log_probability.mean()


class HierarchicalRankModel(nn.Module):
    """Centered additive effects with identity at the all-zero parameter."""

    def __init__(self, variant: str, regime_dim: int = 4, hours: int = 24, zones: int = 10):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.use_hour = variant != "global"
        self.use_zone = variant in {"hour_zone", "additive", "full", "full_no_shrink"}
        self.use_regime = variant in {"hour_regime", "additive", "full", "full_no_shrink"}
        self.use_zone_slope = variant in {"full", "full_no_shrink"}
        self.global_effect = nn.Parameter(torch.zeros(2))
        self.hour_effect = nn.Parameter(torch.zeros(hours, 2)) if self.use_hour else None
        self.zone_effect = nn.Parameter(torch.zeros(zones, 2)) if self.use_zone else None
        self.regime_effect = nn.Parameter(torch.zeros(regime_dim, 2)) if self.use_regime else None
        self.zone_spread_slope = nn.Parameter(torch.zeros(zones, 2)) if self.use_zone_slope else None

    @staticmethod
    def _center(effect: torch.Tensor) -> torch.Tensor:
        return effect - effect.mean(dim=0, keepdim=True)

    def predictors(
        self,
        regime: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        value = self.global_effect.expand(len(regime), -1)
        if self.hour_effect is not None:
            value = value + self._center(self.hour_effect)[hour]
        if self.zone_effect is not None:
            value = value + self._center(self.zone_effect)[zone]
        if self.regime_effect is not None:
            value = value + regime @ self.regime_effect
        if self.zone_spread_slope is not None:
            slopes = self._center(self.zone_spread_slope)[zone]
            value = value + slopes * regime[:, 1:2]
        return value.mul(float(strength)).clamp(-5.0, 5.0)

    def forward(
        self,
        regime: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictors = self.predictors(regime, hour, zone, strength)
        mean = torch.sigmoid(predictors[:, 0])
        concentration = 2.0 * torch.exp(predictors[:, 1])
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        return alpha.clamp_min(1e-4), beta.clamp_min(1e-4), predictors

    def penalty(self) -> torch.Tensor:
        terms: list[torch.Tensor] = [self.global_effect.square().mean()]
        if self.hour_effect is not None:
            centered = self._center(self.hour_effect)
            cyclic_difference = centered - torch.roll(centered, shifts=1, dims=0)
            terms.extend((0.25 * centered.square().mean(), cyclic_difference.square().mean()))
        if self.zone_effect is not None:
            terms.append(self._center(self.zone_effect).square().mean())
        if self.regime_effect is not None:
            terms.append(self.regime_effect.square().mean())
        if self.zone_spread_slope is not None:
            terms.append(2.0 * self._center(self.zone_spread_slope).square().mean())
        return torch.stack(terms).sum()


__all__ = [
    "VARIANTS",
    "HierarchicalRankModel",
    "beta_binomial_log_pmf",
    "censored_beta_binomial_nll",
]
