"""Hierarchical atom, miss-risk gate, and interior rank models for CAA."""

from __future__ import annotations

import torch
from torch import nn

from rahc.v2_model import HierarchicalRankModel


class CenteredEffectModel(nn.Module):
    """Low-rank hierarchical effects with centered hour/zone identifiability."""

    def __init__(self, output_dim: int, regime_dim: int, *, zone_spread: bool = True) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.global_effect = nn.Parameter(torch.zeros(output_dim))
        self.hour_effect = nn.Parameter(torch.zeros(24, output_dim))
        self.zone_effect = nn.Parameter(torch.zeros(10, output_dim))
        self.regime_effect = nn.Parameter(torch.zeros(regime_dim, output_dim))
        self.zone_spread = (
            nn.Parameter(torch.zeros(10, output_dim)) if zone_spread else None
        )

    @staticmethod
    def _center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=0, keepdim=True)

    def forward(
        self,
        regime: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        value = self.global_effect.expand(len(regime), -1)
        value = value + self._center(self.hour_effect)[hour]
        value = value + self._center(self.zone_effect)[zone]
        value = value + regime @ self.regime_effect
        if self.zone_spread is not None:
            value = value + self._center(self.zone_spread)[zone] * regime[:, 1:2]
        return value.mul(float(strength)).clamp(-6.0, 6.0)

    def penalty(self) -> torch.Tensor:
        hour = self._center(self.hour_effect)
        cyclic = hour - torch.roll(hour, shifts=1, dims=0)
        terms = [
            self.global_effect.square().mean(),
            0.25 * hour.square().mean(),
            cyclic.square().mean(),
            self._center(self.zone_effect).square().mean(),
            self.regime_effect.square().mean(),
        ]
        if self.zone_spread is not None:
            terms.append(2.0 * self._center(self.zone_spread).square().mean())
        return torch.stack(terms).sum()


class CAAStateModels(nn.Module):
    """Three independent components sharing only the forecast feature schema."""

    def __init__(self, regime_dim: int) -> None:
        super().__init__()
        # Validation contains enough zero events for a state model.  The upper
        # atom is deliberately not parameterized because y=1 is essentially
        # absent in development data; its raw conditional mass is retained.
        self.zero_atom = CenteredEffectModel(1, regime_dim)
        self.miss_gate = CenteredEffectModel(1, regime_dim)
        self.interior_rank = HierarchicalRankModel(
            variant="full", regime_dim=regime_dim, hours=24, zones=10
        )

    def penalty(self, component: str) -> torch.Tensor:
        if component == "zero_atom":
            return self.zero_atom.penalty()
        if component == "miss_gate":
            return self.miss_gate.penalty()
        if component == "interior_rank":
            return self.interior_rank.penalty()
        raise ValueError(f"unknown component {component!r}")


__all__ = ["CAAStateModels", "CenteredEffectModel"]
