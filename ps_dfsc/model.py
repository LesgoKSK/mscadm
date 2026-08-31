from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CalibrationTensorOutput:
    scenarios: torch.Tensor
    probabilities: torch.Tensor
    ess: torch.Tensor
    entropy: torch.Tensor
    transport_cost: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor


class PSDFSCNetwork(nn.Module):
    """Permutation-equivariant scenario reweighter with monotone transport."""

    def __init__(
        self,
        *,
        zones: int = 10,
        hours: int = 24,
        embedding: int = 64,
        interaction_rank: int = 4,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if zones != 10 or hours != 24:
            raise ValueError("PS-DFSC currently requires 10 zones and 24 hours")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.zones = zones
        self.hours = hours
        self.rank = interaction_rank
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.Conv1d(zones, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, embedding, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(embedding * 3, embedding),
            nn.SiLU(),
            nn.Linear(embedding, 1),
        )
        parameter_count = 2 * (
            zones + hours + zones * interaction_rank + hours * interaction_rank
        )
        self.transport_head = nn.Sequential(
            nn.Linear(embedding * 2, embedding),
            nn.SiLU(),
            nn.Linear(embedding, parameter_count),
        )
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.logit_head[-1].bias)
        nn.init.zeros_(self.transport_head[-1].weight)
        nn.init.zeros_(self.transport_head[-1].bias)

    def _surface(
        self, raw: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, int]:
        batch = raw.shape[0]
        z = raw[:, offset : offset + self.zones]
        offset += self.zones
        h = raw[:, offset : offset + self.hours]
        offset += self.hours
        left = raw[:, offset : offset + self.zones * self.rank].reshape(
            batch, self.zones, self.rank
        )
        offset += self.zones * self.rank
        right = raw[:, offset : offset + self.hours * self.rank].reshape(
            batch, self.hours, self.rank
        )
        offset += self.hours * self.rank
        interaction = torch.einsum("bzr,bhr->bzh", left, right) / max(self.rank, 1)
        return z[:, :, None] + h[:, None, :] + interaction, offset

    @staticmethod
    def monotone_transport(
        scenarios: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        *,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        interior = scenarios.clamp(epsilon, 1.0 - epsilon)
        logits = torch.log(interior) - torch.log1p(-interior)
        moved = torch.sigmoid(scale[:, None] * logits + shift[:, None])
        moved = torch.where(scenarios == 0.0, torch.zeros_like(moved), moved)
        moved = torch.where(scenarios == 1.0, torch.ones_like(moved), moved)
        return moved

    def forward(self, scenarios: torch.Tensor) -> CalibrationTensorOutput:
        if scenarios.ndim != 4 or scenarios.shape[2:] != (self.zones, self.hours):
            raise ValueError("scenarios must have shape [batch,member,10,24]")
        if not torch.isfinite(scenarios).all():
            raise ValueError("scenarios contain non-finite values")
        batch, members = scenarios.shape[:2]
        embedded = self.encoder(scenarios.reshape(batch * members, 10, 24))
        embedded = embedded.reshape(batch, members, -1)
        pooled_mean = embedded.mean(dim=1)
        pooled_max = embedded.max(dim=1).values
        context = torch.cat((pooled_mean, pooled_max), dim=-1)
        expanded = torch.cat(
            (
                embedded,
                pooled_mean[:, None].expand(-1, members, -1),
                pooled_max[:, None].expand(-1, members, -1),
            ),
            dim=-1,
        )
        logits = self.logit_head(expanded).squeeze(-1)
        probabilities = torch.softmax(logits / self.temperature, dim=1)
        raw_transport = self.transport_head(context)
        scale_raw, offset = self._surface(raw_transport, 0)
        shift_raw, offset = self._surface(raw_transport, offset)
        if offset != raw_transport.shape[1]:
            raise AssertionError("transport parameter layout mismatch")
        scale = 1.0 + 0.1 * torch.tanh(scale_raw)
        shift = 0.15 * torch.tanh(shift_raw)
        moved = self.monotone_transport(scenarios, scale, shift)
        ess = 1.0 / probabilities.square().sum(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        distance = (moved - scenarios).abs().mean(dim=(2, 3))
        transport_cost = (probabilities * distance).sum(dim=1)
        return CalibrationTensorOutput(
            scenarios=moved,
            probabilities=probabilities,
            ess=ess,
            entropy=entropy,
            transport_cost=transport_cost,
            scale=scale,
            shift=shift,
        )


__all__ = ["CalibrationTensorOutput", "PSDFSCNetwork"]
