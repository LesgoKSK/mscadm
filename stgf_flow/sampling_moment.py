from __future__ import annotations

import torch

from .graph import decode_field
from .model import STGFFlow
from .sampling import integrate_heun


@torch.no_grad()
def sample_stgf_moment_anchored(
    model: STGFFlow,
    condition: torch.Tensor,
    *,
    members: int,
    steps: int,
    member_chunk: int,
    seed: int,
    residual_temperature: float,
    physical_mean_anchor: float,
) -> torch.Tensor:
    if not 0.0 < residual_temperature <= 2.0:
        raise ValueError("residual_temperature must lie in (0, 2]")
    if not 0.0 <= physical_mean_anchor <= 1.0:
        raise ValueError("physical_mean_anchor must lie in [0, 1]")
    model.eval()
    batch = len(condition)
    center_standardized = model.center(condition)
    center_physical = model.physical_from_standardized(center_standardized)
    spectral_condition = model.spectral_condition(condition)
    generator = torch.Generator(device=condition.device).manual_seed(int(seed))
    chunks: list[torch.Tensor] = []
    completed = 0
    while completed < members:
        current = min(member_chunk, members - completed)
        repeated_spectral = spectral_condition[:, None].expand(
            -1, current, -1, -1, -1
        ).reshape(batch * current, 10, 24, condition.shape[-1])
        initial = torch.randn(
            batch * current,
            10,
            24,
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        residual = integrate_heun(
            model, initial, repeated_spectral, steps=steps
        )
        physical_residual = decode_field(
            residual, model.graph_basis, model.temporal_basis
        )
        repeated_center = center_standardized[:, None].expand(
            -1, current, -1, -1
        ).reshape(batch * current, 10, 24)
        raw = model.physical_from_standardized(
            repeated_center + physical_residual
        ).reshape(batch, current, 10, 24)
        chunks.append(raw)
        completed += current
    raw = torch.cat(chunks, dim=1)
    raw_mean = raw.mean(dim=1, keepdim=True)
    anchored_mean = raw_mean + physical_mean_anchor * (
        center_physical[:, None] - raw_mean
    )
    scenarios = anchored_mean + residual_temperature * (raw - raw_mean)
    return scenarios.clamp(0.0, 1.0)


__all__ = ["sample_stgf_moment_anchored"]
