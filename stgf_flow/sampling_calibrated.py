from __future__ import annotations

import torch

from .graph import decode_field
from .model import STGFFlow
from .sampling import integrate_heun


@torch.no_grad()
def sample_stgf_calibrated(
    model: STGFFlow,
    condition: torch.Tensor,
    *,
    members: int,
    steps: int,
    member_chunk: int,
    seed: int,
    residual_temperature: float,
    mean_anchor: float,
) -> torch.Tensor:
    if not 0.0 < residual_temperature <= 2.0:
        raise ValueError("residual_temperature must lie in (0, 2]")
    if not 0.0 <= mean_anchor <= 1.0:
        raise ValueError("mean_anchor must lie in [0, 1]")
    if members < 2 or member_chunk < 1:
        raise ValueError("invalid ensemble shape")
    model.eval()
    batch = len(condition)
    center = model.center(condition)
    spectral_condition = model.spectral_condition(condition)
    generator = torch.Generator(device=condition.device).manual_seed(int(seed))
    latent_chunks: list[torch.Tensor] = []
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
        ).reshape(batch, current, 10, 24)
        latent_chunks.append(residual)
        completed += current
    residual = torch.cat(latent_chunks, dim=1)
    residual = residual - mean_anchor * residual.mean(dim=1, keepdim=True)
    residual = residual * residual_temperature
    physical_residual = decode_field(
        residual.reshape(batch * members, 10, 24),
        model.graph_basis,
        model.temporal_basis,
    ).reshape(batch, members, 10, 24)
    generated = model.physical_from_standardized(
        center[:, None] + physical_residual
    )
    return generated.clamp(0.0, 1.0)


__all__ = ["sample_stgf_calibrated"]
