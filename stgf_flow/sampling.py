from __future__ import annotations

import torch

from .graph import decode_field
from .model import STGFFlow


@torch.no_grad()
def integrate_heun(
    model: STGFFlow,
    initial: torch.Tensor,
    spectral_condition: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("flow steps must be positive")
    state = initial
    delta = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (len(state),),
            index / steps,
            dtype=state.dtype,
            device=state.device,
        )
        velocity = model.flow(
            state,
            time,
            spectral_condition,
            model.graph_eigenvalues,
            model.temporal_frequencies,
        )
        proposal = state + delta * velocity
        if index + 1 == steps:
            state = proposal
        else:
            next_time = torch.full(
                (len(state),),
                (index + 1) / steps,
                dtype=state.dtype,
                device=state.device,
            )
            next_velocity = model.flow(
                proposal,
                next_time,
                spectral_condition,
                model.graph_eigenvalues,
                model.temporal_frequencies,
            )
            state = state + 0.5 * delta * (velocity + next_velocity)
    return state


@torch.no_grad()
def sample_stgf(
    model: STGFFlow,
    condition: torch.Tensor,
    *,
    members: int,
    steps: int,
    member_chunk: int,
    seed: int,
) -> torch.Tensor:
    if members < 2:
        raise ValueError("at least two ensemble members are required")
    if member_chunk < 1:
        raise ValueError("member_chunk must be positive")
    model.eval()
    batch = len(condition)
    center = model.center(condition)
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
        repeated_center = center[:, None].expand(
            -1, current, -1, -1
        ).reshape(batch * current, 10, 24)
        generated = model.physical_from_standardized(
            repeated_center + physical_residual
        ).reshape(batch, current, 10, 24)
        chunks.append(generated)
        completed += current
    return torch.cat(chunks, dim=1).clamp(0.0, 1.0)


__all__ = ["integrate_heun", "sample_stgf"]
