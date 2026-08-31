from __future__ import annotations

import math
from typing import Literal

import torch

from .data import INTERIOR_STATE, MASK_STATE, ONE_STATE, ZERO_STATE
from .model import MMJDWind, MixedMeasureOutput


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def sample_independent_states(
    probabilities: torch.Tensor, *, members: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent states from analytic mixed-measure probabilities."""

    if probabilities.shape[-1] != 3:
        raise ValueError("state probabilities must end in three classes")
    batch, zones, hours, _ = probabilities.shape
    expanded = probabilities[:, None].expand(-1, members, -1, -1, -1)
    flat = expanded.reshape(-1, 3)
    generator = _generator(probabilities.device, seed)
    sampled = torch.multinomial(flat, 1, replacement=True, generator=generator)
    states = sampled.reshape(batch, members, zones, hours)
    return states, expanded


@torch.no_grad()
def sample_correlated_states(
    model: MMJDWind,
    condition: torch.Tensor,
    base_probabilities: torch.Tensor,
    *,
    members: int,
    steps: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively reveal a correlated state field from the absorbing mask."""

    if steps <= 0:
        raise ValueError("jump sampling steps must be positive")
    batch, zones, hours, features = condition.shape
    total = zones * hours
    repeated_condition = condition[:, None].expand(
        -1, members, -1, -1, -1
    ).reshape(batch * members, zones, hours, features)
    repeated_base = base_probabilities[:, None].expand(
        -1, members, -1, -1, -1
    ).reshape(batch * members, zones, hours, 3)
    state = torch.full(
        (batch * members, zones, hours),
        MASK_STATE,
        dtype=torch.long,
        device=condition.device,
    )
    final_probabilities = repeated_base
    generator = _generator(condition.device, seed)
    for index in range(steps):
        time = torch.full(
            (batch * members,),
            index / max(steps, 1),
            device=condition.device,
        )
        logits = model.jump(repeated_condition, state, time, repeated_base)
        probabilities = torch.softmax(logits, dim=-1)
        candidates = torch.multinomial(
            probabilities.reshape(-1, 3),
            1,
            replacement=True,
            generator=generator,
        ).reshape(batch * members, zones, hours)
        confidence = probabilities.gather(-1, candidates[..., None])[..., 0]
        masked = state == MASK_STATE
        desired = total if index == steps - 1 else math.ceil(
            total * (index + 1) / steps
        )
        current = (~masked).flatten(1).sum(1)
        for row in range(len(state)):
            count = min(int(desired - int(current[row])), int(masked[row].sum()))
            if count <= 0:
                continue
            row_scores = confidence[row].masked_fill(~masked[row], -1.0).flatten()
            selected = torch.topk(row_scores, k=count, sorted=False).indices
            flat_state = state[row].flatten()
            flat_candidate = candidates[row].flatten()
            flat_state[selected] = flat_candidate[selected]
        final_probabilities = probabilities
    if bool((state == MASK_STATE).any()):
        raise RuntimeError("correlated state sampler left masked cells")
    return (
        state.reshape(batch, members, zones, hours),
        final_probabilities.reshape(batch, members, zones, hours, 3),
    )


@torch.no_grad()
def mass_preserving_states(
    sampled: torch.Tensor,
    member_probabilities: torch.Tensor,
    analytic_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Match finite-M atom counts while preserving member-wise probability ranks."""

    if sampled.ndim != 4 or member_probabilities.shape != (*sampled.shape, 3):
        raise ValueError("sample/member probability shapes do not align")
    batch, members, zones, hours = sampled.shape
    if analytic_probabilities.shape != (batch, zones, hours, 3):
        raise ValueError("analytic probabilities do not align")
    result = torch.full_like(sampled, INTERIOR_STATE)
    zero_count = torch.round(members * analytic_probabilities[..., ZERO_STATE]).long()
    one_count = torch.round(members * analytic_probabilities[..., ONE_STATE]).long()
    excess = (zero_count + one_count - members).clamp_min(0)
    zero_count = (zero_count - excess).clamp_min(0)
    for day in range(batch):
        for zone in range(zones):
            for hour in range(hours):
                n_one = int(one_count[day, zone, hour])
                n_zero = int(zero_count[day, zone, hour])
                available = torch.ones(members, dtype=torch.bool, device=sampled.device)
                if n_one:
                    one_scores = member_probabilities[day, :, zone, hour, ONE_STATE]
                    selected = torch.topk(one_scores, n_one, sorted=False).indices
                    result[day, selected, zone, hour] = ONE_STATE
                    available[selected] = False
                if n_zero:
                    zero_scores = member_probabilities[
                        day, :, zone, hour, ZERO_STATE
                    ].masked_fill(~available, -1.0)
                    selected = torch.topk(zero_scores, n_zero, sorted=False).indices
                    result[day, selected, zone, hour] = ZERO_STATE
    return result


def integrate_rectified_flow(
    model: MMJDWind,
    noise: torch.Tensor,
    condition: torch.Tensor,
    state: torch.Tensor,
    *,
    steps: int,
    method: Literal["euler", "heun"] = "heun",
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("flow sampling steps must be positive")
    value = noise
    delta = 1.0 / steps
    for index in range(steps):
        t0 = torch.full(
            (len(value),), index / steps, device=value.device, dtype=value.dtype
        )
        velocity = model.flow(value, t0, condition, state)
        if method == "euler" or index == steps - 1:
            value = value + delta * velocity
            continue
        proposal = value + delta * velocity
        t1 = torch.full(
            (len(value),),
            (index + 1) / steps,
            device=value.device,
            dtype=value.dtype,
        )
        corrected = model.flow(proposal, t1, condition, state)
        value = value + 0.5 * delta * (velocity + corrected)
    return value


def sample_joint(
    model: MMJDWind,
    condition: torch.Tensor,
    *,
    members: int,
    jump_steps: int,
    flow_steps: int,
    seed: int,
    state_mode: Literal["none", "independent", "correlated", "mass_preserving"],
    flow_method: Literal["euler", "heun"] = "heun",
    member_chunk: int = 25,
) -> tuple[torch.Tensor, MixedMeasureOutput, torch.Tensor]:
    """Generate [day,member,zone,hour] scenarios and analytic atom probabilities."""

    model.eval()
    statistics = model.statistics(condition)
    base = statistics.state_probabilities
    if state_mode == "none":
        states = torch.full(
            (len(condition), members, 10, 24),
            INTERIOR_STATE,
            device=condition.device,
            dtype=torch.long,
        )
    elif state_mode == "independent":
        states, _ = sample_independent_states(base, members=members, seed=seed + 1)
    else:
        states, member_probabilities = sample_correlated_states(
            model,
            condition,
            base,
            members=members,
            steps=jump_steps,
            seed=seed + 1,
        )
        if state_mode == "mass_preserving":
            states = mass_preserving_states(states, member_probabilities, base)
        elif state_mode != "correlated":
            raise ValueError(f"unknown state mode: {state_mode}")
    chunks: list[torch.Tensor] = []
    generator = _generator(condition.device, seed + 2)
    for start in range(0, members, member_chunk):
        stop = min(start + member_chunk, members)
        width = stop - start
        repeated_condition = condition[:, None].expand(
            -1, width, -1, -1, -1
        ).reshape(-1, 10, 24, condition.shape[-1])
        repeated_state = states[:, start:stop].reshape(-1, 10, 24)
        noise = torch.randn(
            repeated_state.shape,
            generator=generator,
            device=condition.device,
            dtype=condition.dtype,
        )
        residual = integrate_rectified_flow(
            model,
            noise,
            repeated_condition,
            repeated_state,
            steps=flow_steps,
            method=flow_method,
        )
        repeated_statistics = MixedMeasureOutput(
            location=statistics.location[:, None]
            .expand(-1, width, -1, -1)
            .reshape(-1, 10, 24),
            scale=statistics.scale[:, None]
            .expand(-1, width, -1, -1)
            .reshape(-1, 10, 24),
            zero_logit=statistics.zero_logit[:, None]
            .expand(-1, width, -1, -1)
            .reshape(-1, 10, 24),
            upper_conditional_logit=statistics.upper_conditional_logit[:, None]
            .expand(-1, width, -1, -1)
            .reshape(-1, 10, 24),
        )
        reconstructed = model.reconstruct(
            residual, repeated_state, repeated_statistics
        ).reshape(len(condition), width, 10, 24)
        chunks.append(reconstructed)
    return torch.cat(chunks, dim=1), statistics, states


__all__ = [
    "integrate_rectified_flow",
    "mass_preserving_states",
    "sample_correlated_states",
    "sample_independent_states",
    "sample_joint",
]
