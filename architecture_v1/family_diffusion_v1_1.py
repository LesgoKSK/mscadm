"""Family-v1.1 masked joint diffusion with v-prediction.

This module is intentionally separate from :mod:`family_diffusion`: family-v1
is immutable after its epsilon-prediction sampler-contract No-Go.  The only
algorithmic revision here is the prediction target and its exact DDIM algebra.
The schedule, raw-logit latent, model topology, atom allocation and EMA trainer
remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from .atom import AtomAllocation
from .family_diffusion import (
    DiffusionLossSample,
    FamilyEMATrainer,
    MaskedJointDDPM,
    _clean_interior_logit,
    _extract_schedule,
    _validate_joint_training_inputs,
    ddim_timestep_grid,
)
from .model import JointRectifiedFlowBase, ScenarioBatch


class VPredictionJointDDPM(MaskedJointDDPM):
    """Masked cosine diffusion whose shared network predicts ``v``.

    With ``a = alpha_bar[t]`` the registered parameterization is

    ``v = sqrt(a) * epsilon - sqrt(1-a) * x0``.

    Sampling uses the orthogonal inverse, avoiding division by
    ``sqrt(alpha_bar)`` at the noisy endpoint.
    """

    prediction_parameterization = "v"

    def loss(
        self,
        model: JointRectifiedFlowBase,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        states: torch.Tensor,
        observed_mask: torch.Tensor,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_sample: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], DiffusionLossSample]:
        active = _validate_joint_training_inputs(
            model, condition, observation, states, observed_mask
        )
        clean = _clean_interior_logit(model, observation, active)
        if timestep is None:
            timestep = torch.randint(
                self.timesteps,
                (len(clean),),
                dtype=torch.long,
                device=clean.device,
                generator=generator,
            )
        noisy, masked_noise = self.q_sample(
            clean, timestep, active, noise=noise, generator=generator
        )
        sqrt_alpha = _extract_schedule(self.sqrt_alpha_bar, timestep, clean)
        sqrt_one_minus = _extract_schedule(
            self.sqrt_one_minus_alpha_bar, timestep, clean
        )
        target_v = sqrt_alpha * masked_noise - sqrt_one_minus * clean
        target_v = torch.where(active, target_v, torch.zeros_like(target_v))

        encoded = model.encode_condition(condition)
        context = model.condition_context(encoded)
        normalized_time = timestep.to(clean.dtype) / float(self.timesteps - 1)
        predicted_v = model.flow(noisy, normalized_time, context, active)
        squared = (predicted_v - target_v).square()
        active_float = active.to(squared.dtype)
        v_mse = (squared * active_float).sum() / active_float.sum()
        inactive_max = (
            predicted_v[~active].abs().max()
            if bool((~active).any())
            else predicted_v.sum() * 0.0
        )
        losses = {
            "loss": v_mse,
            "v_mse": v_mse,
            "active_fraction": active_float.mean(),
            "inactive_v_max": inactive_max,
        }
        if not return_sample:
            return losses
        return losses, DiffusionLossSample(
            timestep=timestep,
            noise=masked_noise,
            noisy_latent=noisy,
            clean_latent=clean,
            active_mask=active,
        )

    @torch.no_grad()
    def sample_ddim(
        self,
        model: JointRectifiedFlowBase,
        condition: torch.Tensor,
        *,
        members: int,
        steps: int = 31,
        eta: float = 0.0,
        seed: int,
        member_chunk: int = 25,
        allocation: AtomAllocation | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> ScenarioBatch:
        if eta != 0.0:
            raise ValueError("family-v1.1 registers deterministic DDIM with eta=0 only")
        if members < 1 or member_chunk < 1:
            raise ValueError("members and member_chunk must be positive")
        grid = ddim_timestep_grid(self.timesteps, steps)
        was_training = model.training
        model.eval()
        try:
            if condition.dtype != torch.float32:
                raise TypeError("family-v1.1 sampling requires FP32 condition")
            encoded = model.encode_condition(condition)
            statistics = model.atom(encoded)
            if allocation is None:
                allocation = model.atom.allocate(
                    statistics, members=members, seed=int(seed) + 1
                )
            expected = (len(condition), members, model.zones, model.hours)
            if allocation.states.shape != expected:
                raise ValueError("provided atom allocation has the wrong shape")
            if allocation.states.device != condition.device:
                raise ValueError("provided atom allocation is on the wrong device")
            if not torch.equal(
                allocation.analytic_probabilities, statistics.probabilities
            ):
                raise ValueError("provided allocation does not match current E/A law")
            states = allocation.states
            active = allocation.active_mask
            if initial_noise is None:
                generator = torch.Generator(device=condition.device)
                generator.manual_seed(int(seed) + 2)
                initial_noise = torch.randn(
                    expected,
                    dtype=condition.dtype,
                    device=condition.device,
                    generator=generator,
                )
            if initial_noise.shape != expected or initial_noise.dtype != condition.dtype:
                raise ValueError("initial noise must match scenario shape and dtype")
            if initial_noise.device != condition.device:
                raise ValueError("initial noise is on the wrong device")
            if not bool(torch.isfinite(initial_noise).all()):
                raise ValueError("initial noise contains non-finite values")

            context = model.condition_context(encoded)
            schedule = self.alpha_bar.to(
                device=condition.device, dtype=condition.dtype
            )
            reverse_grid = grid.flip(0).tolist()
            chunks: list[torch.Tensor] = []
            batched_forward_calls = 0
            for start in range(0, members, member_chunk):
                stop = min(start + member_chunk, members)
                width = stop - start
                chunk_active = active[:, start:stop].reshape(
                    -1, model.zones, model.hours
                )
                latent = torch.where(
                    active[:, start:stop],
                    initial_noise[:, start:stop],
                    torch.zeros_like(initial_noise[:, start:stop]),
                ).reshape(-1, model.zones, model.hours)
                chunk_context = context[:, None].expand(
                    -1, width, -1, -1, -1
                ).reshape(-1, model.zones, model.hours, model.encoder_dim)
                for index, current_timestep in enumerate(reverse_grid):
                    normalized_time = torch.full(
                        (len(latent),),
                        current_timestep / float(self.timesteps - 1),
                        dtype=latent.dtype,
                        device=latent.device,
                    )
                    predicted_v = model.flow(
                        latent, normalized_time, chunk_context, chunk_active
                    )
                    batched_forward_calls += 1
                    alpha_current = schedule[current_timestep]
                    sqrt_alpha = torch.sqrt(alpha_current)
                    sqrt_one_minus = torch.sqrt(1.0 - alpha_current)
                    predicted_clean = sqrt_alpha * latent - sqrt_one_minus * predicted_v
                    predicted_epsilon = sqrt_one_minus * latent + sqrt_alpha * predicted_v
                    if index + 1 < len(reverse_grid):
                        alpha_previous = schedule[reverse_grid[index + 1]]
                        latent = (
                            torch.sqrt(alpha_previous) * predicted_clean
                            + torch.sqrt(1.0 - alpha_previous) * predicted_epsilon
                        )
                    else:
                        latent = predicted_clean
                    latent = torch.where(
                        chunk_active, latent, torch.zeros_like(latent)
                    )
                    if not bool(torch.isfinite(latent).all()):
                        raise FloatingPointError(
                            f"non-finite v-DDIM latent after timestep {current_timestep}"
                        )
                    if bool((latent[~chunk_active] != 0.0).any()):
                        raise RuntimeError("inactive v-DDIM coordinates moved from zero")
                chunks.append(
                    latent.reshape(len(condition), width, model.zones, model.hours)
                )
            latent = torch.cat(chunks, dim=1)
            expected_calls = steps * ((members + member_chunk - 1) // member_chunk)
            if batched_forward_calls != expected_calls:
                raise RuntimeError("v-DDIM network-call accounting invariant failed")
            values = model.atom.reconstruct(latent, states)
            if bool(((values[active] <= 0.0) | (values[active] >= 1.0)).any()):
                raise FloatingPointError(
                    "family-v1.1 sampler contract failed: interior decoded to boundary"
                )
            return ScenarioBatch(
                values=values,
                states=states,
                active_mask=active,
                interior_latent=latent,
                atom_statistics=statistics,
                atom_allocation=allocation,
                per_path_nfe=steps,
                batched_forward_calls=batched_forward_calls,
            )
        finally:
            model.train(was_training)


__all__ = ["FamilyEMATrainer", "VPredictionJointDDPM"]
