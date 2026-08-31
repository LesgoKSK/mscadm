"""Family-v1 masked joint DDPM and common EMA training primitives.

This module implements the frozen Flow-vs-Joint-DDPM comparison contract.  D0
does not introduce a second denoiser architecture: the exact same
``JointTimeDomainVelocity`` tensor graph used by F0 predicts epsilon for D0.
The diffusion schedule is therefore a parameter-free wrapper around an
architecture-v1 model.

The continuous process exists only on observed/sampled interior coordinates.
Exact zero/one atoms and missing training cells are set to latent zero before
and after every network/sampler operation.  Decoding is delegated to the
shared exact-atom module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .atom import AtomAllocation
from .model import JointRectifiedFlowBase, ScenarioBatch
from .training import (
    ArchitectureBatch,
    canonical_sha256,
    configure_stage,
    file_sha256,
    parameter_manifest,
    shared_ea_state_sha256,
    tensor_state_sha256,
)


Family = Literal["F0", "D0"]
FAMILY_CHECKPOINT_SCHEMA = "architecture_v1_family_ema_checkpoint_v1"
EMA_SCHEMA = "architecture_v1_common_ema_v1"


def cosine_betas(
    timesteps: int = 250,
    *,
    offset: float = 0.008,
    beta_min: float = 1e-8,
    beta_max: float = 0.999,
) -> torch.Tensor:
    """Return the registered improved-DDPM cosine schedule in FP32."""

    if timesteps < 2:
        raise ValueError("diffusion timesteps must be at least two")
    if not 0.0 <= offset < 1.0:
        raise ValueError("cosine offset must lie in [0,1)")
    if not 0.0 < beta_min < beta_max < 1.0:
        raise ValueError("beta bounds must satisfy 0 < min < max < 1")
    positions = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    phase = ((positions / timesteps + offset) / (1.0 + offset)) * (
        torch.pi / 2.0
    )
    alpha_bar = torch.cos(phase).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(beta_min, beta_max).to(torch.float32)


def ddim_timestep_grid(timesteps: int, steps: int) -> torch.Tensor:
    """Unique rounded linspace required by family-v1, in ascending order."""

    if timesteps < 2:
        raise ValueError("diffusion timesteps must be at least two")
    if not 2 <= steps <= timesteps:
        raise ValueError("DDIM steps must lie in [2, diffusion_timesteps]")
    grid = torch.linspace(0, timesteps - 1, steps, dtype=torch.float64).round().long()
    if len(torch.unique_consecutive(grid)) != steps:
        raise RuntimeError("rounded DDIM grid contains duplicate timesteps")
    if int(grid[0]) != 0 or int(grid[-1]) != timesteps - 1:
        raise RuntimeError("DDIM grid must include both schedule endpoints")
    return grid


def _validate_joint_training_inputs(
    model: JointRectifiedFlowBase,
    condition: torch.Tensor,
    observation: torch.Tensor,
    states: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    if condition.dtype != torch.float32 or observation.dtype != torch.float32:
        raise TypeError("family-v1 training requires FP32 condition and observation")
    expected = (len(condition), model.zones, model.hours)
    if condition.ndim != 4 or condition.shape[1:] != (
        model.zones,
        model.hours,
        model.condition_dim,
    ):
        raise ValueError("condition has the wrong joint-day shape")
    if observation.shape != expected or states.shape != expected:
        raise ValueError("observation/states do not align with condition")
    if observed_mask.shape != expected or observed_mask.dtype != torch.bool:
        raise ValueError("observed_mask must be boolean and align with observation")
    if states.dtype != torch.long:
        raise TypeError("states must use torch.long codes")
    for name, value in (
        ("observation", observation),
        ("condition", condition),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if bool(((observation < 0.0) | (observation > 1.0)).any()):
        raise ValueError("observation must lie in [0,1]")
    active = model.atom.active_mask(states, observed_mask)
    if not bool(active.any()):
        raise ValueError("diffusion loss requires an observed interior target")
    return active


def _clean_interior_logit(
    model: JointRectifiedFlowBase,
    observation: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    safe = torch.where(active, observation, torch.full_like(observation, 0.5))
    clean = torch.logit(safe.clamp(model.logit_epsilon, 1.0 - model.logit_epsilon))
    return torch.where(active, clean, torch.zeros_like(clean))


def _extract_schedule(
    values: torch.Tensor,
    timestep: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    selected = values.to(device=reference.device, dtype=reference.dtype).index_select(
        0, timestep
    )
    return selected.reshape(len(timestep), *([1] * (reference.ndim - 1)))


@dataclass(frozen=True)
class DiffusionLossSample:
    """Auditable random variables used by one masked DDPM loss evaluation."""

    timestep: torch.Tensor
    noise: torch.Tensor
    noisy_latent: torch.Tensor
    clean_latent: torch.Tensor
    active_mask: torch.Tensor


class MaskedJointDDPM(nn.Module):
    """Parameter-free masked Gaussian diffusion over a complete joint day."""

    def __init__(
        self,
        *,
        timesteps: int = 250,
        cosine_offset: float = 0.008,
        beta_min: float = 1e-8,
        beta_max: float = 0.999,
    ) -> None:
        super().__init__()
        betas = cosine_betas(
            timesteps,
            offset=cosine_offset,
            beta_min=beta_min,
            beta_max=beta_max,
        )
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.cosine_offset = float(cosine_offset)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def extra_repr(self) -> str:
        return (
            f"timesteps={self.timesteps}, cosine_offset={self.cosine_offset:g}, "
            f"beta_clip=({self.beta_min:g},{self.beta_max:g})"
        )

    def q_sample(
        self,
        clean_latent: torch.Tensor,
        timestep: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``q(x_t | x_0)`` while keeping inactive coordinates exactly zero."""

        if clean_latent.ndim != 3:
            raise ValueError("clean latent must have shape [B,Z,T]")
        if active_mask.shape != clean_latent.shape or active_mask.dtype != torch.bool:
            raise ValueError("active mask must be boolean and align with clean latent")
        if timestep.shape != (len(clean_latent),) or timestep.dtype != torch.long:
            raise ValueError("timestep must be torch.long with shape [B]")
        if timestep.device != clean_latent.device:
            raise ValueError("timestep and clean latent must share a device")
        if bool(((timestep < 0) | (timestep >= self.timesteps)).any()):
            raise ValueError("timestep lies outside the diffusion schedule")
        if not bool(torch.isfinite(clean_latent).all()):
            raise ValueError("clean latent contains non-finite values")
        clean = torch.where(active_mask, clean_latent, torch.zeros_like(clean_latent))
        if noise is None:
            noise = torch.randn(
                clean.shape,
                dtype=clean.dtype,
                device=clean.device,
                generator=generator,
            )
        elif noise.shape != clean.shape or noise.dtype != clean.dtype:
            raise ValueError("noise must match clean latent shape and dtype")
        if noise.device != clean.device:
            raise ValueError("noise and clean latent must share a device")
        if not bool(torch.isfinite(noise).all()):
            raise ValueError("noise contains non-finite values")
        masked_noise = torch.where(active_mask, noise, torch.zeros_like(noise))
        noisy = (
            _extract_schedule(self.sqrt_alpha_bar, timestep, clean) * clean
            + _extract_schedule(self.sqrt_one_minus_alpha_bar, timestep, clean)
            * masked_noise
        )
        noisy = torch.where(active_mask, noisy, torch.zeros_like(noisy))
        return noisy, masked_noise

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
        """Compute one-network-call epsilon MSE on observed interior cells only."""

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
        encoded = model.encode_condition(condition)
        context = model.condition_context(encoded)
        normalized_time = timestep.to(clean.dtype) / float(self.timesteps - 1)
        predicted_noise = model.flow(noisy, normalized_time, context, active)
        squared = (predicted_noise - masked_noise).square()
        active_float = active.to(squared.dtype)
        epsilon_mse = (squared * active_float).sum() / active_float.sum()
        inactive_max = (
            predicted_noise[~active].abs().max()
            if bool((~active).any())
            else predicted_noise.sum() * 0.0
        )
        losses = {
            "loss": epsilon_mse,
            "epsilon_mse": epsilon_mse,
            "active_fraction": active_float.mean(),
            "inactive_epsilon_max": inactive_max,
        }
        if not return_sample:
            return losses
        sample = DiffusionLossSample(
            timestep=timestep,
            noise=masked_noise,
            noisy_latent=noisy,
            clean_latent=clean,
            active_mask=active,
        )
        return losses, sample

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
        """Generate joint scenarios using deterministic DDIM (eta exactly zero)."""

        if eta != 0.0:
            raise ValueError("family-v1 registers deterministic DDIM with eta=0 only")
        if members < 1 or member_chunk < 1:
            raise ValueError("members and member_chunk must be positive")
        grid = ddim_timestep_grid(self.timesteps, steps)
        was_training = model.training
        model.eval()
        try:
            if condition.dtype != torch.float32:
                raise TypeError("family-v1 sampling requires FP32 condition")
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
                raise ValueError("provided allocation does not match the current E/A law")
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
                    epsilon = model.flow(
                        latent, normalized_time, chunk_context, chunk_active
                    )
                    batched_forward_calls += 1
                    alpha_current = schedule[current_timestep]
                    predicted_clean = (
                        latent - torch.sqrt(1.0 - alpha_current) * epsilon
                    ) / torch.sqrt(alpha_current)
                    if index + 1 < len(reverse_grid):
                        previous_timestep = reverse_grid[index + 1]
                        alpha_previous = schedule[previous_timestep]
                        latent = (
                            torch.sqrt(alpha_previous) * predicted_clean
                            + torch.sqrt(1.0 - alpha_previous) * epsilon
                        )
                    else:
                        # alpha_bar immediately before t=0 is one.
                        latent = predicted_clean
                    latent = torch.where(
                        chunk_active, latent, torch.zeros_like(latent)
                    )
                    if not bool(torch.isfinite(latent).all()):
                        raise FloatingPointError(
                            f"non-finite DDIM latent after timestep {current_timestep}"
                        )
                    if bool((latent[~chunk_active] != 0.0).any()):
                        raise RuntimeError("inactive DDIM coordinates moved away from zero")
                chunks.append(
                    latent.reshape(len(condition), width, model.zones, model.hours)
                )
            latent = torch.cat(chunks, dim=1)
            expected_calls = steps * ((members + member_chunk - 1) // member_chunk)
            if batched_forward_calls != expected_calls:
                raise RuntimeError("DDIM network-call accounting invariant failed")
            values = model.atom.reconstruct(latent, states)
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


def _transport_parameter_names(model: JointRectifiedFlowBase) -> tuple[str, ...]:
    names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("encoder.")
        and not name.startswith("atom.")
    )
    if not names:
        raise RuntimeError("no flow-stage transport parameters were selected")
    return names


class CommonEMA:
    """EMA over all flow-stage parameters, including matched dormant controls."""

    def __init__(
        self,
        model: JointRectifiedFlowBase,
        *,
        decay: float = 0.999,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        names = tuple(parameter_names or _transport_parameter_names(model))
        if len(set(names)) != len(names):
            raise ValueError("EMA parameter names must be unique")
        parameters = dict(model.named_parameters())
        missing = sorted(set(names) - set(parameters))
        if missing:
            raise ValueError(f"EMA parameter names are absent from model: {missing[:8]}")
        if any(name.startswith(("encoder.", "atom.")) for name in names):
            raise ValueError("shared E/A parameters must not be EMA tracked")
        self.parameter_names = names
        self.shadow = {
            name: parameters[name].detach().clone() for name in self.parameter_names
        }
        self.num_updates = 0
        self._validate(model)

    def _validate(self, model: JointRectifiedFlowBase) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters).intersection(self.parameter_names) != set(self.parameter_names):
            raise ValueError("EMA/model parameter topology mismatch")
        for name in self.parameter_names:
            parameter = parameters[name]
            shadow = self.shadow[name]
            if shadow.shape != parameter.shape or shadow.dtype != parameter.dtype:
                raise ValueError(f"EMA tensor metadata mismatch for {name}")
            if shadow.device != parameter.device:
                raise ValueError(f"EMA tensor device mismatch for {name}")
            if not bool(torch.isfinite(shadow).all()):
                raise FloatingPointError(f"EMA tensor is non-finite: {name}")

    @torch.no_grad()
    def update(self, model: JointRectifiedFlowBase) -> None:
        self._validate(model)
        parameters = dict(model.named_parameters())
        one_minus = 1.0 - self.decay
        for name in self.parameter_names:
            parameter = parameters[name]
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"online parameter is non-finite: {name}")
            self.shadow[name].mul_(self.decay).add_(parameter, alpha=one_minus)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: JointRectifiedFlowBase) -> None:
        self._validate(model)
        parameters = dict(model.named_parameters())
        for name in self.parameter_names:
            parameters[name].copy_(self.shadow[name])

    @contextmanager
    def average_parameters(
        self, model: JointRectifiedFlowBase
    ) -> Iterator[JointRectifiedFlowBase]:
        """Temporarily expose EMA weights and restore online weights exactly."""

        self._validate(model)
        parameters = dict(model.named_parameters())
        backup = {
            name: parameters[name].detach().clone() for name in self.parameter_names
        }
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name in self.parameter_names:
                    parameters[name].copy_(backup[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": EMA_SCHEMA,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "parameter_names": list(self.parameter_names),
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
            "tensor_sha256": tensor_state_sha256(self.shadow),
        }

    def load_state_dict(
        self, model: JointRectifiedFlowBase, state: Mapping[str, Any]
    ) -> None:
        if state.get("schema") != EMA_SCHEMA:
            raise ValueError("EMA checkpoint schema mismatch")
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay differs from the frozen trainer configuration")
        names = tuple(str(name) for name in state["parameter_names"])
        if names != self.parameter_names:
            raise ValueError("EMA parameter-name topology mismatch")
        raw_shadow = state["shadow"]
        if not isinstance(raw_shadow, Mapping) or set(raw_shadow) != set(names):
            raise ValueError("EMA shadow state is incomplete")
        expected_hash = str(state["tensor_sha256"])
        if tensor_state_sha256(raw_shadow) != expected_hash:
            raise ValueError("EMA shadow tensor hash mismatch")
        parameters = dict(model.named_parameters())
        loaded: dict[str, torch.Tensor] = {}
        for name in names:
            value = raw_shadow[name]
            if not torch.is_tensor(value) or value.shape != parameters[name].shape:
                raise ValueError(f"EMA checkpoint tensor mismatch for {name}")
            if value.dtype != parameters[name].dtype:
                raise ValueError(f"EMA checkpoint dtype mismatch for {name}")
            loaded[name] = value.detach().to(parameters[name].device).clone()
        self.shadow = loaded
        self.num_updates = int(state["num_updates"])
        if self.num_updates < 0:
            raise ValueError("EMA update count cannot be negative")
        self._validate(model)

    def tensor_sha256(self) -> str:
        return tensor_state_sha256(self.shadow)


def _assert_scalar_losses(losses: Mapping[str, torch.Tensor], *, where: str) -> None:
    if "loss" not in losses:
        raise KeyError(f"{where} did not return loss")
    for name, value in losses.items():
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"{where}.{name} must be a scalar tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{where}.{name} must be FP32")
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"{where}.{name} is non-finite")


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


class FamilyEMATrainer:
    """One common finite-checked AdamW+EMA trainer for both F0 and D0."""

    def __init__(
        self,
        model: JointRectifiedFlowBase,
        *,
        family: Family,
        diffusion: MaskedJointDDPM | None = None,
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gradient_clip: float = 5.0,
        ema_decay: float = 0.999,
    ) -> None:
        if family not in ("F0", "D0"):
            raise ValueError("family must be F0 or D0")
        if learning_rate <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid AdamW hyperparameters")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if family == "D0" and diffusion is None:
            diffusion = MaskedJointDDPM()
        self.model = model
        self.family: Family = family
        self.diffusion = diffusion
        self.gradient_clip = float(gradient_clip)
        parameters = configure_stage(model, "flow")
        self.parameter_names = _transport_parameter_names(model)
        if {id(parameter) for parameter in parameters} != {
            id(dict(model.named_parameters())[name]) for name in self.parameter_names
        }:
            raise RuntimeError("optimizer/EMA flow-stage parameter sets differ")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        self.ema = CommonEMA(
            model, decay=ema_decay, parameter_names=self.parameter_names
        )
        self.optimizer_updates = 0

    def _loss(
        self,
        batch: ArchitectureBatch,
        *,
        generator: torch.Generator | None,
    ) -> Mapping[str, torch.Tensor]:
        if self.family == "F0":
            return self.model.flow_loss(
                batch.condition,
                batch.target,
                states=batch.state,
                observed_mask=batch.observed_mask,
                generator=generator,
            )
        assert self.diffusion is not None
        result = self.diffusion.loss(
            self.model,
            batch.condition,
            batch.target,
            states=batch.state,
            observed_mask=batch.observed_mask,
            generator=generator,
        )
        assert isinstance(result, dict)
        return result

    def train_step(
        self,
        batch: ArchitectureBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """Run one online update followed immediately by one common EMA update."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses = self._loss(batch, generator=generator)
        _assert_scalar_losses(losses, where=f"{self.family}_loss")
        losses["loss"].backward()
        named_parameters = dict(self.model.named_parameters())
        gradients = {
            name: named_parameters[name].grad
            for name in self.parameter_names
            if named_parameters[name].grad is not None
        }
        if not gradients:
            raise RuntimeError("family loss produced no transport gradients")
        for name, gradient in gradients.items():
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"non-finite gradient: {name}")
        parameters = [named_parameters[name] for name in self.parameter_names]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite pre-clip gradient norm")
        self.optimizer.step()
        for name in self.parameter_names:
            if not bool(torch.isfinite(named_parameters[name]).all()):
                raise FloatingPointError(f"non-finite updated parameter: {name}")
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("non-finite AdamW state")
        self.ema.update(self.model)
        self.optimizer_updates += 1
        if self.ema.num_updates != self.optimizer_updates:
            raise RuntimeError("EMA and optimizer update counters diverged")
        return {
            **{name: float(value.detach().cpu()) for name, value in losses.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "gradient_was_clipped": float(gradient_norm > self.gradient_clip),
            "optimizer_updates": float(self.optimizer_updates),
            "wall_seconds": float(time.perf_counter() - started),
        }

    @torch.no_grad()
    def evaluate_loss(
        self,
        batches: Sequence[ArchitectureBatch],
        *,
        seed: int,
    ) -> dict[str, float]:
        """Evaluate a fixed-seed bank using EMA weights only."""

        if not batches:
            raise ValueError("evaluation requires at least one batch")
        device = batches[0].condition.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        was_training = self.model.training
        totals: dict[str, float] = {}
        total_weight = 0
        with self.ema.average_parameters(self.model):
            self.model.eval()
            try:
                for batch in batches:
                    losses = self._loss(batch, generator=generator)
                    _assert_scalar_losses(losses, where=f"EMA_{self.family}_loss")
                    weight = int(
                        (
                            batch.observed_mask
                            & (batch.state == 1)
                        ).sum().item()
                    )
                    if weight < 1:
                        raise ValueError("evaluation batch has no active target")
                    total_weight += weight
                    for name, value in losses.items():
                        totals[name] = totals.get(name, 0.0) + float(value.cpu()) * weight
            finally:
                self.model.train(was_training)
        return {name: value / total_weight for name, value in totals.items()}

    @contextmanager
    def ema_weights(self) -> Iterator[JointRectifiedFlowBase]:
        """Context used for validation, best-checkpoint selection and sampling."""

        with self.ema.average_parameters(self.model) as model:
            yield model

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": (
                [value.cpu() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
        }

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda", [])
        if cuda_state:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])

    def checkpoint_state(
        self, *, identity: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        model_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        identity_dict = dict(identity or {})
        return {
            "schema": FAMILY_CHECKPOINT_SCHEMA,
            "family": self.family,
            "identity": identity_dict,
            "identity_sha256": canonical_sha256(identity_dict),
            "optimizer_updates": self.optimizer_updates,
            "model_state_dict": model_state,
            "model_tensor_sha256": tensor_state_sha256(model_state),
            "shared_EA_state_sha256": shared_ea_state_sha256(self.model),
            "parameter_manifest": parameter_manifest(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "rng_state": self._rng_state(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any] | None = None,
    ) -> str:
        return _atomic_save(self.checkpoint_state(identity=identity), Path(path))

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_identity: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> Mapping[str, Any]:
        checkpoint = Path(path)
        sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
        if not checkpoint.is_file() or not sidecar.is_file():
            raise FileNotFoundError("checkpoint or SHA256 sidecar is missing")
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if len(fields) != 2 or fields[1] != checkpoint.name:
            raise ValueError("checkpoint SHA256 sidecar is malformed")
        if file_sha256(checkpoint) != fields[0]:
            raise ValueError("checkpoint file SHA256 mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != FAMILY_CHECKPOINT_SCHEMA:
            raise ValueError("family checkpoint schema mismatch")
        if payload.get("family") != self.family:
            raise ValueError("checkpoint belongs to a different model family")
        expected = dict(expected_identity or {})
        if payload.get("identity_sha256") != canonical_sha256(expected):
            raise ValueError("checkpoint experiment identity mismatch")
        if payload.get("identity") != expected:
            raise ValueError("checkpoint identity payload mismatch")
        manifest = parameter_manifest(self.model)
        if (
            payload["parameter_manifest"]["structural_sha256"]
            != manifest["structural_sha256"]
        ):
            raise ValueError("checkpoint model topology mismatch")
        model_state = payload["model_state_dict"]
        if tensor_state_sha256(model_state) != payload["model_tensor_sha256"]:
            raise ValueError("checkpoint online tensor hash mismatch")
        self.model.load_state_dict(model_state, strict=True)
        if shared_ea_state_sha256(self.model) != payload["shared_EA_state_sha256"]:
            raise ValueError("restored shared E/A identity mismatch")
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        device = next(self.model.parameters()).device
        _optimizer_to(self.optimizer, device)
        self.ema.load_state_dict(self.model, payload["ema_state_dict"])
        self.optimizer_updates = int(payload["optimizer_updates"])
        if self.optimizer_updates < 0 or self.ema.num_updates != self.optimizer_updates:
            raise ValueError("checkpoint optimizer/EMA counters are inconsistent")
        if restore_rng:
            self._restore_rng_state(payload["rng_state"])
        return payload


__all__ = [
    "CommonEMA",
    "DiffusionLossSample",
    "EMA_SCHEMA",
    "FAMILY_CHECKPOINT_SCHEMA",
    "Family",
    "FamilyEMATrainer",
    "MaskedJointDDPM",
    "cosine_betas",
    "ddim_timestep_grid",
]
