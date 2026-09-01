"""Frozen-backbone Temporal Utility Probe for architecture-v1 family-v1.2.

The probe does not train a new diffusion backbone.  It loads one retained
family-v1.1 D0-v best-EMA model, freezes every backbone tensor, and trains only
a small ordered residual on the condition context.  During sampling that
residual is enabled at one registered block of the 31-step DDIM trajectory.

The module deliberately keeps three semantics separate:

* the atom branch reads the unmodified frozen encoder output;
* the context adapter is evaluated once per day/member-independent condition;
* the frozen velocity network remains inside autograd so gradients can flow
  from the loss through its context input to the adapter, while no backbone
  parameter can receive a gradient or optimizer update.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .atom import AtomAllocation
from .family_diffusion import (
    CommonEMA,
    _clean_interior_logit,
    _extract_schedule,
    _validate_joint_training_inputs,
    ddim_timestep_grid,
)
from .family_diffusion_v1_1 import VPredictionJointDDPM
from .model import JointRectifiedFlowBase, ScenarioBatch
from .training import (
    ArchitectureBatch,
    canonical_sha256,
    file_sha256,
    parameter_manifest,
    shared_ea_state_sha256,
    tensor_state_sha256,
)


PROBE_CHECKPOINT_SCHEMA = "architecture_v1_family_v1_2_adapter_checkpoint_v1"
NWP_REGISTRY_SCHEMA = "architecture_v1_family_v1_2_nwp_dynamicity_v1"


@dataclass(frozen=True)
class DDIMStageBlock:
    """One contiguous block in reverse DDIM execution order."""

    index: int
    label: str
    timesteps: tuple[int, ...]
    logsnr: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("stage-block index must be non-negative")
        if not self.label:
            raise ValueError("stage-block label cannot be empty")
        if not self.timesteps or len(self.timesteps) != len(self.logsnr):
            raise ValueError("stage-block timesteps and log-SNR values must align")
        if len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("stage-block timesteps must be unique")
        if any(left <= right for left, right in zip(self.timesteps, self.timesteps[1:])):
            raise ValueError("stage-block timesteps must follow descending reverse order")
        if not all(np.isfinite(value) for value in self.logsnr):
            raise ValueError("stage-block log-SNR values must be finite")

    def contains(self, timestep: int) -> bool:
        return int(timestep) in self.timesteps

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "timesteps_reverse_order": list(self.timesteps),
            "logSNR": list(self.logsnr),
            "points": len(self.timesteps),
            "logSNR_min": float(min(self.logsnr)),
            "logSNR_max": float(max(self.logsnr)),
        }


def build_ddim_stage_blocks(
    diffusion: VPredictionJointDDPM,
    *,
    steps: int,
    block_sizes: Sequence[int],
    labels: Sequence[str] | None = None,
) -> tuple[DDIMStageBlock, ...]:
    """Build contiguous blocks from the exact registered reverse DDIM grid."""

    sizes = tuple(int(value) for value in block_sizes)
    if not sizes or any(value < 1 for value in sizes) or sum(sizes) != int(steps):
        raise ValueError("positive stage-block sizes must sum to DDIM steps")
    names = tuple(labels or (f"B{index + 1}" for index in range(len(sizes))))
    if len(names) != len(sizes) or len(set(names)) != len(names):
        raise ValueError("stage-block labels must be unique and align with sizes")
    reverse = ddim_timestep_grid(diffusion.timesteps, int(steps)).flip(0)
    logsnr = torch.log(diffusion.alpha_bar) - torch.log1p(-diffusion.alpha_bar)
    output: list[DDIMStageBlock] = []
    start = 0
    for index, (size, label) in enumerate(zip(sizes, names)):
        selected = reverse[start : start + size]
        output.append(
            DDIMStageBlock(
                index=index,
                label=str(label),
                timesteps=tuple(int(value) for value in selected.tolist()),
                logsnr=tuple(float(logsnr[int(value)]) for value in selected.tolist()),
            )
        )
        start += size
    flattened = tuple(value for block in output for value in block.timesteps)
    if flattened != tuple(int(value) for value in reverse.tolist()):
        raise RuntimeError("stage blocks do not exactly partition the reverse DDIM grid")
    return tuple(output)


def validate_stage_registry(
    diffusion: VPredictionJointDDPM,
    *,
    steps: int,
    records: Sequence[Mapping[str, Any]],
    atol: float = 1e-6,
) -> tuple[DDIMStageBlock, ...]:
    """Fail closed if a JSON stage registry differs from the actual schedule."""

    if not records:
        raise ValueError("stage registry is empty")
    sizes = [len(record["timesteps_reverse_order"]) for record in records]
    labels = [str(record["label"]) for record in records]
    actual = build_ddim_stage_blocks(
        diffusion, steps=int(steps), block_sizes=sizes, labels=labels
    )
    for index, (block, record) in enumerate(zip(actual, records)):
        if int(record["index"]) != index or block.index != index:
            raise ValueError("stage registry indices are not contiguous")
        if list(block.timesteps) != [int(value) for value in record["timesteps_reverse_order"]]:
            raise ValueError(f"stage registry timestep drift in block {index}")
        registered = np.asarray(record["logSNR"], dtype=np.float64)
        calculated = np.asarray(block.logsnr, dtype=np.float64)
        if registered.shape != calculated.shape or not np.allclose(
            registered, calculated, rtol=0.0, atol=float(atol)
        ):
            raise ValueError(f"stage registry log-SNR drift in block {index}")
    return actual


def validate_permutation_bank(
    bank: Sequence[Sequence[int]],
    *,
    hours: int = 24,
    require_derangement: bool = True,
    maximum_adjacent_edges: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Validate registered adjacency-destroying permutations."""

    expected = list(range(int(hours)))
    result: list[tuple[int, ...]] = []
    for index, raw in enumerate(bank):
        order = tuple(int(value) for value in raw)
        if sorted(order) != expected:
            raise ValueError(f"permutation {index} is not a full hour bijection")
        if require_derangement and any(position == hour for position, hour in enumerate(order)):
            raise ValueError(f"permutation {index} retains a fixed hour position")
        adjacent = sum(abs(left - right) == 1 for left, right in zip(order, order[1:]))
        if adjacent > int(maximum_adjacent_edges):
            raise ValueError(f"permutation {index} retains {adjacent} true adjacent edges")
        result.append(order)
    if not result or len(set(result)) != len(result):
        raise ValueError("permutation bank must be non-empty and unique")
    return tuple(result)


def permutation_bank_sha256(bank: Sequence[Sequence[int]]) -> str:
    values = validate_permutation_bank(bank)
    return canonical_sha256({"hours": 24, "permutations": [list(value) for value in values]})


class TemporalContextResidualAdapter(nn.Module):
    """Small ordered GRU residual on frozen condition context.

    Outputs are scattered back to their physical hour positions.  Consequently
    a shuffled order changes only recurrent adjacency, not tensor layout or the
    hour labels already embedded in the frozen context.
    """

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int,
        hours: int = 24,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if context_dim < 1 or hidden_dim < 1 or hours < 2:
            raise ValueError("adapter dimensions must be positive")
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual scale must be finite and positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.hours = int(hours)
        self.residual_scale = float(residual_scale)
        self.input_norm = nn.LayerNorm(self.context_dim)
        self.recurrent = nn.GRUCell(self.context_dim, self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.context_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _order(self, hour_order: torch.Tensor | Sequence[int] | None, device: torch.device) -> torch.Tensor:
        if hour_order is None:
            return torch.arange(self.hours, dtype=torch.long, device=device)
        order = torch.as_tensor(hour_order, dtype=torch.long, device=device)
        if order.shape != (self.hours,) or not torch.equal(
            torch.sort(order).values,
            torch.arange(self.hours, dtype=torch.long, device=device),
        ):
            raise ValueError("adapter hour order must be a complete permutation")
        return order

    def forward(
        self,
        context: torch.Tensor,
        *,
        hour_order: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if context.ndim != 4 or context.shape[2] != self.hours or context.shape[3] != self.context_dim:
            raise ValueError("adapter context must have shape [B,Z,hours,context_dim]")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("adapter context contains non-finite values")
        order = self._order(hour_order, context.device)
        normalized = self.input_norm(context).index_select(2, order)
        batch, zones, _, _ = normalized.shape
        sequence = normalized.reshape(batch * zones, self.hours, self.context_dim)
        hidden = torch.zeros(
            (batch * zones, self.hidden_dim),
            dtype=context.dtype,
            device=context.device,
        )
        outputs: list[torch.Tensor] = []
        for position in range(self.hours):
            hidden = self.recurrent(sequence[:, position], hidden)
            outputs.append(hidden)
        ordered = torch.stack(outputs, dim=1).reshape(
            batch, zones, self.hours, self.hidden_dim
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(self.hours, dtype=torch.long, device=order.device)
        chronological_layout = ordered.index_select(2, inverse)
        projected = self.output_projection(self.output_norm(chronological_layout))
        residual = self.residual_scale * torch.tanh(projected)
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("adapter produced non-finite context residual")
        return residual

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def output_is_exactly_zero(self) -> bool:
        return bool(
            torch.count_nonzero(self.output_projection.weight).item() == 0
            and torch.count_nonzero(self.output_projection.bias).item() == 0
        )


@dataclass(frozen=True)
class ProbeLossSample:
    timestep: torch.Tensor
    noise: torch.Tensor
    noisy_latent: torch.Tensor
    clean_latent: torch.Tensor
    active_mask: torch.Tensor
    base_context: torch.Tensor
    context_residual: torch.Tensor
    adjusted_context: torch.Tensor


@dataclass(frozen=True)
class ProbeScenarioBatch(ScenarioBatch):
    stage_block_index: int
    stage_block_label: str
    intervention_timesteps: tuple[int, ...]
    context_intervention_calls: int
    context_baseline_calls: int
    context_residual_abs_max: float


class TemporalUtilityProbe(nn.Module):
    """Frozen D0-v plus one trainable stage-conditioned context adapter."""

    def __init__(
        self,
        backbone: JointRectifiedFlowBase,
        diffusion: VPredictionJointDDPM,
        stage_block: DDIMStageBlock,
        *,
        adapter_hidden_dim: int = 32,
        residual_scale: float = 1.0,
        registered_ddim_steps: int = 31,
    ) -> None:
        super().__init__()
        if diffusion.prediction_parameterization != "v":
            raise ValueError("family-v1.2 requires v-prediction diffusion")
        if any(value < 0 or value >= diffusion.timesteps for value in stage_block.timesteps):
            raise ValueError("stage block contains an invalid diffusion timestep")
        grid = tuple(
            int(value)
            for value in ddim_timestep_grid(diffusion.timesteps, int(registered_ddim_steps))
            .flip(0)
            .tolist()
        )
        if not set(stage_block.timesteps).issubset(grid):
            raise ValueError("stage block is not a subset of the registered DDIM grid")
        self.backbone = backbone
        self.diffusion = diffusion
        self.stage_block = stage_block
        self.registered_ddim_steps = int(registered_ddim_steps)
        self.adapter = TemporalContextResidualAdapter(
            backbone.encoder_dim,
            hidden_dim=int(adapter_hidden_dim),
            hours=backbone.hours,
            residual_scale=float(residual_scale),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.backbone.eval()
        self._frozen_backbone_sha256 = self.backbone_tensor_sha256()

    def train(self, mode: bool = True) -> "TemporalUtilityProbe":
        super().train(mode)
        self.backbone.eval()
        self.adapter.train(mode)
        return self

    def backbone_tensor_sha256(self) -> str:
        return tensor_state_sha256(
            {name: value.detach().cpu() for name, value in self.backbone.state_dict().items()}
        )

    def assert_backbone_unchanged(self) -> None:
        if self.backbone_tensor_sha256() != self._frozen_backbone_sha256:
            raise RuntimeError("frozen D0-v backbone tensor hash changed")
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"frozen backbone parameter became trainable: {name}")
            if parameter.grad is not None:
                raise RuntimeError(f"frozen backbone parameter received a gradient: {name}")

    @property
    def frozen_backbone_sha256(self) -> str:
        return self._frozen_backbone_sha256

    def _contexts(
        self,
        condition: torch.Tensor,
        *,
        hour_order: torch.Tensor | Sequence[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # E and the memoryless base-context transform are frozen.  It is safe
        # and cheaper to detach them.  The frozen flow below must *not* be put
        # under no_grad because its context-input derivative trains the adapter.
        with torch.no_grad():
            encoded = self.backbone.encode_condition(condition)
            base_context = self.backbone.condition_context(encoded)
        residual = self.adapter(base_context, hour_order=hour_order)
        adjusted = base_context + residual
        return encoded, base_context, residual, adjusted

    def sample_registered_timestep(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("timestep sampling requires a positive batch size")
        choices = torch.as_tensor(
            self.stage_block.timesteps, dtype=torch.long, device=device
        )
        selected = torch.randint(
            len(choices),
            (int(batch_size),),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        return choices.index_select(0, selected)

    def loss(
        self,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        states: torch.Tensor,
        observed_mask: torch.Tensor,
        hour_order: torch.Tensor | Sequence[int] | None = None,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_sample: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ProbeLossSample]:
        active = _validate_joint_training_inputs(
            self.backbone, condition, observation, states, observed_mask
        )
        clean = _clean_interior_logit(self.backbone, observation, active)
        if timestep is None:
            timestep = self.sample_registered_timestep(
                len(clean), device=clean.device, generator=generator
            )
        if timestep.shape != (len(clean),) or timestep.dtype != torch.long:
            raise ValueError("probe timestep must have one int64 value per calendar day")
        allowed = torch.as_tensor(
            self.stage_block.timesteps, dtype=torch.long, device=timestep.device
        )
        if not bool((timestep[:, None] == allowed[None]).any(dim=1).all()):
            raise ValueError("probe loss received a timestep outside its registered block")
        noisy, masked_noise = self.diffusion.q_sample(
            clean, timestep, active, noise=noise, generator=generator
        )
        sqrt_alpha = _extract_schedule(
            self.diffusion.sqrt_alpha_bar, timestep, clean
        )
        sqrt_one_minus = _extract_schedule(
            self.diffusion.sqrt_one_minus_alpha_bar, timestep, clean
        )
        target_v = sqrt_alpha * masked_noise - sqrt_one_minus * clean
        target_v = torch.where(active, target_v, torch.zeros_like(target_v))
        _, base_context, residual, adjusted = self._contexts(
            condition, hour_order=hour_order
        )
        normalized_time = timestep.to(clean.dtype) / float(self.diffusion.timesteps - 1)
        predicted_v = self.backbone.flow(
            noisy, normalized_time, adjusted, active
        )
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
            "context_residual_abs_max": residual.abs().max(),
        }
        if not return_sample:
            return losses
        return losses, ProbeLossSample(
            timestep=timestep,
            noise=masked_noise,
            noisy_latent=noisy,
            clean_latent=clean,
            active_mask=active,
            base_context=base_context,
            context_residual=residual,
            adjusted_context=adjusted,
        )

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: torch.Tensor,
        *,
        members: int,
        steps: int = 31,
        eta: float = 0.0,
        seed: int,
        member_chunk: int = 25,
        allocation: AtomAllocation | None = None,
        initial_noise: torch.Tensor | None = None,
        hour_order: torch.Tensor | Sequence[int] | None = None,
        intervention: bool = True,
    ) -> ProbeScenarioBatch:
        if int(steps) != self.registered_ddim_steps:
            raise ValueError("probe sampling requires the registered DDIM step count")
        if eta != 0.0:
            raise ValueError("family-v1.2 registers deterministic DDIM with eta=0 only")
        if members < 1 or member_chunk < 1:
            raise ValueError("members and member_chunk must be positive")
        grid = ddim_timestep_grid(self.diffusion.timesteps, int(steps))
        was_training = self.training
        self.eval()
        try:
            if condition.dtype != torch.float32:
                raise TypeError("family-v1.2 sampling requires FP32 condition")
            encoded = self.backbone.encode_condition(condition)
            statistics = self.backbone.atom(encoded)
            base_context = self.backbone.condition_context(encoded)
            residual = (
                self.adapter(base_context, hour_order=hour_order)
                if intervention
                else torch.zeros_like(base_context)
            )
            adjusted_context = base_context + residual
            if allocation is None:
                allocation = self.backbone.atom.allocate(
                    statistics, members=members, seed=int(seed) + 1
                )
            expected = (
                len(condition), members, self.backbone.zones, self.backbone.hours
            )
            if allocation.states.shape != expected:
                raise ValueError("provided atom allocation has the wrong shape")
            if allocation.states.device != condition.device:
                raise ValueError("provided atom allocation is on the wrong device")
            if not torch.equal(
                allocation.analytic_probabilities, statistics.probabilities
            ):
                raise ValueError("provided allocation does not match frozen E/A law")
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

            schedule = self.diffusion.alpha_bar.to(
                device=condition.device, dtype=condition.dtype
            )
            reverse_grid = tuple(int(value) for value in grid.flip(0).tolist())
            intervention_set = set(self.stage_block.timesteps) if intervention else set()
            if intervention_set - set(reverse_grid):
                raise RuntimeError("intervention block is absent from sampling grid")
            chunks: list[torch.Tensor] = []
            batched_forward_calls = 0
            intervention_calls = 0
            baseline_calls = 0
            for start in range(0, members, member_chunk):
                stop = min(start + member_chunk, members)
                width = stop - start
                chunk_active = active[:, start:stop].reshape(
                    -1, self.backbone.zones, self.backbone.hours
                )
                latent = torch.where(
                    active[:, start:stop],
                    initial_noise[:, start:stop],
                    torch.zeros_like(initial_noise[:, start:stop]),
                ).reshape(-1, self.backbone.zones, self.backbone.hours)
                base_chunk = base_context[:, None].expand(
                    -1, width, -1, -1, -1
                ).reshape(
                    -1,
                    self.backbone.zones,
                    self.backbone.hours,
                    self.backbone.encoder_dim,
                )
                adjusted_chunk = adjusted_context[:, None].expand(
                    -1, width, -1, -1, -1
                ).reshape_as(base_chunk)
                for index, current_timestep in enumerate(reverse_grid):
                    normalized_time = torch.full(
                        (len(latent),),
                        current_timestep / float(self.diffusion.timesteps - 1),
                        dtype=latent.dtype,
                        device=latent.device,
                    )
                    in_block = current_timestep in intervention_set
                    context = adjusted_chunk if in_block else base_chunk
                    predicted_v = self.backbone.flow(
                        latent, normalized_time, context, chunk_active
                    )
                    batched_forward_calls += 1
                    intervention_calls += int(in_block)
                    baseline_calls += int(not in_block)
                    alpha_current = schedule[current_timestep]
                    sqrt_alpha = torch.sqrt(alpha_current)
                    sqrt_one_minus = torch.sqrt(1.0 - alpha_current)
                    predicted_clean = (
                        sqrt_alpha * latent - sqrt_one_minus * predicted_v
                    )
                    predicted_epsilon = (
                        sqrt_one_minus * latent + sqrt_alpha * predicted_v
                    )
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
                            f"non-finite probe latent after timestep {current_timestep}"
                        )
                    if bool((latent[~chunk_active] != 0.0).any()):
                        raise RuntimeError("inactive probe coordinates moved from zero")
                chunks.append(
                    latent.reshape(
                        len(condition), width, self.backbone.zones, self.backbone.hours
                    )
                )
            latent = torch.cat(chunks, dim=1)
            chunks_count = (members + member_chunk - 1) // member_chunk
            expected_calls = int(steps) * chunks_count
            if batched_forward_calls != expected_calls:
                raise RuntimeError("probe DDIM network-call accounting failed")
            if intervention_calls != len(intervention_set) * chunks_count:
                raise RuntimeError("probe stage intervention-call accounting failed")
            values = self.backbone.atom.reconstruct(latent, states)
            if bool(((values[active] <= 0.0) | (values[active] >= 1.0)).any()):
                raise FloatingPointError(
                    "family-v1.2 sampler decoded an interior value to a boundary"
                )
            return ProbeScenarioBatch(
                values=values,
                states=states,
                active_mask=active,
                interior_latent=latent,
                atom_statistics=statistics,
                atom_allocation=allocation,
                per_path_nfe=int(steps),
                batched_forward_calls=batched_forward_calls,
                stage_block_index=self.stage_block.index,
                stage_block_label=self.stage_block.label,
                intervention_timesteps=tuple(
                    value for value in reverse_grid if value in intervention_set
                ),
                context_intervention_calls=intervention_calls,
                context_baseline_calls=baseline_calls,
                context_residual_abs_max=float(residual.abs().max().cpu()),
            )
        finally:
            self.train(was_training)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
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


class ProbeEMATrainer:
    """AdamW+EMA trainer whose optimizer can see adapter tensors only."""

    def __init__(
        self,
        probe: TemporalUtilityProbe,
        *,
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gradient_clip: float = 5.0,
        ema_decay: float = 0.999,
        backbone_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if learning_rate <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid adapter AdamW hyperparameters")
        if gradient_clip <= 0.0:
            raise ValueError("adapter gradient clip must be positive")
        self.probe = probe
        self.gradient_clip = float(gradient_clip)
        self.backbone_identity = dict(backbone_identity or {})
        self.parameter_names = tuple(name for name, _ in probe.adapter.named_parameters())
        if not self.parameter_names:
            raise RuntimeError("probe adapter exposes no trainable parameters")
        if any(parameter.requires_grad for parameter in probe.backbone.parameters()):
            raise RuntimeError("probe trainer received a trainable backbone")
        parameters = list(probe.adapter.parameters())
        if any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError("all adapter parameters must be trainable")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            betas=tuple(float(value) for value in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        self.ema = CommonEMA(
            probe.adapter,
            decay=float(ema_decay),
            parameter_names=self.parameter_names,
        )
        self.optimizer_updates = 0

    def train_step(
        self,
        batch: ArchitectureBatch,
        *,
        hour_order: torch.Tensor | Sequence[int] | None,
        generator: torch.Generator | None = None,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.probe.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses = self.probe.loss(
            batch.condition,
            batch.target,
            states=batch.state,
            observed_mask=batch.observed_mask,
            hour_order=hour_order,
            timestep=timestep,
            noise=noise,
            generator=generator,
        )
        assert isinstance(losses, dict)
        for name, value in losses.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(f"probe loss {name} must be a scalar tensor")
            if value.dtype != torch.float32 or not bool(torch.isfinite(value)):
                raise FloatingPointError(f"probe loss {name} is non-finite or non-FP32")
        losses["loss"].backward()
        named = dict(self.probe.adapter.named_parameters())
        gradients = {
            name: named[name].grad
            for name in self.parameter_names
            if named[name].grad is not None
        }
        if not gradients:
            raise RuntimeError("probe loss produced no adapter gradients")
        for name, gradient in gradients.items():
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"non-finite adapter gradient: {name}")
        frozen_gradient_names = [
            name
            for name, parameter in self.probe.backbone.named_parameters()
            if parameter.grad is not None
        ]
        if frozen_gradient_names:
            raise RuntimeError(
                f"frozen backbone received gradients: {frozen_gradient_names[:4]}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(named.values()), self.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite adapter gradient norm")
        output_gradient = self.probe.adapter.output_projection.weight.grad
        output_gradient_norm = (
            float(output_gradient.norm().detach().cpu())
            if output_gradient is not None
            else 0.0
        )
        self.optimizer.step()
        for name, parameter in named.items():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"non-finite updated adapter parameter: {name}")
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("non-finite adapter AdamW state")
        self.ema.update(self.probe.adapter)
        self.optimizer_updates += 1
        if self.ema.num_updates != self.optimizer_updates:
            raise RuntimeError("adapter EMA and optimizer counters diverged")
        return {
            **{name: float(value.detach().cpu()) for name, value in losses.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "gradient_was_clipped": float(gradient_norm > self.gradient_clip),
            "output_projection_gradient_norm": output_gradient_norm,
            "optimizer_updates": float(self.optimizer_updates),
            "wall_seconds": float(time.perf_counter() - started),
        }

    @contextmanager
    def ema_weights(self) -> Iterator[TemporalUtilityProbe]:
        with self.ema.average_parameters(self.probe.adapter):
            yield self.probe

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
                raise RuntimeError("adapter checkpoint contains unavailable CUDA RNG state")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])

    def checkpoint_state(
        self,
        *,
        identity: Mapping[str, Any],
        runner_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.probe.assert_backbone_unchanged()
        adapter_state = {
            name: value.detach().cpu().clone()
            for name, value in self.probe.adapter.state_dict().items()
        }
        identity_dict = dict(identity)
        return {
            "schema": PROBE_CHECKPOINT_SCHEMA,
            "identity": identity_dict,
            "identity_sha256": canonical_sha256(identity_dict),
            "optimizer_updates": self.optimizer_updates,
            "stage_block": self.probe.stage_block.manifest(),
            "backbone_identity": self.backbone_identity,
            "frozen_backbone_tensor_sha256": self.probe.frozen_backbone_sha256,
            "adapter_state_dict": adapter_state,
            "adapter_tensor_sha256": tensor_state_sha256(adapter_state),
            "adapter_parameter_names": list(self.parameter_names),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "rng_state": self._rng_state(),
            "runner_state": dict(runner_state or {}),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any],
        runner_state: Mapping[str, Any] | None = None,
    ) -> str:
        return _atomic_torch_save(
            self.checkpoint_state(identity=identity, runner_state=runner_state),
            Path(path),
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_identity: Mapping[str, Any],
        restore_rng: bool = True,
    ) -> Mapping[str, Any]:
        checkpoint = Path(path)
        sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
        if not checkpoint.is_file() or not sidecar.is_file():
            raise FileNotFoundError("adapter checkpoint or SHA256 sidecar is missing")
        pieces = sidecar.read_text(encoding="ascii").split()
        if len(pieces) != 2 or pieces[1] != checkpoint.name:
            raise ValueError("adapter checkpoint sidecar is malformed")
        if file_sha256(checkpoint) != pieces[0]:
            raise ValueError("adapter checkpoint SHA256 mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != PROBE_CHECKPOINT_SCHEMA:
            raise ValueError("adapter checkpoint schema mismatch")
        expected = dict(expected_identity)
        if payload.get("identity") != expected or payload.get(
            "identity_sha256"
        ) != canonical_sha256(expected):
            raise ValueError("adapter checkpoint experiment identity mismatch")
        if payload.get("frozen_backbone_tensor_sha256") != self.probe.frozen_backbone_sha256:
            raise ValueError("adapter checkpoint belongs to a different frozen backbone")
        if payload.get("backbone_identity") != self.backbone_identity:
            raise ValueError("adapter checkpoint backbone reference mismatch")
        if payload.get("stage_block") != self.probe.stage_block.manifest():
            raise ValueError("adapter checkpoint stage-block mismatch")
        raw = payload["adapter_state_dict"]
        if tensor_state_sha256(raw) != payload["adapter_tensor_sha256"]:
            raise ValueError("adapter checkpoint tensor hash mismatch")
        current = self.probe.adapter.state_dict()
        if set(raw) != set(current):
            raise ValueError("adapter checkpoint tensor topology mismatch")
        for name, value in raw.items():
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"adapter checkpoint tensor metadata mismatch: {name}")
        self.probe.adapter.load_state_dict(raw, strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to(self.optimizer, next(self.probe.adapter.parameters()).device)
        self.ema.load_state_dict(self.probe.adapter, payload["ema_state_dict"])
        self.optimizer_updates = int(payload["optimizer_updates"])
        if self.optimizer_updates < 0 or self.ema.num_updates != self.optimizer_updates:
            raise ValueError("adapter checkpoint update counters are inconsistent")
        if tuple(payload["adapter_parameter_names"]) != self.parameter_names:
            raise ValueError("adapter checkpoint parameter-name order mismatch")
        if restore_rng:
            self._restore_rng_state(payload["rng_state"])
        self.probe.assert_backbone_unchanged()
        return payload


def load_frozen_d0_v_best_ema(
    model: JointRectifiedFlowBase,
    checkpoint: str | Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Load and permanently expose the retained checkpoint's best EMA tensors."""

    path = Path(checkpoint)
    if not path.is_file() or file_sha256(path) != str(expected_file_sha256):
        raise ValueError("D0-v checkpoint file hash mismatch")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split() != [
        str(expected_file_sha256),
        path.name,
    ]:
        raise ValueError("D0-v checkpoint sidecar mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("family") != "D0":
        raise ValueError("family-v1.2 backbone checkpoint is not D0")
    raw_model = payload["model_state_dict"]
    if tensor_state_sha256(raw_model) != payload["model_tensor_sha256"]:
        raise ValueError("D0-v online checkpoint tensor hash mismatch")
    if payload["parameter_manifest"]["structural_sha256"] != parameter_manifest(model)[
        "structural_sha256"
    ]:
        raise ValueError("D0-v checkpoint topology mismatch")
    model.load_state_dict(raw_model, strict=True)
    if shared_ea_state_sha256(model) != payload["shared_EA_state_sha256"]:
        raise ValueError("D0-v shared E/A checkpoint identity mismatch")
    ema = payload["ema_state_dict"]
    shadow = ema["shadow"]
    if tensor_state_sha256(shadow) != ema["tensor_sha256"]:
        raise ValueError("D0-v EMA shadow hash mismatch")
    parameters = dict(model.named_parameters())
    names = tuple(str(name) for name in ema["parameter_names"])
    if set(shadow) != set(names) or not set(names).issubset(parameters):
        raise ValueError("D0-v EMA parameter topology mismatch")
    with torch.no_grad():
        for name in names:
            value = shadow[name]
            if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype:
                raise ValueError(f"D0-v EMA tensor metadata mismatch: {name}")
            parameters[name].copy_(value.to(parameters[name].device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    best_ema_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": str(expected_file_sha256),
        "checkpoint_identity_sha256": payload.get("identity_sha256"),
        "online_optimizer_updates": int(payload["optimizer_updates"]),
        "EMA_updates": int(ema["num_updates"]),
        "best_EMA_backbone_tensor_sha256": tensor_state_sha256(best_ema_state),
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
    }


def _nwp_dynamicity_components(raw_condition: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_condition, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[1:3] != (10, 24) or raw.shape[3] < 4:
        raise ValueError("raw NWP must have shape [day,10,24,feature>=4]")
    if not np.isfinite(raw[..., :4]).all():
        raise ValueError("raw NWP components contain non-finite values")
    u100 = raw[..., 2]
    v100 = raw[..., 3]
    speed = np.hypot(u100, v100)
    rms_dws100 = np.sqrt(np.mean(np.diff(speed, axis=-1) ** 2, axis=(1, 2)))
    dot = u100[..., 1:] * u100[..., :-1] + v100[..., 1:] * v100[..., :-1]
    denominator = speed[..., 1:] * speed[..., :-1]
    cosine = np.divide(
        dot,
        denominator,
        out=np.ones_like(dot),
        where=denominator > 1e-12,
    )
    turn = np.arccos(np.clip(cosine, -1.0, 1.0))
    weight = np.minimum(speed[..., 1:], speed[..., :-1])
    weight_sum = np.sum(weight, axis=(1, 2))
    weighted_turn100 = np.divide(
        np.sum(turn * weight, axis=(1, 2)),
        weight_sum,
        out=np.zeros(len(raw), dtype=np.float64),
        where=weight_sum > 1e-12,
    )
    spatial_dispersion = np.mean(np.std(speed, axis=1), axis=1)
    return {
        "rms_dws100": rms_dws100,
        "weighted_turn100": weighted_turn100,
        "spatial_dispersion": spatial_dispersion,
    }


def fit_nwp_dynamicity_registry(raw_train_condition: np.ndarray) -> dict[str, Any]:
    """Fit the predeclared equal-weight dynamicity score on train NWP only."""

    components = _nwp_dynamicity_components(raw_train_condition)
    names = ("rms_dws100", "weighted_turn100", "spatial_dispersion")
    means = {name: float(np.mean(components[name])) for name in names}
    stds = {name: float(np.std(components[name])) for name in names}
    if any(not np.isfinite(stds[name]) or stds[name] < 1e-12 for name in names):
        raise ValueError("NWP dynamicity component has zero or invalid train spread")
    score = np.mean(
        np.stack(
            [
                (components[name] - means[name]) / stds[name]
                for name in names
            ],
            axis=1,
        ),
        axis=1,
    )
    lower, upper = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(
        score < lower,
        "stable",
        np.where(score < upper, "moderate", "dynamic"),
    )
    core = {
        "schema": NWP_REGISTRY_SCHEMA,
        "fit_role": "train_NWP_only",
        "target_power_used": False,
        "fit_days": int(len(score)),
        "features": list(names),
        "standardization": "train_mean_and_population_std",
        "feature_mean": means,
        "feature_std": stds,
        "score": "equal_weight_mean_of_standardized_features",
        "tertile_cutpoints": [float(lower), float(upper)],
        "labels": ["stable", "moderate", "dynamic"],
        "train_label_counts": {
            label: int(np.count_nonzero(labels == label))
            for label in ("stable", "moderate", "dynamic")
        },
        "train_score_sha256": hashlib.sha256(
            np.ascontiguousarray(score, dtype=np.float64).view(np.uint8)
        ).hexdigest(),
    }
    return {**core, "registry_sha256": canonical_sha256(core)}


def assign_nwp_dynamicity(
    raw_condition: np.ndarray,
    registry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign frozen train-defined NWP labels without reading target power."""

    if registry.get("schema") != NWP_REGISTRY_SCHEMA:
        raise ValueError("unexpected NWP dynamicity registry schema")
    core = {key: value for key, value in registry.items() if key != "registry_sha256"}
    if registry.get("registry_sha256") != canonical_sha256(core):
        raise ValueError("NWP dynamicity registry hash mismatch")
    components = _nwp_dynamicity_components(raw_condition)
    names = tuple(str(value) for value in registry["features"])
    if names != ("rms_dws100", "weighted_turn100", "spatial_dispersion"):
        raise ValueError("NWP dynamicity feature registry drifted")
    score = np.mean(
        np.stack(
            [
                (
                    components[name] - float(registry["feature_mean"][name])
                )
                / float(registry["feature_std"][name])
                for name in names
            ],
            axis=1,
        ),
        axis=1,
    )
    lower, upper = (float(value) for value in registry["tertile_cutpoints"])
    labels = np.where(
        score < lower,
        "stable",
        np.where(score < upper, "moderate", "dynamic"),
    )
    return np.ascontiguousarray(score, dtype=np.float64), np.asarray(labels, dtype="U8")


__all__ = [
    "DDIMStageBlock",
    "NWP_REGISTRY_SCHEMA",
    "PROBE_CHECKPOINT_SCHEMA",
    "ProbeEMATrainer",
    "ProbeLossSample",
    "ProbeScenarioBatch",
    "TemporalContextResidualAdapter",
    "TemporalUtilityProbe",
    "assign_nwp_dynamicity",
    "build_ddim_stage_blocks",
    "fit_nwp_dynamicity_registry",
    "load_frozen_d0_v_best_ema",
    "permutation_bank_sha256",
    "validate_permutation_bank",
    "validate_stage_registry",
]
