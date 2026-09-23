"""Seven-path tiny denoiser and colored-noise DDIM core for TGO-v1.

This module contains no data loader, no target-role access, and no statistical
decision logic.  It implements only the model-level interventions frozen in
``repro_configs/architecture_v1_transition_object_probe.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .family_diffusion import ddim_timestep_grid
from .g0b_tiny_denoiser import (
    TINY_DENOISER_PARAMETERS,
    TinyDenoiser,
)
from .transition_object import (
    MaskConditionedOperator,
    ddim_direct_x0_step,
    iid_forward_sample,
    matched_noise_level_forward,
)


PathId = Literal[
    "LEVEL_IID",
    "TRANSITION_TRUE",
    "LEVEL_MATCHED_METRIC",
    "LEVEL_MATCHED_NOISE",
    "LEVEL_MATCHED_BOTH",
    "ORTHOGONAL_DCT",
    "TRANSITION_WRONG",
]

PATH_IDS: tuple[PathId, ...] = (
    "LEVEL_IID",
    "TRANSITION_TRUE",
    "LEVEL_MATCHED_METRIC",
    "LEVEL_MATCHED_NOISE",
    "LEVEL_MATCHED_BOTH",
    "ORTHOGONAL_DCT",
    "TRANSITION_WRONG",
)

TRANSFORMED_PATHS = frozenset(
    {"TRANSITION_TRUE", "ORTHOGONAL_DCT", "TRANSITION_WRONG"}
)
MATCHED_NOISE_PATHS = frozenset(
    {"LEVEL_MATCHED_NOISE", "LEVEL_MATCHED_BOTH"}
)
MATCHED_METRIC_PATHS = frozenset(
    {"LEVEL_MATCHED_METRIC", "LEVEL_MATCHED_BOTH"}
)


@dataclass(frozen=True)
class TransitionLossSample:
    """Auditable tensors produced by one paired TGO-v1 update."""

    loss: torch.Tensor
    prediction_level: torch.Tensor
    prediction_native: torch.Tensor
    target_native: torch.Tensor
    noisy_native: torch.Tensor
    alpha_bar: torch.Tensor
    log_snr: torch.Tensor


@dataclass(frozen=True)
class TransitionSample:
    """A direct-x0 DDIM residual sample before physical-power decoding."""

    level: torch.Tensor
    native: torch.Tensor
    path_id: PathId
    per_path_nfe: int
    grid: torch.Tensor


def _validate_path(path_id: str) -> PathId:
    if path_id not in PATH_IDS:
        raise ValueError(f"unknown TGO-v1 path: {path_id}")
    return path_id  # type: ignore[return-value]


class TransitionDenoisingSystem(nn.Module):
    """The common 56,058-parameter denoiser with a registered path identity."""

    def __init__(self, path_id: PathId, *, model_seed: int) -> None:
        super().__init__()
        self.path_id = _validate_path(str(path_id))
        self.model_seed = int(model_seed)
        self.denoiser = TinyDenoiser(model_seed=model_seed)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != TINY_DENOISER_PARAMETERS:
            raise RuntimeError(f"TGO-v1 parameter count drifted: {count}")


def _validate_common_inputs(
    system: TransitionDenoisingSystem,
    level_clean: torch.Tensor,
    active_mask: torch.Tensor,
    condition: torch.Tensor,
    timestep: torch.Tensor,
    native_noise: torch.Tensor,
    alpha_bar_schedule: torch.Tensor,
) -> None:
    if level_clean.ndim != 3 or tuple(level_clean.shape[1:]) != (10, 24):
        raise ValueError("level clean target must have shape [batch,10,24]")
    if active_mask.shape != level_clean.shape or active_mask.dtype != torch.bool:
        raise ValueError("active mask must be boolean and align with level target")
    if condition.shape != (len(level_clean), 47):
        raise ValueError("condition must have shape [batch,47]")
    if timestep.shape != (len(level_clean),) or timestep.dtype != torch.long:
        raise ValueError("timestep must be torch.long with shape [batch]")
    if native_noise.shape != level_clean.shape:
        raise ValueError("native noise must align with level target")
    if alpha_bar_schedule.ndim != 1 or len(alpha_bar_schedule) < 2:
        raise ValueError("alpha-bar schedule must be a vector with at least two steps")
    tensors = (level_clean, condition, native_noise, alpha_bar_schedule)
    if any(not torch.is_floating_point(value) for value in tensors):
        raise TypeError("TGO-v1 target, condition, noise, and schedule must be floating point")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("TGO-v1 input contains non-finite values")
    if bool(((timestep < 0) | (timestep >= len(alpha_bar_schedule))).any()):
        raise ValueError("timestep lies outside the alpha-bar schedule")
    devices = {
        level_clean.device,
        active_mask.device,
        condition.device,
        timestep.device,
        native_noise.device,
        alpha_bar_schedule.device,
    }
    if len(devices) != 1:
        raise ValueError("all TGO-v1 tensors must share a device")
    if not bool(active_mask.reshape(len(active_mask), -1).any(dim=1).all()):
        raise ValueError("every calendar-day batch row must have active coordinates")
    if sum(parameter.numel() for parameter in system.parameters()) != TINY_DENOISER_PARAMETERS:
        raise RuntimeError("TGO-v1 system parameter count changed")


def _schedule_batch(
    alpha_bar_schedule: torch.Tensor,
    timestep: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = alpha_bar_schedule.index_select(0, timestep).to(dtype=dtype)
    if bool(((alpha <= 0.0) | (alpha >= 1.0)).any()):
        raise ValueError("selected alpha-bar values must lie strictly in (0,1)")
    log_snr_scalar = torch.log(alpha) - torch.log1p(-alpha)
    log_snr = log_snr_scalar[:, None].expand(-1, 6)
    return alpha, log_snr


def _operator_for_path(
    path_id: PathId,
    *,
    true_operator: MaskConditionedOperator,
    wrong_operator: MaskConditionedOperator,
    dct_operator: MaskConditionedOperator,
) -> MaskConditionedOperator | None:
    if path_id == "TRANSITION_TRUE":
        return true_operator
    if path_id == "TRANSITION_WRONG":
        return wrong_operator
    if path_id == "ORTHOGONAL_DCT":
        return dct_operator
    return None


def _active_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or active_mask.shape != target.shape:
        raise ValueError("active MSE tensors do not align")
    count = active_mask.sum()
    if int(count) == 0:
        raise ValueError("active MSE requires active coordinates")
    error = torch.where(active_mask, prediction - target, torch.zeros_like(target))
    return error.square().sum() / count.to(error.dtype)


def transition_denoising_loss(
    system: TransitionDenoisingSystem,
    level_clean: torch.Tensor,
    active_mask: torch.Tensor,
    condition: torch.Tensor,
    timestep: torch.Tensor,
    native_noise: torch.Tensor,
    alpha_bar_schedule: torch.Tensor,
    *,
    true_operator: MaskConditionedOperator,
    wrong_operator: MaskConditionedOperator,
    dct_operator: MaskConditionedOperator,
) -> TransitionLossSample:
    """Compute one direct-x0 loss under the registered path geometry."""

    _validate_common_inputs(
        system,
        level_clean,
        active_mask,
        condition,
        timestep,
        native_noise,
        alpha_bar_schedule,
    )
    path_id = system.path_id
    clean_level = torch.where(
        active_mask, level_clean, torch.zeros_like(level_clean)
    )
    alpha, log_snr = _schedule_batch(
        alpha_bar_schedule, timestep, dtype=level_clean.dtype
    )
    coordinate_operator = _operator_for_path(
        path_id,
        true_operator=true_operator,
        wrong_operator=wrong_operator,
        dct_operator=dct_operator,
    )
    if coordinate_operator is not None:
        target_native = coordinate_operator.transform(clean_level, active_mask)
        noisy_native = iid_forward_sample(
            target_native, active_mask, alpha, native_noise
        )
    elif path_id in MATCHED_NOISE_PATHS:
        target_native = clean_level
        noisy_native = matched_noise_level_forward(
            clean_level,
            active_mask,
            alpha,
            native_noise,
            true_operator,
        )
    else:
        target_native = clean_level
        noisy_native = iid_forward_sample(
            clean_level, active_mask, alpha, native_noise
        )

    prediction_native = system.denoiser(
        noisy_native, active_mask, condition, log_snr, timestep
    )
    prediction_native = torch.where(
        active_mask, prediction_native, torch.zeros_like(prediction_native)
    )
    if coordinate_operator is not None:
        prediction_level = coordinate_operator.inverse(
            prediction_native, active_mask
        )
    else:
        prediction_level = prediction_native

    if path_id in MATCHED_METRIC_PATHS:
        loss = true_operator.induced_mse(
            prediction_level, clean_level, active_mask
        )
    else:
        loss = _active_mse(prediction_native, target_native, active_mask)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("TGO-v1 denoising loss is non-finite")
    return TransitionLossSample(
        loss=loss,
        prediction_level=prediction_level,
        prediction_native=prediction_native,
        target_native=target_native,
        noisy_native=noisy_native,
        alpha_bar=alpha,
        log_snr=log_snr,
    )


def _initial_native_state(
    path_id: PathId,
    initial_native_noise: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    true_operator: MaskConditionedOperator,
) -> torch.Tensor:
    masked = torch.where(
        active_mask, initial_native_noise, torch.zeros_like(initial_native_noise)
    )
    if path_id in MATCHED_NOISE_PATHS:
        return true_operator.inverse(masked, active_mask)
    return masked


@torch.no_grad()
def sample_transition_ddim(
    system: TransitionDenoisingSystem,
    condition: torch.Tensor,
    active_mask: torch.Tensor,
    initial_native_noise: torch.Tensor,
    alpha_bar_schedule: torch.Tensor,
    *,
    true_operator: MaskConditionedOperator,
    wrong_operator: MaskConditionedOperator,
    dct_operator: MaskConditionedOperator,
    steps: int = 31,
) -> TransitionSample:
    """Generate normalized level residuals with deterministic direct-x0 DDIM."""

    if condition.ndim != 2 or condition.shape[1] != 47:
        raise ValueError("sampling condition must have shape [batch,47]")
    expected = (len(condition), 10, 24)
    if active_mask.shape != expected or active_mask.dtype != torch.bool:
        raise ValueError("sampling active mask must have shape [batch,10,24]")
    if initial_native_noise.shape != expected:
        raise ValueError("initial native noise must have shape [batch,10,24]")
    if alpha_bar_schedule.ndim != 1:
        raise ValueError("sampling alpha-bar schedule must be a vector")
    if not all(
        value.device == condition.device
        for value in (active_mask, initial_native_noise, alpha_bar_schedule)
    ):
        raise ValueError("all sampling tensors must share a device")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (condition, initial_native_noise, alpha_bar_schedule)
    ):
        raise ValueError("sampling tensor contains non-finite values")
    if not bool(active_mask.reshape(len(active_mask), -1).any(dim=1).all()):
        raise ValueError("every sampled row must have active coordinates")

    path_id = system.path_id
    coordinate_operator = _operator_for_path(
        path_id,
        true_operator=true_operator,
        wrong_operator=wrong_operator,
        dct_operator=dct_operator,
    )
    native = _initial_native_state(
        path_id,
        initial_native_noise,
        active_mask,
        true_operator=true_operator,
    )
    grid = ddim_timestep_grid(len(alpha_bar_schedule), steps).to(condition.device)
    reverse_grid = grid.flip(0)
    was_training = system.training
    system.eval()
    try:
        for index, current_index_tensor in enumerate(reverse_grid):
            current_index = int(current_index_tensor)
            timestep = torch.full(
                (len(condition),),
                current_index,
                dtype=torch.long,
                device=condition.device,
            )
            current_alpha, log_snr = _schedule_batch(
                alpha_bar_schedule, timestep, dtype=condition.dtype
            )
            predicted_clean = system.denoiser(
                native, active_mask, condition, log_snr, timestep
            )
            predicted_clean = torch.where(
                active_mask, predicted_clean, torch.zeros_like(predicted_clean)
            )
            if index + 1 < len(reverse_grid):
                previous_index = int(reverse_grid[index + 1])
                previous_alpha = alpha_bar_schedule[previous_index].to(
                    dtype=condition.dtype
                ).expand(len(condition))
                native = ddim_direct_x0_step(
                    native,
                    predicted_clean,
                    active_mask,
                    current_alpha,
                    previous_alpha,
                )
            else:
                native = predicted_clean
            if not bool(torch.isfinite(native).all()):
                raise FloatingPointError(
                    f"non-finite TGO-v1 sample after timestep {current_index}"
                )
            if bool((native[~active_mask] != 0.0).any()):
                raise RuntimeError("inactive TGO-v1 sampler coordinates moved")
        if coordinate_operator is None:
            level = native
        else:
            level = coordinate_operator.inverse(native, active_mask)
        if not bool(torch.isfinite(level).all()):
            raise FloatingPointError("TGO-v1 inverse sample is non-finite")
        return TransitionSample(
            level=level,
            native=native,
            path_id=path_id,
            per_path_nfe=len(reverse_grid),
            grid=grid.detach().cpu(),
        )
    finally:
        system.train(was_training)


__all__ = [
    "MATCHED_METRIC_PATHS",
    "MATCHED_NOISE_PATHS",
    "PATH_IDS",
    "TRANSFORMED_PATHS",
    "PathId",
    "TransitionDenoisingSystem",
    "TransitionLossSample",
    "TransitionSample",
    "sample_transition_ddim",
    "transition_denoising_loss",
]
