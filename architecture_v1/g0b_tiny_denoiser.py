"""Tiny-denoiser components for the frozen architecture-v1 G0-B Probe.

The module deliberately contains no dataset loader and no evaluation logic.
It implements only the six registered corruption paths, their common
capacity-limited denoiser, EMA state, and small numerical helpers used by the
P0/formal runner.  Target-role access remains the runner's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .g0_predictability import ModeGroup
from .g0b_allocation import (
    gaussian_information,
    reverse_water_fill_information,
    snr_from_information,
    weighted_information_budget,
)


PATH_IDS = (
    "IID",
    "CW_GROUP",
    "FIXED_BAND",
    "MULAN_LITE",
    "PA_RWF",
    "PA_SHUFFLE",
)
TINY_DENOISER_INPUT_DIMENSION = 549
TINY_DENOISER_PARAMETERS = 56_058
MULAN_LITE_PARAMETERS = 870


@dataclass(frozen=True)
class ProxyInformationAllocation:
    """One allocation with true budget/reserve and a possibly wrong proxy."""

    baseline_information: np.ndarray
    reserved_information: np.ndarray
    incremental_information: np.ndarray
    information: np.ndarray
    snr: np.ndarray
    alpha_bar: np.ndarray
    baseline_budget: np.ndarray
    allocated_budget: np.ndarray
    budget_error: np.ndarray


@dataclass(frozen=True)
class TorchScheduleBatch:
    """Allocated tensors needed by one tiny-denoiser update."""

    alpha_bar: torch.Tensor
    log_snr: torch.Tensor
    information: torch.Tensor
    baseline_budget: torch.Tensor


@dataclass(frozen=True)
class TinyLossSample:
    """Auditable outputs from one raw-residual reconstruction loss."""

    loss: torch.Tensor
    prediction: torch.Tensor
    noisy: torch.Tensor
    schedule: TorchScheduleBatch


class ConditionStandardizer:
    """Outer-train-only population standardizer for the 47 conditions."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if self.mean.shape != (47,) or self.std.shape != (47,):
            raise ValueError("G0-B condition moments must each have shape [47]")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise FloatingPointError("condition moments must be finite")
        if np.any(self.std <= 0.0):
            raise ValueError("condition standard deviations must be positive")

    @classmethod
    def fit(cls, values: np.ndarray) -> "ConditionStandardizer":
        sample = np.asarray(values, dtype=np.float64)
        if sample.ndim != 2 or sample.shape[1] != 47 or len(sample) < 2:
            raise ValueError("condition fit values must have shape [day,47]")
        if not np.isfinite(sample).all():
            raise FloatingPointError("condition fit values contain non-finite entries")
        mean = sample.mean(axis=0)
        std = sample.std(axis=0, ddof=0)
        std = np.where(std < 1e-10, 1.0, std)
        return cls(mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        sample = np.asarray(values, dtype=np.float64)
        if sample.ndim != 2 or sample.shape[1] != 47:
            raise ValueError("condition values must have shape [day,47]")
        result = (sample - self.mean) / self.std
        if not np.isfinite(result).all():
            raise FloatingPointError("standardized conditions are non-finite")
        return result.astype(np.float32)


def proxy_reserve_then_water_fill(
    true_variance: np.ndarray,
    weights: np.ndarray,
    proxy_variance: np.ndarray,
    baseline_snr: np.ndarray | float,
    *,
    eta: float = 0.5,
    tolerance: float = 1e-12,
) -> ProxyInformationAllocation:
    """Allocate a true IID budget using only ``proxy_variance`` for priority.

    The true variance controls the baseline budget, safety reserve, and final
    information-to-SNR inversion.  The proxy controls only the priority passed
    to reverse water filling.  This distinction is the causal contract shared
    by FIXED_BAND, MULAN_LITE, PA_RWF, and PA_SHUFFLE.
    """

    true = np.asarray(true_variance, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    proxy = np.asarray(proxy_variance, dtype=np.float64)
    if true.ndim != 2 or true.shape[1] != 6:
        raise ValueError("true variance must have shape [batch,6]")
    if weight.shape != true.shape or proxy.shape != true.shape:
        raise ValueError("true variance, weights, and proxy must share shape")
    if not np.isfinite(true).all() or not np.isfinite(proxy).all():
        raise FloatingPointError("allocation variances must be finite")
    if np.any(true <= 0.0) or np.any(proxy <= 0.0):
        raise ValueError("allocation variances must be strictly positive")
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("allocation weights must be finite and positive")
    if not np.allclose(weight.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("allocation weights must sum to one")
    fraction = float(eta)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("eta must lie in [0,1]")

    ratio = np.asarray(baseline_snr, dtype=np.float64)
    if ratio.ndim == 0:
        ratio = np.full(len(true), float(ratio), dtype=np.float64)
    if ratio.shape != (len(true),) or not np.isfinite(ratio).all():
        raise ValueError("baseline SNR must be finite scalar or shape [batch]")
    if np.any(ratio <= 0.0):
        raise ValueError("baseline SNR must be strictly positive")

    baseline_information = gaussian_information(true, ratio[:, None])
    baseline_budget = weighted_information_budget(baseline_information, weight)
    reserved = fraction * baseline_information
    remaining_budget = (1.0 - fraction) * baseline_budget
    proxy_information = gaussian_information(proxy, ratio[:, None])
    priority = proxy * np.exp(-2.0 * fraction * proxy_information)
    incremental, _ = reverse_water_fill_information(
        priority,
        weight,
        remaining_budget,
        tolerance=tolerance,
    )
    information = reserved + incremental
    snr = snr_from_information(true, information)
    alpha_bar = snr / (1.0 + snr)
    allocated_budget = weighted_information_budget(information, weight)
    budget_error = allocated_budget - baseline_budget
    bound = tolerance * (1.0 + np.abs(baseline_budget))
    if np.any(np.abs(budget_error) > bound):
        raise RuntimeError("proxy allocation changed the true information budget")
    if not np.isfinite(alpha_bar).all() or np.any(alpha_bar <= 0.0):
        raise FloatingPointError("proxy allocation produced invalid alpha_bar")
    if np.any(alpha_bar >= 1.0):
        raise FloatingPointError("proxy allocation alpha_bar must remain below one")
    return ProxyInformationAllocation(
        baseline_information=baseline_information,
        reserved_information=reserved,
        incremental_information=incremental,
        information=information,
        snr=snr,
        alpha_bar=alpha_bar,
        baseline_budget=baseline_budget,
        allocated_budget=allocated_budget,
        budget_error=budget_error,
    )


def rank_weighted_fixed_variance(
    variance: np.ndarray, effective_rank: np.ndarray
) -> np.ndarray:
    """Return the registered outer-train rank-weighted six-mode mean."""

    value = np.asarray(variance, dtype=np.float64)
    rank = np.asarray(effective_rank, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6 or rank.shape != value.shape:
        raise ValueError("variance and effective rank must have shape [day,6]")
    if not np.isfinite(value).all() or not np.isfinite(rank).all():
        raise FloatingPointError("fixed-variance inputs must be finite")
    if np.any(value <= 0.0) or np.any(rank <= 0.0):
        raise ValueError("fixed-variance inputs must be positive")
    result = np.sum(rank * value, axis=0) / np.sum(rank, axis=0)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise FloatingPointError("fixed variance is invalid")
    return result


class ModeProjectorBank(nn.Module):
    """The six frozen orthogonal common/local x DCT-band projectors."""

    def __init__(self, groups: Sequence[ModeGroup]) -> None:
        super().__init__()
        if len(groups) != 6:
            raise ValueError("G0-B requires exactly six mode groups")
        spatial = np.stack([group.spatial_projector for group in groups])
        temporal = np.stack([group.temporal_projector for group in groups])
        self.names = tuple(group.name for group in groups)
        self.register_buffer("spatial", torch.as_tensor(spatial, dtype=torch.float32))
        self.register_buffer("temporal", torch.as_tensor(temporal, dtype=torch.float32))

    def project(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (10, 24):
            raise ValueError("projector input must have shape [batch,10,24]")
        return torch.einsum(
            "kij,bjh,khl->bkil", self.spatial, value, self.temporal
        )

    def reconstruct(self, parts: torch.Tensor) -> torch.Tensor:
        if parts.ndim != 4 or tuple(parts.shape[1:]) != (6, 10, 24):
            raise ValueError("projected parts must have shape [batch,6,10,24]")
        return parts.sum(dim=1)

    def whiten(self, value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        if variance.shape != (len(value), 6):
            raise ValueError("whitening variance must have shape [batch,6]")
        if bool((variance <= 0.0).any()):
            raise ValueError("whitening variance must be positive")
        return self.reconstruct(
            self.project(value) / variance.sqrt()[:, :, None, None]
        )

    def unwhiten(self, value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        if variance.shape != (len(value), 6):
            raise ValueError("unwhitening variance must have shape [batch,6]")
        if bool((variance <= 0.0).any()):
            raise ValueError("unwhitening variance must be positive")
        return self.reconstruct(
            self.project(value) * variance.sqrt()[:, :, None, None]
        )


def timestep_fourier_features(
    timestep: torch.Tensor, *, timesteps: int = 250
) -> torch.Tensor:
    """Return the frozen sin/cos features for frequencies 2**0..2**7."""

    if timestep.ndim != 1 or timestep.dtype != torch.long:
        raise ValueError("timestep must be a torch.long vector")
    if timesteps < 2 or bool(((timestep < 0) | (timestep >= timesteps)).any()):
        raise ValueError("timestep lies outside the registered grid")
    frequency = torch.pow(
        torch.tensor(2.0, device=timestep.device),
        torch.arange(8, device=timestep.device, dtype=torch.float32),
    )
    phase = (
        2.0
        * math.pi
        * timestep.to(torch.float32)[:, None]
        * frequency[None]
        / float(timesteps - 1)
    )
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class TinyDenoiser(nn.Module):
    """The registered 56,058-parameter joint x0-prediction MLP."""

    def __init__(self, *, model_seed: int) -> None:
        super().__init__()
        self.model_seed = int(model_seed)
        self.input_norm = nn.LayerNorm(TINY_DENOISER_INPUT_DIMENSION)
        self.hidden_one = nn.Linear(TINY_DENOISER_INPUT_DIMENSION, 64)
        self.hidden_two = nn.Linear(64, 64)
        self.output = nn.Linear(64, 240)
        self.activation = nn.SiLU()
        self.reset_registered_parameters()
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != TINY_DENOISER_PARAMETERS:
            raise RuntimeError(f"tiny-denoiser parameter count drifted: {count}")

    @torch.no_grad()
    def reset_registered_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.model_seed)
        self.input_norm.weight.fill_(1.0)
        self.input_norm.bias.zero_()
        for layer in (self.hidden_one, self.hidden_two):
            nn.init.xavier_uniform_(layer.weight, generator=generator)
            layer.bias.zero_()
        self.output.weight.zero_()
        self.output.bias.zero_()

    def forward(
        self,
        noisy: torch.Tensor,
        active_mask: torch.Tensor,
        condition: torch.Tensor,
        log_snr: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 3 or tuple(noisy.shape[1:]) != (10, 24):
            raise ValueError("noisy input must have shape [batch,10,24]")
        if active_mask.shape != noisy.shape or active_mask.dtype != torch.bool:
            raise ValueError("active mask must be boolean and align with noisy input")
        if condition.shape != (len(noisy), 47):
            raise ValueError("condition must have shape [batch,47]")
        if log_snr.shape != (len(noisy), 6):
            raise ValueError("schedule descriptor must have shape [batch,6]")
        time = timestep_fourier_features(timestep)
        model_input = torch.cat(
            [
                noisy.reshape(len(noisy), 240),
                active_mask.to(noisy.dtype).reshape(len(noisy), 240),
                condition,
                log_snr.clamp(-20.0, 12.0),
                time,
            ],
            dim=1,
        )
        if model_input.shape[1] != TINY_DENOISER_INPUT_DIMENSION:
            raise RuntimeError("tiny-denoiser input dimension drifted")
        hidden = self.activation(self.hidden_one(self.input_norm(model_input)))
        hidden = self.activation(self.hidden_two(hidden))
        result = self.output(hidden).reshape(len(noisy), 10, 24)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("tiny-denoiser prediction is non-finite")
        return result


class MuLANLiteSchedule(nn.Module):
    """Registered 870-parameter bounded condition-to-proxy head."""

    def __init__(self, *, model_seed: int) -> None:
        super().__init__()
        self.model_seed = int(model_seed)
        self.hidden = nn.Linear(47, 16)
        self.output = nn.Linear(16, 6)
        self.activation = nn.SiLU()
        self.reset_registered_parameters()
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != MULAN_LITE_PARAMETERS:
            raise RuntimeError(f"MuLAN-lite parameter count drifted: {count}")

    @torch.no_grad()
    def reset_registered_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.model_seed + 1_000_003)
        nn.init.xavier_uniform_(self.hidden.weight, generator=generator)
        self.hidden.bias.zero_()
        self.output.weight.zero_()
        self.output.bias.zero_()

    def forward(
        self, condition: torch.Tensor, fixed_variance: torch.Tensor
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != 47:
            raise ValueError("MuLAN-lite condition must have shape [batch,47]")
        if fixed_variance.shape != (6,):
            raise ValueError("fixed variance must have shape [6]")
        modulation = self.output(self.activation(self.hidden(condition)))
        proxy = fixed_variance[None] * torch.exp(
            math.log(4.0) * torch.tanh(modulation)
        )
        if not bool(torch.isfinite(proxy).all()) or bool((proxy <= 0.0).any()):
            raise FloatingPointError("MuLAN-lite proxy variance is invalid")
        return proxy


class TinyDenoisingSystem(nn.Module):
    """A base tiny denoiser plus the path-specific registered schedule head."""

    def __init__(self, path_id: str, *, model_seed: int) -> None:
        super().__init__()
        if path_id not in PATH_IDS:
            raise ValueError(f"unknown G0-B path: {path_id}")
        self.path_id = str(path_id)
        self.model_seed = int(model_seed)
        self.denoiser = TinyDenoiser(model_seed=model_seed)
        if path_id == "MULAN_LITE":
            self.scheduler: MuLANLiteSchedule | None = MuLANLiteSchedule(
                model_seed=model_seed
            )
        else:
            self.scheduler = None


def _torch_reverse_water_fill(
    priority: torch.Tensor,
    weights: torch.Tensor,
    budget: torch.Tensor,
) -> torch.Tensor:
    """Differentiable-within-active-set six-mode reverse water filling."""

    if priority.ndim != 2 or priority.shape[1] != 6:
        raise ValueError("priority must have shape [batch,6]")
    if weights.shape != priority.shape or budget.shape != (len(priority),):
        raise ValueError("reverse-water-fill tensors do not align")
    if bool((priority <= 0.0).any()) or bool((weights <= 0.0).any()):
        raise ValueError("priority and weights must be positive")
    log_priority = torch.log(priority)
    ordered_log, order = torch.sort(log_priority, dim=1, descending=True)
    ordered_weight = torch.gather(weights, 1, order)
    cumulative_weight = ordered_weight.cumsum(dim=1)
    cumulative_weighted_log = (ordered_weight * ordered_log).cumsum(dim=1)
    candidates = (
        cumulative_weighted_log - 2.0 * budget[:, None]
    ) / cumulative_weight
    minus_infinity = torch.full(
        (len(priority), 1),
        -torch.inf,
        dtype=priority.dtype,
        device=priority.device,
    )
    next_log = torch.cat([ordered_log[:, 1:], minus_infinity], dim=1)
    valid = candidates >= next_log - 1e-7
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("torch reverse-water-fill active set was not found")
    active_index = valid.to(torch.int64).argmax(dim=1)
    level = torch.gather(candidates, 1, active_index[:, None]).squeeze(1)
    information = 0.5 * torch.relu(log_priority - level[:, None])
    if not bool(torch.isfinite(information).all()):
        raise FloatingPointError("torch reverse-water-fill became non-finite")
    return information


def allocate_torch_schedule(
    path_id: str,
    true_variance: torch.Tensor,
    weights: torch.Tensor,
    timestep: torch.Tensor,
    baseline_alpha_bar: torch.Tensor,
    *,
    proxy_variance: torch.Tensor | None = None,
    eta: float = 0.5,
) -> TorchScheduleBatch:
    """Allocate the registered path schedule for one training batch."""

    if path_id not in PATH_IDS:
        raise ValueError(f"unknown G0-B path: {path_id}")
    if true_variance.ndim != 2 or true_variance.shape[1] != 6:
        raise ValueError("true variance must have shape [batch,6]")
    if weights.shape != true_variance.shape:
        raise ValueError("weights and true variance must align")
    if timestep.shape != (len(true_variance),) or timestep.dtype != torch.long:
        raise ValueError("timestep must be a torch.long batch vector")
    if baseline_alpha_bar.ndim != 1:
        raise ValueError("baseline alpha_bar must be a vector")
    if bool(((timestep < 0) | (timestep >= len(baseline_alpha_bar))).any()):
        raise ValueError("timestep lies outside baseline schedule")
    if bool((true_variance <= 0.0).any()) or bool((weights <= 0.0).any()):
        raise ValueError("true variance and weights must be positive")
    baseline_alpha = baseline_alpha_bar[timestep]
    baseline_snr = baseline_alpha / (1.0 - baseline_alpha)
    baseline_information = 0.5 * torch.log1p(
        true_variance * baseline_snr[:, None]
    )
    baseline_budget = torch.sum(weights * baseline_information, dim=1)

    if path_id == "IID":
        snr = baseline_snr[:, None].expand_as(true_variance)
        information = baseline_information
        alpha = baseline_alpha[:, None].expand_as(true_variance)
    elif path_id == "CW_GROUP":
        scalar_snr = torch.expm1(2.0 * baseline_budget)
        snr = scalar_snr[:, None].expand_as(true_variance)
        information = baseline_budget[:, None].expand_as(true_variance)
        scalar_alpha = scalar_snr / (1.0 + scalar_snr)
        alpha = scalar_alpha[:, None].expand_as(true_variance)
    else:
        if proxy_variance is None or proxy_variance.shape != true_variance.shape:
            raise ValueError(f"{path_id} requires proxy variance [batch,6]")
        if bool((proxy_variance <= 0.0).any()):
            raise ValueError("proxy variance must be positive")
        fraction = float(eta)
        reserve = fraction * baseline_information
        remaining = (1.0 - fraction) * baseline_budget
        proxy_information = 0.5 * torch.log1p(
            proxy_variance * baseline_snr[:, None]
        )
        priority = proxy_variance * torch.exp(
            -2.0 * fraction * proxy_information
        )
        incremental = _torch_reverse_water_fill(priority, weights, remaining)
        information = reserve + incremental
        snr = torch.expm1(2.0 * information) / true_variance
        alpha = snr / (1.0 + snr)

    if not all(
        bool(torch.isfinite(value).all())
        for value in (alpha, snr, information, baseline_budget)
    ):
        raise FloatingPointError("allocated torch schedule is non-finite")
    if bool((alpha <= 0.0).any()) or bool((alpha >= 1.0).any()):
        raise FloatingPointError("allocated alpha_bar must lie strictly in (0,1)")
    if bool((snr <= 0.0).any()):
        raise FloatingPointError("allocated SNR must be strictly positive")
    return TorchScheduleBatch(
        alpha_bar=alpha,
        log_snr=torch.log(snr),
        information=information,
        baseline_budget=baseline_budget,
    )


def tiny_denoising_loss(
    system: TinyDenoisingSystem,
    projectors: ModeProjectorBank,
    residual: torch.Tensor,
    active_mask: torch.Tensor,
    condition: torch.Tensor,
    true_variance: torch.Tensor,
    weights: torch.Tensor,
    fixed_variance: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    baseline_alpha_bar: torch.Tensor,
    *,
    shuffle_proxy: torch.Tensor | None = None,
    eta: float = 0.5,
) -> TinyLossSample:
    """Compute one path's paired direct-x0 raw-residual training loss."""

    if residual.ndim != 3 or tuple(residual.shape[1:]) != (10, 24):
        raise ValueError("residual must have shape [batch,10,24]")
    if active_mask.shape != residual.shape or active_mask.dtype != torch.bool:
        raise ValueError("active mask must be boolean and align with residual")
    if noise.shape != residual.shape:
        raise ValueError("noise and residual must align")
    if fixed_variance.shape != (6,):
        raise ValueError("fixed variance must have shape [6]")
    if bool((~active_mask).all(dim=(1, 2)).any()):
        raise RuntimeError("a tiny-denoiser training day has no active cells")
    path_id = system.path_id
    proxy: torch.Tensor | None = None
    if path_id == "FIXED_BAND":
        proxy = fixed_variance[None].expand_as(true_variance)
    elif path_id == "MULAN_LITE":
        if system.scheduler is None:
            raise RuntimeError("MuLAN-lite system lost its scheduler")
        proxy = system.scheduler(condition, fixed_variance)
    elif path_id == "PA_RWF":
        proxy = true_variance
    elif path_id == "PA_SHUFFLE":
        if shuffle_proxy is None:
            raise ValueError("PA_SHUFFLE requires a wrong-day proxy")
        proxy = shuffle_proxy

    schedule = allocate_torch_schedule(
        path_id,
        true_variance,
        weights,
        timestep,
        baseline_alpha_bar,
        proxy_variance=proxy,
        eta=eta,
    )
    if path_id == "CW_GROUP":
        clean_internal = projectors.whiten(residual, true_variance)
        scalar_alpha = schedule.alpha_bar[:, 0, None, None]
        noisy = (
            scalar_alpha.sqrt() * clean_internal
            + (1.0 - scalar_alpha).sqrt() * noise
        )
        predicted_internal = system.denoiser(
            noisy, active_mask, condition, schedule.log_snr, timestep
        )
        prediction = projectors.unwhiten(predicted_internal, true_variance)
    else:
        residual_parts = projectors.project(residual)
        noise_parts = projectors.project(noise)
        alpha = schedule.alpha_bar[:, :, None, None]
        noisy = projectors.reconstruct(
            alpha.sqrt() * residual_parts + (1.0 - alpha).sqrt() * noise_parts
        )
        prediction = system.denoiser(
            noisy, active_mask, condition, schedule.log_snr, timestep
        )

    squared_error = torch.square(prediction - residual)
    active_float = active_mask.to(squared_error.dtype)
    loss = torch.sum(squared_error * active_float) / torch.sum(active_float)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("tiny-denoiser loss is non-finite")
    return TinyLossSample(loss, prediction, noisy, schedule)


class TinyEMA:
    """Exact checkpointable EMA over every trainable system parameter."""

    def __init__(self, system: TinyDenoisingSystem, *, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        self.parameter_names = tuple(
            name for name, parameter in system.named_parameters() if parameter.requires_grad
        )
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in system.named_parameters()
            if parameter.requires_grad
        }
        self.num_updates = 0
        self._validate(system)

    def _validate(self, system: TinyDenoisingSystem) -> None:
        parameters = dict(system.named_parameters())
        if set(parameters) != set(self.parameter_names):
            raise ValueError("EMA/system parameter topology mismatch")
        for name in self.parameter_names:
            value = parameters[name]
            shadow = self.shadow[name]
            if value.shape != shadow.shape or value.dtype != shadow.dtype:
                raise ValueError(f"EMA tensor metadata mismatch: {name}")
            if value.device != shadow.device:
                raise ValueError(f"EMA tensor device mismatch: {name}")
            if not bool(torch.isfinite(shadow).all()):
                raise FloatingPointError(f"EMA tensor is non-finite: {name}")

    @torch.no_grad()
    def update(self, system: TinyDenoisingSystem) -> None:
        self._validate(system)
        parameters = dict(system.named_parameters())
        for name in self.parameter_names:
            if not bool(torch.isfinite(parameters[name]).all()):
                raise FloatingPointError(f"online parameter is non-finite: {name}")
            self.shadow[name].mul_(self.decay).add_(
                parameters[name], alpha=1.0 - self.decay
            )
        self.num_updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "parameter_names": self.parameter_names,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(
        self, state: Mapping[str, object], system: TinyDenoisingSystem
    ) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay mismatch during resume")
        names = tuple(str(name) for name in state["parameter_names"])
        if names != self.parameter_names:
            raise ValueError("EMA parameter names mismatch during resume")
        raw_shadow = state["shadow"]
        if not isinstance(raw_shadow, Mapping):
            raise TypeError("EMA shadow checkpoint must be a mapping")
        self.shadow = {
            name: torch.as_tensor(raw_shadow[name]).detach().clone().to(
                dict(system.named_parameters())[name].device
            )
            for name in self.parameter_names
        }
        self.num_updates = int(state["num_updates"])
        self._validate(system)


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and raw little-endian CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())
