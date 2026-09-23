"""Mask-conditioned reversible coordinates for the TGO-v1 Probe.

The operators in this module act independently on every zone and on every
maximal contiguous run of active hours.  Missing cells and exact boundary
atoms are therefore never converted into artificial zero-valued neighbours.

For an active mask ``M`` and level coordinate ``u`` the registered true
transition operator is ``z = B_M u``.  ``B_M`` stores one anchor followed by
first differences inside each active run.  The inverse is a segmentwise
cumulative sum.  A frozen wrong-adjacency operator and a segmentwise
orthonormal DCT-II provide the two representation controls.

The module deliberately exposes the pullback identities used by the matched
metric and matched noise controls.  It does not load data, fit a model, or
decide which target role may be accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np
import torch


OperatorKind = Literal[
    "identity",
    "transition_true",
    "transition_wrong",
    "orthogonal_dct",
]

OPERATOR_KINDS: tuple[OperatorKind, ...] = (
    "identity",
    "transition_true",
    "transition_wrong",
    "orthogonal_dct",
)


@dataclass(frozen=True)
class WrongAdjacencyAudit:
    """Aggregate edge-destruction accounting for the frozen wrong order."""

    eligible_segments: int
    eligible_edges: int
    retained_true_edges: int

    @property
    def retained_true_edge_fraction(self) -> float:
        if self.eligible_edges == 0:
            return 0.0
        return self.retained_true_edges / self.eligible_edges


def _validate_kind(kind: str) -> OperatorKind:
    if kind not in OPERATOR_KINDS:
        raise ValueError(f"unknown transition-object operator: {kind}")
    return kind  # type: ignore[return-value]


def _active_runs(active: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("one-dimensional active mask required")
    padded = np.concatenate(
        [np.asarray([False]), mask, np.asarray([False])]
    )
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        np.arange(start, stop, dtype=np.int64)
        for start, stop in changes.reshape(-1, 2)
    )


def wrong_adjacency_order(length: int) -> tuple[int, ...]:
    """Return the frozen outside-in order ``0,L-1,1,L-2,...``."""

    if length < 1:
        raise ValueError("segment length must be positive")
    left = 0
    right = length - 1
    order: list[int] = []
    while left <= right:
        order.append(left)
        left += 1
        if left <= right:
            order.append(right)
            right -= 1
    return tuple(order)


def _dct_ii_matrix(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("DCT length must be positive")
    position = np.arange(length, dtype=np.float64)[None, :]
    frequency = np.arange(length, dtype=np.float64)[:, None]
    matrix = np.sqrt(2.0 / length) * np.cos(
        math.pi * (position + 0.5) * frequency / length
    )
    matrix[0] = math.sqrt(1.0 / length)
    return matrix


def _decode_mask(code: int, hours: int) -> np.ndarray:
    if code < 0 or code >= 1 << hours:
        raise ValueError("active-mask code lies outside the hour dimension")
    return np.asarray([(code >> index) & 1 for index in range(hours)], dtype=bool)


@lru_cache(maxsize=16384)
def _base_matrix(
    kind: OperatorKind,
    mask_code: int,
    hours: int,
    inverse: bool,
) -> np.ndarray:
    """Return an unscaled operator matrix for one zone mask.

    Inactive rows and columns are exactly zero.  The inverse is an inverse only
    on the active subspace, which is the intended continuous-target domain.
    """

    _validate_kind(kind)
    active = _decode_mask(mask_code, hours)
    matrix = np.zeros((hours, hours), dtype=np.float64)
    for run in _active_runs(active):
        if kind == "identity":
            matrix[run, run] = 1.0
            continue
        if kind == "orthogonal_dct":
            dct = _dct_ii_matrix(len(run))
            matrix[np.ix_(run, run)] = dct.T if inverse else dct
            continue
        if kind == "transition_true":
            order = run
        elif kind == "transition_wrong":
            order = run[np.asarray(wrong_adjacency_order(len(run)), dtype=np.int64)]
        else:  # pragma: no cover - _validate_kind makes this unreachable.
            raise AssertionError(kind)

        if inverse:
            for position, node in enumerate(order):
                matrix[node, order[: position + 1]] = 1.0
        else:
            matrix[order[0], order[0]] = 1.0
            for position in range(1, len(order)):
                node = order[position]
                previous = order[position - 1]
                matrix[node, node] = 1.0
                matrix[node, previous] = -1.0
    matrix.setflags(write=False)
    return matrix


def _mask_codes(mask: torch.Tensor) -> np.ndarray:
    hours = mask.shape[-1]
    if hours > 62:
        raise ValueError("bit-coded masks support at most 62 hours")
    # Matrix construction and the cached bit codes are CPU-side metadata.
    # Moving the small boolean mask before integer arithmetic also avoids the
    # unsupported CUDA int64 matmul kernel on some PyTorch builds.
    flat = mask.detach().reshape(-1, hours).to(device="cpu", dtype=torch.int64)
    powers = torch.bitwise_left_shift(
        torch.ones(hours, dtype=torch.int64),
        torch.arange(hours, dtype=torch.int64),
    )
    return torch.matmul(flat, powers).numpy()


def _validate_value_and_mask(
    value: torch.Tensor, active_mask: torch.Tensor
) -> tuple[int, int]:
    if not torch.is_floating_point(value):
        raise TypeError("operator value must be floating point")
    if value.ndim < 2:
        raise ValueError("operator value must end in [zone,hour]")
    if active_mask.shape != value.shape or active_mask.dtype != torch.bool:
        raise ValueError("active mask must be boolean and align with value")
    if value.shape[-1] < 1 or value.shape[-2] < 1:
        raise ValueError("zone and hour dimensions must be positive")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("operator value contains non-finite entries")
    return int(value.shape[-2]), int(value.shape[-1])


class MaskConditionedOperator:
    """A reversible linear coordinate map on mask-selected active runs."""

    def __init__(self, kind: OperatorKind, *, scale: float = 1.0) -> None:
        self.kind = _validate_kind(str(kind))
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("operator scale must be finite and positive")

    def matrix(
        self,
        active_mask: torch.Tensor,
        *,
        inverse: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build matrices with shape ``[...,zone,hour,hour]``."""

        if active_mask.ndim < 2 or active_mask.dtype != torch.bool:
            raise ValueError("active mask must be boolean and end in [zone,hour]")
        zones = int(active_mask.shape[-2])
        hours = int(active_mask.shape[-1])
        codes = _mask_codes(active_mask)
        arrays = [
            _base_matrix(self.kind, int(code), hours, bool(inverse))
            for code in codes
        ]
        if arrays:
            stacked = np.stack(arrays, axis=0)
        else:  # pragma: no cover - tensors cannot have a zero zone dimension.
            stacked = np.empty((0, hours, hours), dtype=np.float64)
        result = torch.as_tensor(stacked.copy(), dtype=dtype, device=active_mask.device)
        leading = active_mask.shape[:-2]
        return result.reshape(*leading, zones, hours, hours)

    def transform(
        self, value: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        zones, hours = _validate_value_and_mask(value, active_mask)
        flat_value = value.reshape(-1, hours)
        matrix = self.matrix(
            active_mask, inverse=False, dtype=value.dtype
        ).reshape(-1, hours, hours)
        if len(matrix) != len(flat_value):
            raise RuntimeError("operator matrix/value batch mismatch")
        result = torch.bmm(matrix, flat_value[..., None]).squeeze(-1)
        result = (self.scale * result).reshape_as(value)
        result = torch.where(active_mask, result, torch.zeros_like(result))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("operator transform produced non-finite values")
        if tuple(result.shape[-2:]) != (zones, hours):
            raise RuntimeError("operator transform changed the zone/hour layout")
        return result

    def inverse(
        self, coefficient: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        zones, hours = _validate_value_and_mask(coefficient, active_mask)
        flat = coefficient.reshape(-1, hours) / self.scale
        matrix = self.matrix(
            active_mask, inverse=True, dtype=coefficient.dtype
        ).reshape(-1, hours, hours)
        if len(matrix) != len(flat):
            raise RuntimeError("inverse operator matrix/value batch mismatch")
        result = torch.bmm(matrix, flat[..., None]).squeeze(-1).reshape_as(coefficient)
        result = torch.where(active_mask, result, torch.zeros_like(result))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("operator inverse produced non-finite values")
        if tuple(result.shape[-2:]) != (zones, hours):
            raise RuntimeError("operator inverse changed the zone/hour layout")
        return result

    def induced_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must align")
        error = self.transform(prediction - target, active_mask)
        count = active_mask.sum()
        if int(count) == 0:
            raise ValueError("induced MSE requires active coordinates")
        return error.square().sum() / count.to(error.dtype)


def fit_level_rms(value: torch.Tensor, active_mask: torch.Tensor) -> float:
    """Fit the frozen scalar ``s_r`` from active outer-training values."""

    _validate_value_and_mask(value, active_mask)
    count = active_mask.sum()
    if int(count) == 0:
        raise ValueError("level RMS requires active coordinates")
    squared = torch.where(active_mask, value.square(), torch.zeros_like(value))
    rms = torch.sqrt(squared.sum(dtype=torch.float64) / count.to(torch.float64))
    result = float(rms.detach().cpu())
    if not math.isfinite(result) or result <= 0.0:
        raise FloatingPointError("level RMS is not finite and positive")
    return result


def fit_operator_scale(
    level_value: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    kind: OperatorKind,
) -> float:
    """Scale an operator so its active coefficient RMS matches level RMS."""

    _validate_value_and_mask(level_value, active_mask)
    base = MaskConditionedOperator(kind, scale=1.0)
    transformed = base.transform(level_value, active_mask)
    level_rms = fit_level_rms(level_value, active_mask)
    coefficient_rms = fit_level_rms(transformed, active_mask)
    result = level_rms / coefficient_rms
    if not math.isfinite(result) or result <= 0.0:
        raise FloatingPointError("fitted operator scale is invalid")
    return result


def wrong_adjacency_audit(active_mask: torch.Tensor) -> WrongAdjacencyAudit:
    """Count physical edges retained by the frozen wrong-adjacency order."""

    if active_mask.ndim < 2 or active_mask.dtype != torch.bool:
        raise ValueError("active mask must be boolean and end in [zone,hour]")
    flat = active_mask.detach().cpu().numpy().reshape(-1, active_mask.shape[-1])
    segments = 0
    edges = 0
    retained = 0
    for row in flat:
        for run in _active_runs(row):
            if len(run) < 3:
                continue
            order = run[np.asarray(wrong_adjacency_order(len(run)), dtype=np.int64)]
            segments += 1
            edges += len(order) - 1
            retained += int(np.sum(np.abs(np.diff(order)) == 1))
    return WrongAdjacencyAudit(segments, edges, retained)


def _schedule_factor(alpha_bar: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    expected = value.shape[:-2]
    if alpha_bar.shape != expected:
        raise ValueError(
            f"alpha_bar must have leading shape {tuple(expected)}, got {tuple(alpha_bar.shape)}"
        )
    if not torch.is_floating_point(alpha_bar):
        raise TypeError("alpha_bar must be floating point")
    if bool(((alpha_bar <= 0.0) | (alpha_bar >= 1.0)).any()):
        raise ValueError("alpha_bar must lie strictly in (0,1)")
    return alpha_bar.to(dtype=value.dtype, device=value.device)[..., None, None]


def iid_forward_sample(
    clean: torch.Tensor,
    active_mask: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Scalar-schedule Gaussian corruption in the supplied coordinate."""

    _validate_value_and_mask(clean, active_mask)
    if noise.shape != clean.shape or not torch.is_floating_point(noise):
        raise ValueError("noise must be floating point and align with clean")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise contains non-finite entries")
    alpha = _schedule_factor(alpha_bar, clean)
    result = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
    return torch.where(active_mask, result, torch.zeros_like(result))


def transition_native_forward(
    level_clean: torch.Tensor,
    active_mask: torch.Tensor,
    alpha_bar: torch.Tensor,
    native_noise: torch.Tensor,
    operator: MaskConditionedOperator,
) -> torch.Tensor:
    """Corrupt ``B_M u`` with IID noise in the transition coordinate."""

    clean = operator.transform(level_clean, active_mask)
    return iid_forward_sample(clean, active_mask, alpha_bar, native_noise)


def matched_noise_level_forward(
    level_clean: torch.Tensor,
    active_mask: torch.Tensor,
    alpha_bar: torch.Tensor,
    native_noise: torch.Tensor,
    operator: MaskConditionedOperator,
) -> torch.Tensor:
    """Corrupt level coordinates with the exact ``B_M^{-1}`` noise law."""

    pulled_noise = operator.inverse(native_noise, active_mask)
    return iid_forward_sample(level_clean, active_mask, alpha_bar, pulled_noise)


def ddim_direct_x0_step(
    noisy: torch.Tensor,
    predicted_clean: torch.Tensor,
    active_mask: torch.Tensor,
    alpha_current: torch.Tensor,
    alpha_previous: torch.Tensor,
) -> torch.Tensor:
    """One deterministic direct-x0 DDIM step in an arbitrary coordinate."""

    _validate_value_and_mask(noisy, active_mask)
    if predicted_clean.shape != noisy.shape:
        raise ValueError("predicted clean coordinate must align with noisy input")
    current = _schedule_factor(alpha_current, noisy)
    previous = _schedule_factor(alpha_previous, noisy)
    predicted_noise = (
        noisy - current.sqrt() * predicted_clean
    ) / (1.0 - current).sqrt()
    result = previous.sqrt() * predicted_clean + (1.0 - previous).sqrt() * predicted_noise
    result = torch.where(active_mask, result, torch.zeros_like(result))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("DDIM step produced non-finite values")
    return result


__all__ = [
    "MaskConditionedOperator",
    "OPERATOR_KINDS",
    "OperatorKind",
    "WrongAdjacencyAudit",
    "ddim_direct_x0_step",
    "fit_level_rms",
    "fit_operator_scale",
    "iid_forward_sample",
    "matched_noise_level_forward",
    "transition_native_forward",
    "wrong_adjacency_audit",
    "wrong_adjacency_order",
]
