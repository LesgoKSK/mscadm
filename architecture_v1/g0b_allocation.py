"""Schedule-only allocation utilities for the G0-B0 feasibility audit.

This module contains no denoiser, optimizer, dataset loader, or target access.
It treats the six G0-A mode-group variance forecasts as diagonal Gaussian
reference variances and reallocates the information of an IID VP channel.

The primary construction reserves a fixed fraction of every mode's IID
Gaussian mutual information and reverse-water-fills only the remainder.  It is
therefore bounded between two auditable endpoints:

``eta = 1``
    Exact IID information allocation.

``eta = 0``
    Unfloored reverse water-filling, retained only as a theoretical oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InformationAllocation:
    """One batched reserve-then-water-fill allocation."""

    eta: float
    baseline_information: np.ndarray
    reserved_information: np.ndarray
    incremental_information: np.ndarray
    information: np.ndarray
    snr: np.ndarray
    alpha_bar: np.ndarray
    log_water_level: np.ndarray
    baseline_budget: np.ndarray
    allocated_budget: np.ndarray
    budget_error: np.ndarray


def _validated_mode_matrices(
    variance: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(variance, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 2:
        raise ValueError("variance must have shape [batch,mode] with at least two modes")
    if weight.shape != value.shape:
        raise ValueError("weights must have the same [batch,mode] shape as variance")
    if not np.isfinite(value).all() or not np.isfinite(weight).all():
        raise FloatingPointError("variance and weights must be finite")
    if np.any(value <= 0.0):
        raise ValueError("every Gaussian reference variance must be strictly positive")
    if np.any(weight <= 0.0):
        raise ValueError("every information weight must be strictly positive")
    totals = weight.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("information weights must sum to one on every batch row")
    return value, weight


def gaussian_information(
    variance: np.ndarray,
    snr: np.ndarray | float,
) -> np.ndarray:
    """Return ``0.5 * log(1 + variance * nominal_snr)`` in nats."""

    value = np.asarray(variance, dtype=np.float64)
    ratio = np.asarray(snr, dtype=np.float64)
    if not np.isfinite(value).all() or not np.isfinite(ratio).all():
        raise FloatingPointError("variance and SNR must be finite")
    if np.any(value <= 0.0) or np.any(ratio < 0.0):
        raise ValueError("variance must be positive and SNR must be non-negative")
    information = 0.5 * np.log1p(value * ratio)
    if not np.isfinite(information).all():
        raise FloatingPointError("Gaussian information became non-finite")
    return information


def gaussian_posterior_variance(
    variance: np.ndarray,
    snr: np.ndarray | float,
) -> np.ndarray:
    """Return the scalar Gaussian posterior variance for a noisy channel."""

    value = np.asarray(variance, dtype=np.float64)
    ratio = np.asarray(snr, dtype=np.float64)
    if not np.isfinite(value).all() or not np.isfinite(ratio).all():
        raise FloatingPointError("variance and SNR must be finite")
    if np.any(value <= 0.0) or np.any(ratio < 0.0):
        raise ValueError("variance must be positive and SNR must be non-negative")
    posterior = value / (1.0 + value * ratio)
    if not np.isfinite(posterior).all() or np.any(posterior <= 0.0):
        raise FloatingPointError("Gaussian posterior variance is invalid")
    return posterior


def snr_from_information(
    variance: np.ndarray,
    information: np.ndarray,
) -> np.ndarray:
    """Invert Gaussian channel information to nominal SNR."""

    value = np.asarray(variance, dtype=np.float64)
    amount = np.asarray(information, dtype=np.float64)
    if value.shape != amount.shape:
        raise ValueError("variance and information shapes must match")
    if not np.isfinite(value).all() or not np.isfinite(amount).all():
        raise FloatingPointError("variance and information must be finite")
    if np.any(value <= 0.0) or np.any(amount < 0.0):
        raise ValueError("variance must be positive and information non-negative")
    snr = np.expm1(2.0 * amount) / value
    if not np.isfinite(snr).all() or np.any(snr < 0.0):
        raise FloatingPointError("information-to-SNR inversion failed")
    return snr


def weighted_information_budget(
    information: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the dimension-weighted information on each batch row."""

    amount = np.asarray(information, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if amount.ndim != 2 or amount.shape != weight.shape:
        raise ValueError("information and weights must share shape [batch,mode]")
    return np.sum(weight * amount, axis=1)


def reverse_water_fill_information(
    effective_variance: np.ndarray,
    weights: np.ndarray,
    budget: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the diagonal Gaussian reverse-water-filling problem exactly.

    The optimization is

    ``min sum_k w_k a_k exp(-2 J_k)``

    subject to ``J_k >= 0`` and ``sum_k w_k J_k = budget``.  The water level
    is solved in log space by enumerating the at-most-six active sets rather
    than by an iterative optimizer.
    """

    value, weight = _validated_mode_matrices(effective_variance, weights)
    target = np.asarray(budget, dtype=np.float64)
    if target.shape != (len(value),):
        raise ValueError("budget must have shape [batch]")
    if not np.isfinite(target).all():
        raise FloatingPointError("information budget must be finite")
    if np.any(target < -tolerance):
        raise ValueError("information budget cannot be negative")
    target = np.maximum(target, 0.0)

    log_value = np.log(value)
    information = np.zeros_like(value)
    log_water_level = np.empty(len(value), dtype=np.float64)

    for row in range(len(value)):
        if target[row] <= tolerance:
            log_water_level[row] = float(np.max(log_value[row]))
            continue

        order = np.argsort(-log_value[row], kind="mergesort")
        ordered_log = log_value[row, order]
        ordered_weight = weight[row, order]
        cumulative_weight = np.cumsum(ordered_weight)
        cumulative_weighted_log = np.cumsum(ordered_weight * ordered_log)
        selected_level: float | None = None
        for active in range(1, value.shape[1] + 1):
            candidate = (
                cumulative_weighted_log[active - 1] - 2.0 * target[row]
            ) / cumulative_weight[active - 1]
            next_log = (
                ordered_log[active]
                if active < value.shape[1]
                else -np.inf
            )
            if candidate >= next_log - tolerance:
                selected_level = float(candidate)
                break
        if selected_level is None:
            raise RuntimeError("reverse-water-filling active-set search failed")
        log_water_level[row] = selected_level
        information[row] = 0.5 * np.maximum(
            log_value[row] - selected_level,
            0.0,
        )

    realized = weighted_information_budget(information, weight)
    error = realized - target
    bound = tolerance * (1.0 + np.abs(target))
    if np.any(np.abs(error) > bound):
        raise RuntimeError("reverse-water-filling did not conserve its budget")
    return information, log_water_level


def reserve_then_water_fill(
    variance: np.ndarray,
    weights: np.ndarray,
    baseline_snr: np.ndarray | float,
    *,
    eta: float,
    tolerance: float = 1e-12,
) -> InformationAllocation:
    """Reserve an IID information fraction, then water-fill the remainder."""

    value, weight = _validated_mode_matrices(variance, weights)
    fraction = float(eta)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("eta must be finite and lie in [0,1]")
    ratio = np.asarray(baseline_snr, dtype=np.float64)
    if ratio.ndim == 0:
        ratio = np.full(len(value), float(ratio), dtype=np.float64)
    if ratio.shape != (len(value),):
        raise ValueError("baseline_snr must be scalar or have shape [batch]")
    if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise ValueError("G0-B0 baseline SNR must be finite and strictly positive")

    baseline_information = gaussian_information(value, ratio[:, None])
    baseline_budget = weighted_information_budget(baseline_information, weight)
    reserved = fraction * baseline_information
    remaining_budget = (1.0 - fraction) * baseline_budget
    effective_variance = value * np.exp(-2.0 * reserved)
    incremental, log_water_level = reverse_water_fill_information(
        effective_variance,
        weight,
        remaining_budget,
        tolerance=tolerance,
    )
    information = reserved + incremental
    snr = snr_from_information(value, information)
    alpha_bar = snr / (1.0 + snr)
    allocated_budget = weighted_information_budget(information, weight)
    budget_error = allocated_budget - baseline_budget

    if not np.isfinite(alpha_bar).all() or np.any(alpha_bar < 0.0):
        raise FloatingPointError("allocated VP alpha_bar is invalid")
    if np.any(alpha_bar >= 1.0):
        raise FloatingPointError("allocated VP alpha_bar must stay below one")
    bound = tolerance * (1.0 + np.abs(baseline_budget))
    if np.any(np.abs(budget_error) > bound):
        raise RuntimeError("reserve-then-water-fill changed the information budget")

    return InformationAllocation(
        eta=fraction,
        baseline_information=baseline_information,
        reserved_information=reserved,
        incremental_information=incremental,
        information=information,
        snr=snr,
        alpha_bar=alpha_bar,
        log_water_level=log_water_level,
        baseline_budget=baseline_budget,
        allocated_budget=allocated_budget,
        budget_error=budget_error,
    )


__all__ = [
    "InformationAllocation",
    "gaussian_information",
    "gaussian_posterior_variance",
    "reserve_then_water_fill",
    "reverse_water_fill_information",
    "snr_from_information",
    "weighted_information_budget",
]
