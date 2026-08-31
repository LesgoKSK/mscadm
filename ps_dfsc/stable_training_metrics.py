"""Finite-gradient surrogates for proper-score training constraints.

Validation and test metrics remain the exact discrete definitions in
``ps_dfsc.metrics``.  These functions only smooth the non-differentiable
zero-distance points that otherwise produce NaN gradients for duplicated
scenarios and exact zero atoms.
"""

from __future__ import annotations

import torch


SMOOTHING_EPSILON = 1.0e-6
SURROGATE_SCHEMA = "ps_dfsc_finite_gradient_scores_v1"


def smooth_energy_score(
    scenarios: torch.Tensor,
    probability: torch.Tensor,
    truth: torch.Tensor,
    *,
    epsilon: float = SMOOTHING_EPSILON,
) -> torch.Tensor:
    flat = scenarios.flatten(2)
    observed = truth.flatten(1)

    def smooth_norm(value: torch.Tensor) -> torch.Tensor:
        return (
            value.square().sum(dim=-1).add(epsilon**2).sqrt()
            - epsilon
        )

    first = (
        probability * smooth_norm(flat - observed[:, None])
    ).sum(dim=1)
    pairwise = smooth_norm(flat[:, :, None] - flat[:, None, :])
    second = torch.einsum(
        "bi,bj,bij->b", probability, probability, pairwise
    )
    return (first - 0.5 * second).mean()


def smooth_variogram_score(
    scenarios: torch.Tensor,
    probability: torch.Tensor,
    truth: torch.Tensor,
    *,
    epsilon: float = SMOOTHING_EPSILON,
) -> torch.Tensor:
    def smooth_half_power(value: torch.Tensor) -> torch.Tensor:
        return (
            (value.square() + epsilon**2).pow(0.25)
            - epsilon**0.5
        )

    temporal_truth = smooth_half_power(
        truth[:, :, 1:] - truth[:, :, :-1]
    )
    temporal_sample = smooth_half_power(
        scenarios[:, :, :, 1:] - scenarios[:, :, :, :-1]
    )
    temporal_expected = torch.einsum(
        "bm,bmzt->bzt", probability, temporal_sample
    )
    spatial_truth = smooth_half_power(
        truth[:, 1:] - truth[:, :-1]
    )
    spatial_sample = smooth_half_power(
        scenarios[:, :, 1:] - scenarios[:, :, :-1]
    )
    spatial_expected = torch.einsum(
        "bm,bmzt->bzt", probability, spatial_sample
    )
    return (
        (temporal_truth - temporal_expected).square().mean()
        + (spatial_truth - spatial_expected).square().mean()
    )


__all__ = [
    "SMOOTHING_EPSILON",
    "SURROGATE_SCHEMA",
    "smooth_energy_score",
    "smooth_variogram_score",
]
