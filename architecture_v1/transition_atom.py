"""Shared train-only atom nuisance model for the TGO-v1 comparison.

Every continuous-coordinate path must receive exactly the same sampled
zero/interior/one states.  This module contains the architecture-v1 NWP
encoder and exact atom head only; it deliberately has no continuous transport
network.  Target-role boundaries and fitting loops belong to the runner.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .atom import AtomAllocation, AtomStatistics, ExactAtomModule
from .model import JointNWPEncoder


class TransitionAtomNuisance(nn.Module):
    """The shared architecture-v1 E/A topology without a transport model."""

    def __init__(
        self,
        *,
        fixed_one_probability: float | None,
        location_mean: float,
        location_std: float,
    ) -> None:
        super().__init__()
        if fixed_one_probability is not None and not 0.0 < fixed_one_probability < 1.0:
            raise ValueError("fixed one probability must lie in (0,1)")
        if not math.isfinite(location_mean):
            raise ValueError("location mean must be finite")
        if not math.isfinite(location_std) or location_std <= 0.0:
            raise ValueError("location std must be finite and positive")
        self.fixed_one_probability = fixed_one_probability
        self.location_mean = float(location_mean)
        self.location_std = float(location_std)
        self.encoder = JointNWPEncoder(
            20,
            model_dim=64,
            depth=2,
            heads=4,
            ff_multiplier=4,
            dropout=0.0,
            zones=10,
            hours=24,
        )
        self.atom = ExactAtomModule(
            64,
            hidden_dim=64,
            shared_priority_weight=0.5,
            fixed_one_probability=fixed_one_probability,
        )

    def encode(self, condition: torch.Tensor) -> torch.Tensor:
        return self.encoder(condition)

    def forward(self, condition: torch.Tensor) -> AtomStatistics:
        return self.atom(self.encode(condition))

    def loss(
        self,
        condition: torch.Tensor,
        observation: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observation.shape != (len(condition), 10, 24):
            raise ValueError("atom observation must have shape [batch,10,24]")
        if observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool:
            raise ValueError("observed mask must be boolean and align with observation")
        states = self.atom.states_from_observation(observation)
        return self.atom.loss(
            self(condition),
            states,
            observed_mask=observed_mask,
            location_target=observation,
            location_weight=0.25,
            location_mean=self.location_mean,
            location_std=self.location_std,
            location_logit_epsilon=1e-4,
            location_smooth_l1_beta=1.0,
        )

    @torch.no_grad()
    def allocate(
        self,
        condition: torch.Tensor,
        *,
        members: int,
        seed: int,
    ) -> tuple[AtomStatistics, AtomAllocation]:
        statistics = self(condition)
        allocation = self.atom.allocate(statistics, members=members, seed=seed)
        return statistics, allocation


def train_only_atom_contract(
    observation: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    one_support_minimum: int = 20,
) -> dict[str, float | int | bool | None]:
    """Return train-only upper-atom and interior-logit nuisance constants."""

    if observation.ndim != 3 or tuple(observation.shape[1:]) != (10, 24):
        raise ValueError("observation must have shape [day,10,24]")
    if observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool:
        raise ValueError("observed mask must be boolean and align with observation")
    if one_support_minimum < 1:
        raise ValueError("one support minimum must be positive")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("observation contains non-finite entries")
    if bool(((observation < 0.0) | (observation > 1.0)).any()):
        raise ValueError("observation must lie in [0,1]")
    observed_count = int(observed_mask.sum())
    if observed_count == 0:
        raise ValueError("atom contract requires observed targets")
    one_count = int((observed_mask & (observation == 1.0)).sum())
    fixed = one_count < int(one_support_minimum)
    fixed_probability = (
        (one_count + 0.5) / (observed_count + 1.0) if fixed else None
    )
    interior = observed_mask & (observation > 0.0) & (observation < 1.0)
    if not bool(interior.any()):
        raise ValueError("atom contract requires observed interior targets")
    latent = torch.logit(observation[interior].clamp(1e-4, 1.0 - 1e-4)).double()
    location_mean = float(latent.mean())
    location_std = float(latent.std(unbiased=False))
    if not math.isfinite(location_std) or location_std <= 0.0:
        raise FloatingPointError("interior logit standard deviation is invalid")
    return {
        "observed_count": observed_count,
        "one_count": one_count,
        "one_support_minimum": int(one_support_minimum),
        "fixed_one_probability": fixed_probability,
        "upper_atom_fixed": fixed,
        "location_mean": location_mean,
        "location_std": location_std,
    }


__all__ = ["TransitionAtomNuisance", "train_only_atom_contract"]
