"""Exact-boundary state semantics for the architecture-v1 model family.

The continuous transport is defined only on ``INTERIOR_STATE`` coordinates.
This module owns the complementary responsibilities: estimating boundary-state
probabilities, turning those probabilities into finite ensemble state fields,
and reconstructing exact zeros/ones without inventing a continuous latent value
at an atom coordinate.

State codes deliberately match :mod:`mm_jdwind.data` so existing prepared
targets can be consumed without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ZERO_STATE = 0
INTERIOR_STATE = 1
ONE_STATE = 2
NUM_STATES = 3


def _validate_states(states: torch.Tensor) -> None:
    if states.ndim not in (3, 4):
        raise ValueError("states must have shape [B,Z,T] or [B,M,Z,T]")
    if states.dtype != torch.long:
        raise TypeError("states must use torch.long state codes")
    if bool(((states < ZERO_STATE) | (states > ONE_STATE)).any()):
        raise ValueError("states contain an unknown atom code")


@dataclass(frozen=True)
class AtomStatistics:
    """Analytic categorical state distribution for every zone-hour cell."""

    logits: torch.Tensor
    location_logit: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.logits.ndim != 4 or self.logits.shape[-1] != NUM_STATES:
            raise ValueError("atom logits must have shape [B,Z,T,3]")
        if not bool(torch.isfinite(self.logits).all()):
            raise ValueError("atom logits contain non-finite values")
        if self.location_logit is not None:
            if self.location_logit.shape != self.logits.shape[:-1]:
                raise ValueError("location logit must align with atom cells")
            if not bool(torch.isfinite(self.location_logit).all()):
                raise ValueError("location logit contains non-finite values")

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def zero_probability(self) -> torch.Tensor:
        return self.probabilities[..., ZERO_STATE]

    @property
    def interior_probability(self) -> torch.Tensor:
        return self.probabilities[..., INTERIOR_STATE]

    @property
    def one_probability(self) -> torch.Tensor:
        return self.probabilities[..., ONE_STATE]


@dataclass(frozen=True)
class AtomAllocation:
    """A finite-member state field and the mask seen by the continuous flow."""

    states: torch.Tensor
    active_mask: torch.Tensor
    analytic_probabilities: torch.Tensor
    realized_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        _validate_states(self.states)
        if self.states.ndim != 4:
            raise ValueError("allocated states must have shape [B,M,Z,T]")
        if self.active_mask.shape != self.states.shape:
            raise ValueError("active mask does not align with allocated states")
        if self.active_mask.dtype != torch.bool:
            raise TypeError("active mask must be boolean")
        if not torch.equal(self.active_mask, self.states == INTERIOR_STATE):
            raise ValueError("active mask must select exactly the interior state")
        expected = (self.states.shape[0], *self.states.shape[2:], NUM_STATES)
        if self.analytic_probabilities.shape != expected:
            raise ValueError("analytic probabilities do not align with states")
        if self.realized_probabilities.shape != expected:
            raise ValueError("realized probabilities do not align with states")


class AtomStateHead(nn.Module):
    """Low-capacity state head operating on the shared NWP representation."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        initial_probabilities: tuple[float, float, float] = (0.05, 0.949, 0.001),
        fixed_one_probability: float | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or feature_dim)
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("atom feature dimensions must be positive")
        prior = torch.as_tensor(initial_probabilities, dtype=torch.float32)
        if prior.shape != (NUM_STATES,) or bool((prior <= 0.0).any()):
            raise ValueError("initial atom probabilities must be three positive values")
        prior = prior / prior.sum()
        if fixed_one_probability is not None and not 0.0 < fixed_one_probability < 1.0:
            raise ValueError("fixed one probability must lie in (0,1)")
        self.fixed_one_probability = (
            None if fixed_one_probability is None else float(fixed_one_probability)
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.hidden = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        output_states = NUM_STATES if self.fixed_one_probability is None else 2
        self.output = nn.Linear(hidden_dim, output_states)
        self.location = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.location[-1].weight)
        nn.init.zeros_(self.location[-1].bias)
        with torch.no_grad():
            if self.fixed_one_probability is None:
                self.output.bias.copy_(prior.log())
            else:
                conditional = prior[:2] / prior[:2].sum()
                self.output.bias.copy_(conditional.log())

    def forward(self, encoded_nwp: torch.Tensor) -> AtomStatistics:
        if encoded_nwp.ndim != 4:
            raise ValueError("encoded NWP must have shape [B,Z,T,D]")
        raw = self.output(self.hidden(self.input_norm(encoded_nwp)))
        if self.fixed_one_probability is None:
            logits = raw
        else:
            conditional = torch.softmax(raw, dim=-1)
            fixed_one = torch.as_tensor(
                self.fixed_one_probability,
                dtype=conditional.dtype,
                device=conditional.device,
            )
            probabilities = torch.cat(
                (
                    (1.0 - fixed_one) * conditional,
                    torch.full_like(conditional[..., :1], fixed_one),
                ),
                dim=-1,
            )
            logits = probabilities.log()
        location_logit = self.location(encoded_nwp)[..., 0]
        return AtomStatistics(logits=logits, location_logit=location_logit)


class ExactAtomModule(nn.Module):
    """Complete atom head/allocation/mask/reconstruction boundary module.

    ``allocate`` uses balanced finite-member rounding.  Every cell receives the
    closest categorical counts whose sum is exactly ``members``.  A seeded,
    partially shared member priority turns those counts into joint state fields;
    this keeps member identity meaningful while making the procedure exactly
    reproducible.  It is an allocation mechanism, not a claim that the learned
    state-field dependence problem is already solved.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        initial_probabilities: tuple[float, float, float] = (0.05, 0.949, 0.001),
        shared_priority_weight: float = 0.75,
        fixed_one_probability: float | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= shared_priority_weight <= 1.0:
            raise ValueError("shared_priority_weight must lie in [0,1]")
        self.shared_priority_weight = float(shared_priority_weight)
        self.allocation_semantics = "balanced_shared_priority_control"
        self.head = AtomStateHead(
            feature_dim,
            hidden_dim=hidden_dim,
            initial_probabilities=initial_probabilities,
            fixed_one_probability=fixed_one_probability,
        )

    def forward(self, encoded_nwp: torch.Tensor) -> AtomStatistics:
        return self.head(encoded_nwp)

    @staticmethod
    def states_from_observation(observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 3:
            raise ValueError("observation must have shape [B,Z,T]")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains non-finite values")
        if bool(((observation < 0.0) | (observation > 1.0)).any()):
            raise ValueError("observation must lie in [0,1]")
        states = torch.full_like(observation, INTERIOR_STATE, dtype=torch.long)
        states = states.masked_fill(observation == 0.0, ZERO_STATE)
        states = states.masked_fill(observation == 1.0, ONE_STATE)
        return states

    @staticmethod
    def active_mask(
        states: torch.Tensor, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _validate_states(states)
        active = states == INTERIOR_STATE
        if observed_mask is not None:
            if observed_mask.shape != states.shape or observed_mask.dtype != torch.bool:
                raise ValueError(
                    "observed mask must be boolean and align with states"
                )
            active = active & observed_mask
        return active

    @staticmethod
    def loss(
        statistics: AtomStatistics,
        target_states: torch.Tensor,
        *,
        observed_mask: torch.Tensor,
        location_target: torch.Tensor | None = None,
        location_weight: float = 0.0,
        location_mean: float = 0.0,
        location_std: float = 1.0,
        location_logit_epsilon: float = 1e-4,
        location_smooth_l1_beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        _validate_states(target_states)
        if target_states.ndim != 3:
            raise ValueError("atom targets must have shape [B,Z,T]")
        if statistics.logits.shape[:-1] != target_states.shape:
            raise ValueError("atom targets do not align with logits")
        if observed_mask.shape != target_states.shape or observed_mask.dtype != torch.bool:
            raise ValueError(
                "observed mask must be boolean and align with atom targets"
            )
        if not bool(observed_mask.any()):
            raise ValueError("atom loss requires at least one observed target")
        if location_weight < 0.0:
            raise ValueError("location auxiliary weight must be non-negative")
        if location_std <= 0.0 or not torch.isfinite(torch.tensor(location_std)):
            raise ValueError("location auxiliary standard deviation must be positive")
        if not 0.0 < location_logit_epsilon < 0.5:
            raise ValueError("location logit epsilon must lie in (0,0.5)")
        if location_smooth_l1_beta <= 0.0:
            raise ValueError("location SmoothL1 beta must be positive")
        atom_nll = F.cross_entropy(
            statistics.logits[observed_mask], target_states[observed_mask]
        )
        zero_scalar = atom_nll * 0.0
        location_loss = zero_scalar
        interior_count = zero_scalar
        if location_target is not None:
            if location_target.shape != target_states.shape:
                raise ValueError("location target does not align with atom targets")
            if not bool(torch.isfinite(location_target).all()):
                raise ValueError("location target contains non-finite values")
            interior = observed_mask & (target_states == INTERIOR_STATE)
            interior_count = interior.float().sum()
            if location_weight > 0.0 and not bool(interior.any()):
                raise ValueError("location auxiliary requires observed interior targets")
            if bool(interior.any()):
                assert statistics.location_logit is not None
                safe = location_target[interior].clamp(
                    location_logit_epsilon, 1.0 - location_logit_epsilon
                )
                normalized = (torch.logit(safe) - location_mean) / location_std
                location_loss = F.smooth_l1_loss(
                    statistics.location_logit[interior],
                    normalized,
                    beta=location_smooth_l1_beta,
                )
        elif location_weight > 0.0:
            raise ValueError("positive location weight requires location_target")
        loss = atom_nll + float(location_weight) * location_loss
        prediction = statistics.logits.argmax(dim=-1)
        observed_float = observed_mask.float()
        observed_count = observed_float.sum().clamp_min(1.0)
        correct = ((prediction == target_states) & observed_mask).float().sum()
        zero_error = (
            statistics.zero_probability
            - (target_states == ZERO_STATE).float()
        ).square()
        return {
            "loss": loss,
            "atom_nll": atom_nll,
            "interior_location_smooth_l1": location_loss,
            "interior_location_count": interior_count,
            "accuracy": correct / observed_count,
            "zero_brier": (zero_error * observed_float).sum() / observed_count,
            "observed_fraction": observed_float.mean(),
        }

    @staticmethod
    def _balanced_counts(probabilities: torch.Tensor, members: int) -> torch.Tensor:
        """Largest-remainder categorical counts, vectorized over all cells."""

        raw = probabilities * members
        counts = torch.floor(raw).to(torch.long)
        flat_counts = counts.reshape(-1, NUM_STATES)
        flat_fraction = (raw - counts).reshape(-1, NUM_STATES)
        remainder = members - flat_counts.sum(dim=-1)
        if bool(((remainder < 0) | (remainder >= NUM_STATES)).any()):
            raise RuntimeError("categorical rounding produced an invalid remainder")
        order = torch.argsort(flat_fraction, dim=-1, descending=True, stable=True)
        rows = torch.arange(len(flat_counts), device=probabilities.device)
        for offset in range(NUM_STATES - 1):
            selected = remainder > offset
            if bool(selected.any()):
                flat_counts[
                    rows[selected], order[selected, offset]
                ] += 1
        return flat_counts.reshape_as(counts)

    @torch.no_grad()
    def allocate(
        self,
        statistics: AtomStatistics,
        *,
        members: int,
        seed: int,
    ) -> AtomAllocation:
        if members < 1:
            raise ValueError("members must be positive")
        probabilities = statistics.probabilities
        batch, zones, hours, _ = probabilities.shape
        counts = self._balanced_counts(probabilities, members)
        generator = torch.Generator(device=probabilities.device)
        generator.manual_seed(int(seed))

        # A shared priority preserves joint member identity; cell-specific jitter
        # prevents every zone-hour from receiving an identical state ordering.
        shared = torch.rand(
            batch,
            members,
            1,
            1,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        )
        local = torch.rand(
            batch,
            members,
            zones,
            hours,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        )
        weight = self.shared_priority_weight
        priority = weight * shared + (1.0 - weight) * local
        order = torch.argsort(priority, dim=1, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            members, dtype=order.dtype, device=order.device
        ).view(1, members, 1, 1).expand_as(order)
        ranks.scatter_(1, order, rank_values)

        zero_count = counts[..., ZERO_STATE][:, None]
        one_count = counts[..., ONE_STATE][:, None]
        states = torch.full_like(ranks, INTERIOR_STATE, dtype=torch.long)
        states = states.masked_fill(ranks < zero_count, ZERO_STATE)
        states = states.masked_fill(ranks >= members - one_count, ONE_STATE)
        active = states == INTERIOR_STATE
        realized = F.one_hot(states, num_classes=NUM_STATES).float().mean(dim=1)
        return AtomAllocation(
            states=states,
            active_mask=active,
            analytic_probabilities=probabilities,
            realized_probabilities=realized,
        )

    @staticmethod
    def reconstruct(interior_latent: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Map interior logits to power while retaining exact boundary atoms."""

        _validate_states(states)
        if interior_latent.shape != states.shape:
            raise ValueError("interior latent does not align with atom states")
        continuous = torch.sigmoid(interior_latent)
        values = torch.where(
            states == ZERO_STATE,
            torch.zeros_like(continuous),
            continuous,
        )
        return torch.where(
            states == ONE_STATE,
            torch.ones_like(values),
            values,
        )

    def extra_repr(self) -> str:
        return (
            f"allocation_semantics={self.allocation_semantics}, "
            f"shared_priority_weight={self.shared_priority_weight:g}"
        )


__all__ = [
    "AtomAllocation",
    "AtomStateHead",
    "AtomStatistics",
    "ExactAtomModule",
    "INTERIOR_STATE",
    "NUM_STATES",
    "ONE_STATE",
    "ZERO_STATE",
]
