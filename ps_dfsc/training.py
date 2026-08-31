from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from repro.suc import rts24

from .differentiable_suc import DifferentiableSUC, commitment_transitions
from .mapping import WindFarmMapping
from .model import PSDFSCNetwork


@dataclass(frozen=True)
class TrainingConfig:
    beta: float = 0.25
    epochs: int = 50
    proper_warmup_epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    dual_learning_rate: float = 0.1
    augmented_penalty: float = 10.0
    strong_convexity: float = 1e-4
    ess_floor: float = 50.0
    entropy_fraction_floor: float = 0.85
    transport_budget: float = 0.02
    seed: int = 0


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)


class AugmentedLagrangian:
    def __init__(self, names: tuple[str, ...], *, penalty: float, dual_lr: float):
        self.penalty = float(penalty)
        self.dual_lr = float(dual_lr)
        self.dual = {name: 0.0 for name in names}

    def loss(self, constraints: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = []
        for name, value in constraints.items():
            positive = torch.relu(value)
            terms.append(
                self.dual[name] * value + 0.5 * self.penalty * positive.square()
            )
        return torch.stack(terms).sum()

    def update(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.dual[name] = max(
                0.0, self.dual[name] + self.dual_lr * float(value)
            )


def _weighted_crps(
    scenarios: torch.Tensor, probability: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    first = (
        probability[:, :, None, None] * (scenarios - truth[:, None]).abs()
    ).sum(dim=1)
    pairwise = (scenarios[:, :, None] - scenarios[:, None, :]).abs()
    second = torch.einsum(
        "bi,bj,bijzt->bzt", probability, probability, pairwise
    )
    return (first - 0.5 * second).mean()


def _energy_score(
    scenarios: torch.Tensor, probability: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    flat = scenarios.flatten(2)
    observed = truth.flatten(1)
    first = (
        probability
        * torch.linalg.vector_norm(flat - observed[:, None], dim=-1)
    ).sum(dim=1)
    pairwise = torch.linalg.vector_norm(
        flat[:, :, None] - flat[:, None, :], dim=-1
    )
    second = torch.einsum("bi,bj,bij->b", probability, probability, pairwise)
    return (first - 0.5 * second).mean()


def _variogram_score(
    scenarios: torch.Tensor, probability: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    temporal_truth = (truth[:, :, 1:] - truth[:, :, :-1]).abs().sqrt()
    temporal_sample = (
        scenarios[:, :, :, 1:] - scenarios[:, :, :, :-1]
    ).abs().sqrt()
    temporal_expected = torch.einsum(
        "bm,bmzt->bzt", probability, temporal_sample
    )
    spatial_truth = (truth[:, 1:] - truth[:, :-1]).abs().sqrt()
    spatial_sample = (
        scenarios[:, :, 1:] - scenarios[:, :, :-1]
    ).abs().sqrt()
    spatial_expected = torch.einsum(
        "bm,bmzt->bzt", probability, spatial_sample
    )
    return (
        (temporal_truth - temporal_expected).square().mean()
        + (spatial_truth - spatial_expected).square().mean()
    )


def _ramp_crps(
    scenarios: torch.Tensor, probability: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    aggregate = scenarios.mean(dim=2)
    aggregate_truth = truth.mean(dim=1)
    return _weighted_crps(
        torch.diff(aggregate, dim=2)[:, :, None],
        probability,
        torch.diff(aggregate_truth, dim=1)[:, None],
    )


def _zero_brier(
    scenarios: torch.Tensor, probability: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    zero_probability = (
        probability[:, :, None, None] * (scenarios == 0.0).to(scenarios.dtype)
    ).sum(dim=1)
    return (
        zero_probability - (truth == 0.0).to(scenarios.dtype)
    ).square().mean()


def _soft_weighted_quantile(
    scenarios: torch.Tensor,
    probability: torch.Tensor,
    quantile: float,
    *,
    temperature: float = 0.02,
) -> torch.Tensor:
    sorted_values, order = torch.sort(scenarios, dim=1)
    expanded_probability = probability[:, :, None, None].expand_as(scenarios)
    sorted_probability = torch.gather(expanded_probability, 1, order)
    cumulative = sorted_probability.cumsum(dim=1)
    selection = torch.softmax(
        -(cumulative - quantile).abs() / temperature, dim=1
    )
    return (selection * sorted_values).sum(dim=1)


def _soft_coverage(
    scenarios: torch.Tensor,
    probability: torch.Tensor,
    truth: torch.Tensor,
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    lower = _soft_weighted_quantile(scenarios, probability, 0.05)
    upper = _soft_weighted_quantile(scenarios, probability, 0.95)
    above = torch.sigmoid((truth - lower) / temperature)
    below = torch.sigmoid((upper - truth) / temperature)
    return (above * below).mean()


def soft_cvar(cost: torch.Tensor, alpha: float = 0.90) -> torch.Tensor:
    if cost.ndim != 1 or len(cost) < 1:
        raise ValueError("cost must be a non-empty vector")
    eta = torch.quantile(cost.detach(), alpha)
    scale = cost.detach().std().clamp_min(1.0) * 0.02
    excess = torch.nn.functional.softplus((cost - eta) / scale) * scale
    return eta + excess.mean() / (1.0 - alpha)


def _mapping_matrix(mapping: WindFarmMapping, device, dtype) -> torch.Tensor:
    matrix = torch.zeros((6, 10), device=device, dtype=dtype)
    for farm, group in enumerate(mapping.groups):
        matrix[farm, list(group)] = mapping.zone_capacity_mw
    return matrix


def _reduce_torch(
    scenarios_mw: torch.Tensor,
    probability: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    representatives = []
    masses = []
    for cluster in range(clusters):
        selected = (assignments == cluster).to(probability.dtype)
        mass = (probability * selected).sum().clamp_min(1e-10)
        representatives.append(
            (
                scenarios_mw
                * (probability * selected)[:, None, None]
            ).sum(dim=0)
            / mass
        )
        masses.append(mass)
    reduced_probability = torch.stack(masses)
    reduced_probability = reduced_probability / reduced_probability.sum()
    return torch.stack(representatives), reduced_probability


def _realized_cost(
    layer: DifferentiableSUC,
    planning,
    realized,
    truth_wind: torch.Tensor,
    startup: torch.Tensor,
) -> torch.Tensor:
    grid = layer.system
    dtype, device = truth_wind.dtype, truth_wind.device
    energy = torch.as_tensor(
        [item.energy_cost for item in grid.generators], dtype=dtype, device=device
    )
    startup_cost = torch.as_tensor(
        [item.startup_cost for item in grid.generators], dtype=dtype, device=device
    )
    cost = (startup * startup_cost[:, None]).sum()
    cost = cost + (
        planning.reserve_up
        * (layer.reserve_up_cost_fraction * energy[:, None])
    ).sum()
    cost = cost + (
        planning.reserve_down
        * (layer.reserve_down_cost_fraction * energy[:, None])
    ).sum()
    cost = cost + layer.reserve_shortage_penalty * (
        planning.reserve_shortage_up.sum()
        + planning.reserve_shortage_down.sum()
    )
    cost = cost + (realized.dispatch[0] * energy[:, None]).sum()
    cost = cost + layer.shedding_penalty * realized.load_shedding.sum()
    cost = cost + layer.curtailment_penalty * (
        truth_wind.sum() - realized.used_wind.sum()
    )
    return cost


def train_calibrator(
    model: PSDFSCNetwork,
    base_scenarios: np.ndarray,
    observations: np.ndarray,
    assignments: np.ndarray,
    commitments: np.ndarray,
    mapping: WindFarmMapping,
    *,
    baseline_scores: dict[str, float],
    baseline_mean_cost: float,
    baseline_cvar90: float,
    config: TrainingConfig = TrainingConfig(),
    device: str | torch.device = "cpu",
    commitment_refresh: Callable[[PSDFSCNetwork], np.ndarray] | None = None,
) -> TrainingHistory:
    """Train PS-DFSC using fixed-commitment differentiable QP recourse."""

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    target_device = torch.device(device)
    model.to(target_device)
    scenarios = torch.as_tensor(
        base_scenarios, dtype=torch.float32, device=target_device
    )
    truth = torch.as_tensor(
        observations, dtype=torch.float32, device=target_device
    )
    labels = torch.as_tensor(assignments, dtype=torch.long, device=target_device)
    cached_commitment = np.asarray(commitments, dtype=np.float64)
    if scenarios.ndim != 5 or scenarios.shape[2:] != (10, 24):
        raise ValueError("base_scenarios must have shape [day,member,10,24]")
    if truth.shape != (len(scenarios), 10, 24):
        raise ValueError("observations do not align with scenarios")
    if labels.shape != scenarios.shape[:2]:
        raise ValueError("assignments must have shape [day,member]")
    if cached_commitment.shape != (len(scenarios), 12, 24):
        raise ValueError("commitments must have shape [day,12,24]")
    clusters = int(labels.max()) + 1
    layer = DifferentiableSUC(
        scenarios=clusters, strong_convexity=config.strong_convexity
    )
    matrix = _mapping_matrix(mapping, target_device, torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    constraint_names = (
        "CRPS",
        "ES",
        "VS",
        "ramp_CRPS",
        "zero_Brier",
        "coverage_lower",
        "coverage_upper",
        "ESS",
        "entropy",
        "transport",
    )
    augmented = AugmentedLagrangian(
        constraint_names,
        penalty=config.augmented_penalty,
        dual_lr=config.dual_learning_rate,
    )
    history = TrainingHistory()
    middle = config.proper_warmup_epochs + (
        config.epochs - config.proper_warmup_epochs
    ) // 2
    for epoch in range(config.epochs):
        if (
            epoch == middle
            and commitment_refresh is not None
            and epoch >= config.proper_warmup_epochs
        ):
            cached_commitment = np.asarray(
                commitment_refresh(model), dtype=np.float64
            )
        order = rng.permutation(len(scenarios))
        epoch_losses: list[float] = []
        epoch_constraints: dict[str, list[float]] = {
            name: [] for name in constraint_names
        }
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            output = model(scenarios[index])
            constraints = {
                "CRPS": _weighted_crps(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 1.005 * baseline_scores["CRPS"],
                "ES": _energy_score(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 1.01 * baseline_scores["ES"],
                "VS": _variogram_score(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 1.01 * baseline_scores["VS"],
                "ramp_CRPS": _ramp_crps(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 1.01 * baseline_scores["ramp_CRPS"],
                "zero_Brier": _zero_brier(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 1.01 * baseline_scores["zero_Brier"],
                "coverage_lower": 0.88
                - _soft_coverage(
                    output.scenarios, output.probabilities, truth[index]
                ),
                "coverage_upper": _soft_coverage(
                    output.scenarios, output.probabilities, truth[index]
                )
                - 0.92,
                "ESS": config.ess_floor - output.ess.min(),
                "entropy": config.entropy_fraction_floor
                * np.log(scenarios.shape[1])
                - output.entropy.min(),
                "transport": output.transport_cost.max()
                - config.transport_budget,
            }
            decision_costs = []
            if epoch >= config.proper_warmup_epochs:
                for local, day in enumerate(index):
                    farm_scenarios = torch.einsum(
                        "fz,mzt->mft",
                        matrix,
                        output.scenarios[local].to(torch.float64),
                    )
                    reduced, reduced_probability = _reduce_torch(
                        farm_scenarios,
                        output.probabilities[local].to(torch.float64),
                        labels[day],
                        clusters,
                    )
                    commitment_np = cached_commitment[day]
                    startup_np, shutdown_np = commitment_transitions(commitment_np)
                    commitment = torch.as_tensor(
                        commitment_np, dtype=torch.float64, device=target_device
                    )
                    startup = torch.as_tensor(
                        startup_np, dtype=torch.float64, device=target_device
                    )
                    shutdown = torch.as_tensor(
                        shutdown_np, dtype=torch.float64, device=target_device
                    )
                    planning = layer.plan(
                        reduced,
                        reduced_probability,
                        commitment,
                        startup,
                        shutdown,
                    )
                    truth_wind = torch.einsum(
                        "fz,zt->ft", matrix, truth[day].to(torch.float64)
                    )
                    realized = layer.realize(
                        truth_wind,
                        commitment,
                        startup,
                        shutdown,
                        planning.day_ahead_dispatch,
                        planning.reserve_up,
                        planning.reserve_down,
                    )
                    decision_costs.append(
                        _realized_cost(
                            layer, planning, realized, truth_wind, startup
                        )
                    )
            if decision_costs:
                costs = torch.stack(decision_costs)
                decision_loss = (
                    costs.mean() / baseline_mean_cost
                    + config.beta * soft_cvar(costs) / baseline_cvar90
                )
            else:
                decision_loss = torch.zeros(
                    (), dtype=scenarios.dtype, device=target_device
                )
            loss = decision_loss + augmented.loss(constraints)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            for name, value in constraints.items():
                epoch_constraints[name].append(float(value.detach().cpu()))
        mean_constraints = {
            name: float(np.mean(values))
            for name, values in epoch_constraints.items()
        }
        augmented.update(mean_constraints)
        history.epochs.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(epoch_losses)),
                **{f"constraint_{key}": value for key, value in mean_constraints.items()},
                **{f"dual_{key}": value for key, value in augmented.dual.items()},
            }
        )
    return history


__all__ = [
    "AugmentedLagrangian",
    "TrainingConfig",
    "TrainingHistory",
    "soft_cvar",
    "train_calibrator",
]
