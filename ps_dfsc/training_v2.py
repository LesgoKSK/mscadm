from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .differentiable_suc import DifferentiableSUC, commitment_transitions
from .mapping import WindFarmMapping
from .model import PSDFSCNetwork
from .training import (
    AugmentedLagrangian,
    TrainingConfig,
    TrainingHistory,
    _energy_score,
    _mapping_matrix,
    _ramp_crps,
    _realized_cost,
    _reduce_torch,
    _soft_coverage,
    _variogram_score,
    _weighted_crps,
    _zero_brier,
    soft_cvar,
)


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
    """Corrected v2 trainer for [day,member,10,24] scenario archives."""

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
    if scenarios.ndim != 4 or scenarios.shape[2:] != (10, 24):
        raise ValueError("base_scenarios must have shape [day,member,10,24]")
    if truth.shape != (len(scenarios), 10, 24):
        raise ValueError("observations do not align with scenarios")
    if labels.shape != scenarios.shape[:2]:
        raise ValueError("assignments must have shape [day,member]")
    if cached_commitment.shape != (len(scenarios), 12, 24):
        raise ValueError("commitments must have shape [day,12,24]")
    if baseline_mean_cost <= 0.0 or baseline_cvar90 <= 0.0:
        raise ValueError("cost normalizers must be positive")
    clusters = int(labels.max()) + 1
    if not all(
        np.array_equal(np.unique(day_labels.cpu()), np.arange(clusters))
        for day_labels in labels
    ):
        raise ValueError("every day must contain all contiguous cluster labels")
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
        refresh_applied = False
        if (
            epoch == middle
            and commitment_refresh is not None
            and epoch >= config.proper_warmup_epochs
        ):
            refreshed = np.asarray(
                commitment_refresh(model), dtype=np.float64
            )
            if refreshed.shape != cached_commitment.shape:
                raise ValueError("commitment refresh returned an invalid shape")
            cached_commitment = refreshed
            refresh_applied = True
        order = rng.permutation(len(scenarios))
        epoch_losses: list[float] = []
        epoch_decision: list[float] = []
        epoch_constraints: dict[str, list[float]] = {
            name: [] for name in constraint_names
        }
        for begin in range(0, len(order), config.batch_size):
            index = order[begin : begin + config.batch_size]
            output = model(scenarios[index])
            soft_coverage = _soft_coverage(
                output.scenarios, output.probabilities, truth[index]
            )
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
                "coverage_lower": 0.88 - soft_coverage,
                "coverage_upper": soft_coverage - 0.92,
                "ESS": config.ess_floor - output.ess.min(),
                "entropy": config.entropy_fraction_floor
                * np.log(scenarios.shape[1])
                - output.entropy.min(),
                "transport": output.transport_cost.max()
                - config.transport_budget,
            }
            decision_costs = []
            if epoch >= config.proper_warmup_epochs:
                for local, day_value in enumerate(index):
                    day = int(day_value)
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
            epoch_decision.append(float(decision_loss.detach().cpu()))
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
                "decision_loss": float(np.mean(epoch_decision)),
                "commitment_refresh_applied": float(refresh_applied),
                **{
                    f"constraint_{key}": value
                    for key, value in mean_constraints.items()
                },
                **{
                    f"dual_{key}": value
                    for key, value in augmented.dual.items()
                },
            }
        )
    return history


__all__ = ["train_calibrator"]
