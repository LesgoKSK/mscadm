"""Batched publication trainer using parallel differentiable QPs."""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import torch

from .differentiable_suc import commitment_transitions
from .fast_differentiable_suc import stable_soft_cvar
from .fast_differentiable_suc_publication_v4 import (
    BatchedClarabelPublicationDifferentiableSUC,
)
from .mapping import WindFarmMapping
from .model import PSDFSCNetwork
from .training import (
    AugmentedLagrangian,
    TrainingConfig,
    TrainingHistory,
    _energy_score,
    _mapping_matrix,
    _ramp_crps,
    _reduce_torch,
    _soft_coverage,
    _variogram_score,
    _weighted_crps,
    _zero_brier,
)


def _batched_realized_cost(
    layer,
    planning,
    realized,
    truth_wind,
    startup,
):
    dtype, device = truth_wind.dtype, truth_wind.device
    energy = torch.as_tensor(
        [item.energy_cost for item in layer.system.generators],
        dtype=dtype,
        device=device,
    )
    startup_cost = torch.as_tensor(
        [item.startup_cost for item in layer.system.generators],
        dtype=dtype,
        device=device,
    )
    cost = (startup * startup_cost[None, :, None]).sum(dim=(1, 2))
    cost = cost + (
        planning.reserve_up
        * (layer.reserve_up_cost_fraction * energy[None, :, None])
    ).sum(dim=(1, 2))
    cost = cost + (
        planning.reserve_down
        * (layer.reserve_down_cost_fraction * energy[None, :, None])
    ).sum(dim=(1, 2))
    cost = cost + layer.reserve_shortage_penalty * (
        planning.reserve_shortage_up.sum(dim=1)
        + planning.reserve_shortage_down.sum(dim=1)
    )
    cost = cost + (
        realized.dispatch[:, 0]
        * energy[None, :, None]
    ).sum(dim=(1, 2))
    cost = cost + layer.shedding_penalty * (
        realized.load_shedding.sum(dim=(1, 2))
    )
    cost = cost + layer.curtailment_penalty * (
        truth_wind.sum(dim=(1, 2))
        - realized.used_wind.sum(dim=(1, 2, 3))
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
    resume_state: dict | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> TrainingHistory:
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
    labels = torch.as_tensor(
        assignments, dtype=torch.long, device=target_device
    )
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
        np.array_equal(np.unique(day.cpu()), np.arange(clusters))
        for day in labels
    ):
        raise ValueError("every day must contain all cluster labels")
    layer = BatchedClarabelPublicationDifferentiableSUC(
        scenarios=clusters,
        strong_convexity=config.strong_convexity,
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
    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        augmented.dual.update(resume_state["augmented_dual"])
        rng.bit_generator.state = resume_state["numpy_rng_state"]
        cached_commitment = np.asarray(
            resume_state["cached_commitment"], dtype=np.float64
        )
        history.epochs.extend(resume_state["history_epochs"])
        start_epoch = int(resume_state["next_epoch"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if (
            target_device.type == "cuda"
            and resume_state.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state(
                resume_state["cuda_rng_state"], device=target_device
            )
    middle = config.proper_warmup_epochs + (
        config.epochs - config.proper_warmup_epochs
    ) // 2
    for epoch in range(start_epoch, config.epochs):
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
        epoch_losses = []
        epoch_decision = []
        epoch_constraints = {name: [] for name in constraint_names}
        for begin in range(0, len(order), config.batch_size):
            index = order[begin : begin + config.batch_size]
            output = model(scenarios[index])
            coverage = _soft_coverage(
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
                "coverage_lower": 0.88 - coverage,
                "coverage_upper": coverage - 0.92,
                "ESS": config.ess_floor - output.ess.min(),
                "entropy": config.entropy_fraction_floor
                * np.log(scenarios.shape[1])
                - output.entropy.min(),
                "transport": output.transport_cost.max()
                - config.transport_budget,
            }
            if epoch >= config.proper_warmup_epochs:
                reduced_days = []
                reduced_probabilities = []
                commitment_days = []
                startup_days = []
                shutdown_days = []
                truth_days = []
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
                    startup_np, shutdown_np = commitment_transitions(
                        commitment_np
                    )
                    reduced_days.append(reduced)
                    reduced_probabilities.append(reduced_probability)
                    commitment_days.append(
                        torch.as_tensor(
                            commitment_np,
                            dtype=torch.float64,
                            device=target_device,
                        )
                    )
                    startup_days.append(
                        torch.as_tensor(
                            startup_np,
                            dtype=torch.float64,
                            device=target_device,
                        )
                    )
                    shutdown_days.append(
                        torch.as_tensor(
                            shutdown_np,
                            dtype=torch.float64,
                            device=target_device,
                        )
                    )
                    truth_days.append(
                        torch.einsum(
                            "fz,zt->ft",
                            matrix,
                            truth[day].to(torch.float64),
                        )
                    )
                reduced_batch = torch.stack(reduced_days)
                probability_batch = torch.stack(reduced_probabilities)
                commitment_batch = torch.stack(commitment_days)
                startup_batch = torch.stack(startup_days)
                shutdown_batch = torch.stack(shutdown_days)
                truth_batch = torch.stack(truth_days)
                planning = layer.plan(
                    reduced_batch,
                    probability_batch,
                    commitment_batch,
                    startup_batch,
                    shutdown_batch,
                )
                realized = layer.realize(
                    truth_batch,
                    commitment_batch,
                    startup_batch,
                    shutdown_batch,
                    planning.day_ahead_dispatch,
                    planning.reserve_up,
                    planning.reserve_down,
                )
                costs = _batched_realized_cost(
                    layer,
                    planning,
                    realized,
                    truth_batch,
                    startup_batch,
                )
                decision_loss = (
                    costs.mean() / baseline_mean_cost
                    + config.beta
                    * stable_soft_cvar(costs)
                    / baseline_cvar90
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
        record = {
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
        history.epochs.append(record)
        print(json.dumps({"phase": "epoch", **record}), flush=True)
        if epoch_callback is not None:
            epoch_callback(
                {
                    "schema": "ps_dfsc_epoch_resume_v1",
                    "next_epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "augmented_dual": dict(augmented.dual),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(target_device)
                        if target_device.type == "cuda"
                        else None
                    ),
                    "cached_commitment": cached_commitment,
                    "history_epochs": list(history.epochs),
                }
            )
    return history


__all__ = ["train_calibrator"]
