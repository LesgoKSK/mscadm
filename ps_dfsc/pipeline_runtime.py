from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .exact_suc import solve_two_stage_suc
from .inference import calibrate
from .metrics import weighted_per_date_metrics
from .model import PSDFSCNetwork
from .reduction import fit_fixed_assignments
from .selection import empirical_cvar
from .training import TrainingConfig
from .training_v2 import train_calibrator


def train_command(args) -> None:
    """Full training command with the pre-registered midpoint commitment refresh."""

    import repro_scripts.run_ps_dfsc as pipeline

    source = np.load(args.input)
    scenarios = source[args.scenario_key]
    observations = source[args.observation_key]
    mapping = pipeline._mapping_from_args(args)
    assignments = (
        source["assignments"]
        if "assignments" in source
        else np.stack(
            [
                fit_fixed_assignments(mapping.transform(values), args.clusters)
                for values in scenarios
            ]
        )
    )
    if "commitments" in source:
        commitments = source["commitments"]
    else:
        commitments = pipeline._initial_commitments(
            scenarios,
            mapping,
            args.clusters,
            mip_gap=args.mip_gap,
            time_limit=args.time_limit,
        )
    uniform = np.full(scenarios.shape[:2], 1.0 / scenarios.shape[1])
    baseline_metrics = weighted_per_date_metrics(
        scenarios, uniform, observations
    )
    baseline_scores = {
        key: float(np.mean(baseline_metrics[key]))
        for key in ("CRPS", "ES", "VS", "ramp_CRPS", "zero_Brier")
    }
    if "baseline_realized_cost" not in source:
        raise ValueError(
            "training archive must contain baseline_realized_cost for normalization"
        )
    baseline_cost = source["baseline_realized_cost"].astype(np.float64)
    config = TrainingConfig(
        beta=args.beta,
        epochs=args.epochs,
        proper_warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        strong_convexity=args.strong_convexity,
        seed=args.seed,
    )
    model = PSDFSCNetwork()
    refresh_calls = 0

    def refresh(current_model: PSDFSCNetwork) -> np.ndarray:
        nonlocal refresh_calls
        refresh_calls += 1
        refreshed = []
        for day, values in enumerate(scenarios):
            distribution = calibrate(
                current_model,
                values,
                mapping,
                clusters=args.clusters,
                assignments=assignments[day],
                device=args.device,
            )
            solved = solve_two_stage_suc(
                distribution.suc_scenarios,
                distribution.suc_probabilities,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
            refreshed.append(solved.first_stage.commitment)
            print(
                json.dumps(
                    {
                        "phase": "commitment_refresh",
                        "day_index": day,
                        "days": len(scenarios),
                        "mip_gap": solved.mip_gap,
                        "solve_time_seconds": solved.solve_time_seconds,
                    }
                ),
                flush=True,
            )
        return np.stack(refreshed)

    history = train_calibrator(
        model,
        scenarios,
        observations,
        assignments,
        commitments,
        mapping,
        baseline_scores=baseline_scores,
        baseline_mean_cost=float(baseline_cost.mean()),
        baseline_cvar90=empirical_cvar(baseline_cost, 0.90),
        config=config,
        device=args.device,
        commitment_refresh=refresh,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ps_dfsc_checkpoint_v2",
            "model_state": model.state_dict(),
            "training_config": asdict(config),
            "baseline_scores": baseline_scores,
            "baseline_mean_cost": float(baseline_cost.mean()),
            "baseline_cvar90": empirical_cvar(baseline_cost, 0.90),
            "commitment_refresh_calls": refresh_calls,
            "mapping": {
                "groups": mapping.groups,
                "wind_buses": mapping.wind_buses,
                "zone_capacity_mw": mapping.zone_capacity_mw,
            },
            "training_relaxation": "fixed exact commitment, copper-plate LP-QP recourse",
            "exact_evaluation": "binary DC-network MILP",
        },
        output,
    )
    pipeline._write_json(
        output.with_suffix(".history.json"),
        {
            "epochs": history.epochs,
            "commitment_refresh_calls": refresh_calls,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "epochs": args.epochs,
                "commitment_refresh_calls": refresh_calls,
            }
        )
    )


__all__ = ["train_command"]
