"""Audited and restartable publication training command."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .exact_suc_publication import solve_two_stage_suc
from .inference import calibrate
from .manifest import file_sha256
from .metrics import weighted_per_date_metrics
from .model import PSDFSCNetwork
from .reduction import fit_fixed_assignments
from .selection import empirical_cvar
from .training import TrainingConfig
from .training_v2 import train_calibrator


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().view(np.uint8))
    return digest.hexdigest()


def _day_sha256(scenarios, assignments) -> str:
    digest = hashlib.sha256()
    for value in (scenarios, assignments):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def train_command(args) -> None:
    import repro_scripts.run_ps_dfsc as pipeline

    source = np.load(args.input, allow_pickle=False)
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
    if "commitments" not in source:
        raise ValueError(
            "publication training requires audited initial commitments"
        )
    commitments = source["commitments"]
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
            "training archive must contain baseline_realized_cost"
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
    output = Path(args.output)
    refresh_cache = output.with_suffix(".refresh_cache")
    refresh_cache.mkdir(parents=True, exist_ok=True)
    mapping_path = getattr(args, "mapping", None)
    mapping_sha256 = (
        file_sha256(mapping_path)
        if mapping_path is not None
        else "fit_from_registered_training_archive"
    )
    refresh_calls = 0
    refresh_records: list[dict] = []

    def refresh(current_model: PSDFSCNetwork) -> np.ndarray:
        nonlocal refresh_calls, refresh_records
        refresh_calls += 1
        midpoint_model_sha256 = _model_sha256(current_model)
        refreshed = []
        current_records = []
        for day, values in enumerate(scenarios):
            day_hash = _day_sha256(values, assignments[day])
            cache_path = refresh_cache / f"day_{day:03d}.npz"
            cached = None
            if cache_path.exists():
                candidate = np.load(cache_path, allow_pickle=False)
                if (
                    str(candidate["midpoint_model_sha256"])
                    == midpoint_model_sha256
                    and str(candidate["day_sha256"]) == day_hash
                    and str(candidate["mapping_sha256"]) == mapping_sha256
                    and float(candidate["requested_mip_gap"]) == args.mip_gap
                    and float(candidate["time_limit_seconds"])
                    == args.time_limit
                ):
                    cached = candidate
            if cached is None:
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
                payload = {
                    "schema": np.asarray(
                        "ps_dfsc_midpoint_commitment_cache_v1"
                    ),
                    "midpoint_model_sha256": np.asarray(
                        midpoint_model_sha256
                    ),
                    "day_sha256": np.asarray(day_hash),
                    "mapping_sha256": np.asarray(mapping_sha256),
                    "requested_mip_gap": np.asarray(args.mip_gap),
                    "time_limit_seconds": np.asarray(args.time_limit),
                    "commitment": solved.first_stage.commitment,
                    "mip_gap": np.asarray(solved.mip_gap),
                    "solver_success": np.asarray(solved.success),
                    "solver_status": np.asarray(solved.status),
                    "dual_bound": np.asarray(solved.mip_dual_bound),
                    "node_count": np.asarray(solved.mip_node_count),
                    "solve_time_seconds": np.asarray(
                        solved.solve_time_seconds
                    ),
                    "cache_origin": np.asarray("solved"),
                }
                np.savez_compressed(cache_path, **payload)
                cached = np.load(cache_path, allow_pickle=False)
            record = {
                "day_index": day,
                "mip_gap": float(cached["mip_gap"]),
                "solver_success": bool(cached["solver_success"]),
                "solver_status": str(cached["solver_status"]),
                "dual_bound": float(cached["dual_bound"]),
                "node_count": int(cached["node_count"]),
                "solve_time_seconds": float(cached["solve_time_seconds"]),
                "cache_origin": str(cached["cache_origin"]),
            }
            refreshed.append(cached["commitment"])
            current_records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "commitment_refresh",
                        "days": len(scenarios),
                        **record,
                    }
                ),
                flush=True,
            )
        refresh_records = current_records
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
    output.parent.mkdir(parents=True, exist_ok=True)
    refresh_summary = {
        "cases": len(refresh_records),
        "solver_success_rate": (
            float(np.mean([row["solver_success"] for row in refresh_records]))
            if refresh_records
            else 0.0
        ),
        "target_gap_pass_rate": (
            float(
                np.mean(
                    [
                        row["mip_gap"]
                        <= args.mip_gap * (1.0 + 1e-6) + 1e-12
                        for row in refresh_records
                    ]
                )
            )
            if refresh_records
            else 0.0
        ),
        "maximum_mip_gap": (
            float(max(row["mip_gap"] for row in refresh_records))
            if refresh_records
            else np.nan
        ),
    }
    torch.save(
        {
            "schema": "ps_dfsc_checkpoint_v3",
            "model_state": model.state_dict(),
            "model_state_sha256": _model_sha256(model),
            "training_config": asdict(config),
            "training_input": str(Path(args.input).resolve()),
            "training_input_sha256": file_sha256(args.input),
            "mapping_sha256": mapping_sha256,
            "baseline_scores": baseline_scores,
            "baseline_mean_cost": float(baseline_cost.mean()),
            "baseline_cvar90": empirical_cvar(baseline_cost, 0.90),
            "commitment_refresh_calls": refresh_calls,
            "commitment_refresh_summary": refresh_summary,
            "mapping": {
                "groups": mapping.groups,
                "wind_buses": mapping.wind_buses,
                "zone_capacity_mw": mapping.zone_capacity_mw,
            },
            "training_relaxation": (
                "fixed exact commitment, strictly-convex copper-plate QP"
            ),
            "exact_evaluation": "binary DC-network MILP",
        },
        output,
    )
    pipeline._write_json(
        output.with_suffix(".history.json"),
        {
            "epochs": history.epochs,
            "commitment_refresh_calls": refresh_calls,
            "commitment_refresh_summary": refresh_summary,
            "commitment_refresh_records": refresh_records,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "epochs": args.epochs,
                "commitment_refresh_calls": refresh_calls,
                **refresh_summary,
            }
        )
    )


__all__ = ["train_command"]
