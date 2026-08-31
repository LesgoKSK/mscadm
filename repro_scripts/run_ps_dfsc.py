from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ps_dfsc.evaluation import run_exact_cases, summarize_exact_cases
from ps_dfsc.exact_suc import solve_two_stage_suc
from ps_dfsc.inference import calibrate
from ps_dfsc.manifest import build_lock_manifest, verify_lock_manifest
from ps_dfsc.mapping import WindFarmMapping, fit_wind_farm_mapping
from ps_dfsc.metrics import (
    make_conditional_assignments,
    weighted_per_date_metrics,
)
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.reduction import fit_fixed_assignments
from ps_dfsc.safety import SafetyThresholds, evaluate_safety_gate
from ps_dfsc.splits import build_ps_dfsc_split_registry
from ps_dfsc.training import TrainingConfig, train_calibrator


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _load_mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value.get("wind_buses", (3, 5, 7, 16, 21, 23))),
        zone_capacity_mw=float(value.get("zone_capacity_mw", 120.0)),
    )


def _save_mapping(path: str | Path, mapping: WindFarmMapping) -> None:
    _write_json(
        path,
        {
            "schema": "ps_dfsc_wind_farm_mapping_v1",
            "groups": [list(group) for group in mapping.groups],
            "wind_buses": list(mapping.wind_buses),
            "zone_capacity_mw": mapping.zone_capacity_mw,
        },
    )


def _mapping_from_args(args) -> WindFarmMapping:
    if getattr(args, "mapping", None):
        return _load_mapping(args.mapping)
    reference = np.load(args.mapping_training_archive)
    key = getattr(args, "mapping_training_key", "observations")
    return fit_wind_farm_mapping(reference[key])


def freeze_splits(args) -> None:
    registry = build_ps_dfsc_split_registry(
        args.data_dir,
        mm_registry=args.mm_registry,
        stgf_registry=args.stgf_registry,
        seed=args.seed,
    )
    _write_json(args.output, registry)
    print(json.dumps({"output": str(Path(args.output).resolve()), "status": "frozen"}))


def fit_mapping(args) -> None:
    archive = np.load(args.input)
    mapping = fit_wind_farm_mapping(
        archive[args.key], zone_capacity_mw=args.zone_capacity
    )
    _save_mapping(args.output, mapping)
    print(json.dumps({"output": str(Path(args.output).resolve()), "groups": mapping.groups}))


def _load_model(checkpoint: str | Path, device: str) -> PSDFSCNetwork:
    model = PSDFSCNetwork()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)
    return model.to(device)


def calibrate_archive(args) -> None:
    source = np.load(args.input)
    scenarios = source[args.scenario_key]
    mapping = _mapping_from_args(args)
    model = _load_model(args.checkpoint, args.device)
    distributions = [
        calibrate(
            model,
            values,
            mapping,
            clusters=args.clusters,
            ess_floor=args.ess_floor,
            normalized_entropy_floor=args.entropy_floor,
            transport_budget=args.transport_budget,
            device=args.device,
        )
        for values in scenarios
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        full_scenarios=np.stack([item.full_scenarios for item in distributions]),
        probabilities=np.stack([item.probabilities for item in distributions]),
        suc_scenarios=np.stack([item.suc_scenarios for item in distributions]),
        suc_probabilities=np.stack(
            [item.suc_probabilities for item in distributions]
        ),
        ess=np.asarray([item.ess for item in distributions]),
        entropy=np.asarray([item.entropy for item in distributions]),
        transport_cost=np.asarray([item.transport_cost for item in distributions]),
        used_fallback=np.asarray([item.used_fallback for item in distributions]),
        fallback_reason=np.asarray([item.fallback_reason for item in distributions]),
        groups=np.asarray(
            [
                list(group) + [-1] * (2 - len(group))
                for group in mapping.groups
            ],
            dtype=np.int64,
        ),
        days=source["days"] if "days" in source else np.arange(len(scenarios)),
    )
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "days": len(distributions),
                "fallback_days": sum(item.used_fallback for item in distributions),
            }
        )
    )


def _assignments_for_metrics(args, observations: np.ndarray, days: np.ndarray):
    reference = np.load(args.regime_reference)
    reference_truth = reference[args.observation_key]
    aggregate = reference_truth.mean(axis=1)
    wind_thresholds = tuple(np.quantile(aggregate.mean(axis=1), [1 / 3, 2 / 3]))
    ramp_thresholds = tuple(
        np.quantile(np.abs(np.diff(aggregate, axis=1)).mean(axis=1), [1 / 3, 2 / 3])
    )
    return make_conditional_assignments(
        observations,
        days=days,
        wind_thresholds=wind_thresholds,
        ramp_thresholds=ramp_thresholds,
    )


def safety_gate(args) -> None:
    candidate = np.load(args.candidate)
    baseline = np.load(args.baseline)
    truth_archive = np.load(args.truth)
    observations = truth_archive[args.observation_key]
    days = truth_archive["days"]
    assignments = _assignments_for_metrics(args, observations, days)
    candidate_metrics = weighted_per_date_metrics(
        candidate["full_scenarios"],
        candidate["probabilities"],
        observations,
        assignments=assignments,
    )
    baseline_metrics = weighted_per_date_metrics(
        baseline["full_scenarios"],
        baseline["probabilities"],
        observations,
        assignments=assignments,
    )
    structural_ok = bool(
        np.all(candidate["ess"] >= args.ess_floor)
        and np.all(
            candidate["entropy"]
            / np.log(candidate["probabilities"].shape[1])
            >= args.entropy_floor
        )
        and np.all(candidate["transport_cost"] <= args.transport_budget)
    )
    decision = evaluate_safety_gate(
        candidate_metrics,
        baseline_metrics,
        thresholds=SafetyThresholds(),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        structural_ok=structural_ok,
    )
    result = asdict(decision)
    result["candidate"] = str(Path(args.candidate).resolve())
    result["baseline"] = str(Path(args.baseline).resolve())
    _write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "passed": decision.passed}))


def _load_distributions(archive) -> list:
    from ps_dfsc.types import CalibratedDistribution

    return [
        CalibratedDistribution(
            full_scenarios=archive["full_scenarios"][day],
            probabilities=archive["probabilities"][day],
            suc_scenarios=archive["suc_scenarios"][day],
            suc_probabilities=archive["suc_probabilities"][day],
            ess=float(archive["ess"][day]),
            entropy=float(archive["entropy"][day]),
            transport_cost=float(archive["transport_cost"][day]),
            used_fallback=bool(archive["used_fallback"][day]),
            fallback_reason=str(archive["fallback_reason"][day]),
        )
        for day in range(len(archive["full_scenarios"]))
    ]


def exact_evaluate(args) -> None:
    calibrated = np.load(args.calibrated)
    truth = np.load(args.truth)
    mapping = _mapping_from_args(args)
    observations = truth[args.observation_key]
    farm_truth = mapping.transform(observations)
    frame = run_exact_cases(
        _load_distributions(calibrated),
        farm_truth,
        outer=args.outer,
        method=args.method,
        dates=truth["days"] if "days" in truth else None,
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = summarize_exact_cases(frame)
    _write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps({"output": str(output.resolve()), **summary}))


def _initial_commitments(
    scenarios: np.ndarray,
    mapping: WindFarmMapping,
    clusters: int,
    *,
    mip_gap: float,
    time_limit: float,
) -> np.ndarray:
    result = []
    uniform = np.full(scenarios.shape[1], 1.0 / scenarios.shape[1])
    for values in scenarios:
        mapped = mapping.transform(values)
        labels = fit_fixed_assignments(mapped, clusters)
        from ps_dfsc.reduction import weighted_cluster_reduction

        reduced, probability = weighted_cluster_reduction(mapped, uniform, labels)
        solved = solve_two_stage_suc(
            reduced,
            probability,
            mip_gap=mip_gap,
            time_limit=time_limit,
        )
        result.append(solved.first_stage.commitment)
    return np.stack(result)


def train(args) -> None:
    source = np.load(args.input)
    scenarios = source[args.scenario_key]
    observations = source[args.observation_key]
    mapping = _mapping_from_args(args)
    assignments = np.stack(
        [
            fit_fixed_assignments(mapping.transform(values), args.clusters)
            for values in scenarios
        ]
    )
    if "commitments" in source:
        commitments = source["commitments"]
    else:
        commitments = _initial_commitments(
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
    from ps_dfsc.selection import empirical_cvar

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
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ps_dfsc_checkpoint_v1",
            "model_state": model.state_dict(),
            "training_config": asdict(config),
            "baseline_scores": baseline_scores,
            "mapping": {
                "groups": mapping.groups,
                "wind_buses": mapping.wind_buses,
                "zone_capacity_mw": mapping.zone_capacity_mw,
            },
        },
        output,
    )
    _write_json(
        output.with_suffix(".history.json"),
        {"epochs": history.epochs, "commitment_refresh_applied": False},
    )
    print(json.dumps({"output": str(output.resolve()), "epochs": args.epochs}))


def lock(args) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    safety = json.loads(Path(args.safety).read_text(encoding="utf-8"))
    manifest = build_lock_manifest(
        split_registry=args.splits,
        config=config,
        model_paths=args.models,
        outer=args.outer,
        beta=args.beta,
        candidate_id=args.candidate_id,
        safety_decision=safety,
    )
    verify_lock_manifest(manifest)
    _write_json(args.output, manifest)
    print(json.dumps({"output": str(Path(args.output).resolve()), "status": "locked"}))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="PS-DFSC experiment pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-splits")
    freeze.add_argument("--data-dir", default="Data")
    freeze.add_argument(
        "--mm-registry",
        default="repro_configs/mm_jdwind_confirmation_splits.json",
    )
    freeze.add_argument(
        "--stgf-registry",
        default="repro_configs/stgf_confirmation_splits.json",
    )
    freeze.add_argument("--seed", type=int, default=20260801)
    freeze.add_argument(
        "--output", default="repro_configs/ps_dfsc_splits.json"
    )
    freeze.set_defaults(function=freeze_splits)

    mapping = commands.add_parser("fit-mapping")
    mapping.add_argument("--input", required=True)
    mapping.add_argument("--key", default="observations")
    mapping.add_argument("--zone-capacity", type=float, default=120.0)
    mapping.add_argument("--output", required=True)
    mapping.set_defaults(function=fit_mapping)

    train_command = commands.add_parser("train")
    train_command.add_argument("--input", required=True)
    train_command.add_argument("--scenario-key", default="base_scenarios")
    train_command.add_argument("--observation-key", default="observations")
    train_command.add_argument("--mapping")
    train_command.add_argument("--mapping-training-archive")
    train_command.add_argument("--mapping-training-key", default="observations")
    train_command.add_argument("--clusters", type=int, default=20)
    train_command.add_argument("--beta", type=float, required=True)
    train_command.add_argument("--epochs", type=int, default=50)
    train_command.add_argument("--warmup-epochs", type=int, default=20)
    train_command.add_argument("--batch-size", type=int, default=4)
    train_command.add_argument("--learning-rate", type=float, default=1e-3)
    train_command.add_argument("--strong-convexity", type=float, default=1e-4)
    train_command.add_argument("--mip-gap", type=float, default=0.001)
    train_command.add_argument("--time-limit", type=float, default=600.0)
    train_command.add_argument("--seed", type=int, default=0)
    train_command.add_argument("--device", default="cpu")
    train_command.add_argument("--output", required=True)
    train_command.set_defaults(function=train)

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--input", required=True)
    calibration.add_argument("--scenario-key", default="scenarios")
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--mapping")
    calibration.add_argument("--mapping-training-archive")
    calibration.add_argument("--mapping-training-key", default="observations")
    calibration.add_argument("--clusters", type=int, default=20)
    calibration.add_argument("--ess-floor", type=float, default=50.0)
    calibration.add_argument("--entropy-floor", type=float, default=0.85)
    calibration.add_argument("--transport-budget", type=float, default=0.02)
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(function=calibrate_archive)

    gate = commands.add_parser("safety-gate")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--truth", required=True)
    gate.add_argument("--regime-reference", required=True)
    gate.add_argument("--observation-key", default="observations")
    gate.add_argument("--ess-floor", type=float, default=50.0)
    gate.add_argument("--entropy-floor", type=float, default=0.85)
    gate.add_argument("--transport-budget", type=float, default=0.02)
    gate.add_argument("--bootstrap-samples", type=int, default=10000)
    gate.add_argument("--seed", type=int, default=0)
    gate.add_argument("--output", required=True)
    gate.set_defaults(function=safety_gate)

    exact = commands.add_parser("exact-evaluate")
    exact.add_argument("--calibrated", required=True)
    exact.add_argument("--truth", required=True)
    exact.add_argument("--observation-key", default="observations")
    exact.add_argument("--mapping")
    exact.add_argument("--mapping-training-archive")
    exact.add_argument("--mapping-training-key", default="observations")
    exact.add_argument("--outer", type=int, required=True)
    exact.add_argument("--method", default="PS-DFSC")
    exact.add_argument("--mip-gap", type=float, default=0.001)
    exact.add_argument("--time-limit", type=float, default=600.0)
    exact.add_argument("--output", required=True)
    exact.set_defaults(function=exact_evaluate)

    lock_command = commands.add_parser("lock")
    lock_command.add_argument("--splits", required=True)
    lock_command.add_argument("--config", default="repro_configs/ps_dfsc.json")
    lock_command.add_argument("--models", nargs="+", required=True)
    lock_command.add_argument("--safety", required=True)
    lock_command.add_argument("--outer", type=int, required=True)
    lock_command.add_argument("--beta", type=float, required=True)
    lock_command.add_argument("--candidate-id", required=True)
    lock_command.add_argument("--output", required=True)
    lock_command.set_defaults(function=lock)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
