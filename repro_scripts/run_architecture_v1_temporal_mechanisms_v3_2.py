#!/usr/bin/env python3
"""Run the frozen v3.2 three-seed chronological/shuffle adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_2.json"
BASE_CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2_2_1.json"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.formal_evaluation import aggregate_sampling_replicates
from architecture_v1.formal_training import (
    FormalEpochTrainer,
    canonical_sha256,
    file_sha256,
    shared_ea_state_sha256,
    tensor_state_sha256,
)
from architecture_v1.mechanism_evaluation import (
    average_training_seeds,
    chronological_vs_control_gate,
)
from architecture_v1.model import (
    R0JointRectifiedFlow,
    T0StableSourceRectifiedFlow,
    T1FeatureStableSourceRectifiedFlow,
    T1SourceStableRectifiedFlow,
    T1StableShuffleRectifiedFlow,
)
from repro_scripts.run_architecture_v1_formal_r0 import (
    _configure_runtime,
    _derive_train_quantities,
    _load_best_checkpoint,
    _make_validation_bank,
    _model_kwargs,
)
from repro_scripts.run_architecture_v1_temporal_mechanisms_v3_0 import (
    _atomic_json,
    _jsonable,
    _read,
    _sample_primary,
    _verified,
)
from repro_scripts.run_architecture_v1_temporal_mechanisms_v3_1_pilot import (
    _hard_gate,
    _initial_bank,
)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(path)
    if config.get("schema") != "architecture_v1_temporal_mechanisms_v3_2":
        raise ValueError("unexpected v3.2 config schema")
    if config.get("status") != "frozen_before_any_v3_2_training_update":
        raise RuntimeError("v3.2 config is not frozen")
    _verified(path, file_sha256(path))
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("v3.2 role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    if roles["selection_access_authorized"] is not False:
        raise RuntimeError("v3.2 cannot authorize selection access")
    lineage = config["lineage"]
    for key in ("v3_1_config", "v3_1_pilot_result", "v3_1_pilot_freeze"):
        _verified(ROOT / lineage[key], lineage[f"{key}_sha256"])
    pilot = _read(ROOT / lineage["v3_1_pilot_result"])
    freeze = _read(ROOT / lineage["v3_1_pilot_freeze"])
    required = lineage["v3_1_required_status"]
    if pilot.get("status") != required or freeze.get("status") != required:
        raise RuntimeError("v3.1 pilot did not authorize v3.2")
    _verified(ROOT / lineage["shared_EA_checkpoint"], lineage["shared_EA_checkpoint_sha256"])
    imports = config["seed0_imports"]
    for name in ("T0", "T1_feature", "T1_source"):
        _verified(ROOT / imports[name]["checkpoint"], imports[name]["checkpoint_sha256"])
        completion = _read(ROOT / imports[name]["completion"])
        if completion.get("status") != "complete":
            raise RuntimeError(f"imported completion is not complete: {name}")
        if completion.get("best_checkpoint_sha256") != imports[name]["checkpoint_sha256"]:
            raise RuntimeError(f"imported completion/checkpoint mismatch: {name}")
    declared = [(str(name), int(seed)) for name, seed in config["execution_order_remaining_12"]]
    expected = [
        (name, seed)
        for seed in (0, 1, 2)
        for name in config["candidates"]
        if not (seed == 0 and name in ("T0", "T1_feature", "T1_source"))
    ]
    if len(declared) != 12 or set(declared) != set(expected):
        raise RuntimeError("remaining-12 execution matrix is incomplete")
    return config, _read(BASE_CONFIG)


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(), config_path.resolve(), BASE_CONFIG,
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "formal_training.py",
        ROOT / "architecture_v1" / "mechanism_evaluation.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_formal_r0.py",
        ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanisms_v3_0.py",
        ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanisms_v3_1_pilot.py",
    ]
    records = [
        {"path": item.relative_to(ROOT).as_posix(), "bytes": item.stat().st_size, "sha256": file_sha256(item)}
        for item in paths
    ]
    core = {"schema": "architecture_v1_temporal_mechanism_v3_2_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _candidate(name: str, kwargs: Mapping[str, Any], config: Mapping[str, Any]):
    common = dict(kwargs)
    if name == "T0":
        return T0StableSourceRectifiedFlow(**common)
    if name == "T1_feature":
        return T1FeatureStableSourceRectifiedFlow(**common)
    if name == "T1_source":
        return T1SourceStableRectifiedFlow(**common)
    order = config["shuffle_control"]["hour_order_zero_based"]
    if name == "T1_feature_shuffle":
        return T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="feature", **common
        )
    if name == "T1_source_shuffle":
        return T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="source", **common
        )
    raise ValueError(f"unknown v3.2 candidate: {name}")


def _fit_new(
    *, name: str, seed: int, model: Any, bundle: Any, bank: Any,
    config: Mapping[str, Any], config_sha: str, code_sha: str,
    initial_hash: str, root: Path, device: torch.device, resume: bool,
) -> tuple[dict[str, Any], float]:
    stage = config["training"]
    run_root = root / "runs" / name / f"seed{seed}"
    trainer = FormalEpochTrainer(
        model,
        run_root,
        resolved_config={
            "schema": "architecture_v1_temporal_mechanism_v3_2_training_identity_v1",
            "formal_config_sha256": config_sha,
            "candidate": name,
            "training_seed": seed,
            "training": stage,
            "initial_tensor_bank_sha256": initial_hash,
        },
        protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
        data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
        code_sha256=code_sha,
        validation_bank=bank,
        training_seed=seed,
        shuffle_seed=int(stage["common_epoch_shuffle_seed"]),
        path_seed=int(stage["common_flow_path_seed"]),
        run_id=f"temporal_v3_2_{name}_seed{seed}",
        device=device,
        extra_identity_hashes={
            "formal_config_file_sha256": config_sha,
            "shared_ea_checkpoint_sha256": config["lineage"]["shared_EA_checkpoint_sha256"],
            "initial_tensor_bank_sha256": initial_hash,
        },
        training_freeze_path=root / "training.freeze.json",
    )
    zero_loss = float(bank.evaluate(model, bundle.validation, stage="flow", device=device)["loss"])
    completion_path = run_root / "flow" / "completion.json"
    if completion_path.exists():
        if not resume:
            raise FileExistsError(f"existing run requires --resume: {name}/seed{seed}")
        completion = _read(completion_path)
        _load_best_checkpoint(run_root / "flow" / "best.pt", model, device)
    else:
        completion = trainer.fit_stage(
            "flow", bundle.train, bundle.validation,
            max_epochs=int(stage["maximum_epochs"]),
            batch_days=int(stage["batch_days"]),
            learning_rate=float(stage["learning_rate"]),
            weight_decay=float(stage["weight_decay"]),
            gradient_clip=float(stage["gradient_clip"]),
            validate_every_epochs=int(stage["validate_every_epochs"]),
            early_stopping_patience=int(stage["early_stopping_patience_validations"]),
            early_stopping_minimum_delta=float(stage["early_stopping_minimum_delta"]),
            early_stopping_relative_delta=float(stage["early_stopping_relative_delta"]),
            minimum_epochs_before_early_stop=int(stage["minimum_epochs_before_early_stop"]),
            resume=bool(resume and (run_root / "flow" / "latest_safe.pt").exists()),
            restore_best=True,
            record_update_preclip_gradient_norms=True,
            gradient_audit_epoch_start=int(stage["gradient_audit_epoch_start_one_based"]),
            gradient_audit_gate={
                "gradient_clip_fraction_max": float(stage["gradient_clip_fraction_max"]),
                "preclip_gradient_norm_p99_to_median_max": float(stage["preclip_gradient_norm_p99_to_median_max"]),
                "preclip_gradient_norm_max_over_all_flow_updates": float(stage["preclip_gradient_norm_max"]),
                "require_all_finite": True,
            },
        )
    return completion, zero_loss


@torch.no_grad()
def _source_diagnostic(model: Any, bundle: Any, *, seed: int, device: torch.device) -> dict[str, float]:
    total = total_square = 0.0
    count = saturation_count = rho_count = 0
    inactive_max = rho_max = 0.0
    members = 100
    for start in range(0, len(bundle.validation), 2):
        stop = min(start + 2, len(bundle.validation))
        condition = torch.from_numpy(
            np.ascontiguousarray(bundle.validation.condition[start:stop], dtype=np.float32)
        ).to(device)
        encoded = model.encode_condition(condition)
        statistics = model.atom(encoded)
        allocation = model.atom.allocate(statistics, members=members, seed=seed + start * 1009)
        active = allocation.active_mask
        generator = torch.Generator(device=device).manual_seed(seed + 1 + start * 1009)
        iid = torch.randn(active.shape, dtype=condition.dtype, device=device, generator=generator)
        flat_iid = torch.where(active, iid, torch.zeros_like(iid)).reshape(-1, model.zones, model.hours)
        flat_encoded = encoded[:, None].expand(-1, members, -1, -1, -1).reshape(
            -1, model.zones, model.hours, model.encoder_dim
        )
        flat_active = active.reshape(-1, model.zones, model.hours)
        source = model.prepare_source_noise(flat_iid, flat_encoded, flat_active)
        values = source[flat_active].double()
        total += float(values.sum())
        total_square += float(values.square().sum())
        count += int(values.numel())
        if bool((~flat_active).any()):
            inactive_max = max(inactive_max, float(source[~flat_active].abs().max()))
        rho = model.stable_source.correlation(encoded).abs()
        rho_max = max(rho_max, float(rho.max()))
        saturation_count += int((rho >= 0.94).sum())
        rho_count += int(rho.numel())
    mean = total / count
    variance = (total_square - count * mean * mean) / (count - 1)
    return {
        "active_count": count,
        "empirical_mean": mean,
        "empirical_variance": variance,
        "inactive_max_abs": inactive_max,
        "rho_abs_max": rho_max,
        "fraction_abs_rho_ge_0.94": saturation_count / rho_count,
    }


def _seed_stability(seed_gates: Mapping[int, Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: np.asarray(
            [seed_gates[seed]["validation_aggregate"][name] for seed in sorted(seed_gates)],
            dtype=np.float64,
        )
        for name in ("level_CRPS", "ramp_CRPS", "normalized_joint_ES", "coverage90")
    }
    spreads = {name: float(values.max() - values.min()) for name, values in metrics.items()}
    joint_relative = spreads["normalized_joint_ES"] / float(np.median(metrics["normalized_joint_ES"]))
    checks = {
        "level_CRPS": spreads["level_CRPS"] <= float(rules["level_CRPS_max_minus_min_max"]),
        "ramp_CRPS": spreads["ramp_CRPS"] <= float(rules["ramp_CRPS_max_minus_min_max"]),
        "normalized_joint_ES": joint_relative <= float(rules["normalized_joint_ES_relative_max_minus_min_max"]),
        "coverage90": spreads["coverage90"] <= float(rules["coverage90_max_minus_min_max"]),
    }
    return {
        "passed": bool(all(checks.values())), "checks": checks,
        "spreads": spreads, "normalized_joint_ES_relative_spread": joint_relative,
        "values_by_seed": {name: values.tolist() for name, values in metrics.items()},
    }


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, base = _load_config(config_path)
    root = ROOT / config["output_root"]
    if (root / "training.freeze.json").exists():
        raise RuntimeError("v3.2 is already frozen")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("v3.2 output is non-empty; use --resume")
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    config_sha = file_sha256(config_path)
    code = _code_manifest(config_path)
    bundle = build_architecture_v1_fit_data(config_path=BASE_CONFIG)
    derived = _derive_train_quantities(bundle, base)
    kwargs = _model_kwargs(base, derived)
    kwargs["stable_source_rho_max"] = float(config["common_model"]["stable_source_rho_max"])
    bank = _make_validation_bank(bundle, base, evaluation_batch_days=int(config["training"]["batch_days"]))
    shared_kwargs = {key: value for key, value in kwargs.items() if key != "stable_source_rho_max"}
    shared = R0JointRectifiedFlow(**shared_kwargs).to(device)
    _load_best_checkpoint(ROOT / config["lineage"]["shared_EA_checkpoint"], shared, device)
    shared_hash = shared_ea_state_sha256(shared)
    if shared_hash != config["lineage"]["shared_EA_state_sha256"]:
        raise RuntimeError("shared E/A identity changed")

    initial_banks: dict[int, dict[str, torch.Tensor]] = {}
    initial_hashes: dict[int, str] = {}
    for seed in config["training"]["model_seeds"]:
        state, digest = _initial_bank(int(seed), shared, kwargs)
        initial_banks[int(seed)] = state
        initial_hashes[int(seed)] = digest
    if initial_hashes[0] != config["seed0_imports"]["initial_tensor_bank_sha256"]:
        raise RuntimeError("seed0 initial bank no longer matches v3.1")

    completions: dict[tuple[str, int], dict[str, Any]] = {}
    zero_losses: dict[tuple[str, int], float] = {}
    # First perform exactly the remaining 12 new trainings in registered order.
    for name, raw_seed in config["execution_order_remaining_12"]:
        seed = int(raw_seed)
        model = _candidate(name, kwargs, config)
        model.load_state_dict(initial_banks[seed], strict=True)
        if tensor_state_sha256(model.state_dict()) != initial_hashes[seed]:
            raise RuntimeError(f"initial tensor mismatch: {name}/seed{seed}")
        model.freeze_shared(True)
        completion, zero_loss = _fit_new(
            name=name, seed=seed, model=model, bundle=bundle, bank=bank,
            config=config, config_sha=config_sha, code_sha=code["code_sha256"],
            initial_hash=initial_hashes[seed], root=root, device=device, resume=resume,
        )
        completions[(name, seed)] = completion
        zero_losses[(name, seed)] = zero_loss

    # Register three previously observed seed0 candidates without retraining.
    for name in ("T0", "T1_feature", "T1_source"):
        imported = config["seed0_imports"][name]
        completions[(name, 0)] = _read(ROOT / imported["completion"])
        initial_model = _candidate(name, kwargs, config).to(device)
        initial_model.load_state_dict(initial_banks[0], strict=True)
        initial_model.freeze_shared(True)
        zero_losses[(name, 0)] = float(
            bank.evaluate(initial_model, bundle.validation, stage="flow", device=device)["loss"]
        )
        _atomic_json(
            root / "runs" / name / "seed0" / "import_record.json",
            {
                "schema": "architecture_v1_v3_2_seed0_import_v1", "candidate": name,
                "training_seed": 0, "classification": config["seed0_imports"]["classification"],
                "checkpoint": str((ROOT / imported["checkpoint"]).resolve()),
                "checkpoint_sha256": imported["checkpoint_sha256"],
                "selection_target_accessed": False, "calibration_target_accessed": False,
            },
        )

    sampling_base = dict(base)
    sampling_base["formal_sampling"] = dict(base["formal_sampling"])
    for field in ("members", "integration_steps", "method", "member_chunk"):
        sampling_base["formal_sampling"][field] = config["evaluation"][field]

    reports: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in config["candidates"]}
    per_day: dict[str, dict[int, Mapping[str, np.ndarray]]] = {name: {} for name in config["candidates"]}
    for name in config["candidates"]:
        for raw_seed in config["training"]["model_seeds"]:
            seed = int(raw_seed)
            model = _candidate(name, kwargs, config).to(device)
            if seed == 0 and name in config["seed0_imports"]:
                checkpoint = ROOT / config["seed0_imports"][name]["checkpoint"]
            else:
                checkpoint = root / "runs" / name / f"seed{seed}" / "flow" / "best.pt"
            _load_best_checkpoint(checkpoint, model, device)
            if shared_ea_state_sha256(model) != shared_hash:
                raise RuntimeError(f"shared E/A changed: {name}/seed{seed}")
            replicate_metrics: list[Mapping[str, np.ndarray]] = []
            sample_records: list[dict[str, Any]] = []
            for sampling_seed in config["evaluation"]["sampling_seeds"]:
                sampling_seed = int(sampling_seed)
                output_dir = root / "runs" / name / f"seed{seed}" / "validation"
                record_path = output_dir / f"sampling{sampling_seed}_record.json"
                if record_path.exists():
                    if not resume:
                        raise FileExistsError(record_path)
                    record = _read(record_path)
                    metrics = {key: np.asarray(value, dtype=np.float64) for key, value in record["per_day"].items()}
                else:
                    record = _sample_primary(
                        model, bundle, sampling_base, candidate=name, training_seed=seed,
                        sampling_seed=sampling_seed,
                        checkpoint_sha=str(completions[(name, seed)]["best_checkpoint_sha256"]),
                        config_sha=config_sha, code_sha=code["code_sha256"],
                        output_dir=output_dir, device=device,
                    )
                    metrics = record["per_day"]
                    _atomic_json(record_path, record)
                replicate_metrics.append(metrics)
                sample_records.append({key: value for key, value in record.items() if key != "per_day"})
            aggregate = aggregate_sampling_replicates(replicate_metrics)
            source = _source_diagnostic(
                model, bundle,
                seed=int(config["evaluation"]["source_diagnostic_seed"]), device=device,
            )
            gate = _hard_gate(
                completion=completions[(name, seed)], zero_loss=zero_losses[(name, seed)],
                aggregate=aggregate["aggregate"], source=source,
                shared_unchanged=shared_ea_state_sha256(model) == shared_hash,
                rules=config["per_seed_hard_gates"],
            )
            report = {
                "schema": "architecture_v1_temporal_mechanism_v3_2_seed_result_v1",
                "candidate": name, "training_seed": seed,
                "evidence_classification": (
                    config["seed0_imports"]["classification"]
                    if seed == 0 and name in config["seed0_imports"] else "new_v3_2_development_run"
                ),
                "initial_tensor_bank_sha256": initial_hashes[seed],
                "checkpoint_sha256": completions[(name, seed)]["best_checkpoint_sha256"],
                "gate": gate, "sample_records": sample_records,
                "per_day": aggregate["per_day"],
                "selection_target_accessed": False, "calibration_target_accessed": False,
            }
            _atomic_json(root / "runs" / name / f"seed{seed}" / "V3_2_RESULT.json", report)
            reports[name][seed] = report
            per_day[name][seed] = aggregate["per_day"]

    candidate_results: dict[str, Any] = {}
    averaged: dict[str, dict[str, np.ndarray]] = {}
    for name in config["candidates"]:
        seed_gates = {seed: report["gate"] for seed, report in reports[name].items()}
        stability = _seed_stability(seed_gates, config["go_no_go"]["seed_stability"])
        averaged[name] = average_training_seeds(per_day[name])
        candidate_results[name] = {
            "all_seed_hard_gates_passed": all(item["passed"] for item in seed_gates.values()),
            "seed_stability": stability,
            "passed": bool(all(item["passed"] for item in seed_gates.values()) and stability["passed"]),
            "per_seed_gate": seed_gates,
        }
    repetitions = int(config["evaluation"]["paired_calendar_day_bootstrap_repetitions"])
    feature_gate = chronological_vs_control_gate(
        averaged["T0"], averaged["T1_feature"], averaged["T1_feature_shuffle"],
        rules=config["go_no_go"], repetitions=repetitions, seed=41001,
    )
    source_gate = chronological_vs_control_gate(
        averaged["T0"], averaged["T1_source"], averaged["T1_source_shuffle"],
        rules=config["go_no_go"], repetitions=repetitions, seed=42001,
    )
    feature_go = bool(
        candidate_results["T1_feature"]["passed"]
        and candidate_results["T1_feature_shuffle"]["passed"] and feature_gate["passed"]
    )
    source_go = bool(
        candidate_results["T1_source"]["passed"]
        and candidate_results["T1_source_shuffle"]["passed"] and source_gate["passed"]
    )
    status = (
        "V3_2_BOTH_GO" if feature_go and source_go else
        "V3_2_FEATURE_GO" if feature_go else
        "V3_2_SOURCE_GO" if source_go else "V3_2_T1_NO_GO"
    )
    result = {
        "schema": "architecture_v1_temporal_mechanism_v3_2_result_v1",
        "status": status, "formal_config": str(config_path.resolve()),
        "formal_config_sha256": config_sha, "code_manifest": code, "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": shared_hash,
        "initial_tensor_bank_sha256_by_seed": {str(key): value for key, value in initial_hashes.items()},
        "candidate_results": candidate_results,
        "feature_gate": feature_gate, "source_gate": source_gate,
        "feature_authorized": feature_go, "source_authorized": source_go,
        "selection_authorized": False, "selection_state": "sealed",
        "selection_target_accessed": False, "calibration_state": "sealed",
        "calibration_target_accessed": False,
        "next_action": "freeze_retained_temporal_mechanism_then_implement_matched_joint_DDPM",
    }
    result_path = root / "V3_2_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    freeze = {
        "schema": "architecture_v1_temporal_mechanism_v3_2_freeze_v1",
        "status": status, "result": str(result_path.resolve()),
        "result_sha256": result_sha, "training_closed": True,
        "selection_authorized": False,
        "checkpoint_sha256": {
            name: {str(seed): reports[name][seed]["checkpoint_sha256"] for seed in reports[name]}
            for name in reports
        },
    }
    freeze_sha = _atomic_json(root / "training.freeze.json", freeze)
    return {**result, "result_sha256": result_sha, "freeze_sha256": freeze_sha}


def dry_run(config_path: Path) -> dict[str, Any]:
    config, _ = _load_config(config_path)
    return {
        "schema": "architecture_v1_temporal_mechanism_v3_2_dry_run_v1",
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config": str(config_path.resolve()), "config_sha256": file_sha256(config_path),
        "candidates": list(config["candidates"]),
        "training_seeds": config["training"]["model_seeds"],
        "imported_seed0": ["T0", "T1_feature", "T1_source"],
        "new_training_count": len(config["execution_order_remaining_12"]),
        "execution_order_remaining_12": config["execution_order_remaining_12"],
        "target_roles_if_executed": ["train", "validation"],
        "selection_state": "sealed", "calibration_state": "sealed",
        "next_flag": "--execute-training",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute_training:
        parser.error("--resume requires --execute-training")
    path = args.config.resolve()
    result = execute(path, resume=args.resume) if args.execute_training else dry_run(path)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
