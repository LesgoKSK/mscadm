#!/usr/bin/env python3
"""Run the sealed single-seed v3.1 temporal-mechanism hard-gate pilot."""

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

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_1.json"
BASE_CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2_2_1.json"
SMOKE_RESULT = ROOT / "outputs" / "architecture_v1_temporal_mechanism_v3_1_smoke" / "SMOKE_RESULT.json"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.formal_evaluation import aggregate_sampling_replicates
from architecture_v1.formal_training import (
    FormalEpochTrainer,
    canonical_sha256,
    file_sha256,
    shared_ea_state_sha256,
    tensor_state_sha256,
)
from architecture_v1.model import (
    R0JointRectifiedFlow,
    T0StableSourceRectifiedFlow,
    T1FeatureStableSourceRectifiedFlow,
    T1SourceStableRectifiedFlow,
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


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(path)
    if config.get("schema") != "architecture_v1_temporal_mechanisms_v3_1":
        raise ValueError("unexpected v3.1 mechanism config schema")
    if config.get("status") != "frozen_before_any_retained_v3_1_pilot_training":
        raise RuntimeError("v3.1 pilot config is not frozen")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("pilot role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    lineage = config["lineage"]
    _verified(ROOT / lineage["v3_0_freeze"], lineage["v3_0_freeze_sha256"])
    if _read(ROOT / lineage["v3_0_freeze"]).get("status") != lineage["v3_0_status"]:
        raise RuntimeError("v3.0 No-Go lineage changed")
    result_path = ROOT / lineage["R0_D_result"]
    _verified(result_path, lineage["R0_D_result_sha256"])
    if _read(result_path).get("status") != lineage["R0_D_status_required"]:
        raise RuntimeError("R0-D did not authorize the mechanism pilot")
    _verified(ROOT / lineage["shared_EA_checkpoint"], lineage["shared_EA_checkpoint_sha256"])
    smoke = _read(SMOKE_RESULT)
    if smoke.get("status") != "PASS" or not all(smoke.get("gates", {}).values()):
        raise RuntimeError("v3.1 synthetic smoke is absent or failed")
    if smoke.get("config_sha256") != file_sha256(path):
        raise RuntimeError("v3.1 smoke used a different config")
    return config, _read(BASE_CONFIG)


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        BASE_CONFIG,
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "formal_training.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_formal_r0.py",
        ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanisms_v3_0.py",
        ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanism_v3_1_smoke.py",
    ]
    records = [
        {"path": item.relative_to(ROOT).as_posix(), "bytes": item.stat().st_size, "sha256": file_sha256(item)}
        for item in paths
    ]
    core = {"schema": "architecture_v1_temporal_mechanism_v3_1_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _candidate(name: str, kwargs: Mapping[str, Any]):
    if name == "T0":
        return T0StableSourceRectifiedFlow(**dict(kwargs))
    if name == "T1_feature":
        return T1FeatureStableSourceRectifiedFlow(**dict(kwargs))
    if name == "T1_source":
        return T1SourceStableRectifiedFlow(**dict(kwargs))
    raise ValueError(f"candidate is not in the v3.1 hard-gate pilot: {name}")


def _initial_bank(
    seed: int,
    shared: R0JointRectifiedFlow,
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    torch.manual_seed(12000 + seed)
    torch.cuda.manual_seed_all(12000 + seed)
    model = T0StableSourceRectifiedFlow(**dict(kwargs))
    model.load_shared_from(shared, freeze=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, tensor_state_sha256(state)


def _fit(
    *,
    name: str,
    model: Any,
    bundle: Any,
    bank: Any,
    config: Mapping[str, Any],
    config_sha: str,
    code_sha: str,
    initial_hash: str,
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], float]:
    stage = config["pilot_training"]
    run_root = root / "runs" / name / "seed0"
    trainer = FormalEpochTrainer(
        model,
        run_root,
        resolved_config={
            "schema": "architecture_v1_temporal_mechanism_v3_1_pilot_identity_v1",
            "formal_config_sha256": config_sha,
            "candidate": name,
            "training": stage,
            "initial_tensor_bank_sha256": initial_hash,
        },
        protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
        data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
        code_sha256=code_sha,
        validation_bank=bank,
        training_seed=0,
        shuffle_seed=int(stage["common_epoch_shuffle_seed"]),
        path_seed=int(stage["common_flow_path_seed"]),
        run_id=f"temporal_v3_1_pilot_{name}_seed0",
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
            raise FileExistsError(f"candidate exists; use --resume: {name}")
        completion = _read(completion_path)
        _load_best_checkpoint(run_root / "flow" / "best.pt", model, device)
    else:
        completion = trainer.fit_stage(
            "flow",
            bundle.train,
            bundle.validation,
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
            gradient_audit_epoch_start=31,
            gradient_audit_gate={
                "gradient_clip_fraction_max": float(config["pilot_hard_gates"]["gradient_clip_fraction_max"]),
                "preclip_gradient_norm_p99_to_median_max": 5.0,
                "preclip_gradient_norm_max_over_all_flow_updates": 50.0,
                "require_all_finite": True,
            },
        )
    return completion, zero_loss


@torch.no_grad()
def _source_diagnostic(model: Any, bundle: Any, *, seed: int, device: torch.device) -> dict[str, float]:
    total = 0.0
    total_square = 0.0
    count = 0
    inactive_max = 0.0
    rho_max = 0.0
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
        rho_max = max(rho_max, float(model.stable_source.correlation(encoded).abs().max()))
    mean = total / count
    variance = (total_square - count * mean * mean) / (count - 1)
    return {
        "active_count": count,
        "empirical_mean": mean,
        "empirical_variance": variance,
        "inactive_max_abs": inactive_max,
        "rho_abs_max": rho_max,
    }


def _hard_gate(
    *,
    completion: Mapping[str, Any],
    zero_loss: float,
    aggregate: Mapping[str, float],
    source: Mapping[str, float],
    shared_unchanged: bool,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    audit = completion.get("preclip_gradient_audit", {})
    clip_fraction = float(audit.get("clip_fraction", float("inf")))
    finite_values = [*aggregate.values(), *source.values(), zero_loss, float(completion["best_validation_loss"])]
    checks = {
        "completion": completion.get("status") == "complete",
        "finite_training_sampling_and_metrics": bool(np.isfinite(finite_values).all()),
        "shared_EA_unchanged": shared_unchanged,
        "flow_improvement": float(completion["best_validation_loss"]) < zero_loss,
        "gradient_clip_fraction": clip_fraction <= float(rules["gradient_clip_fraction_max"]),
        "coverage90_lower": float(aggregate["coverage90"]) >= float(rules["coverage90_min"]),
        "coverage90_upper": float(aggregate["coverage90"]) <= float(rules["coverage90_max"]),
        "width90_lower": float(aggregate["width90"]) >= float(rules["width90_min"]),
        "width90_upper": float(aggregate["width90"]) <= float(rules["width90_max"]),
        "level_CRPS": float(aggregate["level_CRPS"]) <= float(rules["level_CRPS_max"]),
        "ramp_CRPS": float(aggregate["ramp_CRPS"]) <= float(rules["ramp_CRPS_max"]),
        "normalized_joint_ES": float(aggregate["normalized_joint_ES"]) <= float(rules["normalized_joint_ES_max"]),
        "source_inactive_zero": float(source["inactive_max_abs"]) <= float(rules["source_inactive_max_abs"]),
        "source_rho_bounded": float(source["rho_abs_max"]) <= float(rules["source_rho_abs_max"]),
        "source_mean": abs(float(source["empirical_mean"])) <= float(rules["source_empirical_mean_abs_max"]),
        "source_variance_lower": float(source["empirical_variance"]) >= float(rules["source_empirical_variance_min"]),
        "source_variance_upper": float(source["empirical_variance"]) <= float(rules["source_empirical_variance_max"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "gradient_clip_fraction": clip_fraction,
        "zero_velocity_fixed_bank_loss": zero_loss,
        "best_fixed_bank_loss": float(completion["best_validation_loss"]),
        "validation_aggregate": dict(aggregate),
        "source_diagnostic": dict(source),
    }


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, base = _load_config(config_path)
    root = ROOT / config["output_root"]
    if (root / "training.freeze.json").exists():
        raise RuntimeError("v3.1 pilot is already frozen")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("pilot output is non-empty; use --resume")
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    config_sha = file_sha256(config_path)
    code = _code_manifest(config_path)
    bundle = build_architecture_v1_fit_data(config_path=BASE_CONFIG)
    derived = _derive_train_quantities(bundle, base)
    kwargs = _model_kwargs(base, derived)
    kwargs["stable_source_rho_max"] = float(config["common_model"]["stable_source_rho_max"])
    bank = _make_validation_bank(bundle, base, evaluation_batch_days=int(config["pilot_training"]["batch_days"]))
    shared = R0JointRectifiedFlow(**{key: value for key, value in kwargs.items() if key != "stable_source_rho_max"}).to(device)
    _load_best_checkpoint(ROOT / config["lineage"]["shared_EA_checkpoint"], shared, device)
    shared_hash = shared_ea_state_sha256(shared)
    if shared_hash != config["lineage"]["shared_EA_state_sha256"]:
        raise RuntimeError("shared E/A identity changed")
    initial, initial_hash = _initial_bank(0, shared, kwargs)

    # Make the imported sampler use the prospectively registered v3.1 chunk.
    sampling_base = dict(base)
    sampling_base["formal_sampling"] = dict(base["formal_sampling"])
    sampling_base["formal_sampling"]["members"] = int(config["pilot_evaluation"]["members"])
    sampling_base["formal_sampling"]["integration_steps"] = int(config["pilot_evaluation"]["integration_steps"])
    sampling_base["formal_sampling"]["method"] = str(config["pilot_evaluation"]["method"])
    sampling_base["formal_sampling"]["member_chunk"] = int(config["pilot_evaluation"]["primary_member_chunk"])

    reports: dict[str, Any] = {}
    completions: dict[str, Any] = {}
    for name in config["pilot_candidates"]:
        model = _candidate(name, kwargs)
        model.load_state_dict(initial, strict=True)
        if tensor_state_sha256(model.state_dict()) != initial_hash:
            raise RuntimeError("candidate initial tensor bank differs")
        model.freeze_shared(True)
        completion, zero_loss = _fit(
            name=name,
            model=model,
            bundle=bundle,
            bank=bank,
            config=config,
            config_sha=config_sha,
            code_sha=code["code_sha256"],
            initial_hash=initial_hash,
            root=root,
            device=device,
            resume=resume,
        )
        completions[name] = completion
        output_dir = root / "runs" / name / "seed0" / "validation"
        record_path = output_dir / "sampling21000_record.json"
        if record_path.exists():
            if not resume:
                raise FileExistsError(record_path)
            record = _read(record_path)
            per_day = {key: np.asarray(value, dtype=np.float64) for key, value in record["per_day"].items()}
        else:
            record = _sample_primary(
                model,
                bundle,
                sampling_base,
                candidate=name,
                training_seed=0,
                sampling_seed=int(config["pilot_evaluation"]["sampling_seed"]),
                checkpoint_sha=str(completion["best_checkpoint_sha256"]),
                config_sha=config_sha,
                code_sha=code["code_sha256"],
                output_dir=output_dir,
                device=device,
            )
            per_day = record["per_day"]
            _atomic_json(record_path, record)
        aggregate = aggregate_sampling_replicates([per_day])["aggregate"]
        source = _source_diagnostic(
            model,
            bundle,
            seed=int(config["pilot_evaluation"]["source_diagnostic_seed"]),
            device=device,
        )
        gate = _hard_gate(
            completion=completion,
            zero_loss=zero_loss,
            aggregate=aggregate,
            source=source,
            shared_unchanged=shared_ea_state_sha256(model) == shared_hash,
            rules=config["pilot_hard_gates"],
        )
        report = {
            "schema": "architecture_v1_temporal_mechanism_v3_1_candidate_pilot_v1",
            "candidate": name,
            "training_seed": 0,
            "initial_tensor_bank_sha256": initial_hash,
            "checkpoint_sha256": completion["best_checkpoint_sha256"],
            "gate": gate,
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
        }
        _atomic_json(root / "runs" / name / "seed0" / "PILOT_RESULT.json", report)
        reports[name] = report

    passed = bool(all(item["gate"]["passed"] for item in reports.values()))
    status = "PILOT_GO_EXPAND_SHUFFLES_AND_THREE_SEEDS" if passed else "PILOT_NO_GO"
    result = {
        "schema": "architecture_v1_temporal_mechanism_v3_1_pilot_result_v1",
        "status": status,
        "formal_config": str(config_path.resolve()),
        "formal_config_sha256": config_sha,
        "code_manifest": code,
        "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": shared_hash,
        "initial_tensor_bank_sha256": initial_hash,
        "candidate_reports": reports,
        "all_three_required": True,
        "all_three_passed": passed,
        "selection_authorized": False,
        "selection_state": "sealed",
        "selection_target_accessed": False,
        "calibration_state": "sealed",
        "calibration_target_accessed": False,
        "next_action": config["pilot_hard_gates"]["if_pass" if passed else "if_fail"],
    }
    result_path = root / "PILOT_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    freeze = {
        "schema": "architecture_v1_temporal_mechanism_v3_1_pilot_freeze_v1",
        "status": status,
        "result": str(result_path.resolve()),
        "result_sha256": result_sha,
        "training_closed": True,
        "selection_authorized": False,
        "checkpoint_sha256": {name: item["best_checkpoint_sha256"] for name, item in completions.items()},
    }
    freeze_sha = _atomic_json(root / "training.freeze.json", freeze)
    return {**result, "result_sha256": result_sha, "freeze_sha256": freeze_sha}


def dry_run(config_path: Path) -> dict[str, Any]:
    config, _ = _load_config(config_path)
    return {
        "schema": "architecture_v1_temporal_mechanism_v3_1_pilot_dry_run_v1",
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "candidates": list(config["pilot_candidates"]),
        "training_seeds": [config["pilot_training"]["model_seed"]],
        "target_roles_if_executed": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
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
