#!/usr/bin/env python3
"""Train and evaluate the sealed T0/T1 temporal-mechanism panel.

Default invocation is predictor-only.  The retained-weight path requires
``--execute-training`` and materializes only train/validation targets through
the already frozen architecture-v1 formal-v2 fit protocol.
"""

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

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_0.json"
BASE_CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2_2_1.json"
SMOKE_RESULT = ROOT / "outputs" / "architecture_v1_temporal_mechanism_smoke_v3_0" / "SMOKE_RESULT.json"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.formal_evaluation import (
    aggregate_sampling_replicates,
    validation_per_day_metrics,
)
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
    T0MemorylessRectifiedFlow,
    T1FeatureRectifiedFlow,
    T1ShuffleRectifiedFlow,
    T1SourceRectifiedFlow,
)
from repro_scripts.run_architecture_v1_formal_r0 import (
    _atomic_validation_npz,
    _configure_runtime,
    _derive_train_quantities,
    _dispersion_gate,
    _load_best_checkpoint,
    _make_validation_bank,
    _model_kwargs,
    _per_seed_g1,
    _sample_validation_v2_2_chunk_audit,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _verified(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f"registered file identity changed: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        words = sidecar.read_text(encoding="ascii").strip().split()
        if len(words) != 2 or words != [digest, path.name]:
            raise RuntimeError(f"invalid SHA sidecar: {path}")
    return digest


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(path)
    if config.get("schema") != "architecture_v1_temporal_mechanisms_v3_0":
        raise ValueError("unexpected mechanism config schema")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("mechanism role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    if config["go_no_go"]["decision"]["selection_access_authorized"] is not False:
        raise RuntimeError("mechanism experiment cannot authorize selection")
    lineage = config["lineage"]
    result_path = ROOT / lineage["R0_D_result"]
    _verified(result_path, lineage["R0_D_result_sha256"])
    if _read(result_path).get("status") != lineage["R0_D_status_required"]:
        raise RuntimeError("R0-D did not authorize the common-shell experiment")
    _verified(
        ROOT / lineage["shared_EA_checkpoint"],
        lineage["shared_EA_checkpoint_sha256"],
    )
    smoke = _read(SMOKE_RESULT)
    if smoke.get("status") != "SMOKE_PASS" or not all(smoke.get("checks", {}).values()):
        raise RuntimeError("temporal mechanism smoke is absent or failed")
    if smoke.get("config_sha256") != file_sha256(path):
        raise RuntimeError("smoke was not executed under the current mechanism config")
    return config, _read(BASE_CONFIG)


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        BASE_CONFIG,
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "formal_training.py",
        ROOT / "architecture_v1" / "mechanism_evaluation.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_formal_r0.py",
    ]
    records = [
        {
            "path": item.relative_to(ROOT).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in paths
    ]
    core = {"schema": "architecture_v1_temporal_mechanism_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _candidate(name: str, kwargs: Mapping[str, Any], config: Mapping[str, Any]):
    common = dict(kwargs)
    if name == "T0":
        return T0MemorylessRectifiedFlow(**common)
    if name == "T1_feature":
        return T1FeatureRectifiedFlow(**common)
    if name == "T1_source":
        return T1SourceRectifiedFlow(**common)
    order = config["shuffle_control"]["hour_order_zero_based"]
    if name == "T1_feature_shuffle":
        return T1ShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="feature", **common
        )
    if name == "T1_source_shuffle":
        return T1ShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="source", **common
        )
    raise ValueError(f"unknown mechanism candidate: {name}")


def _initial_bank(
    seed: int,
    shared: R0JointRectifiedFlow,
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    torch.manual_seed(12000 + int(seed))
    torch.cuda.manual_seed_all(12000 + int(seed))
    model = T0MemorylessRectifiedFlow(**dict(kwargs))
    model.load_shared_from(shared, freeze=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, tensor_state_sha256(state)


def _fit_candidate_seed(
    *,
    candidate: str,
    seed: int,
    initial_state: Mapping[str, torch.Tensor],
    initial_hash: str,
    shared_hash: str,
    kwargs: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha: str,
    code_sha: str,
    bundle: Any,
    bank: Any,
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[Any, dict[str, Any], float]:
    model = _candidate(candidate, kwargs, config)
    model.load_state_dict(initial_state, strict=True)
    if tensor_state_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError("candidate did not load the common initial tensor bank")
    model.freeze_shared(True)
    if shared_ea_state_sha256(model) != shared_hash:
        raise RuntimeError("candidate shared E/A differs before training")
    stage = config["training"]
    run_root = root / "runs" / candidate / f"seed{seed}"
    trainer = FormalEpochTrainer(
        model,
        run_root,
        resolved_config={
            "schema": "architecture_v1_temporal_mechanism_training_identity_v1",
            "formal_config_sha256": config_sha,
            "candidate": candidate,
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
        run_id=f"temporal_v3_0_{candidate}_seed{seed}",
        device=device,
        extra_identity_hashes={
            "formal_config_file_sha256": config_sha,
            "shared_ea_checkpoint_sha256": config["lineage"]["shared_EA_checkpoint_sha256"],
            "initial_tensor_bank_sha256": initial_hash,
        },
        training_freeze_path=root / "training.freeze.json",
    )
    zero_velocity = float(
        bank.evaluate(model, bundle.validation, stage="flow", device=device)["loss"]
    )
    completion_path = run_root / "flow" / "completion.json"
    if completion_path.exists():
        if not resume:
            raise FileExistsError(f"candidate seed exists; use --resume: {candidate}/{seed}")
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
                "gradient_clip_fraction_max": 0.25,
                "preclip_gradient_norm_p99_to_median_max": 5.0,
                "preclip_gradient_norm_max_over_all_flow_updates": 50.0,
                "require_all_finite": True,
            },
        )
    if shared_ea_state_sha256(model) != shared_hash:
        raise RuntimeError("shared E/A changed during candidate training")
    return model, completion, zero_velocity


def _sample_primary(
    model: Any,
    bundle: Any,
    base: Mapping[str, Any],
    *,
    candidate: str,
    training_seed: int,
    sampling_seed: int,
    checkpoint_sha: str,
    config_sha: str,
    code_sha: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    sampling = base["formal_sampling"]
    parts: list[np.ndarray] = []
    states: list[np.ndarray] = []
    zeros: list[np.ndarray] = []
    ones: list[np.ndarray] = []
    for start in range(0, len(bundle.validation), 2):
        stop = min(start + 2, len(bundle.validation))
        condition = torch.from_numpy(
            np.ascontiguousarray(bundle.validation.condition[start:stop], dtype=np.float32)
        ).to(device)
        scenario = model.sample(
            condition,
            members=int(sampling["members"]),
            steps=int(sampling["integration_steps"]),
            seed=int(sampling_seed + start * 1009),
            method=str(sampling["method"]),
            member_chunk=int(sampling["member_chunk"]),
        )
        parts.append(scenario.values.cpu().numpy().astype(np.float32))
        states.append(scenario.states.cpu().numpy().astype(np.int8))
        zeros.append(scenario.atom_statistics.zero_probability.cpu().numpy().astype(np.float32))
        ones.append(scenario.atom_statistics.one_probability.cpu().numpy().astype(np.float32))
    values = np.concatenate(parts)
    state = np.concatenate(states)
    zero = np.concatenate(zeros)
    one = np.concatenate(ones)
    metrics = validation_per_day_metrics(
        values,
        bundle.validation.target.astype(np.float32),
        bundle.validation.observed_mask.astype(bool),
        zero_probability=zero,
        one_probability=one,
    )
    path = output_dir / f"{candidate}_seed{training_seed}_sampling{sampling_seed}.npz"
    metadata = {
        "schema": "architecture_v1_temporal_mechanism_validation_archive_v1",
        "candidate": candidate,
        "training_seed": training_seed,
        "sampling_seed": sampling_seed,
        "split_role": "validation",
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    archive_sha = _atomic_validation_npz(
        path,
        {
            "scenarios": values,
            "states": state,
            "zero_probability": zero,
            "one_probability": one,
            "observations": bundle.validation.target.astype(np.float32),
            "observed_mask": bundle.validation.observed_mask.astype(bool),
            "day": bundle.validation.day.astype("datetime64[D]"),
            "zones": bundle.validation.zones.astype(np.int64),
        },
        metadata,
    )
    return {
        "sampling_seed": sampling_seed,
        "archive": str(path.resolve()),
        "archive_sha256": archive_sha,
        "per_day": metrics,
        "chunk_audit": None,
    }


def _mechanism_chunk_gate(pair: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rule = config["evaluation"]["chunk_gate"]
    audit = pair["audit"]
    exact_names = (
        "state_exact", "active_mask_exact", "analytic_probability_exact",
        "realized_probability_exact", "atom_probability_exact",
        "zero_probability_exact", "one_probability_exact",
        "same_seed_primary_replay_bitwise_exact", "atom_latent_strictly_zero",
        "atom_boundary_values_exact",
    )
    checks = {name: audit.get(name) is True for name in exact_names}
    checks.update({f"{name}_exact": value is True for name, value in pair["exact_metric_checks"].items()})
    checks["continuous_value_tolerance"] = (
        float(audit["value_absolute_difference"]["maximum"])
        <= float(rule["continuous_value_global_max_abs_difference_max"])
    )
    checks["proper_score_tolerance"] = all(
        float(value) <= float(rule["proper_score_each_aggregate_abs_difference_max"])
        for value in pair["proper_score_absolute_deltas"].values()
    )
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "global_max_abs_difference": float(audit["value_absolute_difference"]["maximum"]),
        "proper_score_absolute_deltas": dict(pair["proper_score_absolute_deltas"]),
        "legacy_allclose_diagnostic": bool(audit["value_allclose"]),
    }


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, base = _load_config(config_path)
    root = ROOT / config["output_root"]
    if (root / "training.freeze.json").exists():
        raise RuntimeError("temporal mechanism training is already frozen")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("mechanism output is non-empty; use --resume")
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    config_sha = file_sha256(config_path)
    code = _code_manifest(config_path)
    bundle = build_architecture_v1_fit_data(config_path=BASE_CONFIG)
    derived = _derive_train_quantities(bundle, base)
    kwargs = _model_kwargs(base, derived)
    bank = _make_validation_bank(
        bundle,
        base,
        evaluation_batch_days=int(config["training"]["batch_days"]),
    )
    shared = R0JointRectifiedFlow(**kwargs).to(device)
    _load_best_checkpoint(ROOT / config["lineage"]["shared_EA_checkpoint"], shared, device)
    shared_hash = shared_ea_state_sha256(shared)
    if shared_hash != config["lineage"]["shared_EA_state_sha256"]:
        raise RuntimeError("shared E/A tensor identity changed")

    candidates = list(config["candidates"])
    seeds = [int(seed) for seed in config["training"]["model_seeds"]]
    per_candidate_seed: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in candidates}
    completions: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in candidates}
    initial_hashes: dict[int, str] = {}
    for seed in seeds:
        initial, initial_hash = _initial_bank(seed, shared, kwargs)
        initial_hashes[seed] = initial_hash
        for candidate in candidates:
            model, completion, zero_velocity = _fit_candidate_seed(
                candidate=candidate,
                seed=seed,
                initial_state=initial,
                initial_hash=initial_hash,
                shared_hash=shared_hash,
                kwargs=kwargs,
                config=config,
                config_sha=config_sha,
                code_sha=code["code_sha256"],
                bundle=bundle,
                bank=bank,
                root=root,
                device=device,
                resume=resume,
            )
            completions[candidate][seed] = completion
            samples: list[dict[str, Any]] = []
            metrics: list[Mapping[str, np.ndarray]] = []
            for sampling_seed in config["evaluation"]["sampling_seeds"]:
                sampling_seed = int(sampling_seed)
                output_dir = root / "runs" / candidate / f"seed{seed}" / "validation"
                audit_scope = config["evaluation"]["chunk_audit_scope"]
                audited = (
                    seed in audit_scope["training_seeds"]
                    and sampling_seed in audit_scope["sampling_seeds"]
                    and candidate in audit_scope["candidates"]
                )
                record_path = output_dir / f"sampling{sampling_seed}_record.json"
                if record_path.exists():
                    if not resume:
                        raise FileExistsError(record_path)
                    record = _read(record_path)
                    record["per_day"] = {
                        key: np.asarray(value, dtype=np.float64)
                        for key, value in record["per_day"].items()
                    }
                elif audited:
                    pair = _sample_validation_v2_2_chunk_audit(
                        model,
                        bundle,
                        base,
                        training_seed=seed,
                        sampling_seed=sampling_seed,
                        checkpoint_sha256=str(completion["best_checkpoint_sha256"]),
                        config_sha256=config_sha,
                        code_sha256=code["code_sha256"],
                        output_dir=output_dir,
                        device=device,
                        candidate_id=candidate,
                    )
                    gate = _mechanism_chunk_gate(pair, config)
                    record = {
                        "sampling_seed": sampling_seed,
                        "archive": pair["primary_archive"],
                        "archive_sha256": pair["primary_archive_sha256"],
                        "per_day": pair["per_day"],
                        "chunk_audit": gate,
                    }
                    _atomic_json(record_path, record)
                else:
                    record = _sample_primary(
                        model,
                        bundle,
                        base,
                        candidate=candidate,
                        training_seed=seed,
                        sampling_seed=sampling_seed,
                        checkpoint_sha=str(completion["best_checkpoint_sha256"]),
                        config_sha=config_sha,
                        code_sha=code["code_sha256"],
                        output_dir=output_dir,
                        device=device,
                    )
                    _atomic_json(record_path, record)
                if record["chunk_audit"] is not None and not record["chunk_audit"]["passed"]:
                    raise RuntimeError(f"chunk audit failed: {candidate}/seed{seed}/{sampling_seed}")
                samples.append({key: value for key, value in record.items() if key != "per_day"})
                metrics.append(record["per_day"])
            aggregate = aggregate_sampling_replicates(metrics)
            semantics = {
                "passed": True,
                "scope": "shared sampler implementation audited at registered seed/candidate pairs",
                "sample_records": samples,
            }
            g1 = _per_seed_g1(
                completion,
                zero_velocity,
                aggregate,
                semantics,
                shared_hash,
                base,
            )
            report = {
                "schema": "architecture_v1_temporal_mechanism_seed_result_v1",
                "candidate": candidate,
                "training_seed": seed,
                "initial_tensor_bank_sha256": initial_hash,
                "G1": g1,
                "per_day": aggregate["per_day"],
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
            }
            _atomic_json(root / "runs" / candidate / f"seed{seed}" / "validation_result.json", report)
            per_candidate_seed[candidate][seed] = report

    candidate_results: dict[str, Any] = {}
    averaged: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        seed_g1 = {seed: item["G1"] for seed, item in per_candidate_seed[candidate].items()}
        dispersion = _dispersion_gate(seed_g1, base)
        averaged[candidate] = average_training_seeds(
            {seed: item["per_day"] for seed, item in per_candidate_seed[candidate].items()}
        )
        candidate_results[candidate] = {
            "all_seed_G1_passed": all(item["passed"] for item in seed_g1.values()),
            "seed_stability": dispersion,
            "passed": bool(all(item["passed"] for item in seed_g1.values()) and dispersion["passed"]),
            "per_seed_G1": seed_g1,
        }

    repetitions = int(config["evaluation"]["paired_calendar_day_bootstrap_repetitions"])
    feature_gate = chronological_vs_control_gate(
        averaged["T0"],
        averaged["T1_feature"],
        averaged["T1_feature_shuffle"],
        rules=config["go_no_go"],
        repetitions=repetitions,
        seed=41001,
    )
    source_gate = chronological_vs_control_gate(
        averaged["T0"],
        averaged["T1_source"],
        averaged["T1_source_shuffle"],
        rules=config["go_no_go"],
        repetitions=repetitions,
        seed=42001,
    )
    feature_pass = bool(candidate_results["T1_feature"]["passed"] and candidate_results["T1_feature_shuffle"]["passed"] and feature_gate["passed"])
    source_pass = bool(candidate_results["T1_source"]["passed"] and candidate_results["T1_source_shuffle"]["passed"] and source_gate["passed"])
    status = (
        "T1_BOTH_GO" if feature_pass and source_pass
        else "T1_FEATURE_GO" if feature_pass
        else "T1_SOURCE_GO" if source_pass
        else "T1_NO_GO"
    )
    result = {
        "schema": "architecture_v1_temporal_mechanism_result_v1",
        "status": status,
        "formal_config": str(config_path.resolve()),
        "formal_config_sha256": config_sha,
        "code_manifest": code,
        "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": shared_hash,
        "initial_tensor_bank_sha256_by_seed": {str(key): value for key, value in initial_hashes.items()},
        "candidate_results": candidate_results,
        "feature_gate": feature_gate,
        "source_gate": source_gate,
        "feature_authorized": feature_pass,
        "source_authorized": source_pass,
        "selection_authorized": False,
        "selection_state": "sealed",
        "selection_target_accessed": False,
        "calibration_state": "sealed",
        "calibration_target_accessed": False,
        "next_action": "implement_matched_joint_DDPM_after_mechanism_result_without_opening_selection",
    }
    result_path = root / "TEMPORAL_MECHANISM_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    freeze = {
        "schema": "architecture_v1_temporal_mechanism_training_freeze_v1",
        "status": status,
        "result": str(result_path.resolve()),
        "result_sha256": result_sha,
        "training_closed": True,
        "selection_authorized": False,
        "checkpoint_sha256": {
            candidate: {
                str(seed): completion["best_checkpoint_sha256"]
                for seed, completion in values.items()
            }
            for candidate, values in completions.items()
        },
    }
    freeze_sha = _atomic_json(root / "training.freeze.json", freeze)
    return {**result, "result_sha256": result_sha, "freeze_sha256": freeze_sha}


def dry_run(config_path: Path) -> dict[str, Any]:
    config, _ = _load_config(config_path)
    return {
        "schema": "architecture_v1_temporal_mechanism_dry_run_v1",
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "candidates": list(config["candidates"]),
        "training_seeds": config["training"]["model_seeds"],
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
