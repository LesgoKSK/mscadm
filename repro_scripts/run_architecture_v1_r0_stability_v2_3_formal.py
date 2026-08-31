#!/usr/bin/env python3
"""Prospective formal verification of the minimal R0 stability repair D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "repro_configs" / "architecture_v1_r0_stability_v2_3_formal.json"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.formal_evaluation import aggregate_sampling_replicates
from architecture_v1.formal_training import (
    FormalEpochTrainer,
    canonical_sha256,
    file_sha256,
    shared_ea_state_sha256,
)
from architecture_v1.model import R0JointRectifiedFlow
from repro_scripts.run_architecture_v1_formal_r0 import (
    _configure_runtime,
    _derive_train_quantities,
    _dispersion_gate,
    _load_best_checkpoint,
    _make_validation_bank,
    _model_kwargs,
    _per_seed_g1,
    _sample_validation_v2_2_chunk_audit,
)


RESULT_SCHEMA = "architecture_v1_r0_stability_v2_3_formal_result_v1"


def _read_json(path: Path) -> dict[str, Any]:
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = sidecar.with_name(sidecar.name + ".tmp")
    side_tmp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    side_tmp.replace(sidecar)
    return digest


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, digest: str) -> None:
    if not path.is_file() or file_sha256(path) != digest:
        raise RuntimeError(f"registered identity changed: {path}")


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = _read_json(path)
    if config.get("schema") != "architecture_v1_r0_stability_v2_3_formal_v1":
        raise ValueError("unexpected v2.3 formal schema")
    if config["role_access"]["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("formal D role allowlist changed")
    if config["role_access"]["selection_state"] != "sealed" or config["role_access"]["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    lineage = config["lineage"]
    base_path = _resolve(lineage["base_formal_config"])
    _verify(base_path, lineage["base_formal_config_sha256"])
    _verify(_resolve(lineage["base_gate"]), lineage["base_gate_sha256"])
    pilot_path = _resolve(lineage["pilot_result"])
    _verify(pilot_path, lineage["pilot_result_sha256"])
    if _read_json(pilot_path).get("status") != "PILOT_COMPLETE_ATTRIBUTION_ONLY":
        raise RuntimeError("v2.3 attribution pilot identity/status changed")
    _verify(_resolve(config["shared_ea"]["checkpoint"]), config["shared_ea"]["checkpoint_sha256"])
    _verify(_resolve(config["shared_ea"]["G0_result"]), config["shared_ea"]["G0_result_sha256"])
    if _read_json(_resolve(config["shared_ea"]["G0_result"])).get("passed") is not True:
        raise RuntimeError("registered shared E/A G0 did not pass")
    return config, _read_json(base_path), base_path


def _code_manifest(config_path: Path, base_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        base_path.resolve(),
        _resolve(config["lineage"]["base_gate"]),
        PROJECT_ROOT / "architecture_v1" / "data.py",
        PROJECT_ROOT / "architecture_v1" / "formal_evaluation.py",
        PROJECT_ROOT / "architecture_v1" / "formal_training.py",
        PROJECT_ROOT / "architecture_v1" / "model.py",
        PROJECT_ROOT / "architecture_v1" / "training.py",
        PROJECT_ROOT / "repro_scripts" / "run_architecture_v1_formal_r0.py",
    ]
    records = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    core = {"schema": "architecture_v1_r0_stability_v2_3_formal_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _new_sampling_gate(pair: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rule = config["sampling_semantics_gate"]
    audit = pair["audit"]
    exact_audit = (
        "state_exact",
        "active_mask_exact",
        "analytic_probability_exact",
        "realized_probability_exact",
        "atom_probability_exact",
        "zero_probability_exact",
        "one_probability_exact",
        "same_seed_primary_replay_bitwise_exact",
        "atom_latent_strictly_zero",
        "atom_boundary_values_exact",
    )
    exact_metric = pair["exact_metric_checks"]
    checks = {name: bool(audit.get(name) is True) for name in exact_audit}
    checks.update({f"{name}_exact": bool(value) for name, value in exact_metric.items()})
    checks["continuous_value_global_max_abs_difference"] = (
        float(audit["value_absolute_difference"]["maximum"])
        <= float(rule["continuous_value_global_max_abs_difference_max"])
    )
    checks["proper_score_deltas"] = all(
        float(value) <= float(rule["proper_score_each_aggregate_abs_difference_max"])
        for value in pair["proper_score_absolute_deltas"].values()
    )
    return {
        "schema": "architecture_v1_r0_stability_v2_3_sampling_gate_v1",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "per_cell_allclose_diagnostic": bool(audit["value_allclose"]),
        "global_max_abs_difference": float(audit["value_absolute_difference"]["maximum"]),
        "proper_score_absolute_deltas": dict(pair["proper_score_absolute_deltas"]),
        "thresholds": {
            "continuous_value_global_max_abs_difference_max": rule["continuous_value_global_max_abs_difference_max"],
            "proper_score_each_aggregate_abs_difference_max": rule["proper_score_each_aggregate_abs_difference_max"],
        },
    }


def _load_pair(path: Path) -> dict[str, Any]:
    words = path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip().split()
    if len(words) != 2 or words[1] != path.name or words[0] != file_sha256(path):
        raise RuntimeError(f"pair JSON sidecar mismatch: {path}")
    result = _read_json(path)
    result["per_day"] = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in result["per_day"].items()
    }
    return result


def _fit_or_load(
    *,
    seed: int,
    shared: R0JointRectifiedFlow,
    kwargs: Mapping[str, Any],
    bundle: Any,
    bank: Any,
    config: Mapping[str, Any],
    config_sha: str,
    code_sha: str,
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[R0JointRectifiedFlow, dict[str, Any], float]:
    stage = config["flow_training"]
    torch.manual_seed(12000 + seed)
    torch.cuda.manual_seed_all(12000 + seed)
    model = R0JointRectifiedFlow(**kwargs)
    model.load_shared_from(shared, freeze=True)
    shared_hash = shared_ea_state_sha256(shared)
    if shared_ea_state_sha256(model) != shared_hash:
        raise RuntimeError("shared E/A changed while constructing formal D")
    run_root = root / "runs" / f"seed{seed}"
    trainer = FormalEpochTrainer(
        model,
        run_root,
        resolved_config={
            "schema": "architecture_v1_r0_stability_v2_3_formal_flow_identity_v1",
            "formal_config_sha256": config_sha,
            "candidate_id": config["candidate_id"],
            "flow_training": stage,
        },
        protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
        data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
        code_sha256=code_sha,
        validation_bank=bank,
        training_seed=seed,
        shuffle_seed=int(stage["common_epoch_shuffle_seed"]),
        path_seed=int(stage["common_flow_path_seed"]),
        run_id=f"r0_stability_v2_3_formal_D_seed{seed}",
        device=device,
        extra_identity_hashes={
            "formal_config_file_sha256": config_sha,
            "shared_ea_checkpoint_sha256": config["shared_ea"]["checkpoint_sha256"],
        },
        training_freeze_path=root / "training.freeze.json",
    )
    zero_velocity = bank.evaluate(model, bundle.validation, stage="flow", device=device)["loss"]
    completion_path = run_root / "flow" / "completion.json"
    if completion_path.is_file():
        if not resume:
            raise FileExistsError(f"formal D seed exists; use --resume: {seed}")
        completion = _read_json(completion_path)
        _load_best_checkpoint(run_root / "flow" / "best.pt", model, device)
    else:
        latest = run_root / "flow" / "latest_safe.pt"
        gradient = stage["gradient_audit"]
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
            resume=bool(resume and latest.is_file()),
            restore_best=True,
            record_update_preclip_gradient_norms=True,
            gradient_audit_epoch_start=int(gradient["epoch_start_1_based_inclusive"]),
            gradient_audit_gate={
                "gradient_clip_fraction_max": gradient["gradient_clip_fraction_max"],
                "preclip_gradient_norm_p99_to_median_max": gradient["preclip_gradient_norm_p99_to_median_max"],
                "preclip_gradient_norm_max_over_all_flow_updates": gradient["preclip_gradient_norm_max_over_all_flow_updates"],
                "require_all_finite": True,
            },
        )
    if shared_ea_state_sha256(model) != shared_hash:
        raise RuntimeError("shared E/A changed during formal D flow training")
    return model, completion, float(zero_velocity)


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, base, base_path = _load_config(config_path)
    root = _resolve(config["output_root"])
    if (root / "training.freeze.json").exists():
        raise RuntimeError("formal D is already frozen")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("formal D output is non-empty; use --resume")
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    config_sha = file_sha256(config_path)
    code = _code_manifest(config_path, base_path, config)
    bundle = build_architecture_v1_fit_data(config_path=base_path)
    derived = _derive_train_quantities(bundle, base)
    kwargs = _model_kwargs(base, derived)
    bank = _make_validation_bank(
        bundle, base, evaluation_batch_days=int(config["flow_training"]["batch_days"])
    )
    shared = R0JointRectifiedFlow(**kwargs).to(device)
    _load_best_checkpoint(_resolve(config["shared_ea"]["checkpoint"]), shared, device)
    shared_hash = shared_ea_state_sha256(shared)

    seed_reports: dict[int, dict[str, Any]] = {}
    completions: dict[int, dict[str, Any]] = {}
    for seed in config["flow_training"]["model_seeds"]:
        seed = int(seed)
        model, completion, zero_velocity = _fit_or_load(
            seed=seed,
            shared=shared,
            kwargs=kwargs,
            bundle=bundle,
            bank=bank,
            config=config,
            config_sha=config_sha,
            code_sha=code["code_sha256"],
            root=root,
            device=device,
            resume=resume,
        )
        completions[seed] = completion
        replicates = []
        pairs = []
        for sampling_seed in config["evaluation"]["sampling_seeds"]:
            pair_path = root / "runs" / f"seed{seed}" / "chunk_audit" / f"sampling{sampling_seed}_pair.json"
            if pair_path.is_file():
                if not resume:
                    raise FileExistsError(pair_path)
                pair = _load_pair(pair_path)
            else:
                pair = _sample_validation_v2_2_chunk_audit(
                    model,
                    bundle,
                    base,
                    training_seed=seed,
                    sampling_seed=int(sampling_seed),
                    checkpoint_sha256=str(completion["best_checkpoint_sha256"]),
                    config_sha256=config_sha,
                    code_sha256=code["code_sha256"],
                    output_dir=pair_path.parent,
                    device=device,
                )
                pair["formal_D_gate"] = _new_sampling_gate(pair, config)
                _atomic_json(pair_path, pair)
            if "formal_D_gate" not in pair:
                pair["formal_D_gate"] = _new_sampling_gate(pair, config)
            replicates.append(pair["per_day"])
            pairs.append(pair)
        aggregate = aggregate_sampling_replicates(replicates)
        semantics = {
            "schema": "architecture_v1_r0_stability_v2_3_formal_sampling_semantics_v1",
            "pair_results": [
                {
                    "sampling_seed": pair["sampling_seed"],
                    "primary_archive": pair["primary_archive"],
                    "alternate_archive": pair["alternate_archive"],
                    "formal_D_gate": pair["formal_D_gate"],
                }
                for pair in pairs
            ],
            "passed": bool(all(pair["formal_D_gate"]["passed"] for pair in pairs)),
        }
        g1 = _per_seed_g1(
            completion,
            zero_velocity,
            aggregate,
            semantics,
            shared_hash,
            base,
        )
        seed_report = {
            "schema": "architecture_v1_r0_stability_v2_3_formal_seed_v1",
            "training_seed": seed,
            "G1": g1,
            "per_day": aggregate["per_day"],
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
        }
        _atomic_json(root / "runs" / f"seed{seed}" / "G1_validation.json", seed_report)
        seed_reports[seed] = g1

    dispersion = _dispersion_gate(seed_reports, base)
    overall = bool(all(item["passed"] for item in seed_reports.values()) and dispersion["passed"])
    status = "D_G1_GO" if overall else "D_G1_NO_GO"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "overall_passed": overall,
        "candidate_id": config["candidate_id"],
        "formal_config": str(config_path.resolve()),
        "formal_config_sha256": config_sha,
        "code_manifest": code,
        "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": shared_hash,
        "per_seed_G1": {str(seed): report for seed, report in seed_reports.items()},
        "three_seed_dispersion": dispersion,
        "T0_common_shell_implementation_authorized": overall,
        "selection_authorized": False,
        "selection_state": "sealed",
        "selection_target_accessed": False,
        "calibration_state": "sealed",
        "calibration_target_accessed": False,
        "next_action": (
            "freeze_T0_common_shell_code_and_training_plan_without_opening_selection"
            if overall
            else "stop_before_T0_and_audit_formal_D"
        ),
    }
    result_path = root / "FORMAL_D_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    freeze = {
        "schema": "architecture_v1_r0_stability_v2_3_formal_freeze_v1",
        "status": "formal_D_frozen",
        "formal_D_result": str(result_path.resolve()),
        "formal_D_result_sha256": result_sha,
        "candidate_id": config["candidate_id"],
        "training_closed": True,
        "T0_common_shell_implementation_authorized": overall,
        "selection_authorized": False,
        "checkpoint_sha256_by_seed": {
            str(seed): completion["best_checkpoint_sha256"]
            for seed, completion in completions.items()
        },
    }
    freeze_sha = _atomic_json(root / "training.freeze.json", freeze)
    return {**result, "path": str(result_path.resolve()), "file_sha256": result_sha, "freeze_sha256": freeze_sha}


def dry_run(config_path: Path) -> dict[str, Any]:
    config, _, base_path = _load_config(config_path)
    return {
        "schema": "architecture_v1_r0_stability_v2_3_formal_dry_run_v1",
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "base_config": str(base_path.resolve()),
        "candidate_id": config["candidate_id"],
        "flow_seeds": config["flow_training"]["model_seeds"],
        "common_shuffle_seed": config["flow_training"]["common_epoch_shuffle_seed"],
        "common_path_seed": config["flow_training"]["common_flow_path_seed"],
        "target_roles_if_executed": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_flag": "--execute-formal-D",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-formal-D", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute_formal_D:
        parser.error("--resume requires --execute-formal-D")
    path = args.config.resolve()
    result = execute(path, resume=args.resume) if args.execute_formal_D else dry_run(path)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
