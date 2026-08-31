#!/usr/bin/env python3
"""Validation-only R0 stability attribution pilot.

This runner never materializes calibration, selection, R-SEEN, or final targets.
It compares two three-seed panels with one shared E/A checkpoint:

* B: independent shuffle/path randomness across flow seeds;
* C: common shuffle/path randomness across flow seeds.

The seed-0 flow is identical in B and C and is loaded from formal-v2.2.1.
The pilot is engineering evidence only and cannot authorize selection or T0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "repro_configs" / "architecture_v1_r0_stability_v2_3_pilot.json"

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
)
from architecture_v1.model import R0JointRectifiedFlow
from repro_scripts.run_architecture_v1_formal_r0 import (
    _configure_runtime,
    _derive_train_quantities,
    _load_best_checkpoint,
    _make_validation_bank,
    _model_kwargs,
    _sample_validation_once,
)


SCHEMA = "architecture_v1_r0_stability_v2_3_pilot_result_v1"
PRIMARY = (
    "level_CRPS",
    "ramp_CRPS",
    "normalized_joint_ES",
    "coverage90",
    "width90",
    "zero_Brier",
    "atom_state_Brier",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_tmp.replace(sidecar)
    return digest


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verified(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"registered file identity changed: {path}")


def _load_config(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    source = path.resolve()
    config = _read_json(source)
    if config.get("schema") != "architecture_v1_r0_stability_v2_3_pilot_v1":
        raise ValueError("unexpected v2.3 pilot schema")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("pilot target-role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection and calibration must remain sealed")
    base_path = _resolve(config["base_formal_config"]["path"])
    result_path = _resolve(config["base_result"]["path"])
    _verified(base_path, config["base_formal_config"]["sha256"])
    _verified(result_path, config["base_result"]["sha256"])
    base = _read_json(base_path)
    old_result = _read_json(result_path)
    if old_result.get("status") != "G1_NO_GO":
        raise RuntimeError("formal-v2.2.1 No-Go lineage changed")
    for item in (config["shared_ea"], config["reference_seed0"]):
        key = "checkpoint" if "checkpoint" in item else "flow_checkpoint"
        hash_key = "checkpoint_sha256" if key == "checkpoint" else "flow_checkpoint_sha256"
        _verified(_resolve(item[key]), item[hash_key])
    return config, base_path, base, result_path


def _code_manifest(config_path: Path, base_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        base_path.resolve(),
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
    core = {"schema": "architecture_v1_r0_stability_v2_3_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _spread(metrics: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    values = {
        name: np.asarray([metrics[seed][name] for seed in (0, 1, 2)], dtype=np.float64)
        for name in PRIMARY
    }
    absolute = {name: float(value.max() - value.min()) for name, value in values.items()}
    relative_joint = absolute["normalized_joint_ES"] / float(
        np.median(values["normalized_joint_ES"])
    )
    return {
        "values_seed_order_0_1_2": {name: value.tolist() for name, value in values.items()},
        "absolute_spreads": absolute,
        "normalized_joint_ES_relative_spread": relative_joint,
    }


def _target_checks(spread: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    target = config["attribution_decision"]["formal_stability_targets_unchanged"]
    absolute = spread["absolute_spreads"]
    return {
        "level_CRPS": absolute["level_CRPS"] <= target["level_CRPS_max_minus_min_max"],
        "ramp_CRPS": absolute["ramp_CRPS"] <= target["ramp_CRPS_max_minus_min_max"],
        "normalized_joint_ES": spread["normalized_joint_ES_relative_spread"]
        <= target["normalized_joint_ES_relative_max_minus_min_max"],
    }


def _load_replicate(path: Path, base_bundle: Any) -> dict[str, np.ndarray]:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    words = sidecar.read_text(encoding="ascii").strip().split()
    if len(words) != 2 or words[1] != path.name or words[0] != file_sha256(path):
        raise RuntimeError(f"scenario sidecar verification failed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        return validation_per_day_metrics(
            stored["scenarios"],
            base_bundle.validation.target,
            base_bundle.validation.observed_mask,
            zero_probability=stored["zero_probability"],
            one_probability=stored["one_probability"],
        )


def _evaluate_model(
    model: R0JointRectifiedFlow,
    *,
    arm: str,
    training_seed: int,
    checkpoint_sha256: str,
    bundle: Any,
    base_config: Mapping[str, Any],
    pilot_config: Mapping[str, Any],
    config_sha256: str,
    code_sha256: str,
    root: Path,
    device: torch.device,
) -> dict[str, Any]:
    scenario_dir = root / "arms" / arm / f"seed{training_seed}" / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    replicates: list[Mapping[str, np.ndarray]] = []
    archive_records: list[dict[str, Any]] = []
    for sampling_seed in pilot_config["evaluation"]["sampling_seeds"]:
        archive = scenario_dir / f"sampling{sampling_seed}.npz"
        if archive.is_file():
            per_day = _load_replicate(archive, bundle)
            record = {
                "archive": str(archive.resolve()),
                "archive_sha256": file_sha256(archive),
                "sampling_seed": int(sampling_seed),
                "resumed_existing_archive": True,
            }
        else:
            result = _sample_validation_once(
                model,
                bundle,
                base_config,
                training_seed=training_seed,
                sampling_seed=int(sampling_seed),
                checkpoint_sha256=checkpoint_sha256,
                config_sha256=config_sha256,
                code_sha256=code_sha256,
                output_path=archive,
                device=device,
            )
            per_day = result["per_day"]
            record = {
                key: value
                for key, value in result.items()
                if key not in {"per_day", "metadata"}
            }
            record["sampling_seed"] = int(sampling_seed)
            record["resumed_existing_archive"] = False
        replicates.append(per_day)
        archive_records.append(record)
    aggregated = aggregate_sampling_replicates(replicates)
    result = {
        "schema": "architecture_v1_r0_stability_v2_3_arm_seed_v1",
        "arm": arm,
        "training_seed": training_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "aggregate": {name: aggregated["aggregate"][name] for name in PRIMARY},
        "per_day": {name: aggregated["per_day"][name] for name in PRIMARY},
        "archives": archive_records,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    path = root / "arms" / arm / f"seed{training_seed}" / "result.json"
    _atomic_json(path, result)
    return result


def _fit_or_load_flow(
    *,
    arm: str,
    seed: int,
    shared_model: R0JointRectifiedFlow,
    bundle: Any,
    bank: Any,
    kwargs: Mapping[str, Any],
    pilot_config: Mapping[str, Any],
    code_sha256: str,
    config_sha256: str,
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[R0JointRectifiedFlow, dict[str, Any]]:
    arm_spec = pilot_config["arms"][arm]
    init_seed = 12000 + seed
    shuffle_seed = 13000 + seed if arm.startswith("B_") else 13000
    path_seed = 14000 + seed if arm.startswith("B_") else 14000
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    model = R0JointRectifiedFlow(**kwargs)
    model.load_shared_from(shared_model, freeze=True)
    expected_shared = shared_ea_state_sha256(shared_model)
    if shared_ea_state_sha256(model) != expected_shared:
        raise RuntimeError("shared E/A identity changed while building a pilot flow")
    run_root = root / "arms" / arm / f"seed{seed}"
    completion_path = run_root / "flow" / "completion.json"
    trainer = FormalEpochTrainer(
        model,
        run_root,
        resolved_config={
            "schema": "architecture_v1_r0_stability_v2_3_flow_identity_v1",
            "pilot_config_sha256": config_sha256,
            "arm": arm,
            "arm_spec": arm_spec,
            "flow_training": pilot_config["flow_training"],
        },
        protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
        data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
        code_sha256=code_sha256,
        validation_bank=bank,
        training_seed=seed,
        shuffle_seed=shuffle_seed,
        path_seed=path_seed,
        run_id=f"r0_stability_v2_3_{arm}_seed{seed}",
        device=device,
        extra_identity_hashes={
            "pilot_config_file_sha256": config_sha256,
            "shared_ea_checkpoint_sha256": pilot_config["shared_ea"]["checkpoint_sha256"],
        },
        training_freeze_path=root / "pilot.freeze.json",
    )
    if completion_path.is_file():
        if not resume:
            raise FileExistsError(f"completed pilot arm exists; use --resume: {completion_path}")
        completion = _read_json(completion_path)
        _load_best_checkpoint(run_root / "flow" / "best.pt", model, device)
    else:
        latest = run_root / "flow" / "latest_safe.pt"
        stage = pilot_config["flow_training"]
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
        )
    if shared_ea_state_sha256(model) != expected_shared:
        raise RuntimeError("shared E/A changed during pilot flow fitting")
    return model, completion


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    pilot, base_path, base, old_result_path = _load_config(config_path)
    root = _resolve(pilot["output_root"])
    if (root / "pilot.freeze.json").exists():
        raise RuntimeError("v2.3 attribution pilot is already frozen")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("pilot output is non-empty; use --resume")
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    code = _code_manifest(config_path, base_path)
    config_sha = file_sha256(config_path)
    bundle = build_architecture_v1_fit_data(config_path=base_path)
    derived = _derive_train_quantities(bundle, base)
    kwargs = _model_kwargs(base, derived)
    bank = _make_validation_bank(
        bundle,
        base,
        evaluation_batch_days=int(pilot["flow_training"]["batch_days"]),
    )
    shared = R0JointRectifiedFlow(**kwargs).to(device)
    _load_best_checkpoint(_resolve(pilot["shared_ea"]["checkpoint"]), shared, device)
    expected_shared = shared_ea_state_sha256(shared)

    # The existing seed-0 flow is the exact shared reference for both B and C.
    reference = R0JointRectifiedFlow(**kwargs).to(device)
    _load_best_checkpoint(
        _resolve(pilot["reference_seed0"]["flow_checkpoint"]), reference, device
    )
    if shared_ea_state_sha256(reference) != expected_shared:
        raise RuntimeError("reference flow does not contain the registered shared E/A")
    reference_result = _evaluate_model(
        reference,
        arm="shared_reference",
        training_seed=0,
        checkpoint_sha256=pilot["reference_seed0"]["flow_checkpoint_sha256"],
        bundle=bundle,
        base_config=base,
        pilot_config=pilot,
        config_sha256=config_sha,
        code_sha256=code["code_sha256"],
        root=root,
        device=device,
    )

    results: dict[str, dict[int, dict[str, Any]]] = {
        "B_shared_ea_independent_training_path": {0: reference_result},
        "C_shared_ea_common_training_path": {0: reference_result},
    }
    completions: dict[str, dict[int, Any]] = {arm: {} for arm in results}
    for arm in results:
        for seed in (1, 2):
            model, completion = _fit_or_load_flow(
                arm=arm,
                seed=seed,
                shared_model=shared,
                bundle=bundle,
                bank=bank,
                kwargs=kwargs,
                pilot_config=pilot,
                code_sha256=code["code_sha256"],
                config_sha256=config_sha,
                root=root,
                device=device,
                resume=resume,
            )
            completions[arm][seed] = completion
            results[arm][seed] = _evaluate_model(
                model,
                arm=arm,
                training_seed=seed,
                checkpoint_sha256=str(completion["best_checkpoint_sha256"]),
                bundle=bundle,
                base_config=base,
                pilot_config=pilot,
                config_sha256=config_sha,
                code_sha256=code["code_sha256"],
                root=root,
                device=device,
            )

    old = _read_json(old_result_path)
    A_metrics = {
        seed: old["per_seed_gate"][str(seed)]["validation_aggregate"]
        for seed in (0, 1, 2)
    }
    panels: dict[str, Any] = {"A_independent_full_pipeline": _spread(A_metrics)}
    for arm, seed_results in results.items():
        metrics = {seed: item["aggregate"] for seed, item in seed_results.items()}
        panels[arm] = _spread(metrics)
    for panel in panels.values():
        panel["formal_target_checks"] = _target_checks(panel, pilot)
        panel["all_three_primary_targets_pass"] = bool(
            all(panel["formal_target_checks"].values())
        )
    A = panels["A_independent_full_pipeline"]
    B = panels["B_shared_ea_independent_training_path"]
    C = panels["C_shared_ea_common_training_path"]
    attribution = {}
    for metric in ("level_CRPS", "ramp_CRPS"):
        attribution[metric] = {
            "A_minus_B_EA_contribution": A["absolute_spreads"][metric]
            - B["absolute_spreads"][metric],
            "B_minus_C_shuffle_path_contribution": B["absolute_spreads"][metric]
            - C["absolute_spreads"][metric],
            "C_residual_initialization_optimization": C["absolute_spreads"][metric],
        }
    attribution["normalized_joint_ES_relative"] = {
        "A_minus_B_EA_contribution": A["normalized_joint_ES_relative_spread"]
        - B["normalized_joint_ES_relative_spread"],
        "B_minus_C_shuffle_path_contribution": B["normalized_joint_ES_relative_spread"]
        - C["normalized_joint_ES_relative_spread"],
        "C_residual_initialization_optimization": C[
            "normalized_joint_ES_relative_spread"
        ],
    }
    result = {
        "schema": SCHEMA,
        "status": "PILOT_COMPLETE_ATTRIBUTION_ONLY",
        "pilot_config": str(config_path.resolve()),
        "pilot_config_sha256": config_sha,
        "base_result": str(old_result_path.resolve()),
        "base_result_status": old["status"],
        "code_manifest": code,
        "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": expected_shared,
        "panels": panels,
        "attribution": attribution,
        "flow_completions": {
            arm: {
                str(seed): {
                    "best_epoch": item["best_epoch"],
                    "best_validation_loss": item["best_validation_loss"],
                    "best_checkpoint": item["best_checkpoint"],
                    "best_checkpoint_sha256": item["best_checkpoint_sha256"],
                }
                for seed, item in values.items()
            }
            for arm, values in completions.items()
        },
        "candidate_D_frozen": False,
        "T0_authorized": False,
        "selection_state": "sealed",
        "selection_target_accessed": False,
        "calibration_state": "sealed",
        "calibration_target_accessed": False,
        "next_action": "interpret_A_B_C_then_freeze_one_minimal_candidate_D",
    }
    result_path = root / "PILOT_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    freeze = {
        "schema": "architecture_v1_r0_stability_v2_3_pilot_freeze_v1",
        "status": "pilot_frozen",
        "pilot_result": str(result_path.resolve()),
        "pilot_result_sha256": result_sha,
        "candidate_D_frozen": False,
        "selection_authorized": False,
    }
    _atomic_json(root / "pilot.freeze.json", freeze)
    return {**result, "path": str(result_path.resolve()), "file_sha256": result_sha}


def dry_run(config_path: Path) -> dict[str, Any]:
    pilot, base_path, _, _ = _load_config(config_path)
    return {
        "schema": "architecture_v1_r0_stability_v2_3_pilot_dry_run_v1",
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "base_config": str(base_path.resolve()),
        "output_root": str(_resolve(pilot["output_root"])),
        "unique_new_flow_runs": 4,
        "reference_flow_runs_reused": 1,
        "target_roles_if_executed": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_flag": "--execute-pilot"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.resume and not args.execute_pilot:
        parser.error("--resume requires --execute-pilot")
    result = execute(config_path, resume=args.resume) if args.execute_pilot else dry_run(config_path)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
