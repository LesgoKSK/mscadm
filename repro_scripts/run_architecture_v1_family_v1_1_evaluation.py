#!/usr/bin/env python3
"""Matched validation-scenario evaluation for reused F0 versus D0-v.

This runner reuses the frozen family-v1 evaluation definitions while binding
F0 to the three valid family-v1 best EMA checkpoints and D0 to the three new
family-v1.1 v-prediction best EMA checkpoints.  Both candidates use identical
validation days, members, sampling seeds, E/A allocation law and initial-noise
seed formula.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.family_diffusion import FamilyEMATrainer, MaskedJointDDPM
from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import canonical_sha256, file_sha256
import repro_scripts.run_architecture_v1_family_v1_evaluation as base
from repro_scripts.run_architecture_v1_family_v1_1_p0 import _load_config


CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_1.json"
F0_TRAINING_ROOT = ROOT / "outputs" / "architecture_v1_family_v1" / "formal_training"
D0_TRAINING_ROOT = ROOT / "outputs" / "architecture_v1_family_v1_1" / "formal_training"
OUTPUT_ROOT = ROOT / "outputs" / "architecture_v1_family_v1_1" / "formal_evaluation"
ARCHIVE_SCHEMA = "architecture_v1_family_v1_1_validation_scenarios_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verified_result(root: Path, expected_status: str) -> tuple[dict[str, Any], str]:
    path = root / "TRAINING_RESULT.json"
    digest = base._verified_sidecar(path)
    result = _read(path)
    if result.get("status") != expected_status:
        raise RuntimeError(f"unexpected retained-training status: {path}")
    if result.get("all_training_gates_passed") is not True:
        raise RuntimeError(f"retained training gate failed: {path}")
    for role in ("selection", "calibration"):
        if result.get(f"{role}_target_accessed") is not False:
            raise RuntimeError(f"retained training accessed {role}: {path}")
    return result, digest


def _validate_training() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    f0_result, f0_sha = _verified_result(
        F0_TRAINING_ROOT, "SIX_OF_SIX_TRAINING_COMPLETE"
    )
    d0_result, d0_sha = _verified_result(
        D0_TRAINING_ROOT, "THREE_OF_THREE_D0_V_TRAINING_COMPLETE"
    )
    completions: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in (3, 4, 5):
        f0_record = f0_result["runs"][f"F0_seed{seed}"]
        d0_record = d0_result["runs"][f"D0-v_seed{seed}"]
        for family, record in (("F0", f0_record), ("D0", d0_record)):
            completion_path = Path(record["completion"])
            base._verified_sidecar(completion_path)
            completion = _read(completion_path)
            if completion.get("status") != "complete" or completion.get(
                "training_gate_passed"
            ) is not True:
                raise RuntimeError(f"ineligible completion: {family}/seed{seed}")
            base._verified(
                Path(record["best_checkpoint"]), record["best_checkpoint_sha256"]
            )
            if record["best_checkpoint_sha256"] != completion["best_checkpoint_sha256"]:
                raise RuntimeError("completion/checkpoint SHA mismatch")
            completions[(family, seed)] = completion
    core = {
        "schema": "architecture_v1_family_v1_1_composite_training_identity_v1",
        "reused_F0_training_result_sha256": f0_sha,
        "D0_v_training_result_sha256": d0_sha,
    }
    composite_sha = canonical_sha256(core)
    payload = {
        **core,
        "status": "SIX_MATCHED_BEST_EMA_CHECKPOINTS_READY",
        "all_training_gates_passed": True,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    return {
        "path": str((D0_TRAINING_ROOT / "TRAINING_RESULT.json").resolve()),
        "sha256": composite_sha,
        "payload": payload,
        "F0_result_sha256": f0_sha,
        "D0_v_result_sha256": d0_sha,
    }, completions


def _make_trainer(
    model: T0StableSourceRectifiedFlow, *, family: str, config: Mapping[str, Any]
) -> FamilyEMATrainer:
    training = config["training"]
    diffusion_class = MaskedJointDDPM if family == "F0" else VPredictionJointDDPM
    diffusion = diffusion_class(
        timesteps=int(config["D0_diffusion"]["training_timesteps"]),
        cosine_offset=float(config["D0_diffusion"]["cosine_offset"]),
        beta_min=float(config["D0_diffusion"]["beta_clip"][0]),
        beta_max=float(config["D0_diffusion"]["beta_clip"][1]),
    ).to(next(model.parameters()).device)
    return FamilyEMATrainer(
        model,
        family=family,
        diffusion=diffusion,
        learning_rate=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        ema_decay=float(config["common_EMA_amendment"]["decay"]),
    )


def _code_manifest(config_path: Path) -> dict[str, Any]:
    identities = {
        "F0": F0_TRAINING_ROOT / "identity.json",
        "D0-v": D0_TRAINING_ROOT / "identity.json",
    }
    identity_hashes = {}
    for name, path in identities.items():
        identity_hashes[name] = base._verified_sidecar(path)
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        ROOT / "architecture_v1" / "evaluation.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "mechanism_evaluation.py",
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "family_diffusion_v1_1.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_evaluation.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_1_formal.py",
    ]
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    core = {
        "schema": "architecture_v1_family_v1_1_evaluation_code_v1",
        "files": records,
        "formal_training_identity_sha256": identity_hashes,
    }
    return {**core, "code_sha256": canonical_sha256(core)}


def _matched_random_contract_audit() -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_passed = True
    for sampling_seed in (21000, 21001, 21002):
        reference_path = OUTPUT_ROOT / "archives" / f"F0_seed3_sampling{sampling_seed}.npz"
        with np.load(reference_path, allow_pickle=False) as stored:
            reference_states = stored["states"].copy()
            reference_zero = stored["zero_probability"].copy()
            reference_one = stored["one_probability"].copy()
            reference_day = stored["day"].copy()
            reference_truth = stored["observations"].copy()
            reference_mask = stored["observed_mask"].copy()
        checks: dict[str, bool] = {}
        boundary_counts: dict[str, int] = {}
        for family in ("F0", "D0"):
            for seed in (3, 4, 5):
                name = f"{family}_seed{seed}"
                path = OUTPUT_ROOT / "archives" / f"{name}_sampling{sampling_seed}.npz"
                base._verified_sidecar(path)
                with np.load(path, allow_pickle=False) as stored:
                    states = stored["states"]
                    scenarios = stored["scenarios"]
                    checks[name] = all(
                        (
                            np.array_equal(states, reference_states),
                            np.array_equal(stored["zero_probability"], reference_zero),
                            np.array_equal(stored["one_probability"], reference_one),
                            np.array_equal(stored["day"], reference_day),
                            np.array_equal(stored["observations"], reference_truth),
                            np.array_equal(stored["observed_mask"], reference_mask),
                        )
                    )
                    interior = scenarios[states == 1]
                    boundary_counts[name] = int(
                        np.count_nonzero((interior <= 0.0) | (interior >= 1.0))
                    )
        passed = all(checks.values()) and max(boundary_counts.values()) == 0
        all_passed = all_passed and passed
        records[str(sampling_seed)] = {
            "states_atom_probabilities_days_truth_and_masks_exactly_common": checks,
            "interior_decoded_boundary_count": boundary_counts,
            "passed": passed,
        }
    return {
        "validation_days": 50,
        "members": 100,
        "training_seeds": [3, 4, 5],
        "sampling_seeds": [21000, 21001, 21002],
        "common_day_seed_formula": "sampling_seed + zero_based_validation_day_index*1009",
        "atom_allocation_seed_offset": 1,
        "initial_noise_seed_offset": 2,
        "initial_noise_exactly_common_by_identical_generator_shape_dtype_and_seed": True,
        "per_sampling_seed": records,
        "passed": all_passed,
    }


def _finalize_result(raw: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    audit = _matched_random_contract_audit()
    if audit["passed"] is not True:
        raise RuntimeError("matched validation random-contract audit failed")
    result = dict(raw)
    result["schema"] = "architecture_v1_family_v1_1_formal_comparison_v1"
    result["status"] = str(result["status"]).replace("FAMILY_V1_", "FAMILY_V1_1_")
    if result.get("selected_family") == "D0":
        result["selected_family"] = "D0-v"
    result["candidate_labels"] = {"F0": "reused F0 best EMA", "D0": "D0-v best EMA"}
    result["matched_random_contract"] = audit
    result["training_lineage"] = {
        "reused_F0_training_result_sha256": training["F0_result_sha256"],
        "D0_v_training_result_sha256": training["D0_v_result_sha256"],
        "composite_training_identity_sha256": training["sha256"],
    }
    result["next_action"] = (
        "interpret_and_freeze_family_v1_1_result_without_accessing_selection"
    )
    comparison_path = OUTPUT_ROOT / "FAMILY_COMPARISON.json"
    comparison_sha = base._atomic_json(comparison_path, result)
    freeze = {
        "schema": "architecture_v1_family_v1_1_training_evaluation_freeze_v1",
        "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family_comparison": str(comparison_path.resolve()),
        "family_comparison_sha256": comparison_sha,
        "training_lineage": result["training_lineage"],
        "six_best_checkpoint_sha256": {
            key: value["best_checkpoint_sha256"]
            for key, value in {
                **{
                    f"F0_seed{s}": _read(F0_TRAINING_ROOT / "runs" / "F0" / f"seed{s}" / "completion.json")
                    for s in (3, 4, 5)
                },
                **{
                    f"D0-v_seed{s}": _read(D0_TRAINING_ROOT / "runs" / "D0" / f"seed{s}" / "completion.json")
                    for s in (3, 4, 5)
                },
            }.items()
        },
        "matched_random_contract_passed": True,
        "training_closed": True,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    base._atomic_json(D0_TRAINING_ROOT / "training.freeze.json", freeze)
    return result


def _patch_base() -> None:
    base.DEFAULT_CONFIG = CONFIG
    base.FORMAL_ROOT = D0_TRAINING_ROOT
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.ARCHIVE_SCHEMA = ARCHIVE_SCHEMA
    base._load_config = _load_config
    base._validate_training = _validate_training
    base._make_trainer = _make_trainer
    base._code_manifest = _code_manifest


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    _patch_base()
    comparison = OUTPUT_ROOT / "FAMILY_COMPARISON.json"
    freeze = D0_TRAINING_ROOT / "training.freeze.json"
    if comparison.exists() and freeze.exists():
        base._verified_sidecar(comparison)
        base._verified_sidecar(freeze)
        return _read(comparison)
    training, _ = _validate_training()
    raw = base.execute(config_path, resume=resume)
    return _finalize_result(raw, training)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    _patch_base()
    if not args.execute:
        config, _ = _load_config(config_path)
        training, completions = _validate_training()
        code = _code_manifest(config_path)
        print(json.dumps({
            "schema": "architecture_v1_family_v1_1_evaluation_dry_run_v1",
            "training_status": training["payload"]["status"],
            "composite_training_identity_sha256": training["sha256"],
            "evaluation_code_sha256": code["code_sha256"],
            "best_EMA_checkpoints": {
                f"{family if family == 'F0' else 'D0-v'}_seed{seed}": completions[(family, seed)]["best_checkpoint_sha256"]
                for family in ("F0", "D0") for seed in (3, 4, 5)
            },
            "runs": 6,
            "sampling_replicates_per_run": 3,
            "validation_days": 50,
            "members": int(config["evaluation"]["members"]),
            "matched_NFE": int(config["evaluation"]["primary_matched_NFE"]),
            "common_atom_allocation_and_initial_noise": True,
            "target_role": "validation",
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "next_flag": "--execute"
        }, ensure_ascii=False, indent=2))
        return 0
    result = execute(config_path, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
