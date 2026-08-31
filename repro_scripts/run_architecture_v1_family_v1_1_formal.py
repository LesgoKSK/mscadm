#!/usr/bin/env python3
"""Train family-v1.1 v-prediction D0 for seeds 3/4/5 after P0_GO.

The already-valid family-v1 F0 best EMA checkpoints are hash-bound references;
they are not retrained.  All unchanged formal-training machinery is reused from
family-v1, while the diffusion wrapper is prospectively replaced by v-prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.family_diffusion import FamilyEMATrainer
from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import canonical_sha256, file_sha256, shared_ea_state_sha256
import repro_scripts.run_architecture_v1_family_v1_formal as old
from repro_scripts.run_architecture_v1_family_v1_1_p0 import _load_config


CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_1.json"
P0_ROOT = ROOT / "outputs" / "architecture_v1_family_v1_1" / "P0_preflight"
FORMAL_SCHEMA = "architecture_v1_family_v1_1_formal_training_v1"
COMPLETION_SCHEMA = "architecture_v1_family_v1_1_run_completion_v1"
RUNNER_STATE_SCHEMA = "architecture_v1_family_v1_1_epoch_state_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return value


def _make_trainer(
    model: T0StableSourceRectifiedFlow, *, family: str, config: Mapping[str, Any]
) -> FamilyEMATrainer:
    if family != "D0":
        raise ValueError("family-v1.1 formal runner retrains D0-v only")
    training = config["training"]
    diffusion = VPredictionJointDDPM(
        timesteps=int(config["D0_diffusion"]["training_timesteps"]),
        cosine_offset=float(config["D0_diffusion"]["cosine_offset"]),
        beta_min=float(config["D0_diffusion"]["beta_clip"][0]),
        beta_max=float(config["D0_diffusion"]["beta_clip"][1]),
    ).to(next(model.parameters()).device)
    return FamilyEMATrainer(
        model, family="D0", diffusion=diffusion,
        learning_rate=float(training["learning_rate"]),
        betas=tuple(float(v) for v in training["betas"]),
        eps=float(training["eps"]), weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        ema_decay=float(config["common_EMA_amendment"]["decay"]),
    )


def _validate_p0(config: Mapping[str, Any]) -> dict[str, Any]:
    path = P0_ROOT / "P0_RESULT.json"
    digest = old._verified_sidecar(path)
    result = _read(path)
    required = {
        "schema": "architecture_v1_family_v1_1_P0_result_v1", "status": "P0_GO",
        "passed": True, "all_weights_discarded": True,
        "retained_training_authorized": True,
        "validation_target_accessed": False, "calibration_target_accessed": False,
        "selection_target_accessed": False, "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"family-v1.1 P0 authorization drifted: {key}")
    for family in ("F0", "D0"):
        record = result["family_reports"][family]
        old._verified(Path(record["path"]), record["sha256"])
        detail = _read(Path(record["path"]))
        if record["passed"] is not True or detail["sampling"].get("interior_decoded_boundary_count") != 0:
            raise RuntimeError(f"family-v1.1 P0 sampler gate failed: {family}")
    if list(P0_ROOT.rglob("*.pt")) or list(P0_ROOT.rglob("*.pth")):
        raise RuntimeError("family-v1.1 P0 retained forbidden weights")
    return {"path": str(path.resolve()), "sha256": digest, "payload": result}


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(), config_path.resolve(),
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "family_diffusion_v1_1.py",
        ROOT / "architecture_v1" / "model.py", ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_formal.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_1_p0.py",
    ]
    records = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in paths]
    core = {"schema": "architecture_v1_family_v1_1_formal_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _bind_reused_f0(config: Mapping[str, Any]) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for seed in (3, 4, 5):
        record = config["reused_F0_reference"][f"seed{seed}"]
        path = ROOT / record["path"]
        old._verified(path, record["sha256"])
        bound[f"seed{seed}"] = {"path": str(path.resolve()), "sha256": record["sha256"]}
    return bound


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, model_config = _load_config(config_path)
    p0 = _validate_p0(config)
    device, runtime = old._configure_cuda()
    code = _code_manifest(config_path)
    reused_f0 = _bind_reused_f0(config)
    bundle = build_architecture_v1_fit_data(config_path=old.DATA_CONFIG)
    if bundle.materialized_roles != ("train", "validation"):
        raise RuntimeError("family-v1.1 formal runner materialized forbidden roles")
    if bundle.manifest["fit_data_bundle_sha256"] != config["lineage"]["fit_data_bundle_sha256"]:
        raise RuntimeError("fit-data identity mismatch")
    kwargs = old._model_kwargs(config, model_config)
    shared = old._load_shared_ea(config, kwargs, torch.device("cpu"))
    validation_bank = old._validation_bank_manifest(bundle.validation, config)
    validation_batches = old._validation_batches(
        bundle.validation, batch_days=int(config["training"]["batch_calendar_days"]), device=device
    )
    output_root = ROOT / config["output_root"] / "formal_training"
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "training.freeze.json").exists():
        raise RuntimeError("family-v1.1 retained training is already frozen")

    initial_bank: dict[str, str] = {}
    for seed in (3, 4, 5):
        torch.manual_seed(12000 + seed)
        model = T0StableSourceRectifiedFlow(**kwargs)
        model.load_shared_from(shared, freeze=True)
        initial_bank[str(seed)] = old._model_state_hash(model)
    bank_sha = old._atomic_json(output_root / "initial_tensor_bank.json", {
        "schema": "architecture_v1_family_v1_1_initial_tensor_bank_v1",
        "seeds": [3, 4, 5], "candidate": "D0-v", "tensor_sha256_by_seed": initial_bank,
        "weights_stored": False,
    })
    validation_sha = old._atomic_json(output_root / "validation_bank.json", validation_bank)
    identity = {
        "schema": FORMAL_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()), "config_sha256": file_sha256(config_path),
        "P0_result": p0["path"], "P0_result_sha256": p0["sha256"],
        "family_v1_no_go_freeze_sha256": config["family_v1_no_go_freeze_sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "code_manifest": code, "runtime": runtime,
        "initial_tensor_bank_file_sha256": bank_sha,
        "validation_bank_file_sha256": validation_sha,
        "execution_order": [["D0-v", seed] for seed in (3, 4, 5)],
        "reused_F0_best_EMA": reused_f0,
        "materialized_target_roles": ["train", "validation"],
        "selection_state": "sealed", "calibration_state": "sealed",
    }
    identity_path = output_root / "identity.json"
    if identity_path.exists():
        previous = _read(identity_path)
        if {k: v for k, v in previous.items() if k != "created_utc"} != {k: v for k, v in identity.items() if k != "created_utc"}:
            raise RuntimeError("family-v1.1 formal identity drifted before resume")
    else:
        old._atomic_json(identity_path, identity)
    print(f"[formal-v1.1] runtime={runtime['device_name']}; train={len(bundle.train)}; validation={len(bundle.validation)}; P0=GO", flush=True)

    completions: dict[str, Any] = {}
    try:
        for seed in (3, 4, 5):
            completion = old._run_one(
                family="D0", seed=seed, config=config, kwargs=kwargs, shared=shared,
                bundle=bundle, validation_batches=validation_batches,
                validation_bank=validation_bank, initial_hash=initial_bank[str(seed)],
                code_sha256=code["code_sha256"], p0_sha256=p0["sha256"],
                output_root=output_root, device=device, resume=resume,
            )
            completions[f"D0-v_seed{seed}"] = {
                "completion": str((output_root / "runs" / "D0" / f"seed{seed}" / "completion.json").resolve()),
                "best_checkpoint": completion["best_checkpoint"],
                "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
                "training_gate_passed": completion["training_gate_passed"],
            }
        result = {
            "schema": "architecture_v1_family_v1_1_training_result_v1",
            "status": "THREE_OF_THREE_D0_V_TRAINING_COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(), "runs": completions,
            "reused_F0_best_EMA": reused_f0,
            "all_training_gates_passed": all(v["training_gate_passed"] for v in completions.values()),
            "selection_state": "sealed", "calibration_state": "sealed",
            "selection_target_accessed": False, "calibration_target_accessed": False,
            "next_action": "generate_matched_validation_scenarios_for_reused_F0_vs_D0-v"
        }
        old._atomic_json(output_root / "TRAINING_RESULT.json", result)
        return result
    except Exception as error:
        old._atomic_json(output_root / "TRAINING_FAILURE.json", {
            "schema": "architecture_v1_family_v1_1_training_failure_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(error).__name__, "exception_message": str(error),
            "traceback": traceback.format_exc(), "resume_boundary": "last_complete_epoch_latest_safe.pt",
            "selection_target_accessed": False, "calibration_target_accessed": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    old.DEFAULT_CONFIG = args.config.resolve()
    old.P0_ROOT = P0_ROOT
    old.FORMAL_SCHEMA = FORMAL_SCHEMA
    old.COMPLETION_SCHEMA = COMPLETION_SCHEMA
    old.RUNNER_STATE_SCHEMA = RUNNER_STATE_SCHEMA
    old._make_trainer = _make_trainer
    if not args.execute:
        config, _ = _load_config(args.config.resolve())
        p0 = _validate_p0(config)
        print(json.dumps({"schema": "architecture_v1_family_v1_1_formal_dry_run_v1", "P0_status": p0["payload"]["status"], "execution_order": [["D0-v", s] for s in (3,4,5)], "retrain_F0": False, "next_flag": "--execute"}, indent=2))
        return 0
    result = execute(args.config.resolve(), resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
