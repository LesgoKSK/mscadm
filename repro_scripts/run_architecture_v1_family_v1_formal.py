#!/usr/bin/env python3
"""Run six retained family-v1 training jobs after the train-only P0_GO.

The runner trains F0 and D0 for fresh seeds 3/4/5.  It uses the same model
topology, shared frozen E/A, calendar-day minibatch order, AdamW settings and
EMA rule for both families.  Validation loss is evaluated only with EMA
weights and a fixed four-replicate random bank.  Recovery occurs only at a
complete epoch boundary and restores online weights, EMA, optimizer and RNG.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1.json"
DATA_CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2_2_1.json"
P0_ROOT = ROOT / "outputs" / "architecture_v1_family_v1" / "P0_preflight"
FORMAL_SCHEMA = "architecture_v1_family_v1_formal_training_v1"
COMPLETION_SCHEMA = "architecture_v1_family_v1_run_completion_v1"
RUNNER_STATE_SCHEMA = "architecture_v1_family_v1_epoch_state_v1"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.family_diffusion import FamilyEMATrainer, MaskedJointDDPM
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import (
    ArchitectureBatch,
    canonical_sha256,
    file_sha256,
    parameter_manifest,
    shared_ea_state_sha256,
    tensor_state_sha256,
)
from repro_scripts.run_architecture_v1_family_v1 import (
    _atomic_json,
    _configure_cuda,
    _load_config,
    _load_shared_ea,
    _model_kwargs,
    _stable_seed,
    _verified,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verified_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"file or sidecar missing: {path}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
    return _verified(path, pieces[0])


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _code_manifest(config_path: Path) -> dict[str, Any]:
    p0_identity = _read(P0_ROOT / "identity.json")
    p0_files = p0_identity["code_manifest"]["files"]
    for record in p0_files:
        _verified(ROOT / record["path"], record["sha256"])
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        DATA_CONFIG,
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1.py",
        ROOT / "tests" / "test_architecture_v1_family_v1_runner.py",
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
        "schema": "architecture_v1_family_v1_formal_code_v1",
        "files": records,
        "P0_code_sha256": p0_identity["code_manifest"]["code_sha256"],
    }
    return {**core, "code_sha256": canonical_sha256(core)}


def _validate_p0(config: Mapping[str, Any]) -> dict[str, Any]:
    result_path = P0_ROOT / "P0_RESULT.json"
    digest = _verified_sidecar(result_path)
    result = _read(result_path)
    if result.get("schema") != "architecture_v1_family_v1_P0_result_v1":
        raise ValueError("unexpected family-v1 P0 schema")
    required = {
        "status": "P0_GO",
        "passed": True,
        "all_weights_discarded": True,
        "retained_training_authorized": True,
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }
    for name, expected in required.items():
        if result.get(name) != expected:
            raise RuntimeError(f"P0 authorization field changed: {name}")
    for family in ("F0", "D0"):
        record = result["family_reports"][family]
        _verified(Path(record["path"]), record["sha256"])
        if record["passed"] is not True:
            raise RuntimeError(f"P0 family did not pass: {family}")
    if list(P0_ROOT.rglob("*.pt")) or list(P0_ROOT.rglob("*.pth")):
        raise RuntimeError("P0 retained a forbidden weight file")
    if config["freeze_and_resume"]["selection_and_calibration"] != "remain_sealed":
        raise RuntimeError("formal role lock changed")
    return {"path": str(result_path.resolve()), "sha256": digest, "payload": result}


def _model_state_hash(model: torch.nn.Module) -> str:
    return tensor_state_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )


def _take_batch(split: Any, indices: Sequence[int], device: torch.device) -> ArchitectureBatch:
    return ArchitectureBatch.from_split(split, indices).to(device)


def _epoch_order(length: int, *, seed: int, epoch: int) -> np.ndarray:
    order = np.random.default_rng(int(seed) + int(epoch)).permutation(length)
    order = np.ascontiguousarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(length, dtype=np.int64)):
        raise RuntimeError("formal epoch order is not a bijection")
    return order


def _order_sha256(order: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(order, dtype=np.int64).view(np.uint8)).hexdigest()


def _validation_batches(split: Any, *, batch_days: int, device: torch.device) -> list[ArchitectureBatch]:
    day_values = np.asarray(split.day).astype("datetime64[D]").astype(np.int64)
    order = np.argsort(day_values, kind="stable")
    return [
        _take_batch(split, order[start : start + batch_days], device)
        for start in range(0, len(order), batch_days)
    ]


def _validation_bank_manifest(
    split: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    training = config["training"]
    noise = [int(value) for value in training["validation_bank_noise_seeds"]]
    time_values = [
        int(value) for value in training["validation_bank_time_or_timestep_seeds"]
    ]
    combined = [
        _stable_seed("family-v1-formal-validation-bank", n, t)
        for n, t in zip(noise, time_values)
    ]
    days = np.ascontiguousarray(
        np.asarray(split.day).astype("datetime64[D]").astype(np.int64)
    )
    core = {
        "schema": "architecture_v1_family_v1_validation_bank_v1",
        "role": "validation",
        "day_count": int(len(days)),
        "day_index_sha256": hashlib.sha256(days.view(np.uint8)).hexdigest(),
        "replicates": int(training["validation_bank_replicates"]),
        "noise_seeds": noise,
        "time_or_timestep_seeds": time_values,
        "combined_deterministic_seeds": combined,
        "canonical_day_order": "ascending_calendar_day",
        "EMA_weights_only": True,
    }
    if len(noise) != len(time_values) or len(noise) != core["replicates"]:
        raise RuntimeError("formal validation seed registry is inconsistent")
    return {**core, "sha256": canonical_sha256(core)}


def _evaluate_bank(
    trainer: FamilyEMATrainer,
    batches: Sequence[ArchitectureBatch],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        trainer.evaluate_loss(batches, seed=int(seed))
        for seed in bank["combined_deterministic_seeds"]
    ]
    if not records:
        raise RuntimeError("formal validation bank is empty")
    names = tuple(records[0])
    if any(tuple(record) != names for record in records):
        raise RuntimeError("validation metric keys changed across replicates")
    mean = {
        name: float(np.mean([float(record[name]) for record in records]))
        for name in names
    }
    if not all(math.isfinite(value) for value in mean.values()):
        raise FloatingPointError("formal validation bank is non-finite")
    return {"replicates": records, "mean": mean, "selection_loss": mean["loss"]}


def _gradient_audit(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    training = config["training"]
    start = int(training["gradient_audit_epoch_start_one_based"])
    selected = [record for record in records if int(record["epoch_1_based"]) >= start]
    if not selected:
        raise RuntimeError("formal gradient audit window is empty")
    values = np.asarray(
        [float(record["preclip_gradient_norm"]) for record in selected],
        dtype=np.float64,
    )
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError("formal gradient audit contains non-finite values")
    median = float(np.median(values))
    p99 = float(np.quantile(values, 0.99))
    maximum = float(values.max())
    clip = float(training["gradient_clip"])
    clip_fraction = float(np.mean(values > clip))
    ratio = p99 / median if median > 0.0 else math.inf
    gates = {
        "gradient_clip_fraction": clip_fraction
        <= float(training["gradient_clip_fraction_max"]),
        "preclip_gradient_norm_p99_to_median": ratio
        <= float(training["preclip_gradient_norm_p99_to_median_max"]),
        "preclip_gradient_norm_max": maximum
        <= float(training["preclip_gradient_norm_max"]),
    }
    return {
        "audit_epoch_start_one_based": start,
        "update_count": int(len(values)),
        "median": median,
        "p99": p99,
        "maximum": maximum,
        "p99_to_median": ratio,
        "clip_fraction": clip_fraction,
        "thresholds": {
            "clip_fraction_max": float(training["gradient_clip_fraction_max"]),
            "p99_to_median_max": float(
                training["preclip_gradient_norm_p99_to_median_max"]
            ),
            "maximum": float(training["preclip_gradient_norm_max"]),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _make_trainer(
    model: T0StableSourceRectifiedFlow,
    *,
    family: str,
    config: Mapping[str, Any],
) -> FamilyEMATrainer:
    training = config["training"]
    diffusion = MaskedJointDDPM(
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


def _checkpoint_payload(
    trainer: FamilyEMATrainer,
    *,
    identity: Mapping[str, Any],
    next_epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    stale_validations: int,
    history: Sequence[Mapping[str, Any]],
    gradient_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = trainer.checkpoint_state(identity=identity)
    payload["runner_state"] = {
        "schema": RUNNER_STATE_SCHEMA,
        "next_epoch": int(next_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": int(best_epoch),
        "stale_validations": int(stale_validations),
        "history": list(history),
        "gradient_records": list(gradient_records),
    }
    return payload


def _run_one(
    *,
    family: str,
    seed: int,
    config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    shared: Any,
    bundle: Any,
    validation_batches: Sequence[ArchitectureBatch],
    validation_bank: Mapping[str, Any],
    initial_hash: str,
    code_sha256: str,
    p0_sha256: str,
    output_root: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    run_root = output_root / "runs" / family / f"seed{seed}"
    completion_path = run_root / "completion.json"
    latest_path = run_root / "latest_safe.pt"
    best_path = run_root / "best.pt"
    if completion_path.exists():
        completion = _read(completion_path)
        if completion.get("schema") != COMPLETION_SCHEMA:
            raise ValueError(f"unexpected completion schema: {completion_path}")
        _verified_sidecar(completion_path)
        _verified(best_path, completion["best_checkpoint_sha256"])
        print(f"[formal] {family}/seed{seed}: already complete", flush=True)
        return completion
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FileExistsError(f"existing run requires --resume: {family}/seed{seed}")
    run_root.mkdir(parents=True, exist_ok=True)

    initialization_seed = 12000 + int(seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    model = T0StableSourceRectifiedFlow(**kwargs).to(device)
    model.load_shared_from(shared, freeze=True)
    if _model_state_hash(model) != initial_hash:
        raise RuntimeError(f"initial tensor-bank mismatch: {family}/seed{seed}")
    trainer = _make_trainer(model, family=family, config=config)
    identity = {
        "schema": "architecture_v1_family_v1_run_identity_v1",
        "family": family,
        "training_seed": int(seed),
        "initialization_seed": initialization_seed,
        "config_sha256": file_sha256(DEFAULT_CONFIG),
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "initial_model_tensor_sha256": initial_hash,
        "validation_bank_sha256": validation_bank["sha256"],
        "code_sha256": code_sha256,
        "P0_result_sha256": p0_sha256,
    }
    start_epoch = 0
    best_validation_loss = math.inf
    best_epoch = -1
    stale_validations = 0
    history: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    if latest_path.exists():
        if not resume:
            raise FileExistsError(f"resume flag required: {latest_path}")
        payload = trainer.load_checkpoint(
            latest_path, expected_identity=identity, restore_rng=True
        )
        state = payload.get("runner_state", {})
        if state.get("schema") != RUNNER_STATE_SCHEMA:
            raise ValueError("formal runner checkpoint state is missing")
        start_epoch = int(state["next_epoch"])
        best_validation_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        stale_validations = int(state["stale_validations"])
        history = [dict(value) for value in state["history"]]
        gradient_records = [dict(value) for value in state["gradient_records"]]
        print(
            f"[formal] {family}/seed{seed}: resumed at epoch {start_epoch + 1}",
            flush=True,
        )
    else:
        print(f"[formal] {family}/seed{seed}: fresh start", flush=True)

    training = config["training"]
    batch_days = int(training["batch_calendar_days"])
    maximum_epochs = int(training["maximum_epochs"])
    maximum_updates = int(training["maximum_updates"])
    validate_every = int(training["validate_every_epochs"])
    min_epochs = int(training["minimum_epochs_before_early_stop"])
    patience = int(training["early_stopping_patience_validations"])
    shuffle_seed = int(config["fresh_seed_protocol"]["common_epoch_shuffle_seed"])
    path_seed = int(config["fresh_seed_protocol"]["common_path_noise_seed"])
    stopped_early = False
    run_started = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs):
        epoch_started = time.perf_counter()
        order = _epoch_order(len(bundle.train), seed=shuffle_seed, epoch=epoch)
        epoch_losses: list[float] = []
        for batch_index, start in enumerate(range(0, len(order), batch_days)):
            if trainer.optimizer_updates >= maximum_updates:
                break
            indices = order[start : start + batch_days]
            batch = _take_batch(bundle.train, indices, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(
                _stable_seed("family-v1-train-path", path_seed, seed, epoch, batch_index)
            )
            metrics = trainer.train_step(batch, generator=generator)
            epoch_losses.append(metrics["loss"])
            gradient_records.append(
                {
                    "epoch_1_based": epoch + 1,
                    "batch_index_zero_based": batch_index,
                    "optimizer_update": trainer.optimizer_updates,
                    "preclip_gradient_norm": metrics["gradient_norm"],
                    "clipped": metrics["gradient_was_clipped"] == 1.0,
                }
            )
        if not epoch_losses:
            raise RuntimeError("formal epoch completed no optimizer update")
        epoch_record: dict[str, Any] = {
            "epoch_1_based": epoch + 1,
            "optimizer_updates": trainer.optimizer_updates,
            "train_loss_mean": float(np.mean(epoch_losses)),
            "train_loss_min": float(np.min(epoch_losses)),
            "train_loss_max": float(np.max(epoch_losses)),
            "day_order_sha256": _order_sha256(order),
            "wall_seconds": float(time.perf_counter() - epoch_started),
            "validation": None,
        }
        if (epoch + 1) % validate_every == 0:
            validation = _evaluate_bank(trainer, validation_batches, validation_bank)
            if best_epoch < 0:
                replay = _evaluate_bank(trainer, validation_batches, validation_bank)
                if replay != validation:
                    raise RuntimeError("fixed validation bank failed exact replay")
                validation["initial_replay_exact"] = True
            value = float(validation["selection_loss"])
            required_improvement = max(
                float(training["early_stopping_minimum_delta"]),
                float(training["early_stopping_relative_delta"])
                * abs(best_validation_loss)
                if math.isfinite(best_validation_loss)
                else 0.0,
            )
            improved = (not math.isfinite(best_validation_loss)) or (
                best_validation_loss - value >= required_improvement
            )
            if improved:
                best_validation_loss = value
                best_epoch = epoch + 1
                stale_validations = 0
            else:
                stale_validations += 1
            validation.update(
                {
                    "improved": improved,
                    "required_improvement": required_improvement,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "stale_validations": stale_validations,
                }
            )
            epoch_record["validation"] = validation
        history.append(epoch_record)
        payload = _checkpoint_payload(
            trainer,
            identity=identity,
            next_epoch=epoch + 1,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            stale_validations=stale_validations,
            history=history,
            gradient_records=gradient_records,
        )
        _atomic_torch_save(latest_path, payload)
        if epoch_record["validation"] is not None and epoch_record["validation"]["improved"]:
            _atomic_torch_save(best_path, payload)
        if epoch_record["validation"] is not None:
            print(
                f"[formal] {family}/seed{seed}: epoch {epoch + 1}; "
                f"updates={trainer.optimizer_updates}; "
                f"train={epoch_record['train_loss_mean']:.6g}; "
                f"val={epoch_record['validation']['selection_loss']:.6g}; "
                f"best={best_validation_loss:.6g}@{best_epoch}; "
                f"stale={stale_validations}/{patience}",
                flush=True,
            )
        if (
            epoch + 1 >= min_epochs
            and stale_validations >= patience
            and epoch_record["validation"] is not None
        ):
            stopped_early = True
            break
        if trainer.optimizer_updates >= maximum_updates:
            break
    if not best_path.is_file() or best_epoch < 1:
        raise RuntimeError("formal run completed without a validation-selected checkpoint")
    gradient = _gradient_audit(gradient_records, config)
    best_sha = _verified_sidecar(best_path)
    latest_sha = _verified_sidecar(latest_path)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "family": family,
        "training_seed": int(seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(history),
        "optimizer_updates": trainer.optimizer_updates,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": best_sha,
        "latest_safe_checkpoint": str(latest_path.resolve()),
        "latest_safe_checkpoint_sha256": latest_sha,
        "initial_model_tensor_sha256": initial_hash,
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "EMA_updates": trainer.ema.num_updates,
        "gradient_audit": gradient,
        "training_gate_passed": gradient["passed"],
        "validation_bank_sha256": validation_bank["sha256"],
        "identity": identity,
        "parameter_manifest": parameter_manifest(model),
        "wall_seconds_this_process": float(time.perf_counter() - run_started),
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }
    _atomic_json(completion_path, completion)
    print(
        f"[formal] {family}/seed{seed}: COMPLETE; best epoch {best_epoch}; "
        f"training gate={'PASS' if gradient['passed'] else 'FAIL'}",
        flush=True,
    )
    del trainer, model
    torch.cuda.empty_cache()
    return completion


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, model_config = _load_config(config_path)
    p0 = _validate_p0(config)
    device, runtime = _configure_cuda()
    code = _code_manifest(config_path)
    bundle = build_architecture_v1_fit_data(config_path=DATA_CONFIG)
    if bundle.materialized_roles != ("train", "validation"):
        raise RuntimeError("formal family runner materialized forbidden roles")
    if bundle.manifest["fit_data_bundle_sha256"] != config["lineage"]["fit_data_bundle_sha256"]:
        raise RuntimeError("formal fit-data identity mismatch")
    if bundle.protocol.manifest["protocol_sha256"] != config["lineage"]["protocol_sha256"]:
        raise RuntimeError("formal data protocol identity mismatch")
    kwargs = _model_kwargs(config, model_config)
    shared = _load_shared_ea(config, kwargs, torch.device("cpu"))
    validation_bank = _validation_bank_manifest(bundle.validation, config)
    validation_batches = _validation_batches(
        bundle.validation,
        batch_days=int(config["training"]["batch_calendar_days"]),
        device=device,
    )
    output_root = ROOT / config["output_root"] / "formal_training"
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_path = output_root / "training.freeze.json"
    if freeze_path.exists():
        raise RuntimeError("family-v1 retained training is already frozen")

    initial_bank: dict[str, str] = {}
    for seed in (3, 4, 5):
        torch.manual_seed(12000 + seed)
        model = T0StableSourceRectifiedFlow(**kwargs)
        model.load_shared_from(shared, freeze=True)
        initial_bank[str(seed)] = _model_state_hash(model)
        del model
    bank_payload = {
        "schema": "architecture_v1_family_v1_initial_tensor_bank_v1",
        "seeds": [3, 4, 5],
        "F0_D0_class": "T0StableSourceRectifiedFlow",
        "tensor_sha256_by_seed": initial_bank,
        "same_hash_required_within_seed": True,
        "weights_stored": False,
    }
    initial_bank_sha = _atomic_json(output_root / "initial_tensor_bank.json", bank_payload)
    validation_bank_sha = _atomic_json(
        output_root / "validation_bank.json", validation_bank
    )
    identity = {
        "schema": FORMAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "P0_result": p0["path"],
        "P0_result_sha256": p0["sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "code_manifest": code,
        "runtime": runtime,
        "initial_tensor_bank_file_sha256": initial_bank_sha,
        "validation_bank_file_sha256": validation_bank_sha,
        "execution_order": [
            [family, seed] for seed in (3, 4, 5) for family in ("F0", "D0")
        ],
        "materialized_target_roles": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    identity_path = output_root / "identity.json"
    if identity_path.exists():
        previous = _read(identity_path)
        # created_utc is observational rather than a resume identity field.
        previous_core = {key: value for key, value in previous.items() if key != "created_utc"}
        current_core = {key: value for key, value in identity.items() if key != "created_utc"}
        if previous_core != current_core:
            raise RuntimeError("formal training identity drifted before resume")
    else:
        _atomic_json(identity_path, identity)
    print(
        f"[formal] runtime={runtime['device_name']}; train={len(bundle.train)} days; "
        f"validation={len(bundle.validation)} days; P0=GO",
        flush=True,
    )
    completions: dict[str, Any] = {}
    try:
        for seed in (3, 4, 5):
            for family in ("F0", "D0"):
                completion = _run_one(
                    family=family,
                    seed=seed,
                    config=config,
                    kwargs=kwargs,
                    shared=shared,
                    bundle=bundle,
                    validation_batches=validation_batches,
                    validation_bank=validation_bank,
                    initial_hash=initial_bank[str(seed)],
                    code_sha256=code["code_sha256"],
                    p0_sha256=p0["sha256"],
                    output_root=output_root,
                    device=device,
                    resume=resume,
                )
                completions[f"{family}_seed{seed}"] = {
                    "completion": str(
                        (output_root / "runs" / family / f"seed{seed}" / "completion.json").resolve()
                    ),
                    "best_checkpoint": completion["best_checkpoint"],
                    "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
                    "training_gate_passed": completion["training_gate_passed"],
                }
        result = {
            "schema": "architecture_v1_family_v1_training_result_v1",
            "status": "SIX_OF_SIX_TRAINING_COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runs": completions,
            "all_training_gates_passed": all(
                value["training_gate_passed"] for value in completions.values()
            ),
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
            "next_action": "run_family_v1_validation_scenario_evaluation_before_training_freeze",
        }
        _atomic_json(output_root / "TRAINING_RESULT.json", result)
        return result
    except Exception as error:
        _atomic_json(
            output_root / "TRAINING_FAILURE.json",
            {
                "schema": "architecture_v1_family_v1_training_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "resume_boundary": "last_complete_epoch_latest_safe.pt",
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not args.execute:
        config, _ = _load_config(config_path)
        p0 = _validate_p0(config)
        code = _code_manifest(config_path)
        print(
            json.dumps(
                {
                    "schema": "architecture_v1_family_v1_formal_dry_run_v1",
                    "P0_status": p0["payload"]["status"],
                    "P0_result_sha256": p0["sha256"],
                    "formal_code_sha256": code["code_sha256"],
                    "execution_order": [
                        [family, seed]
                        for seed in (3, 4, 5)
                        for family in ("F0", "D0")
                    ],
                    "allowed_target_roles": config["role_access"][
                        "formal_allowed_target_roles"
                    ],
                    "selection_state": "sealed",
                    "calibration_state": "sealed",
                    "next_flag": "--execute",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = execute(config_path, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
