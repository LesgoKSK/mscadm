#!/usr/bin/env python3
"""Retained-training runner for the frozen TGO-v1 attribution Probe.

Default execution is target free.  ``--execute-training`` first fits one
shared train-only atom nuisance per outer fold and then trains the registered
84 denoisers.  Only final EMA states are retained.  Outer-held-out
reconstruction, scenario generation, and path comparison are deliberately not
implemented here; they become authorized only after an exact 84/84 freeze.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    ROOT / "repro_configs" / "architecture_v1_transition_object_probe.json"
)
RUNNER_PATH = Path(__file__).resolve()
FORMAL_ROOT_NAME = "formal_training"
GLOBAL_SCHEMA = "architecture_v1_tgo_v1_training_identity_v1"
ATOM_IDENTITY_SCHEMA = "architecture_v1_tgo_v1_atom_identity_v1"
ATOM_RESUME_SCHEMA = "architecture_v1_tgo_v1_atom_resume_v1"
ATOM_FINAL_SCHEMA = "architecture_v1_tgo_v1_atom_final_ema_v1"
ATOM_COMPLETION_SCHEMA = "architecture_v1_tgo_v1_atom_completion_v1"
RUN_IDENTITY_SCHEMA = "architecture_v1_tgo_v1_run_identity_v1"
RUN_RESUME_SCHEMA = "architecture_v1_tgo_v1_run_resume_v1"
RUN_FINAL_SCHEMA = "architecture_v1_tgo_v1_final_ema_v1"
RUN_COMPLETION_SCHEMA = "architecture_v1_tgo_v1_run_completion_v1"
PROGRESS_SCHEMA = "architecture_v1_tgo_v1_training_progress_v1"
FREEZE_SCHEMA = "architecture_v1_tgo_v1_training_freeze_v1"
ATOM_CHECKPOINT_INTERVAL = 255

from architecture_v1.g0b_tiny_denoiser import (
    TINY_DENOISER_PARAMETERS,
    TinyEMA,
    module_state_sha256,
    tensor_mapping_sha256,
)
from architecture_v1.protocol import date_list_sha256
from architecture_v1.transition_atom import (
    TransitionAtomNuisance,
    train_only_atom_contract,
)
from architecture_v1.transition_object import (
    MaskConditionedOperator,
    fit_level_rms,
    fit_operator_scale,
)
from architecture_v1.transition_probe import (
    PATH_IDS,
    TransitionDenoisingSystem,
)
from repro_scripts import run_architecture_v1_transition_object_p0 as p0_runner
from repro_scripts.run_architecture_v1_g0_b_tiny_denoiser import (
    _baseline_schedule,
    _replay_nuisance,
)
from repro_scripts.run_architecture_v1_g0_b_tiny_denoiser_formal import (
    _array_sha256,
    _atomic_json,
    _atomic_torch,
    _canonical_sha256,
    _read_json,
    _remove_resume,
    _sha256,
    _verified_sidecar,
)


@dataclass(frozen=True)
class RunSpec:
    outer_fold: int
    model_seed: int
    path_id: str

    @property
    def key(self) -> str:
        return f"fold{self.outer_fold}__seed{self.model_seed}__{self.path_id}"

    def manifest(self) -> dict[str, Any]:
        return {
            "run_key": self.key,
            "outer_fold": self.outer_fold,
            "model_seed": self.model_seed,
            "path_id": self.path_id,
        }


def _matrix(config: Mapping[str, Any]) -> tuple[RunSpec, ...]:
    matrix = config["paired_training_matrix"]
    specs = tuple(
        RunSpec(int(fold), int(seed), str(path_id))
        for fold in matrix["outer_folds"]
        for seed in matrix["model_seeds"]
        for path_id in matrix["paths"]
    )
    if tuple(matrix["paths"]) != PATH_IDS:
        raise RuntimeError("TGO-v1 formal path order drifted")
    if len(specs) != 84 or int(matrix["retained_runs"]) != len(specs):
        raise RuntimeError("TGO-v1 formal matrix must contain exactly 84 runs")
    if len({spec.key for spec in specs}) != len(specs):
        raise RuntimeError("TGO-v1 formal run keys are not unique")
    return specs


def _validate_p0(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    p0_path = ROOT / config["output_root"] / "TGO_V1_P0_RESULT.json"
    digest = _verified_sidecar(p0_path)
    result = _read_json(p0_path)
    if result.get("schema") != p0_runner.RESULT_SCHEMA:
        raise RuntimeError("TGO-v1 P0 schema drifted")
    if result.get("status") != config["P0"]["go_status"]:
        raise RuntimeError("retained training requires TGO-v1 P0 Go")
    if result.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("TGO-v1 P0/config identity drifted")
    if result.get("role_access", {}).get("materialized_roles") != ["train"]:
        raise RuntimeError("TGO-v1 P0 materialized a non-train target role")
    if (
        result.get("role_access", {}).get("outer_test_TGO_metrics_constructed")
        is not False
        or result.get("artifacts", {}).get("weights_written") is not False
        or result.get("artifacts", {}).get("checkpoint_written") is not False
        or not all(value is True for value in result.get("hard_gates", {}).values())
    ):
        raise RuntimeError("TGO-v1 P0 boundary or hard-gate record drifted")
    expected_code = {
        "transition_object": _sha256(
            ROOT / "architecture_v1" / "transition_object.py"
        ),
        "transition_probe": _sha256(
            ROOT / "architecture_v1" / "transition_probe.py"
        ),
        "transition_atom": _sha256(
            ROOT / "architecture_v1" / "transition_atom.py"
        ),
        "runner": _sha256(
            ROOT / "repro_scripts" / "run_architecture_v1_transition_object_p0.py"
        ),
    }
    if result.get("code_sha256") != expected_code:
        raise RuntimeError("TGO-v1 implementation changed after P0")
    return {
        "path": str(p0_path.resolve()),
        "sha256": digest,
        "status": result["status"],
        "device": result["device"],
    }


def _code_identity(config_path: Path) -> dict[str, str]:
    config = _read_json(config_path)
    validated = p0_runner._validate_config(config, config_path)
    return {
        "config_sha256": validated["config_sha256"],
        "transition_object_sha256": _sha256(
            ROOT / "architecture_v1" / "transition_object.py"
        ),
        "transition_probe_sha256": _sha256(
            ROOT / "architecture_v1" / "transition_probe.py"
        ),
        "transition_atom_sha256": _sha256(
            ROOT / "architecture_v1" / "transition_atom.py"
        ),
        "P0_runner_sha256": _sha256(
            ROOT / "repro_scripts" / "run_architecture_v1_transition_object_p0.py"
        ),
        "formal_runner_sha256": _sha256(RUNNER_PATH),
    }


def _configure_device(
    device_name: str, *, allow_cpu: bool
) -> tuple[torch.device, dict[str, Any]]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "formal TGO-v1 training resolved to CPU; pass --allow-cpu only if intentional"
        )
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device, {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "CUDA_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "automatic_mixed_precision": False,
        "deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _rng_state() -> dict[str, Any]:
    return p0_runner._rng_state()


def _set_rng_state(state: Mapping[str, Any]) -> None:
    p0_runner._set_rng_state(state)


def _ema_system_state(
    system: torch.nn.Module, ema: TinyEMA
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
    }
    for name, value in ema.shadow.items():
        if name not in state:
            raise RuntimeError(f"EMA parameter absent from system state: {name}")
        state[name] = value.detach().cpu().clone()
    return state


def _global_identity(
    config: Mapping[str, Any],
    config_path: Path,
    p0: Mapping[str, Any],
    code: Mapping[str, str],
    bundle: Any,
    nuisance_replay: Mapping[str, Any],
    runtime: Mapping[str, Any],
    matrix: Sequence[RunSpec],
) -> dict[str, Any]:
    return {
        "schema": GLOBAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": code["config_sha256"],
        "P0": dict(p0),
        "code_identity": dict(code),
        "runtime": dict(runtime),
        "data_identity": {
            "train_days": int(len(bundle.train)),
            "train_date_sha256": date_list_sha256(bundle.train.day),
            "train_split_array_sha256": bundle.manifest["data_audit"][
                "split_array_sha256"
            ]["train"],
            "train_only_data_bundle_sha256": bundle.manifest[
                "train_only_data_bundle_sha256"
            ],
        },
        "nuisance_replay": dict(nuisance_replay),
        "matrix_sha256": _canonical_sha256(
            {"runs": [spec.manifest() for spec in matrix]}
        ),
        "expected_atom_models": 6,
        "expected_denoiser_runs": len(matrix),
        "materialized_target_roles": ["train"],
        "outer_test_TGO_metrics_constructed": False,
        "scenario_bank_constructed": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }


def _without_created(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_utc"}


def _prepare_global_identity(
    formal_root: Path, identity: Mapping[str, Any], *, resume: bool
) -> str:
    path = formal_root / "training.identity.json"
    if path.exists():
        if not resume:
            raise FileExistsError("formal training identity exists; use --resume")
        _verified_sidecar(path)
        previous = _read_json(path)
        if _without_created(previous) != _without_created(identity):
            raise RuntimeError("TGO-v1 formal global identity drifted")
        return _sha256(path)
    if resume and formal_root.exists() and any(formal_root.iterdir()):
        raise RuntimeError("cannot resume non-empty formal root without identity")
    return _atomic_json(path, identity)


def _operator_context(
    fold: Any, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, MaskConditionedOperator], dict[str, Any]]:
    residual = torch.from_numpy(fold.residual_train).double()
    active_cpu = torch.from_numpy(fold.active_train).bool()
    level_rms = fit_level_rms(residual, active_cpu)
    level = residual / level_rms
    scales = {
        kind: fit_operator_scale(level, active_cpu, kind=kind)
        for kind in (
            "transition_true",
            "transition_wrong",
            "orthogonal_dct",
        )
    }
    operators = {
        kind: MaskConditionedOperator(kind, scale=scale)
        for kind, scale in scales.items()
    }
    arrays = {
        "level": np.ascontiguousarray(level.numpy(), dtype=np.float32),
        "active": np.ascontiguousarray(fold.active_train, dtype=bool),
        "condition": np.ascontiguousarray(fold.condition_train, dtype=np.float32),
    }
    tensors = {
        name: torch.as_tensor(value, device=device)
        for name, value in arrays.items()
    }
    manifest = {
        "outer_train_days": int(len(fold.outer_train)),
        "outer_train_index_sha256": _array_sha256(
            outer_train=np.ascontiguousarray(fold.outer_train)
        ),
        "training_array_sha256": _array_sha256(**arrays),
        "level_RMS": level_rms,
        "operator_scales": scales,
    }
    return tensors, operators, manifest


def _atom_index_bank(
    *, fold: int, train_days: int, updates: int, batch_size: int, seed_root: int = 52000
) -> tuple[torch.Tensor, dict[str, Any]]:
    if min(train_days, updates, batch_size) < 1:
        raise ValueError("atom index-bank dimensions must be positive")
    required = updates * batch_size
    pieces: list[torch.Tensor] = []
    total = 0
    epoch = 0
    while total < required:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_root + 100 * int(fold) + epoch)
        permutation = torch.randperm(train_days, generator=generator)
        pieces.append(permutation)
        total += len(permutation)
        epoch += 1
    bank = torch.cat(pieces)[:required].reshape(updates, batch_size)
    return bank, {
        "seed_formula": "52000+100*outer_fold+epoch_zero_based",
        "epochs_touched": epoch,
        "updates": updates,
        "batch_size": batch_size,
        "sha256": tensor_mapping_sha256({"index": bank}),
    }


def _run_root(formal_root: Path, spec: RunSpec) -> Path:
    return (
        formal_root
        / "runs"
        / f"fold{spec.outer_fold}"
        / f"seed{spec.model_seed}"
        / spec.path_id
    )


def _atom_root(formal_root: Path, fold: int) -> Path:
    return formal_root / "atoms" / f"fold{fold}"


def _atom_identity(
    *,
    config: Mapping[str, Any],
    fold: Any,
    global_identity_sha: str,
    contract: Mapping[str, Any],
    bank_manifest: Mapping[str, Any],
    condition: np.ndarray,
    observation: np.ndarray,
    observed_mask: np.ndarray,
    initialization_sha: str,
    parameters: int,
) -> dict[str, Any]:
    atom = config["shared_atom_contract"]
    return {
        "schema": ATOM_IDENTITY_SCHEMA,
        "outer_fold": int(fold.fold),
        "global_training_identity_sha256": global_identity_sha,
        "outer_train_days": int(len(fold.outer_train)),
        "outer_train_index_sha256": _array_sha256(
            outer_train=np.ascontiguousarray(fold.outer_train)
        ),
        "training_array_sha256": _array_sha256(
            condition=np.ascontiguousarray(condition),
            observation=np.ascontiguousarray(observation),
            observed_mask=np.ascontiguousarray(observed_mask),
        ),
        "train_only_contract": dict(contract),
        "random_bank": dict(bank_manifest),
        "initialization_seed": 51000 + int(fold.fold),
        "initialization_sha256": initialization_sha,
        "parameters": parameters,
        "optimizer": {
            "name": atom["optimizer"],
            "learning_rate": float(atom["learning_rate"]),
            "betas": [float(value) for value in atom["betas"]],
            "eps": float(atom["eps"]),
            "weight_decay": float(atom["weight_decay"]),
            "gradient_clip": float(atom["gradient_clip"]),
        },
        "updates": int(atom["fit_updates"]),
        "EMA_decay": float(atom["EMA_decay"]),
        "checkpoint_interval_updates": ATOM_CHECKPOINT_INTERVAL,
        "selection": atom["selection"],
        "continuous_transport_parameters": "absent",
        "target_role": "train_outer_train_only",
    }


def _atom_summary() -> dict[str, Any]:
    return {
        "updates_completed": 0,
        "loss_first": None,
        "loss_final": None,
        "loss_min": None,
        "loss_max": None,
        "atom_nll_final": None,
        "location_loss_final": None,
        "gradient_norm_max": None,
        "wall_seconds_accumulated": 0.0,
    }


def _update_atom_summary(
    summary: dict[str, Any],
    *,
    update: int,
    loss: float,
    atom_nll: float,
    location: float,
    gradient_norm: float,
) -> None:
    values = np.asarray([loss, atom_nll, location, gradient_norm], dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("atom summary received a non-finite value")
    if summary["loss_first"] is None:
        summary["loss_first"] = loss
        summary["loss_min"] = loss
        summary["loss_max"] = loss
        summary["gradient_norm_max"] = gradient_norm
    summary["updates_completed"] = update
    summary["loss_final"] = loss
    summary["loss_min"] = min(float(summary["loss_min"]), loss)
    summary["loss_max"] = max(float(summary["loss_max"]), loss)
    summary["atom_nll_final"] = atom_nll
    summary["location_loss_final"] = location
    summary["gradient_norm_max"] = max(
        float(summary["gradient_norm_max"]), gradient_norm
    )


def _atom_update(
    model: TransitionAtomNuisance,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    condition: torch.Tensor,
    observation: torch.Tensor,
    observed_mask: torch.Tensor,
    index_bank: torch.Tensor,
    update_zero: int,
    device: torch.device,
    *,
    gradient_clip: float,
) -> tuple[float, float, float, float]:
    index = index_bank[update_zero].to(device)
    optimizer.zero_grad(set_to_none=True)
    losses = model.loss(
        condition.index_select(0, index),
        observation.index_select(0, index),
        observed_mask.index_select(0, index),
    )
    loss = losses["loss"]
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if any(value is None for value in gradients):
        raise RuntimeError("shared atom parameter has no gradient")
    if any(not bool(torch.isfinite(value).all()) for value in gradients if value is not None):
        raise FloatingPointError("shared atom gradient is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(gradient_clip)
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("shared atom gradient norm is non-finite")
    optimizer.step()
    if not p0_runner._optimizer_finite(optimizer):
        raise FloatingPointError("shared atom optimizer state is non-finite")
    if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise FloatingPointError("shared atom parameter is non-finite")
    ema.update(model)
    return (
        float(loss.detach().cpu()),
        float(losses["atom_nll"].detach().cpu()),
        float(losses["interior_location_smooth_l1"].detach().cpu()),
        float(gradient_norm.detach().cpu()),
    )


def _atom_checkpoint_payload(
    identity: Mapping[str, Any],
    identity_sha: str,
    model: TransitionAtomNuisance,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ATOM_RESUME_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "completed_updates": int(summary["updates_completed"]),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "EMA": ema.state_dict(),
        "summary": dict(summary),
        "RNG": _rng_state(),
    }


def _load_atom_resume(
    path: Path,
    identity: Mapping[str, Any],
    identity_sha: str,
    model: TransitionAtomNuisance,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
) -> tuple[int, dict[str, Any]]:
    _verified_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != ATOM_RESUME_SCHEMA
        or payload.get("identity_sha256") != identity_sha
        or payload.get("identity") != identity
    ):
        raise RuntimeError("shared atom resume identity drifted")
    completed = int(payload["completed_updates"])
    maximum = int(identity["updates"])
    interval = int(identity["checkpoint_interval_updates"])
    if not 0 < completed < maximum or completed % interval != 0:
        raise RuntimeError("shared atom resume boundary is invalid")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    ema.load_state_dict(payload["EMA"], model)
    summary = dict(payload["summary"])
    if int(summary["updates_completed"]) != completed or ema.num_updates != completed:
        raise RuntimeError("shared atom resume summary drifted")
    _set_rng_state(payload["RNG"])
    return completed, summary


def _atom_completion_payload(
    *,
    identity: Mapping[str, Any],
    identity_sha: str,
    summary: Mapping[str, Any],
    final_path: Path,
    final_sha: str,
    ema_state_sha: str,
    ema_updates: int,
) -> dict[str, Any]:
    return {
        "schema": ATOM_COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": int(identity["outer_fold"]),
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "optimizer_updates": int(identity["updates"]),
        "EMA_updates": int(ema_updates),
        "summary": dict(summary),
        "final_EMA_checkpoint": str(final_path.resolve()),
        "final_EMA_checkpoint_sha256": final_sha,
        "final_EMA_system_state_sha256": ema_state_sha,
        "only_final_EMA_weight_retained": True,
        "resume_checkpoint_deleted": not (final_path.parent / "resume.pt").exists(),
        "target_role": "train_outer_train_only",
        "held_out_target_accessed": False,
        "continuous_transport_parameters": "absent",
    }


def _validate_atom_completion(
    atom_root: Path, *, identity_sha: str, fold: int
) -> dict[str, Any] | None:
    completion_path = atom_root / "run.json"
    final_path = atom_root / "final_ema.pt"
    if not completion_path.exists() and not final_path.exists():
        return None
    if completion_path.exists() and not final_path.exists():
        raise RuntimeError(f"incomplete shared atom artifact pair: fold {fold}")
    if final_path.exists() and not completion_path.exists():
        final_sha = _verified_sidecar(final_path)
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != ATOM_FINAL_SCHEMA
            or payload.get("identity_sha256") != identity_sha
            or int(payload.get("completed_updates", -1)) != 4080
            or int(payload.get("EMA_updates", -1)) != 4080
            or tensor_mapping_sha256(payload["EMA_system_state"])
            != payload.get("EMA_system_state_sha256")
        ):
            raise RuntimeError(f"uncommitted shared atom EMA drifted: fold {fold}")
        _remove_resume(atom_root / "resume.pt")
        _atomic_json(
            completion_path,
            _atom_completion_payload(
                identity=payload["identity"],
                identity_sha=identity_sha,
                summary=payload["summary"],
                final_path=final_path,
                final_sha=final_sha,
                ema_state_sha=payload["EMA_system_state_sha256"],
                ema_updates=int(payload["EMA_updates"]),
            ),
        )
    _verified_sidecar(completion_path)
    final_sha = _verified_sidecar(final_path)
    record = _read_json(completion_path)
    if (
        record.get("schema") != ATOM_COMPLETION_SCHEMA
        or record.get("status") != "complete"
        or int(record.get("outer_fold", -1)) != fold
        or record.get("identity_sha256") != identity_sha
        or record.get("final_EMA_checkpoint_sha256") != final_sha
        or int(record.get("optimizer_updates", -1)) != 4080
        or int(record.get("EMA_updates", -1)) != 4080
    ):
        raise RuntimeError(f"shared atom completion drifted: fold {fold}")
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != ATOM_FINAL_SCHEMA
        or payload.get("identity_sha256") != identity_sha
        or int(payload.get("completed_updates", -1)) != 4080
        or int(payload.get("EMA_updates", -1)) != 4080
        or tensor_mapping_sha256(payload["EMA_system_state"])
        != payload.get("EMA_system_state_sha256")
    ):
        raise RuntimeError(f"shared atom final EMA drifted: fold {fold}")
    retained = sorted(path.name for path in atom_root.glob("*.pt"))
    if retained != ["final_ema.pt"]:
        raise RuntimeError(f"shared atom retained unexpected weights: fold {fold}")
    return record


def _fit_atom_fold(
    *,
    config: Mapping[str, Any],
    bundle: Any,
    fold: Any,
    global_identity_sha: str,
    formal_root: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    outer_train = torch.from_numpy(fold.outer_train).long()
    condition_numpy = np.ascontiguousarray(
        bundle.train.condition[fold.outer_train], dtype=np.float32
    )
    observation_numpy = np.ascontiguousarray(
        bundle.train.target[fold.outer_train], dtype=np.float32
    )
    observed_numpy = np.ascontiguousarray(
        bundle.train.observed_mask[fold.outer_train], dtype=bool
    )
    observation_cpu = torch.from_numpy(observation_numpy)
    observed_cpu = torch.from_numpy(observed_numpy)
    contract = train_only_atom_contract(observation_cpu, observed_cpu)
    atom_config = config["shared_atom_contract"]
    updates = int(atom_config["fit_updates"])
    batch_size = int(atom_config["batch_calendar_days"])
    bank, bank_manifest = _atom_index_bank(
        fold=int(fold.fold),
        train_days=len(outer_train),
        updates=updates,
        batch_size=batch_size,
    )
    initialization_seed = 51000 + int(fold.fold)
    torch.manual_seed(initialization_seed)
    model = TransitionAtomNuisance(
        fixed_one_probability=(
            None
            if contract["fixed_one_probability"] is None
            else float(contract["fixed_one_probability"])
        ),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    ).to(device)
    initialization_sha = module_state_sha256(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    identity = _atom_identity(
        config=config,
        fold=fold,
        global_identity_sha=global_identity_sha,
        contract=contract,
        bank_manifest=bank_manifest,
        condition=condition_numpy,
        observation=observation_numpy,
        observed_mask=observed_numpy,
        initialization_sha=initialization_sha,
        parameters=parameters,
    )
    identity_sha = _canonical_sha256(identity)
    root = _atom_root(formal_root, int(fold.fold))
    root.mkdir(parents=True, exist_ok=True)
    complete = _validate_atom_completion(
        root, identity_sha=identity_sha, fold=int(fold.fold)
    )
    if complete is not None:
        if not resume:
            raise FileExistsError(
                f"shared atom fold {fold.fold} exists; use --resume"
            )
        return complete

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(atom_config["learning_rate"]),
        betas=tuple(float(value) for value in atom_config["betas"]),
        eps=float(atom_config["eps"]),
        weight_decay=float(atom_config["weight_decay"]),
    )
    ema = TinyEMA(model, decay=float(atom_config["EMA_decay"]))
    condition = torch.from_numpy(condition_numpy).to(device)
    observation = observation_cpu.to(device)
    observed = observed_cpu.to(device)
    resume_path = root / "resume.pt"
    start = 0
    summary = _atom_summary()
    if resume_path.exists():
        if not resume:
            raise FileExistsError(
                f"shared atom resume exists for fold {fold.fold}; use --resume"
            )
        start, summary = _load_atom_resume(
            resume_path, identity, identity_sha, model, optimizer, ema
        )
    elif any(root.iterdir()):
        allowed = {"failure.json", "failure.json.sha256"}
        if not {path.name for path in root.iterdir()}.issubset(allowed):
            raise RuntimeError(f"unrecognized shared atom artifacts: fold {fold.fold}")

    if start == 0:
        torch.manual_seed(initialization_seed)
        np.random.seed(initialization_seed)
        random.seed(initialization_seed)
    started = time.perf_counter()
    try:
        for update_zero in range(start, updates):
            loss, atom_nll, location, gradient_norm = _atom_update(
                model,
                optimizer,
                ema,
                condition,
                observation,
                observed,
                bank,
                update_zero,
                device,
                gradient_clip=float(atom_config["gradient_clip"]),
            )
            completed = update_zero + 1
            _update_atom_summary(
                summary,
                update=completed,
                loss=loss,
                atom_nll=atom_nll,
                location=location,
                gradient_norm=gradient_norm,
            )
            if completed % ATOM_CHECKPOINT_INTERVAL == 0 and completed < updates:
                summary["wall_seconds_accumulated"] = float(
                    summary["wall_seconds_accumulated"]
                ) + float(time.perf_counter() - started)
                _atomic_torch(
                    resume_path,
                    _atom_checkpoint_payload(
                        identity, identity_sha, model, optimizer, ema, summary
                    ),
                )
                started = time.perf_counter()
        if int(summary["updates_completed"]) != updates or ema.num_updates != updates:
            raise RuntimeError("shared atom stopped before its frozen budget")
        summary["wall_seconds_accumulated"] = float(
            summary["wall_seconds_accumulated"]
        ) + float(time.perf_counter() - started)
        ema_state = _ema_system_state(model, ema)
        ema_state_sha = tensor_mapping_sha256(ema_state)
        final_path = root / "final_ema.pt"
        final_sha = _atomic_torch(
            final_path,
            {
                "schema": ATOM_FINAL_SCHEMA,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "identity": identity,
                "identity_sha256": identity_sha,
                "completed_updates": updates,
                "EMA_updates": ema.num_updates,
                "EMA_system_state": ema_state,
                "EMA_system_state_sha256": ema_state_sha,
                "summary": summary,
                "target_role": "train_outer_train_only",
                "continuous_transport_parameters": "absent",
            },
        )
        _remove_resume(resume_path)
        completion = _atom_completion_payload(
            identity=identity,
            identity_sha=identity_sha,
            summary=summary,
            final_path=final_path,
            final_sha=final_sha,
            ema_state_sha=ema_state_sha,
            ema_updates=ema.num_updates,
        )
        _atomic_json(root / "run.json", completion)
        for failure in (root / "failure.json", root / "failure.json.sha256"):
            if failure.exists():
                failure.unlink()
        return completion
    except Exception as error:
        _atomic_json(
            root / "failure.json",
            {
                "schema": "architecture_v1_tgo_v1_atom_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "outer_fold": int(fold.fold),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "resume_required": True,
                "latest_resume": (
                    str(resume_path.resolve()) if resume_path.exists() else None
                ),
            },
        )
        raise
    finally:
        del model, optimizer, ema, condition, observation, observed
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_identity(
    *,
    config: Mapping[str, Any],
    spec: RunSpec,
    global_identity_sha: str,
    fold_manifest: Mapping[str, Any],
    random_manifest: Mapping[str, Any],
    atom_completion: Mapping[str, Any],
    initialization_sha: str,
) -> dict[str, Any]:
    training = config["common_denoiser"]
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        **spec.manifest(),
        "global_training_identity_sha256": global_identity_sha,
        "fold_training_data": dict(fold_manifest),
        "randomness": dict(random_manifest),
        "shared_atom_final_EMA_checkpoint": atom_completion[
            "final_EMA_checkpoint"
        ],
        "shared_atom_final_EMA_checkpoint_sha256": atom_completion[
            "final_EMA_checkpoint_sha256"
        ],
        "base_initialization_sha256": initialization_sha,
        "trainable_parameters": TINY_DENOISER_PARAMETERS,
        "optimizer": {
            "name": training["optimizer"],
            "learning_rate": float(training["learning_rate"]),
            "betas": [float(value) for value in training["betas"]],
            "eps": float(training["eps"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip": float(training["gradient_clip"]),
        },
        "EMA_decay": float(training["EMA_decay"]),
        "updates": int(training["updates"]),
        "checkpoint_interval_updates": int(
            training["checkpoint_interval_updates"]
        ),
        "prediction_parameterization": training[
            "prediction_parameterization"
        ],
        "precision": training["precision"],
        "AMP": training["AMP"],
        "target_role": "train_outer_train_only",
        "outer_test_reconstruction_evaluation": False,
        "scenario_generation": False,
    }


def _run_summary() -> dict[str, Any]:
    return {
        "updates_completed": 0,
        "loss_first": None,
        "loss_final": None,
        "loss_min": None,
        "loss_max": None,
        "gradient_norm_max": None,
        "wall_seconds_accumulated": 0.0,
    }


def _update_run_summary(
    summary: dict[str, Any], *, update: int, loss: float, gradient_norm: float
) -> None:
    if not np.isfinite(np.asarray([loss, gradient_norm], dtype=np.float64)).all():
        raise FloatingPointError("denoiser summary received a non-finite value")
    if summary["loss_first"] is None:
        summary["loss_first"] = loss
        summary["loss_min"] = loss
        summary["loss_max"] = loss
        summary["gradient_norm_max"] = gradient_norm
    summary["updates_completed"] = update
    summary["loss_final"] = loss
    summary["loss_min"] = min(float(summary["loss_min"]), loss)
    summary["loss_max"] = max(float(summary["loss_max"]), loss)
    summary["gradient_norm_max"] = max(
        float(summary["gradient_norm_max"]), gradient_norm
    )


def _run_checkpoint_payload(
    identity: Mapping[str, Any],
    identity_sha: str,
    system: TransitionDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_RESUME_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "completed_updates": int(summary["updates_completed"]),
        "system": system.state_dict(),
        "optimizer": optimizer.state_dict(),
        "EMA": ema.state_dict(),
        "summary": dict(summary),
        "RNG": _rng_state(),
    }


def _load_run_resume(
    path: Path,
    identity: Mapping[str, Any],
    identity_sha: str,
    system: TransitionDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
) -> tuple[int, dict[str, Any]]:
    _verified_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != RUN_RESUME_SCHEMA
        or payload.get("identity_sha256") != identity_sha
        or payload.get("identity") != identity
    ):
        raise RuntimeError("TGO-v1 denoiser resume identity drifted")
    completed = int(payload["completed_updates"])
    maximum = int(identity["updates"])
    interval = int(identity["checkpoint_interval_updates"])
    if not 0 < completed < maximum or completed % interval != 0:
        raise RuntimeError("TGO-v1 denoiser resume boundary is invalid")
    system.load_state_dict(payload["system"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    ema.load_state_dict(payload["EMA"], system)
    summary = dict(payload["summary"])
    if int(summary["updates_completed"]) != completed or ema.num_updates != completed:
        raise RuntimeError("TGO-v1 denoiser resume summary drifted")
    _set_rng_state(payload["RNG"])
    return completed, summary


def _run_completion_payload(
    *,
    spec: RunSpec,
    identity: Mapping[str, Any],
    identity_sha: str,
    summary: Mapping[str, Any],
    final_path: Path,
    final_sha: str,
    ema_state_sha: str,
    ema_updates: int,
) -> dict[str, Any]:
    return {
        "schema": RUN_COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **spec.manifest(),
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "optimizer_updates": int(identity["updates"]),
        "EMA_updates": int(ema_updates),
        "summary": dict(summary),
        "final_EMA_checkpoint": str(final_path.resolve()),
        "final_EMA_checkpoint_sha256": final_sha,
        "final_EMA_system_state_sha256": ema_state_sha,
        "only_final_EMA_weight_retained": True,
        "resume_checkpoint_deleted": not (final_path.parent / "resume.pt").exists(),
        "materialized_target_roles": ["train"],
        "outer_test_reconstruction_evaluation_performed": False,
        "scenario_generation_performed": False,
        "raw_training_loss_used_as_path_evidence": False,
    }


def _validate_run_completion(
    run_root: Path, *, identity_sha: str, spec: RunSpec
) -> dict[str, Any] | None:
    completion_path = run_root / "run.json"
    final_path = run_root / "final_ema.pt"
    if not completion_path.exists() and not final_path.exists():
        return None
    if completion_path.exists() and not final_path.exists():
        raise RuntimeError(f"incomplete TGO-v1 run artifact pair: {spec.key}")
    if final_path.exists() and not completion_path.exists():
        final_sha = _verified_sidecar(final_path)
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != RUN_FINAL_SCHEMA
            or payload.get("identity_sha256") != identity_sha
            or int(payload.get("completed_updates", -1)) != 1024
            or int(payload.get("EMA_updates", -1)) != 1024
            or tensor_mapping_sha256(payload["EMA_system_state"])
            != payload.get("EMA_system_state_sha256")
        ):
            raise RuntimeError(f"uncommitted TGO-v1 EMA drifted: {spec.key}")
        _remove_resume(run_root / "resume.pt")
        _atomic_json(
            completion_path,
            _run_completion_payload(
                spec=spec,
                identity=payload["identity"],
                identity_sha=identity_sha,
                summary=payload["summary"],
                final_path=final_path,
                final_sha=final_sha,
                ema_state_sha=payload["EMA_system_state_sha256"],
                ema_updates=int(payload["EMA_updates"]),
            ),
        )
    _verified_sidecar(completion_path)
    final_sha = _verified_sidecar(final_path)
    record = _read_json(completion_path)
    if (
        record.get("schema") != RUN_COMPLETION_SCHEMA
        or record.get("status") != "complete"
        or record.get("run_key") != spec.key
        or record.get("identity_sha256") != identity_sha
        or record.get("final_EMA_checkpoint_sha256") != final_sha
        or int(record.get("optimizer_updates", -1)) != 1024
        or int(record.get("EMA_updates", -1)) != 1024
    ):
        raise RuntimeError(f"TGO-v1 completion drifted: {spec.key}")
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != RUN_FINAL_SCHEMA
        or payload.get("identity_sha256") != identity_sha
        or int(payload.get("completed_updates", -1)) != 1024
        or int(payload.get("EMA_updates", -1)) != 1024
        or tensor_mapping_sha256(payload["EMA_system_state"])
        != payload.get("EMA_system_state_sha256")
    ):
        raise RuntimeError(f"TGO-v1 final EMA drifted: {spec.key}")
    retained = sorted(path.name for path in run_root.glob("*.pt"))
    if retained != ["final_ema.pt"]:
        raise RuntimeError(f"TGO-v1 run retained unexpected weights: {spec.key}")
    return record


def _run_one(
    *,
    config: Mapping[str, Any],
    spec: RunSpec,
    tensors: Mapping[str, torch.Tensor],
    operators: Mapping[str, MaskConditionedOperator],
    fold_manifest: Mapping[str, Any],
    atom_completion: Mapping[str, Any],
    alpha_bar: torch.Tensor,
    global_identity_sha: str,
    formal_root: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    training = config["common_denoiser"]
    seed = 63000 + 100 * spec.outer_fold + spec.model_seed
    bank, bank_sha = p0_runner._random_bank(
        seed=seed,
        updates=int(training["updates"]),
        batch_size=int(training["batch_calendar_days"]),
        train_days=int(fold_manifest["outer_train_days"]),
        timesteps=int(training["timesteps"]),
    )
    random_manifest = {
        "seed": seed,
        "sha256": bank_sha,
        "updates": int(training["updates"]),
        "batch_size": int(training["batch_calendar_days"]),
    }
    system = TransitionDenoisingSystem(
        spec.path_id, model_seed=spec.model_seed
    ).to(device)
    if sum(parameter.numel() for parameter in system.parameters()) != TINY_DENOISER_PARAMETERS:
        raise RuntimeError("TGO-v1 denoiser parameter count drifted")
    initialization_sha = module_state_sha256(system)
    identity = _run_identity(
        config=config,
        spec=spec,
        global_identity_sha=global_identity_sha,
        fold_manifest=fold_manifest,
        random_manifest=random_manifest,
        atom_completion=atom_completion,
        initialization_sha=initialization_sha,
    )
    identity_sha = _canonical_sha256(identity)
    root = _run_root(formal_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    complete = _validate_run_completion(root, identity_sha=identity_sha, spec=spec)
    if complete is not None:
        if not resume:
            raise FileExistsError(f"completed run exists; use --resume: {spec.key}")
        return complete

    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema = TinyEMA(system, decay=float(training["EMA_decay"]))
    resume_path = root / "resume.pt"
    start = 0
    summary = _run_summary()
    if resume_path.exists():
        if not resume:
            raise FileExistsError(f"resume checkpoint exists; use --resume: {spec.key}")
        start, summary = _load_run_resume(
            resume_path, identity, identity_sha, system, optimizer, ema
        )
    elif any(root.iterdir()):
        allowed = {"failure.json", "failure.json.sha256"}
        if not {path.name for path in root.iterdir()}.issubset(allowed):
            raise RuntimeError(f"unrecognized partial run artifacts: {spec.key}")

    if start == 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    maximum = int(training["updates"])
    interval = int(training["checkpoint_interval_updates"])
    started = time.perf_counter()
    try:
        for update_zero in range(start, maximum):
            loss, gradient_norm = p0_runner._one_update(
                system,
                optimizer,
                ema,
                tensors,
                bank,
                update_zero,
                device,
                alpha_bar,
                operators["transition_true"],
                operators["transition_wrong"],
                operators["orthogonal_dct"],
                gradient_clip=float(training["gradient_clip"]),
            )
            completed = update_zero + 1
            _update_run_summary(
                summary,
                update=completed,
                loss=loss,
                gradient_norm=gradient_norm,
            )
            if completed % interval == 0 and completed < maximum:
                summary["wall_seconds_accumulated"] = float(
                    summary["wall_seconds_accumulated"]
                ) + float(time.perf_counter() - started)
                _atomic_torch(
                    resume_path,
                    _run_checkpoint_payload(
                        identity, identity_sha, system, optimizer, ema, summary
                    ),
                )
                started = time.perf_counter()
        if int(summary["updates_completed"]) != maximum or ema.num_updates != maximum:
            raise RuntimeError("TGO-v1 run stopped before its frozen budget")
        summary["wall_seconds_accumulated"] = float(
            summary["wall_seconds_accumulated"]
        ) + float(time.perf_counter() - started)
        ema_state = _ema_system_state(system, ema)
        ema_state_sha = tensor_mapping_sha256(ema_state)
        final_path = root / "final_ema.pt"
        final_sha = _atomic_torch(
            final_path,
            {
                "schema": RUN_FINAL_SCHEMA,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "identity": identity,
                "identity_sha256": identity_sha,
                "completed_updates": maximum,
                "EMA_updates": ema.num_updates,
                "EMA_system_state": ema_state,
                "EMA_system_state_sha256": ema_state_sha,
                "summary": summary,
                "outer_test_reconstruction_evaluation_performed": False,
                "scenario_generation_performed": False,
            },
        )
        _remove_resume(resume_path)
        completion = _run_completion_payload(
            spec=spec,
            identity=identity,
            identity_sha=identity_sha,
            summary=summary,
            final_path=final_path,
            final_sha=final_sha,
            ema_state_sha=ema_state_sha,
            ema_updates=ema.num_updates,
        )
        _atomic_json(root / "run.json", completion)
        for failure in (root / "failure.json", root / "failure.json.sha256"):
            if failure.exists():
                failure.unlink()
        return completion
    except Exception as error:
        _atomic_json(
            root / "failure.json",
            {
                "schema": "architecture_v1_tgo_v1_run_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_key": spec.key,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "resume_required": True,
                "latest_resume": (
                    str(resume_path.resolve()) if resume_path.exists() else None
                ),
                "outer_test_reconstruction_evaluation_performed": False,
                "scenario_generation_performed": False,
            },
        )
        raise
    finally:
        del system, optimizer, ema
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _completion_records(
    formal_root: Path, matrix: Sequence[RunSpec]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for spec in matrix:
        path = _run_root(formal_root, spec) / "run.json"
        if not path.exists():
            continue
        _verified_sidecar(path)
        record = _read_json(path)
        if (
            record.get("schema") != RUN_COMPLETION_SCHEMA
            or record.get("status") != "complete"
            or record.get("run_key") != spec.key
        ):
            raise RuntimeError(f"invalid TGO-v1 completion record: {spec.key}")
        records[spec.key] = record
    return records


def _atom_records(formal_root: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for fold in range(6):
        path = _atom_root(formal_root, fold) / "run.json"
        if not path.exists():
            continue
        _verified_sidecar(path)
        record = _read_json(path)
        if (
            record.get("schema") != ATOM_COMPLETION_SCHEMA
            or record.get("status") != "complete"
            or int(record.get("outer_fold", -1)) != fold
        ):
            raise RuntimeError(f"invalid shared atom completion: fold {fold}")
        records[fold] = record
    return records


def _write_progress(
    formal_root: Path,
    matrix: Sequence[RunSpec],
    *,
    selected_this_invocation: Sequence[str],
) -> dict[str, Any]:
    atoms = _atom_records(formal_root)
    complete = _completion_records(formal_root, matrix)
    pending = [spec.key for spec in matrix if spec.key not in complete]
    progress = {
        "schema": PROGRESS_SCHEMA,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_atom_models": 6,
        "completed_atom_models": len(atoms),
        "expected_denoiser_runs": len(matrix),
        "completed_denoiser_runs": len(complete),
        "remaining_denoiser_runs": len(pending),
        "selected_this_invocation": list(selected_this_invocation),
        "completed_run_keys": sorted(complete),
        "pending_run_keys": pending,
        "training_frozen": (formal_root / "training.freeze.json").exists(),
        "outer_test_TGO_metrics_constructed": False,
        "scenario_generation_performed": False,
        "next_action": (
            "complete_shared_atom_models"
            if len(atoms) < 6
            else (
                "complete_remaining_retained_runs"
                if pending
                else "write_and_verify_training_freeze"
            )
        ),
    }
    _atomic_json(formal_root / "training.progress.json", progress)
    return progress


def _try_freeze(
    formal_root: Path,
    matrix: Sequence[RunSpec],
    global_identity_sha: str,
) -> dict[str, Any] | None:
    freeze_path = formal_root / "training.freeze.json"
    if freeze_path.exists():
        _verified_sidecar(freeze_path)
        return _read_json(freeze_path)
    atoms = _atom_records(formal_root)
    completions = _completion_records(formal_root, matrix)
    if len(atoms) != 6 or len(completions) != len(matrix):
        return None

    atom_hashes: dict[str, str] = {}
    for fold, record in atoms.items():
        identity = record["identity"]
        if (
            record["identity_sha256"] != _canonical_sha256(identity)
            or identity["global_training_identity_sha256"]
            != global_identity_sha
            or int(record["optimizer_updates"]) != 4080
            or int(record["EMA_updates"]) != 4080
            or record["only_final_EMA_weight_retained"] is not True
            or record["resume_checkpoint_deleted"] is not True
            or record["held_out_target_accessed"] is not False
            or record["continuous_transport_parameters"] != "absent"
        ):
            raise RuntimeError(f"shared atom freeze validation failed: fold {fold}")
        final_path = Path(record["final_EMA_checkpoint"])
        digest = _verified_sidecar(final_path)
        if digest != record["final_EMA_checkpoint_sha256"]:
            raise RuntimeError(f"shared atom checkpoint hash drifted: fold {fold}")
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if tensor_mapping_sha256(payload["EMA_system_state"]) != record[
            "final_EMA_system_state_sha256"
        ]:
            raise RuntimeError(f"shared atom tensor hash drifted: fold {fold}")
        atom_hashes[str(fold)] = digest

    checkpoint_hashes: dict[str, str] = {}
    paired: dict[tuple[int, int], dict[str, set[str]]] = {}
    for spec in matrix:
        record = completions[spec.key]
        identity = record["identity"]
        if (
            record["identity_sha256"] != _canonical_sha256(identity)
            or identity["global_training_identity_sha256"]
            != global_identity_sha
            or int(record["optimizer_updates"]) != 1024
            or int(record["EMA_updates"]) != 1024
            or record["only_final_EMA_weight_retained"] is not True
            or record["resume_checkpoint_deleted"] is not True
            or record["materialized_target_roles"] != ["train"]
            or record["outer_test_reconstruction_evaluation_performed"]
            is not False
            or record["scenario_generation_performed"] is not False
            or record["raw_training_loss_used_as_path_evidence"] is not False
        ):
            raise RuntimeError(f"TGO-v1 run freeze validation failed: {spec.key}")
        numeric = np.asarray(
            [
                record["summary"]["loss_first"],
                record["summary"]["loss_final"],
                record["summary"]["loss_min"],
                record["summary"]["loss_max"],
                record["summary"]["gradient_norm_max"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"non-finite TGO-v1 summary: {spec.key}")
        final_path = Path(record["final_EMA_checkpoint"])
        digest = _verified_sidecar(final_path)
        if digest != record["final_EMA_checkpoint_sha256"]:
            raise RuntimeError(f"TGO-v1 checkpoint hash drifted: {spec.key}")
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if tensor_mapping_sha256(payload["EMA_system_state"]) != record[
            "final_EMA_system_state_sha256"
        ]:
            raise RuntimeError(f"TGO-v1 tensor hash drifted: {spec.key}")
        checkpoint_hashes[spec.key] = digest
        group = paired.setdefault(
            (spec.outer_fold, spec.model_seed),
            {
                "initialization": set(),
                "random_bank": set(),
                "fold_arrays": set(),
                "operator_scales": set(),
                "atom_checkpoint": set(),
            },
        )
        group["initialization"].add(identity["base_initialization_sha256"])
        group["random_bank"].add(identity["randomness"]["sha256"])
        group["fold_arrays"].add(
            identity["fold_training_data"]["training_array_sha256"]
        )
        group["operator_scales"].add(
            _canonical_sha256(identity["fold_training_data"]["operator_scales"])
        )
        group["atom_checkpoint"].add(
            identity["shared_atom_final_EMA_checkpoint_sha256"]
        )
    if len(paired) != 12 or any(
        len(values) != 1 for group in paired.values() for values in group.values()
    ):
        raise RuntimeError("TGO-v1 cross-path pairing validation failed")

    freeze = {
        "schema": FREEZE_SCHEMA,
        "status": "TGO_V1_84_OF_84_TRAINING_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "global_training_identity": str(
            (formal_root / "training.identity.json").resolve()
        ),
        "global_training_identity_sha256": global_identity_sha,
        "expected_atom_models": 6,
        "completed_atom_models": 6,
        "atom_optimizer_updates": 6 * 4080,
        "expected_denoiser_runs": len(matrix),
        "completed_denoiser_runs": len(completions),
        "denoiser_optimizer_updates": 1024 * len(completions),
        "paired_fold_seed_groups": len(paired),
        "all_cross_path_pairing_checks_passed": True,
        "atom_final_EMA_checkpoint_sha256_by_fold": atom_hashes,
        "denoiser_final_EMA_checkpoint_sha256_by_run": checkpoint_hashes,
        "only_final_EMA_checkpoints_retained": True,
        "training_closed": True,
        "outer_test_TGO_metrics_constructed": False,
        "scenario_generation_performed": False,
        "common_physical_sampling_authorized": True,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_action": "implement_common_physical_scale_scenario_generation_and_frozen_evaluation",
    }
    _atomic_json(freeze_path, freeze)
    return freeze


def dry_run(config_path: Path, *, include_run_keys: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _code_identity(config_path)
    p0 = _validate_p0(config, config_path)
    matrix = _matrix(config)
    formal_root = ROOT / config["output_root"] / FORMAL_ROOT_NAME
    atoms = len(_atom_records(formal_root)) if formal_root.exists() else 0
    completed = (
        len(_completion_records(formal_root, matrix)) if formal_root.exists() else 0
    )
    return {
        "schema": "architecture_v1_tgo_v1_formal_dry_run_v1",
        "mode": "target_free_no_files_created",
        "config_sha256": code["config_sha256"],
        "P0": p0,
        "expected_shared_atom_models": 6,
        "completed_shared_atom_models": atoms,
        "expected_retained_runs": len(matrix),
        "completed_retained_runs": completed,
        "remaining_retained_runs": len(matrix) - completed,
        "matrix_dimensions": {
            "outer_folds": 6,
            "model_seeds": 2,
            "paths": 7,
        },
        "atom_updates_per_fold": int(config["shared_atom_contract"]["fit_updates"]),
        "denoiser_updates_per_run": int(config["common_denoiser"]["updates"]),
        "first_run_key": matrix[0].key,
        "last_run_key": matrix[-1].key,
        "run_keys": (
            [spec.key for spec in matrix]
            if include_run_keys
            else "use --list-run-keys"
        ),
        "formal_training_target_roles_if_executed": ["train"],
        "outer_test_TGO_metrics_constructed": False,
        "scenario_generation_implemented": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "CPU_execution_requires_allow_cpu": True,
        "next_flag": "--execute-training",
    }


def execute_training(
    config_path: Path,
    *,
    resume: bool,
    device_name: str,
    allow_cpu: bool,
    run_keys: Sequence[str],
    max_runs: int | None,
    atom_only: bool,
) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _code_identity(config_path)
    p0 = _validate_p0(config, config_path)
    matrix = _matrix(config)
    by_key = {spec.key: spec for spec in matrix}
    unknown = sorted(set(run_keys) - set(by_key))
    if unknown:
        raise ValueError(f"unknown TGO-v1 run keys: {unknown}")
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("--run-key contains duplicates")
    if max_runs is not None and max_runs < 1:
        raise ValueError("--max-runs must be positive")
    formal_root = ROOT / config["output_root"] / FORMAL_ROOT_NAME
    freeze_path = formal_root / "training.freeze.json"
    if freeze_path.exists():
        _verified_sidecar(freeze_path)
        return _read_json(freeze_path)
    if formal_root.exists() and any(formal_root.iterdir()) and not resume:
        raise FileExistsError("formal training output is non-empty; use --resume")
    formal_root.mkdir(parents=True, exist_ok=True)
    device, runtime = _configure_device(device_name, allow_cpu=allow_cpu)
    bundle, _groups, folds, nuisance_replay, _subsets = _replay_nuisance(
        _read_json(ROOT / config["lineage"]["g0_b_config"])
    )
    alpha_numpy, schedule_audit = _baseline_schedule(
        _read_json(ROOT / config["lineage"]["g0_b_config"])
    )
    identity = _global_identity(
        config,
        config_path,
        p0,
        code,
        bundle,
        nuisance_replay,
        runtime,
        matrix,
    )
    identity["schedule"] = schedule_audit
    global_identity_sha = _prepare_global_identity(
        formal_root, identity, resume=resume
    )

    atom_completions: dict[int, dict[str, Any]] = {}
    for fold in folds:
        atom_completions[int(fold.fold)] = _fit_atom_fold(
            config=config,
            bundle=bundle,
            fold=fold,
            global_identity_sha=global_identity_sha,
            formal_root=formal_root,
            device=device,
            resume=resume,
        )
        _write_progress(formal_root, matrix, selected_this_invocation=[])

    completed_this_invocation: list[str] = []
    if not atom_only:
        already = _completion_records(formal_root, matrix)
        selected = [by_key[key] for key in run_keys] if run_keys else list(matrix)
        selected = [spec for spec in selected if spec.key not in already]
        if max_runs is not None:
            selected = selected[:max_runs]
        fold_contexts: dict[
            int,
            tuple[
                dict[str, torch.Tensor],
                dict[str, MaskConditionedOperator],
                dict[str, Any],
            ],
        ] = {}
        alpha_bar = torch.from_numpy(alpha_numpy).float().to(device)
        for spec in selected:
            if spec.outer_fold not in fold_contexts:
                fold_contexts[spec.outer_fold] = _operator_context(
                    folds[spec.outer_fold], device
                )
            tensors, operators, fold_manifest = fold_contexts[spec.outer_fold]
            _run_one(
                config=config,
                spec=spec,
                tensors=tensors,
                operators=operators,
                fold_manifest=fold_manifest,
                atom_completion=atom_completions[spec.outer_fold],
                alpha_bar=alpha_bar,
                global_identity_sha=global_identity_sha,
                formal_root=formal_root,
                device=device,
                resume=resume,
            )
            completed_this_invocation.append(spec.key)
            _write_progress(
                formal_root,
                matrix,
                selected_this_invocation=completed_this_invocation,
            )

    progress = _write_progress(
        formal_root,
        matrix,
        selected_this_invocation=completed_this_invocation,
    )
    freeze = _try_freeze(formal_root, matrix, global_identity_sha)
    return {
        "schema": "architecture_v1_tgo_v1_training_invocation_result_v1",
        "status": (
            freeze["status"]
            if freeze is not None
            else (
                "TGO_V1_SHARED_ATOMS_COMPLETE"
                if atom_only
                else "TGO_V1_RETAINED_TRAINING_PARTIAL"
            )
        ),
        "completed_atom_models_total": progress["completed_atom_models"],
        "completed_this_invocation": len(completed_this_invocation),
        "completed_runs_total": progress["completed_denoiser_runs"],
        "remaining_runs": progress["remaining_denoiser_runs"],
        "training_freeze_written": freeze is not None,
        "training_freeze": freeze,
        "outer_test_TGO_metrics_constructed": False,
        "scenario_generation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--atom-only", action="store_true")
    parser.add_argument("--list-run-keys", action="store_true")
    arguments = parser.parse_args()
    mutating_options = bool(
        arguments.resume
        or arguments.allow_cpu
        or arguments.run_key
        or arguments.max_runs is not None
        or arguments.atom_only
    )
    if mutating_options and not arguments.execute_training:
        parser.error(
            "--resume/--allow-cpu/--run-key/--max-runs/--atom-only require --execute-training"
        )
    result = (
        execute_training(
            arguments.config.resolve(),
            resume=arguments.resume,
            device_name=arguments.device,
            allow_cpu=arguments.allow_cpu,
            run_keys=arguments.run_key,
            max_runs=arguments.max_runs,
            atom_only=arguments.atom_only,
        )
        if arguments.execute_training
        else dry_run(
            arguments.config.resolve(),
            include_run_keys=arguments.list_run_keys,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
