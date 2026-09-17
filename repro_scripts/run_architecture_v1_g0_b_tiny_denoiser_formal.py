#!/usr/bin/env python3
"""Retained-training runner for the frozen G0-B tiny-denoiser Probe.

The default mode is target-free and only reports the 324-run matrix.  Formal
training requires ``--execute-training``.  This runner never constructs the
outer-test reconstruction bank: it trains on each outer-training subset,
retains only the update-1024 EMA, and writes ``training.freeze.json`` only
after all 324 registered runs are complete and identity-checked.
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
    ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"
)
FORMAL_RUNNER_PATH = Path(__file__).resolve()
FORMAL_ROOT_NAME = "formal_training"
GLOBAL_IDENTITY_SCHEMA = "architecture_v1_g0_b_tiny_training_identity_v1"
RUN_IDENTITY_SCHEMA = "architecture_v1_g0_b_tiny_run_identity_v1"
RESUME_SCHEMA = "architecture_v1_g0_b_tiny_resume_checkpoint_v1"
FINAL_EMA_SCHEMA = "architecture_v1_g0_b_tiny_final_ema_checkpoint_v1"
COMPLETION_SCHEMA = "architecture_v1_g0_b_tiny_run_completion_v1"
PROGRESS_SCHEMA = "architecture_v1_g0_b_tiny_training_progress_v1"
FREEZE_SCHEMA = "architecture_v1_g0_b_tiny_training_freeze_v1"

from architecture_v1.g0b_tiny_denoiser import (
    MULAN_LITE_PARAMETERS,
    PATH_IDS,
    TINY_DENOISER_PARAMETERS,
    ModeProjectorBank,
    TinyDenoisingSystem,
    TinyEMA,
    module_state_sha256,
    tensor_mapping_sha256,
)
from architecture_v1.protocol import date_list_sha256
from repro_scripts import run_architecture_v1_g0_b_tiny_denoiser as p0_runner


@dataclass(frozen=True)
class RunSpec:
    outer_fold: int
    fraction: float
    fraction_index: int
    path: str
    model_seed: int

    @property
    def fraction_label(self) -> str:
        return {0.25: "0p25", 0.5: "0p5", 1.0: "1p0"}[self.fraction]

    @property
    def key(self) -> str:
        return (
            f"fold{self.outer_fold}__fraction{self.fraction_label}__"
            f"{self.path}__seed{self.model_seed}"
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "run_key": self.key,
            "outer_fold": self.outer_fold,
            "fraction": self.fraction,
            "fraction_index": self.fraction_index,
            "path": self.path,
            "model_seed": self.model_seed,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verified_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"artifact or SHA256 sidecar is missing: {path}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
    digest = _sha256(path)
    if digest != pieces[0]:
        raise RuntimeError(f"artifact SHA256 mismatch: {path}")
    return digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _array_sha256(**values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _matrix(config: Mapping[str, Any]) -> tuple[RunSpec, ...]:
    training = config["training"]
    fractions = tuple(float(value) for value in config["learning_curve"]["fractions"])
    specs = tuple(
        RunSpec(fold, fraction, fraction_index, path, int(seed))
        for fold in range(int(training["outer_folds"]))
        for fraction_index, fraction in enumerate(fractions)
        for seed in training["model_seeds"]
        for path in PATH_IDS
    )
    expected = int(training["expected_retained_runs"])
    if len(specs) != expected or len({spec.key for spec in specs}) != expected:
        raise RuntimeError("formal G0-B training matrix drifted")
    if expected != 324:
        raise RuntimeError("formal G0-B training must contain exactly 324 runs")
    return specs


def _validate_p0(
    config: Mapping[str, Any], config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    output_root = ROOT / config["output_root"]
    result_path = output_root / config["planned_implementation_files"]["P0_result"]
    digest = _verified_sidecar(result_path)
    result = _read_json(result_path)
    if result.get("schema") != p0_runner.RESULT_SCHEMA:
        raise RuntimeError("G0-B P0 result schema drifted")
    if result.get("status") != config["P0_preflight"]["P0_GO_status"]:
        raise RuntimeError("formal training requires G0-B P0 GO")
    if result.get("all_hard_gates_passed") is not True:
        raise RuntimeError("G0-B P0 hard gates are not all true")
    if result.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("G0-B P0/config identity drifted")
    if result.get("target_roles_materialized") != ["train"]:
        raise RuntimeError("G0-B P0 materialized a non-train target role")
    if (
        result.get("forbidden_target_arrays_materialized") is not False
        or result.get("validation_bank_constructed") is not False
        or result.get("formal_outer_test_evaluation_performed") is not False
        or result.get("weights_retained") is not False
        or result.get("weight_files_after_run") != []
    ):
        raise RuntimeError("G0-B P0 boundary or retention record drifted")
    current_module = _sha256(ROOT / "architecture_v1" / "g0b_tiny_denoiser.py")
    if result["implementation_identity"]["implementation_module"] != current_module:
        raise RuntimeError("tiny-denoiser implementation changed after P0")
    return {
        "path": str(result_path.resolve()),
        "sha256": digest,
        "status": result["status"],
        "config_sha256": result["config_sha256"],
        "implementation_module_sha256": current_module,
        "cuda_runtime_exercised": bool(result["device"]["cuda_runtime_exercised"]),
    }


def _code_identity(config_path: Path) -> dict[str, str]:
    config = _read_json(config_path)
    base = p0_runner._validate_config(config, config_path)
    return {
        "config_sha256": base["config"],
        "tiny_denoiser_module_sha256": _sha256(
            ROOT / "architecture_v1" / "g0b_tiny_denoiser.py"
        ),
        "g0_predictability_module_sha256": base["g0_predictability_module"],
        "g0_b0_allocation_module_sha256": base["g0_b0_allocation_module"],
        "P0_runner_sha256": _sha256(
            ROOT / "repro_scripts" / "run_architecture_v1_g0_b_tiny_denoiser.py"
        ),
        "formal_runner_sha256": _sha256(FORMAL_RUNNER_PATH),
        "repository_execution_head": base["repository_execution_head"],
    }


def _run_root(formal_root: Path, spec: RunSpec) -> Path:
    return (
        formal_root
        / "runs"
        / f"fold{spec.outer_fold}"
        / f"fraction{spec.fraction_label}"
        / f"seed{spec.model_seed}"
        / spec.path
    )


def _configure_device(device_name: str, *, allow_cpu: bool) -> tuple[torch.device, dict[str, Any]]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "formal 324-run training resolved to CPU; pass --allow-cpu only if this slow execution is intentional"
        )
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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


def _global_identity(
    config: Mapping[str, Any],
    config_path: Path,
    p0: Mapping[str, Any],
    code: Mapping[str, str],
    bundle: Any,
    runtime: Mapping[str, Any],
    matrix: Sequence[RunSpec],
) -> dict[str, Any]:
    training = config["training"]
    return {
        "schema": GLOBAL_IDENTITY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": code["config_sha256"],
        "P0": dict(p0),
        "code_identity": dict(code),
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
        "runtime": dict(runtime),
        "matrix_sha256": _canonical_sha256(
            {"runs": [spec.manifest() for spec in matrix]}
        ),
        "expected_runs": len(matrix),
        "updates_per_run": int(training["optimizer_updates_per_run"]),
        "checkpoint_every_updates": int(training["checkpoint_every_updates"]),
        "retained_checkpoint": training["retained_checkpoint"],
        "materialized_target_roles": ["train"],
        "outer_test_reconstruction_bank_constructed": False,
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
            raise RuntimeError("formal training global identity drifted")
        return _sha256(path)
    if resume and any(formal_root.iterdir()):
        raise RuntimeError("cannot resume a non-empty formal root without identity")
    return _atomic_json(path, identity)


def _subset_context(
    config: Mapping[str, Any],
    fold: Any,
    train_days: np.ndarray,
    spec: RunSpec,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, dict[str, Any]]:
    local = p0_runner._nested_subset_indices(
        train_days,
        fold.outer_train,
        fold=spec.outer_fold,
        fraction=spec.fraction,
        seed_root=int(config["learning_curve"]["subset_seed_root"]),
    )
    selected_days = train_days[fold.outer_train[local]]
    frozen = config["learning_curve"]["subset_registry"][str(spec.outer_fold)][
        f"fraction_{spec.fraction}"
    ]
    subset_sha = date_list_sha256(selected_days)
    if len(local) != int(frozen["days"]) or subset_sha != frozen["date_sha256"]:
        raise RuntimeError("formal training subset identity drifted")
    weights = fold.weights_train[local]
    arrays = {
        "residual": np.ascontiguousarray(fold.residual_train[local], dtype=np.float32),
        "active": np.ascontiguousarray(fold.active_train[local], dtype=bool),
        "condition": np.ascontiguousarray(fold.condition_train[local], dtype=np.float32),
        "variance": np.ascontiguousarray(fold.variance_train[local], dtype=np.float32),
        "weights": np.ascontiguousarray(weights, dtype=np.float32),
    }
    tensors = {
        name: torch.as_tensor(value, device=device)
        for name, value in arrays.items()
    }
    fixed = np.ascontiguousarray(fold.fixed_variance, dtype=np.float32)
    context = {
        "subset_days": int(len(local)),
        "subset_date_sha256": subset_sha,
        "subset_array_sha256": _array_sha256(**arrays, fixed_variance=fixed),
        "complete_outer_train_nuisance_sha256": _array_sha256(
            residual=np.ascontiguousarray(fold.residual_train),
            active=np.ascontiguousarray(fold.active_train),
            variance=np.ascontiguousarray(fold.variance_train),
            effective_rank=np.ascontiguousarray(fold.effective_train),
            condition=np.ascontiguousarray(fold.condition_train),
            fixed_variance=np.ascontiguousarray(fold.fixed_variance),
        ),
    }
    return tensors, fixed, local, context


def _random_context(
    config: Mapping[str, Any], spec: RunSpec, subset_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    training = config["training"]
    seed = (
        int(training["training_random_seed_root"])
        + spec.outer_fold * 10_000
        + spec.fraction_index * 1_000
        + spec.model_seed
    )
    bank, digest = p0_runner._random_bank(
        seed=seed,
        updates=int(training["optimizer_updates_per_run"]),
        batch_size=int(training["batch_calendar_days"]),
        subset_size=subset_size,
        timesteps=int(config["baseline_information_contract"]["timesteps"]),
    )
    return bank, {
        "training_random_seed": seed,
        "common_random_bank_sha256": digest,
        "updates": int(training["optimizer_updates_per_run"]),
        "batch_calendar_days": int(training["batch_calendar_days"]),
    }


def _shuffle_context(
    config: Mapping[str, Any], spec: RunSpec, subset_size: int, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], str]:
    path_spec = next(path for path in config["paths"] if path["id"] == "PA_SHUFFLE")
    count = int(path_spec["training_derangements"])
    permutations = p0_runner._registered_derangements(
        subset_size,
        fold=spec.outer_fold,
        partition="training",
        seed_root=int(path_spec["derangement_seed_root"]),
        count=count,
    )
    stacked = np.stack(permutations).astype(np.int64, copy=False)
    if bool(np.any(stacked == np.arange(subset_size)[None])):
        raise RuntimeError("formal PA_SHUFFLE contains a fixed point")
    tensors = tuple(
        torch.as_tensor(permutation, dtype=torch.long, device=device)
        for permutation in permutations
    )
    return tensors, _array_sha256(permutations=stacked)


def _run_identity(
    config: Mapping[str, Any],
    spec: RunSpec,
    global_identity_sha: str,
    subset_context: Mapping[str, Any],
    random_context: Mapping[str, Any],
    shuffle_sha: str,
    base_initialization_sha: str,
    parameter_count: int,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        **spec.manifest(),
        "global_training_identity_sha256": global_identity_sha,
        "subset": dict(subset_context),
        "randomness": dict(random_context),
        "shuffle_derangements_sha256": shuffle_sha,
        "base_initialization_sha256": base_initialization_sha,
        "trainable_parameters": parameter_count,
        "optimizer": {
            "name": training["optimizer"],
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
        },
        "EMA_decay": float(training["EMA_decay"]),
        "updates": int(training["optimizer_updates_per_run"]),
        "checkpoint_every_updates": int(training["checkpoint_every_updates"]),
        "model_and_corruption_dtype": training["model_and_corruption_dtype"],
        "automatic_mixed_precision": training["automatic_mixed_precision"],
        "validation_early_stopping": training["validation_early_stopping"],
        "outer_test_reconstruction_evaluation": False,
    }


def _running_summary() -> dict[str, Any]:
    return {
        "updates_completed": 0,
        "first_loss": None,
        "final_loss": None,
        "loss_min": None,
        "loss_max": None,
        "gradient_norm_max_before_clipping": None,
        "wall_seconds_accumulated": 0.0,
        "MULAN_online_schedule_audits": [],
    }


def _update_summary(
    summary: dict[str, Any], *, loss: float, gradient_norm: float, update: int
) -> None:
    if not np.isfinite(loss) or not np.isfinite(gradient_norm):
        raise FloatingPointError("formal training summary received a non-finite value")
    if summary["first_loss"] is None:
        summary["first_loss"] = loss
        summary["loss_min"] = loss
        summary["loss_max"] = loss
        summary["gradient_norm_max_before_clipping"] = gradient_norm
    summary["final_loss"] = loss
    summary["loss_min"] = min(float(summary["loss_min"]), loss)
    summary["loss_max"] = max(float(summary["loss_max"]), loss)
    summary["gradient_norm_max_before_clipping"] = max(
        float(summary["gradient_norm_max_before_clipping"]), gradient_norm
    )
    summary["updates_completed"] = update


def _full_fold_mulan_audit(
    system: TinyDenoisingSystem,
    fold: Any,
    baseline_alpha: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    label: str,
    update: int,
) -> dict[str, Any]:
    if system.scheduler is None:
        raise RuntimeError("MuLAN-lite schedule head is absent")
    condition_numpy = np.concatenate(
        [fold.condition_train, fold.condition_test], axis=0
    ).astype(np.float32, copy=False)
    condition = torch.as_tensor(condition_numpy, dtype=torch.float32, device=device)
    fixed = torch.as_tensor(
        fold.fixed_variance, dtype=torch.float32, device=device
    )
    proxy = p0_runner._mulan_proxy(system, condition, fixed)
    true = np.concatenate([fold.variance_train, fold.variance_test], axis=0)
    weights = np.concatenate([fold.weights_train, fold.weights_test], axis=0)
    audit = p0_runner._schedule_audit(
        "MULAN_LITE",
        true,
        weights,
        baseline_alpha,
        config,
        proxy_variance=proxy,
    )
    audit["state"] = label
    audit["after_update"] = update
    return audit


def _ema_system_state(
    system: TinyDenoisingSystem, ema: TinyEMA
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
    }
    for name, value in ema.shadow.items():
        if name not in state:
            raise RuntimeError(f"EMA parameter missing from system state: {name}")
        state[name] = value.detach().cpu().clone()
    return state


def _ema_clone(
    spec: RunSpec,
    system: TinyDenoisingSystem,
    ema: TinyEMA,
    device: torch.device,
) -> TinyDenoisingSystem:
    clone = TinyDenoisingSystem(spec.path, model_seed=spec.model_seed).to(device)
    clone.load_state_dict(_ema_system_state(system, ema), strict=True)
    clone.eval()
    return clone


def _checkpoint_payload(
    identity: Mapping[str, Any],
    identity_sha: str,
    system: TinyDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": dict(identity),
        "run_identity_sha256": identity_sha,
        "completed_updates": int(summary["updates_completed"]),
        "system": system.state_dict(),
        "optimizer": optimizer.state_dict(),
        "EMA": ema.state_dict(),
        "runner_summary": dict(summary),
        "RNG": p0_runner._rng_state(),
    }


def _load_resume(
    path: Path,
    identity: Mapping[str, Any],
    identity_sha: str,
    system: TinyDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    _verified_sidecar(path)
    # RNG-state tensors must remain on CPU; module/optimizer/EMA loaders move
    # their own tensors to the registered parameter device.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != RESUME_SCHEMA:
        raise RuntimeError("formal resume checkpoint schema drifted")
    if (
        payload.get("run_identity_sha256") != identity_sha
        or payload.get("run_identity") != identity
    ):
        raise RuntimeError("formal resume identity drifted")
    completed = int(payload["completed_updates"])
    maximum = int(config["training"]["optimizer_updates_per_run"])
    interval = int(config["training"]["checkpoint_every_updates"])
    if not 0 < completed <= maximum or completed % interval != 0:
        raise RuntimeError("resume update is not a registered boundary")
    system.load_state_dict(payload["system"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    ema.load_state_dict(payload["EMA"], system)
    summary = dict(payload["runner_summary"])
    if int(summary["updates_completed"]) != completed:
        raise RuntimeError("resume runner summary update drifted")
    p0_runner._set_rng_state(payload["RNG"])
    return completed, summary


def _remove_resume(path: Path) -> None:
    for target in (path, path.with_name(path.name + ".sha256")):
        if target.exists():
            target.unlink()


def _completion_payload(
    *,
    spec: RunSpec,
    identity: Mapping[str, Any],
    identity_sha: str,
    summary: Mapping[str, Any],
    final_schedule_audit: Mapping[str, Any] | None,
    final_path: Path,
    final_sha: str,
    ema_state_sha: str,
    ema_updates: int,
) -> dict[str, Any]:
    maximum_updates = int(identity["updates"])
    batch_days = int(identity["randomness"]["batch_calendar_days"])
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **spec.manifest(),
        "run_identity": dict(identity),
        "run_identity_sha256": identity_sha,
        "optimizer_updates": maximum_updates,
        "EMA_updates": int(ema_updates),
        "calendar_day_corruption_exposures": maximum_updates * batch_days,
        "training_summary": dict(summary),
        "MULAN_final_EMA_schedule_audit": final_schedule_audit,
        "final_EMA_checkpoint": str(final_path.resolve()),
        "final_EMA_checkpoint_sha256": final_sha,
        "final_EMA_system_state_sha256": ema_state_sha,
        "only_final_EMA_weight_retained": True,
        "resume_checkpoint_deleted": not (final_path.parent / "resume.pt").exists(),
        "materialized_target_roles": ["train"],
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
        "outer_test_reconstruction_bank_constructed": False,
        "outer_test_reconstruction_evaluation_performed": False,
        "raw_training_loss_used_as_formal_path_evidence": False,
    }


def _recover_or_validate_completion(
    run_root: Path, identity_sha: str, spec: RunSpec
) -> dict[str, Any] | None:
    completion_path = run_root / "run.json"
    final_path = run_root / "final_ema.pt"
    if not completion_path.exists() and not final_path.exists():
        return None
    if completion_path.exists() and not final_path.exists():
        raise RuntimeError(f"completion exists without final EMA: {spec.key}")
    if final_path.exists() and not completion_path.exists():
        # Recover the narrow interruption window after the immutable final EMA
        # was written but before its JSON completion record was committed.
        final_sha = _verified_sidecar(final_path)
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != FINAL_EMA_SCHEMA
            or payload.get("run_identity_sha256") != identity_sha
            or payload.get("run_identity", {}).get("run_key") != spec.key
            or int(payload.get("completed_updates", -1)) != 1024
            or int(payload.get("EMA_updates", -1)) != 1024
            or payload.get("outer_test_reconstruction_evaluation_performed")
            is not False
        ):
            raise RuntimeError(f"uncommitted final EMA identity drifted: {spec.key}")
        actual_state_sha = tensor_mapping_sha256(payload["EMA_system_state"])
        if actual_state_sha != payload.get("EMA_system_state_sha256"):
            raise RuntimeError(f"uncommitted final EMA tensors drifted: {spec.key}")
        _remove_resume(run_root / "resume.pt")
        completion = _completion_payload(
            spec=spec,
            identity=payload["run_identity"],
            identity_sha=identity_sha,
            summary=payload["runner_summary"],
            final_schedule_audit=payload["MULAN_final_EMA_schedule_audit"],
            final_path=final_path,
            final_sha=final_sha,
            ema_state_sha=actual_state_sha,
            ema_updates=int(payload["EMA_updates"]),
        )
        _atomic_json(completion_path, completion)
        for failure in (run_root / "failure.json", run_root / "failure.json.sha256"):
            if failure.exists():
                failure.unlink()
    _verified_sidecar(completion_path)
    final_sha = _verified_sidecar(final_path)
    completion = _read_json(completion_path)
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("run_key") != spec.key
        or completion.get("run_identity_sha256") != identity_sha
        or completion.get("final_EMA_checkpoint_sha256") != final_sha
        or int(completion.get("optimizer_updates", -1)) != 1024
        or completion.get("outer_test_reconstruction_evaluation_performed") is not False
    ):
        raise RuntimeError(f"formal completion identity drifted: {spec.key}")
    unexpected = [
        path.name
        for path in run_root.glob("*.pt")
        if path.name != "final_ema.pt"
    ]
    if unexpected:
        raise RuntimeError(f"completed run retained non-final weights: {unexpected}")
    _validate_final_checkpoint(final_path, completion)
    return completion


def _run_one(
    *,
    config: Mapping[str, Any],
    spec: RunSpec,
    fold: Any,
    train_days: np.ndarray,
    groups: Sequence[Any],
    baseline_alpha_numpy: np.ndarray,
    global_identity_sha: str,
    formal_root: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    run_root = _run_root(formal_root, spec)
    run_root.mkdir(parents=True, exist_ok=True)
    subset, fixed_numpy, local, subset_manifest = _subset_context(
        config, fold, train_days, spec, device
    )
    random_bank, random_manifest = _random_context(config, spec, len(local))
    shuffle_permutations, shuffle_sha = _shuffle_context(
        config, spec, len(local), device
    )
    system = TinyDenoisingSystem(spec.path, model_seed=spec.model_seed).to(device)
    base_sha = module_state_sha256(system.denoiser)
    parameter_count = sum(parameter.numel() for parameter in system.parameters())
    expected_parameters = TINY_DENOISER_PARAMETERS + (
        MULAN_LITE_PARAMETERS if spec.path == "MULAN_LITE" else 0
    )
    if parameter_count != expected_parameters:
        raise RuntimeError("formal tiny-denoiser parameter count drifted")
    identity = _run_identity(
        config,
        spec,
        global_identity_sha,
        subset_manifest,
        random_manifest,
        shuffle_sha,
        base_sha,
        parameter_count,
    )
    identity_sha = _canonical_sha256(identity)
    completion = _recover_or_validate_completion(run_root, identity_sha, spec)
    if completion is not None:
        if not resume:
            raise FileExistsError(f"completed run exists; use --resume: {spec.key}")
        return completion

    training = config["training"]
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema = TinyEMA(system, decay=float(training["EMA_decay"]))
    baseline_alpha = torch.as_tensor(
        baseline_alpha_numpy, dtype=torch.float32, device=device
    )
    fixed = torch.as_tensor(fixed_numpy, dtype=torch.float32, device=device)
    projectors = ModeProjectorBank(groups).to(device)
    resume_path = run_root / "resume.pt"
    start_update = 0
    summary = _running_summary()
    if resume_path.exists():
        if not resume:
            raise FileExistsError(f"resume checkpoint exists; use --resume: {spec.key}")
        start_update, summary = _load_resume(
            resume_path,
            identity,
            identity_sha,
            system,
            optimizer,
            ema,
            config,
            device,
        )
    elif any(run_root.iterdir()):
        allowed = {"failure.json", "failure.json.sha256"}
        present = {path.name for path in run_root.iterdir()}
        if not present.issubset(allowed):
            raise RuntimeError(f"unrecognized partial run artifacts: {spec.key}")

    maximum_updates = int(training["optimizer_updates_per_run"])
    checkpoint_every = int(training["checkpoint_every_updates"])
    process_started = time.perf_counter()
    if start_update == 0:
        torch.manual_seed(int(random_manifest["training_random_seed"]))
        np.random.seed(int(random_manifest["training_random_seed"]))
        random.seed(int(random_manifest["training_random_seed"]))
    path_shuffle = shuffle_permutations if spec.path == "PA_SHUFFLE" else None
    try:
        for update_zero in range(start_update, maximum_updates):
            loss, gradient_norm = p0_runner._training_update(
                system,
                optimizer,
                ema,
                projectors,
                subset,
                random_bank,
                update_zero,
                device,
                baseline_alpha,
                fixed,
                config,
                shuffle_permutations=path_shuffle,
            )
            completed = update_zero + 1
            _update_summary(
                summary,
                loss=loss,
                gradient_norm=gradient_norm,
                update=completed,
            )
            if completed % checkpoint_every == 0:
                if spec.path == "MULAN_LITE":
                    summary["MULAN_online_schedule_audits"].append(
                        _full_fold_mulan_audit(
                            system,
                            fold,
                            baseline_alpha_numpy,
                            config,
                            device,
                            label="online_checkpoint",
                            update=completed,
                        )
                    )
                summary["wall_seconds_accumulated"] = float(
                    summary["wall_seconds_accumulated"]
                ) + float(time.perf_counter() - process_started)
                _atomic_torch(
                    resume_path,
                    _checkpoint_payload(
                        identity, identity_sha, system, optimizer, ema, summary
                    ),
                )
                process_started = time.perf_counter()
        if int(summary["updates_completed"]) != maximum_updates:
            raise RuntimeError("formal run stopped before its frozen update budget")
        if ema.num_updates != maximum_updates:
            raise RuntimeError("formal EMA update count drifted")
        if spec.path == "MULAN_LITE" and len(
            summary["MULAN_online_schedule_audits"]
        ) != maximum_updates // checkpoint_every:
            raise RuntimeError("MuLAN-lite checkpoint schedule audits are incomplete")

        ema_state = _ema_system_state(system, ema)
        ema_state_sha = tensor_mapping_sha256(ema_state)
        final_schedule_audit = None
        if spec.path == "MULAN_LITE":
            ema_system = _ema_clone(spec, system, ema, device)
            final_schedule_audit = _full_fold_mulan_audit(
                ema_system,
                fold,
                baseline_alpha_numpy,
                config,
                device,
                label="final_EMA",
                update=maximum_updates,
            )
            del ema_system
        final_path = run_root / "final_ema.pt"
        final_payload = {
            "schema": FINAL_EMA_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_identity": identity,
            "run_identity_sha256": identity_sha,
            "completed_updates": maximum_updates,
            "EMA_updates": ema.num_updates,
            "EMA_system_state": ema_state,
            "EMA_system_state_sha256": ema_state_sha,
            "runner_summary": summary,
            "MULAN_final_EMA_schedule_audit": final_schedule_audit,
            "outer_test_reconstruction_evaluation_performed": False,
        }
        final_sha = _atomic_torch(final_path, final_payload)
        _remove_resume(resume_path)
        completion = _completion_payload(
            spec=spec,
            identity=identity,
            identity_sha=identity_sha,
            summary=summary,
            final_schedule_audit=final_schedule_audit,
            final_path=final_path,
            final_sha=final_sha,
            ema_state_sha=ema_state_sha,
            ema_updates=ema.num_updates,
        )
        _atomic_json(run_root / "run.json", completion)
        retained = sorted(path.name for path in run_root.glob("*.pt"))
        if retained != ["final_ema.pt"]:
            raise RuntimeError(f"formal run retained unexpected weights: {retained}")
        for failure in (run_root / "failure.json", run_root / "failure.json.sha256"):
            if failure.exists():
                failure.unlink()
        return completion
    except Exception as error:
        _atomic_json(
            run_root / "failure.json",
            {
                "schema": "architecture_v1_g0_b_tiny_run_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_key": spec.key,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "latest_registered_resume_checkpoint": (
                    str(resume_path.resolve()) if resume_path.exists() else None
                ),
                "resume_required": True,
                "outer_test_reconstruction_evaluation_performed": False,
            },
        )
        raise
    finally:
        del ema, optimizer, projectors, system
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _completion_records(
    formal_root: Path, matrix: Sequence[RunSpec]
) -> dict[str, dict[str, Any]]:
    complete: dict[str, dict[str, Any]] = {}
    for spec in matrix:
        path = _run_root(formal_root, spec) / "run.json"
        if not path.exists():
            continue
        _verified_sidecar(path)
        record = _read_json(path)
        if record.get("schema") != COMPLETION_SCHEMA or record.get("status") != "complete":
            raise RuntimeError(f"invalid completion record: {spec.key}")
        if record.get("run_key") != spec.key:
            raise RuntimeError(f"completion run key drifted: {spec.key}")
        complete[spec.key] = record
    return complete


def _write_progress(
    formal_root: Path,
    matrix: Sequence[RunSpec],
    *,
    selected_this_invocation: Sequence[str],
) -> dict[str, Any]:
    complete = _completion_records(formal_root, matrix)
    pending = [spec.key for spec in matrix if spec.key not in complete]
    progress = {
        "schema": PROGRESS_SCHEMA,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": len(matrix),
        "completed_runs": len(complete),
        "remaining_runs": len(pending),
        "selected_this_invocation": list(selected_this_invocation),
        "completed_run_keys": sorted(complete),
        "pending_run_keys": pending,
        "training_frozen": (formal_root / "training.freeze.json").exists(),
        "outer_test_reconstruction_bank_constructed": False,
        "next_action": (
            "complete_remaining_retained_training_runs"
            if pending
            else "write_and_verify_training_freeze"
        ),
    }
    _atomic_json(formal_root / "training.progress.json", progress)
    return progress


def _validate_final_checkpoint(
    path: Path, completion: Mapping[str, Any]
) -> str:
    digest = _verified_sidecar(path)
    if digest != completion["final_EMA_checkpoint_sha256"]:
        raise RuntimeError("final EMA checkpoint/completion hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != FINAL_EMA_SCHEMA
        or int(payload.get("completed_updates", -1)) != 1024
        or int(payload.get("EMA_updates", -1)) != 1024
        or payload.get("run_identity_sha256")
        != completion["run_identity_sha256"]
        or payload.get("EMA_system_state_sha256")
        != completion["final_EMA_system_state_sha256"]
        or payload.get("runner_summary") != completion["training_summary"]
        or payload.get("MULAN_final_EMA_schedule_audit")
        != completion["MULAN_final_EMA_schedule_audit"]
        or payload.get("outer_test_reconstruction_evaluation_performed") is not False
    ):
        raise RuntimeError("final EMA checkpoint payload drifted")
    actual_state_sha = tensor_mapping_sha256(payload["EMA_system_state"])
    if actual_state_sha != payload["EMA_system_state_sha256"]:
        raise RuntimeError("final EMA tensor state hash drifted")
    return digest


def _try_freeze(
    formal_root: Path,
    matrix: Sequence[RunSpec],
    global_identity_sha: str,
) -> dict[str, Any] | None:
    freeze_path = formal_root / "training.freeze.json"
    if freeze_path.exists():
        _verified_sidecar(freeze_path)
        return _read_json(freeze_path)
    completions = _completion_records(formal_root, matrix)
    if len(completions) != len(matrix):
        return None
    checkpoint_hashes: dict[str, str] = {}
    paired: dict[tuple[int, float, int], dict[str, set[str]]] = {}
    for spec in matrix:
        completion = completions[spec.key]
        identity = completion["run_identity"]
        if (
            completion["run_identity_sha256"] != _canonical_sha256(identity)
            or identity["global_training_identity_sha256"] != global_identity_sha
            or identity["run_key"] != spec.key
            or int(completion["optimizer_updates"]) != 1024
            or int(completion["EMA_updates"]) != 1024
            or completion["only_final_EMA_weight_retained"] is not True
            or completion["resume_checkpoint_deleted"] is not True
            or completion["materialized_target_roles"] != ["train"]
            or completion["validation_target_accessed"] is not False
            or completion["calibration_target_accessed"] is not False
            or completion["selection_target_accessed"] is not False
            or completion["raw_training_loss_used_as_formal_path_evidence"]
            is not False
            or completion["outer_test_reconstruction_evaluation_performed"] is not False
        ):
            raise RuntimeError(f"completion failed freeze validation: {spec.key}")
        summary = completion["training_summary"]
        numeric_summary = (
            summary["first_loss"],
            summary["final_loss"],
            summary["loss_min"],
            summary["loss_max"],
            summary["gradient_norm_max_before_clipping"],
        )
        if not np.isfinite(np.asarray(numeric_summary, dtype=np.float64)).all():
            raise RuntimeError(f"non-finite training summary: {spec.key}")
        online_audits = summary["MULAN_online_schedule_audits"]
        final_audit = completion["MULAN_final_EMA_schedule_audit"]
        if spec.path == "MULAN_LITE":
            expected_updates = list(range(128, 1025, 128))
            if (
                [int(item["after_update"]) for item in online_audits]
                != expected_updates
                or final_audit is None
                or final_audit.get("state") != "final_EMA"
            ):
                raise RuntimeError(f"MuLAN schedule audit coverage failed: {spec.key}")
        elif online_audits or final_audit is not None:
            raise RuntimeError(f"non-MuLAN run contains learned-schedule audits: {spec.key}")
        run_root = _run_root(formal_root, spec)
        retained = sorted(path.name for path in run_root.glob("*.pt"))
        if retained != ["final_ema.pt"]:
            raise RuntimeError(f"unexpected retained checkpoints: {spec.key}")
        checkpoint_hashes[spec.key] = _validate_final_checkpoint(
            run_root / "final_ema.pt", completion
        )
        key = (spec.outer_fold, spec.fraction, spec.model_seed)
        group = paired.setdefault(
            key,
            {"initialization": set(), "random_bank": set(), "subset": set()},
        )
        group["initialization"].add(identity["base_initialization_sha256"])
        group["random_bank"].add(
            identity["randomness"]["common_random_bank_sha256"]
        )
        group["subset"].add(identity["subset"]["subset_date_sha256"])
    if any(
        len(values) != 1 for group in paired.values() for values in group.values()
    ):
        raise RuntimeError("cross-path initialization/randomness/subset pairing failed")
    if len(paired) != 6 * 3 * 3:
        raise RuntimeError("paired run-group count drifted")
    freeze = {
        "schema": FREEZE_SCHEMA,
        "status": "G0_B_TINY_DENOISER_324_OF_324_TRAINING_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "global_training_identity": str(
            (formal_root / "training.identity.json").resolve()
        ),
        "global_training_identity_sha256": global_identity_sha,
        "expected_runs": len(matrix),
        "completed_runs": len(completions),
        "completed_updates": 1024 * len(completions),
        "paired_fold_fraction_seed_groups": len(paired),
        "all_cross_path_pairing_checks_passed": True,
        "final_EMA_checkpoint_sha256_by_run": checkpoint_hashes,
        "only_final_EMA_checkpoints_retained": True,
        "training_closed": True,
        "outer_test_reconstruction_bank_constructed": False,
        "outer_test_reconstruction_evaluation_performed": False,
        "outer_test_reconstruction_evaluation_authorized": True,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_action": "implement_and_separately_execute_common_outer_test_reconstruction_evaluation",
    }
    _atomic_json(freeze_path, freeze)
    return freeze


def dry_run(config_path: Path, *, include_run_keys: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _code_identity(config_path)
    p0 = _validate_p0(config, config_path)
    matrix = _matrix(config)
    formal_root = ROOT / config["output_root"] / FORMAL_ROOT_NAME
    completed = (
        len(_completion_records(formal_root, matrix)) if formal_root.exists() else 0
    )
    result = {
        "schema": "architecture_v1_g0_b_tiny_formal_dry_run_v1",
        "mode": "target_free_no_files_created",
        "config_sha256": code["config_sha256"],
        "P0": p0,
        "expected_retained_runs": len(matrix),
        "completed_retained_runs": completed,
        "remaining_retained_runs": len(matrix) - completed,
        "matrix_dimensions": {
            "outer_folds": 6,
            "data_fractions": 3,
            "paths": 6,
            "model_seeds": 3,
        },
        "updates_per_run": int(config["training"]["optimizer_updates_per_run"]),
        "checkpoint_every_updates": int(
            config["training"]["checkpoint_every_updates"]
        ),
        "retained_checkpoint": config["training"]["retained_checkpoint"],
        "first_run_key": matrix[0].key,
        "last_run_key": matrix[-1].key,
        "run_keys": [spec.key for spec in matrix] if include_run_keys else "use --list-run-keys",
        "formal_training_target_roles_if_executed": ["train"],
        "outer_test_reconstruction_evaluation_implemented": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "CPU_execution_requires_allow_cpu": True,
        "next_flag": "--execute-training",
    }
    return result


def execute_training(
    config_path: Path,
    *,
    resume: bool,
    device_name: str,
    allow_cpu: bool,
    run_keys: Sequence[str],
    max_runs: int | None,
) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _code_identity(config_path)
    p0 = _validate_p0(config, config_path)
    matrix = _matrix(config)
    by_key = {spec.key: spec for spec in matrix}
    unknown = sorted(set(run_keys) - set(by_key))
    if unknown:
        raise ValueError(f"unknown formal run keys: {unknown}")
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("--run-key contains duplicates")
    formal_root = ROOT / config["output_root"] / FORMAL_ROOT_NAME
    freeze_path = formal_root / "training.freeze.json"
    if freeze_path.exists():
        _verified_sidecar(freeze_path)
        return _read_json(freeze_path)
    if formal_root.exists() and any(formal_root.iterdir()) and not resume:
        raise FileExistsError("formal training output is non-empty; use --resume")
    formal_root.mkdir(parents=True, exist_ok=True)
    device, runtime = _configure_device(device_name, allow_cpu=allow_cpu)
    bundle, groups, folds, nuisance_audit, subset_replay = p0_runner._replay_nuisance(
        config
    )
    baseline_alpha, baseline_identity = p0_runner._baseline_schedule(config)
    identity = _global_identity(
        config, config_path, p0, code, bundle, runtime, matrix
    )
    identity["nuisance_replay"] = nuisance_audit
    identity["all_18_subset_hashes_replayed"] = len(subset_replay) == 18
    identity["baseline_schedule"] = baseline_identity
    global_identity_sha = _prepare_global_identity(
        formal_root, identity, resume=resume
    )
    already_complete = _completion_records(formal_root, matrix)
    selected = [by_key[key] for key in run_keys] if run_keys else list(matrix)
    selected = [spec for spec in selected if spec.key not in already_complete]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("--max-runs must be positive")
        selected = selected[:max_runs]
    completed_this_invocation: list[str] = []
    for spec in selected:
        _run_one(
            config=config,
            spec=spec,
            fold=folds[spec.outer_fold],
            train_days=bundle.train.day,
            groups=groups,
            baseline_alpha_numpy=baseline_alpha,
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
        "schema": "architecture_v1_g0_b_tiny_training_invocation_result_v1",
        "status": (
            freeze["status"]
            if freeze is not None
            else "G0_B_TINY_DENOISER_RETAINED_TRAINING_PARTIAL"
        ),
        "completed_this_invocation": len(completed_this_invocation),
        "completed_runs_total": progress["completed_runs"],
        "remaining_runs": progress["remaining_runs"],
        "training_freeze_written": freeze is not None,
        "training_freeze": freeze,
        "outer_test_reconstruction_evaluation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow the formal matrix to run on CPU when CUDA is unavailable",
    )
    parser.add_argument(
        "--run-key",
        action="append",
        default=[],
        help="execute only one registered run key; repeat for multiple keys",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="operationally stop after this many pending runs",
    )
    parser.add_argument("--list-run-keys", action="store_true")
    arguments = parser.parse_args()
    if arguments.resume and not arguments.execute_training:
        parser.error("--resume requires --execute-training")
    if arguments.allow_cpu and not arguments.execute_training:
        parser.error("--allow-cpu is valid only with --execute-training")
    if arguments.run_key and not arguments.execute_training:
        parser.error("--run-key requires --execute-training")
    if arguments.max_runs is not None and not arguments.execute_training:
        parser.error("--max-runs requires --execute-training")
    config_path = arguments.config.resolve()
    result = (
        execute_training(
            config_path,
            resume=arguments.resume,
            device_name=arguments.device,
            allow_cpu=arguments.allow_cpu,
            run_keys=arguments.run_key,
            max_runs=arguments.max_runs,
        )
        if arguments.execute_training
        else dry_run(config_path, include_run_keys=arguments.list_run_keys)
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
