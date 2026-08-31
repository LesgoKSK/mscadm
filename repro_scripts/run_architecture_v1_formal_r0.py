#!/usr/bin/env python3
"""Fail-closed formal-v2 R0 preflight, training, and validation runner.

Dry-run is the default.  Preflight and retained-weight training require
separate explicit flags.  This runner imports only the stage-scoped fit data
facade, so calibration, selection, R-SEEN, and final targets are never
materialized by the process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture_v1.data import (  # noqa: E402
    INTERIOR_STATE,
    ONE_STATE,
    ZERO_STATE,
    ArchitectureFitDataBundle,
    build_architecture_v1_fit_data,
)
from architecture_v1.formal_evaluation import (  # noqa: E402
    aggregate_sampling_replicates,
    validation_per_day_metrics,
)
from architecture_v1.formal_training import (  # noqa: E402
    FORMAL_RESUME_SCHEMA,
    FormalEpochTrainer,
    FormalValidationBank,
    common_random_chunk_samples,
    evaluate_sampling_chunk_gate,
    write_training_freeze,
)
from architecture_v1.model import R0JointRectifiedFlow  # noqa: E402
from architecture_v1.protocol import (  # noqa: E402
    build_architecture_protocol,
    canonical_sha256,
    file_sha256,
)
from architecture_v1.training import (  # noqa: E402
    ArchitectureBatch,
    configure_stage,
    shared_ea_state_sha256,
    tensor_state_sha256,
    train_step,
)


DEFAULT_CONFIG = PROJECT_ROOT / "repro_configs" / "architecture_v1_formal_v2.json"
RUNNER_SCHEMA = "architecture_v1_formal_r0_runner_v1"
PREFLIGHT_SCHEMA = "architecture_v1_formal_r0_preflight_v1"
DERIVED_SCHEMA = "architecture_v1_formal_r0_train_derived_v1"
VALIDATION_ARCHIVE_SCHEMA = "architecture_v1_formal_validation_scenarios_v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[D]").astype(str).tolist()
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> str:
    if not overwrite and (path.exists() or path.with_name(path.name + ".sha256").exists()):
        raise FileExistsError(f"refusing to overwrite formal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _verified_sha(path: str | Path) -> str:
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"artifact or SHA sidecar is missing: {source}")
    declared = sidecar.read_text(encoding="ascii").split()[0]
    actual = file_sha256(source)
    if declared != actual:
        raise RuntimeError(f"artifact SHA256 mismatch: {source}")
    return actual


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(
    config_path: str | Path,
    amendment_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    path = Path(config_path).resolve()
    config = _read_json(path)
    revision = config.get("protocol_revision")
    if revision not in {
        "architecture_v1_formal_v2_20260815",
        "architecture_v1_formal_v2_2_20260815",
        "architecture_v1_formal_v2_2_1_20260818",
    }:
        raise ValueError("runner accepts only the frozen formal-v2/v2.2 revisions")
    if config.get("selection_lock", {}).get("state") != "sealed":
        raise RuntimeError("selection must remain sealed during formal R0")
    if config.get("experiment_contract", {}).get("calibration_status") != "sealed":
        raise RuntimeError("calibration must remain sealed during formal R0")
    gate = config.get("gate_registry", {})
    gate_path = _resolve_repo_path(gate["path"])
    if file_sha256(gate_path) != gate["file_sha256"]:
        raise RuntimeError("frozen Go/No-Go registry hash mismatch")
    if _read_json(gate_path).get("schema") != gate["schema"]:
        raise ValueError("unexpected Go/No-Go registry schema")
    if amendment_path is not None:
        if revision != "architecture_v1_formal_v2_20260815":
            raise RuntimeError("formal-v2.2 does not accept the v2.1 amendment mechanism")
        amendment_source = Path(amendment_path).resolve()
        amendment = _read_json(amendment_source)
        if amendment.get("schema") != "architecture_v1_formal_amendment_v1":
            raise ValueError("unexpected formal amendment schema")
        if amendment.get("base_config_file_sha256") != file_sha256(path):
            raise RuntimeError("formal amendment base-config hash mismatch")
        declared_base = _resolve_repo_path(amendment["base_config"]).resolve()
        if declared_base != path:
            raise RuntimeError("formal amendment points to a different base config")
        if amendment.get("scientific_settings_changed") is not False:
            raise RuntimeError("this runner accepts only engineering-only amendments")
        config = dict(config)
        config["output_root"] = amendment["output_root"]
        config["formal_amendment"] = {
            "path": amendment_source.relative_to(PROJECT_ROOT).as_posix(),
            "file_sha256": file_sha256(amendment_source),
            "payload": amendment,
        }
    return path, config, file_sha256(path)


def _code_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "architecture_v1" / name
        for name in (
            "atom.py",
            "data.py",
            "evaluation.py",
            "formal_evaluation.py",
            "formal_training.py",
            "model.py",
            "protocol.py",
            "training.py",
        )
    ]
    paths.extend(
        [
            Path(__file__).resolve(),
            config_path,
            _resolve_repo_path(config["gate_registry"]["path"]),
            PROJECT_ROOT / "repro_configs" / "architecture_v1_smoke_quarantine.json",
        ]
    )
    if config.get("protocol_revision") in {
        "architecture_v1_formal_v2_2_20260815",
        "architecture_v1_formal_v2_2_1_20260818",
    }:
        paths.append(
            PROJECT_ROOT
            / "repro_scripts"
            / "run_architecture_v1_formal_r0_v2_2.py"
        )
    if "formal_amendment" in config:
        paths.append(_resolve_repo_path(config["formal_amendment"]["path"]))
    records = [
        {
            "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(set(paths))
    ]
    core = {"schema": "architecture_v1_formal_code_manifest_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _output_root(config: Mapping[str, Any]) -> Path:
    return _resolve_repo_path(config["output_root"])


def _load_v2_2_p0(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], str] | None:
    if config.get("protocol_revision") not in {
        "architecture_v1_formal_v2_2_20260815",
        "architecture_v1_formal_v2_2_1_20260818",
    }:
        return None
    root = _output_root(config) / "P0_gradient_scale_preflight"
    result_path = root / "P0_RESULT.json"
    result_sha = _verified_sha(result_path)
    result = _read_json(result_path)
    if result.get("status") != "P0_GO" or result.get("passed") is not True:
        raise RuntimeError("formal-v2.2 retained training requires P0_GO")
    if result.get("completed_all_three_seeds") is not True:
        raise RuntimeError("formal-v2.2 P0 did not complete all three seeds")
    if result.get("all_weights_discarded") is not True:
        raise RuntimeError("formal-v2.2 P0 retained a forbidden pilot weight")
    for key in (
        "validation_target_accessed",
        "calibration_target_accessed",
        "selection_target_accessed",
        "r_seen_target_accessed",
        "final_target_accessed",
    ):
        if result.get(key) is not False:
            raise RuntimeError(f"formal-v2.2 P0 forbidden access flag changed: {key}")
    identity_path = root / "identity.json"
    identity_sha = _verified_sha(identity_path)
    identity = _read_json(identity_path)
    if result.get("identity_file_sha256") != identity_sha:
        raise RuntimeError("formal-v2.2 P0 identity hash mismatch")
    checks = {
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
    }
    for name, expected in checks.items():
        if identity.get(name) != expected:
            raise RuntimeError(f"formal-v2.2 P0 identity drift: {name}")
    # P0's code manifest intentionally excludes this retained-training runner,
    # so later implementation cannot retroactively alter the discarded pilot.
    from repro_scripts.run_architecture_v1_formal_r0_v2_2 import (
        _code_manifest as p0_code_manifest,
    )

    current_p0_code = p0_code_manifest(config_path, config)["code_sha256"]
    if identity.get("code_manifest", {}).get("code_sha256") != current_p0_code:
        raise RuntimeError("formal-v2.2 P0 code changed after the P0_GO artifact")
    return result_path, result, result_sha


@dataclass(frozen=True)
class TrainDerived:
    train_observed: int
    train_zero: int
    train_interior: int
    train_one: int
    validation_observed: int
    validation_zero: int
    validation_interior: int
    validation_one: int
    interior_logit_mean: float
    interior_logit_std: float
    upper_one_jeffreys_prior: float
    zero_conditional_jeffreys_prior: float
    upper_one_learnable: bool


def _derive_train_quantities(
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
) -> TrainDerived:
    counts: dict[str, dict[str, int]] = {}
    for role in ("train", "validation"):
        split = bundle.role(role)
        observed = split.observed_mask
        counts[role] = {
            "observed": int(observed.sum()),
            "zero": int((observed & (split.state == ZERO_STATE)).sum()),
            "interior": int((observed & (split.state == INTERIOR_STATE)).sum()),
            "one": int((observed & (split.state == ONE_STATE)).sum()),
        }
    train = bundle.train
    interior = train.observed_mask & (train.state == INTERIOR_STATE)
    epsilon = float(config["shared_ea"]["interior_auxiliary"]["logit_clip_epsilon"])
    values = np.clip(train.target[interior].astype(np.float64), epsilon, 1.0 - epsilon)
    logits = np.log(values / (1.0 - values))
    mean = float(logits.mean())
    std = float(logits.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise FloatingPointError("train-only interior logit normalizer is invalid")
    upper = config["shared_ea"]["upper_atom"]
    learnable = (
        counts["train"]["one"] >= int(upper["train_positive_minimum_for_learning"])
        and counts["validation"]["one"]
        >= int(upper["validation_positive_minimum_for_learning"])
    )
    return TrainDerived(
        train_observed=counts["train"]["observed"],
        train_zero=counts["train"]["zero"],
        train_interior=counts["train"]["interior"],
        train_one=counts["train"]["one"],
        validation_observed=counts["validation"]["observed"],
        validation_zero=counts["validation"]["zero"],
        validation_interior=counts["validation"]["interior"],
        validation_one=counts["validation"]["one"],
        interior_logit_mean=mean,
        interior_logit_std=std,
        upper_one_jeffreys_prior=(counts["train"]["one"] + 0.5)
        / (counts["train"]["observed"] + 1.0),
        zero_conditional_jeffreys_prior=(counts["train"]["zero"] + 0.5)
        / (counts["train"]["zero"] + counts["train"]["interior"] + 1.0),
        upper_one_learnable=bool(learnable),
    )


def _model_kwargs(config: Mapping[str, Any], derived: TrainDerived) -> dict[str, Any]:
    if derived.upper_one_learnable:
        raise RuntimeError(
            "formal-v2 registered bounded contextual upper-atom branch is not "
            "implemented; create a new revision rather than silently using it"
        )
    common = config["model_common_basis"]
    p_one = derived.upper_one_jeffreys_prior
    q_zero = derived.zero_conditional_jeffreys_prior
    probabilities = (
        (1.0 - p_one) * q_zero,
        (1.0 - p_one) * (1.0 - q_zero),
        p_one,
    )
    auxiliary = config["shared_ea"]["interior_auxiliary"]
    return {
        "condition_dim": int(common["condition_dim"]),
        "zones": int(common["zones"]),
        "hours": int(common["hours"]),
        "encoder_dim": int(common["encoder_dim"]),
        "encoder_depth": int(common["encoder_depth"]),
        "flow_dim": int(common["flow_dim"]),
        "flow_depth": int(common["flow_depth"]),
        "heads": int(common["heads"]),
        "ff_multiplier": int(common["ff_multiplier"]),
        "dropout": float(common["dropout"]),
        "atom_hidden_dim": int(common["atom_hidden_dim"]),
        "atom_initial_probabilities": probabilities,
        "atom_fixed_one_probability": p_one,
        "atom_shared_priority_weight": float(common["atom_shared_priority_weight"]),
        "atom_location_auxiliary_weight": float(auxiliary["weight"]),
        "atom_location_mean": derived.interior_logit_mean,
        "atom_location_std": derived.interior_logit_std,
        "atom_location_smooth_l1_beta": float(auxiliary["smooth_l1_beta"]),
        "logit_epsilon": float(common["logit_epsilon"]),
    }


def _configure_runtime(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal R0 execution requires a visible CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_index": int(index),
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "amp": False,
        "tf32": False,
    }


def _batch(split: Any, indices: Sequence[int], device: torch.device) -> ArchitectureBatch:
    return ArchitectureBatch.from_split(split, indices).to(device)


def _run_preflight_benchmark(
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
    derived: TrainDerived,
    *,
    batch_days: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(990000 + batch_days)
    torch.cuda.manual_seed_all(990000 + batch_days)
    model = R0JointRectifiedFlow(**_model_kwargs(config, derived)).to(device)
    training = config["formal_training"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    last_metrics: dict[str, float] = {}
    atom_spec = training["ea_stage"]
    parameters = configure_stage(model, "atom")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(atom_spec["learning_rate"]),
        weight_decay=float(atom_spec["weight_decay"]),
    )
    for update in range(25):
        begin = (update * batch_days) % len(bundle.train)
        indices = np.arange(begin, begin + batch_days) % len(bundle.train)
        last_metrics = train_step(
            model,
            _batch(bundle.train, indices, device),
            optimizer,
            stage="atom",
            gradient_clip=float(atom_spec["gradient_clip"]),
        )
    model.freeze_shared(True)
    flow_spec = training["flow_stage"]
    parameters = configure_stage(model, "flow")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(flow_spec["learning_rate"]),
        weight_decay=float(flow_spec["weight_decay"]),
    )
    for update in range(25):
        begin = (update * batch_days) % len(bundle.train)
        indices = np.arange(begin, begin + batch_days) % len(bundle.train)
        generator = torch.Generator(device=device).manual_seed(991000 + update)
        last_metrics = train_step(
            model,
            _batch(bundle.train, indices, device),
            optimizer,
            stage="flow",
            generator=generator,
            gradient_clip=float(flow_spec["gradient_clip"]),
        )
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    del optimizer, model
    torch.cuda.empty_cache()
    return {
        "batch_days": int(batch_days),
        "updates": 50,
        "atom_updates": 25,
        "flow_updates": 25,
        "wall_seconds_synchronized": float(wall),
        "updates_per_second": float(50.0 / wall),
        "peak_memory_allocated_bytes": peak,
        "last_metrics": last_metrics,
        "weights_retained": False,
    }


def execute_preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    p0 = _load_v2_2_p0(config_path, config)
    bundle = build_architecture_v1_fit_data(config_path=config_path)
    derived = _derive_train_quantities(bundle, config)
    code = _code_manifest(config_path, config)
    branch = config["formal_training"]["batch_size_preflight_branch"]
    primary = int(branch["primary"])
    fallback = int(branch["fallback"])
    fallback_enabled = bool(branch.get("fallback_enabled", True))
    limit = int(4.5 * 1024**3)
    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    try:
        primary_result = _run_preflight_benchmark(
            bundle, config, derived, batch_days=primary, device=device
        )
        attempts.append({"status": "complete", **primary_result})
        if primary_result["peak_memory_allocated_bytes"] <= limit:
            selected = primary
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        attempts.append(
            {
                "status": "cuda_oom",
                "batch_days": primary,
                "exception": str(error),
                "weights_retained": False,
            }
        )
    if selected is None and not fallback_enabled:
        raise RuntimeError(
            "registered batch-16 CUDA preflight failed; formal-v2.2 forbids fallback"
        )
    if selected is None:
        fallback_result = _run_preflight_benchmark(
            bundle, config, derived, batch_days=fallback, device=device
        )
        attempts.append({"status": "complete", **fallback_result})
        if fallback_result["peak_memory_allocated_bytes"] > limit:
            raise RuntimeError("fallback batch still exceeds the preregistered 4.5 GiB limit")
        selected = fallback
    protocol_sha = str(bundle.protocol.manifest["protocol_sha256"])
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete_weights_discarded",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "protocol_sha256": protocol_sha,
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "code_manifest": code,
        "runtime": runtime,
        "train_derived": asdict(derived),
        "attempts": attempts,
        "resolved_batch_days": int(selected),
        "decision_rule": branch["criterion"],
        "memory_limit_bytes": limit,
        "retained_weight_training_started": False,
        "forbidden_target_access": {
            "calibration": False,
            "selection": False,
            "r_seen": False,
            "final": False,
        },
    }
    if p0 is not None:
        payload["formal_v2_2_P0"] = {
            "path": str(p0[0].resolve()),
            "file_sha256": p0[2],
            "status": p0[1]["status"],
            "passed": p0[1]["passed"],
            "execution_order_deviation": (
                "P0 ran before the 50-update capacity preflight; both used "
                "discarded weights and completed before retained training"
            ),
        }
    destination = _output_root(config) / "preflight" / "preflight.json"
    digest = _atomic_json(destination, payload)
    return {**payload, "path": str(destination), "file_sha256": digest}


def _load_preflight(
    config_path: Path,
    config: Mapping[str, Any],
    bundle: ArchitectureFitDataBundle,
) -> tuple[Path, dict[str, Any], str]:
    path = _output_root(config) / "preflight" / "preflight.json"
    digest = _verified_sha(path)
    payload = _read_json(path)
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("status") != "complete_weights_discarded":
        raise RuntimeError("formal preflight is absent or incomplete")
    current_code = _code_manifest(config_path, config)
    checks = {
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
    }
    for name, expected in checks.items():
        if payload.get(name) != expected:
            raise RuntimeError(f"preflight identity drift: {name}")
    if payload.get("code_manifest", {}).get("code_sha256") != current_code["code_sha256"]:
        raise RuntimeError("code changed after preflight; rerun under a new formal revision")
    if payload.get("retained_weight_training_started") is not False:
        raise RuntimeError("preflight retained-weight status is invalid")
    p0 = _load_v2_2_p0(config_path, config)
    if p0 is not None:
        registered = payload.get("formal_v2_2_P0", {})
        if (
            registered.get("file_sha256") != p0[2]
            or registered.get("passed") is not True
        ):
            raise RuntimeError("formal-v2.2 preflight did not bind the P0_GO artifact")
    return path, payload, digest


def _make_validation_bank(
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
    *,
    evaluation_batch_days: int,
) -> FormalValidationBank:
    specification = config["formal_training"]["validation_flow_bank"]
    days = ArchitectureBatch.from_split(
        bundle.validation, np.arange(len(bundle.validation))
    ).day_index
    return FormalValidationBank.from_explicit_seeds(
        days,
        noise_seeds=specification["noise_seeds"],
        time_seeds=specification["time_seeds"],
        plan_id="architecture_v1_formal_v2_fixed_validation_bank",
        evaluation_batch_days=evaluation_batch_days,
    )


def _persist_validation_bank(bank: FormalValidationBank, output_root: Path) -> dict[str, str]:
    path = output_root / "validation_bank" / "fixed_bank.pt"
    manifest_path = output_root / "validation_bank" / "fixed_bank.manifest.json"
    if path.exists() or manifest_path.exists():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if stored.get("manifest") != bank.manifest:
            raise RuntimeError("persisted validation bank identity changed")
        if tensor_state_sha256(stored["tensors"]) != bank.tensor_sha256:
            raise RuntimeError("persisted validation bank tensors changed")
        return {
            "path": str(path.resolve()),
            "file_sha256": _verified_sha(path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _verified_sha(manifest_path),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "manifest": bank.manifest,
            "tensors": {
                "day_index": bank.day_index,
                "noise": bank.noise,
                "flow_time": bank.flow_time,
            },
        },
        temporary,
    )
    temporary.replace(path)
    digest = file_sha256(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    manifest_digest = _atomic_json(manifest_path, bank.manifest)
    return {
        "path": str(path.resolve()),
        "file_sha256": digest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
    }


def _load_best_checkpoint(path: Path, model: R0JointRectifiedFlow, device: torch.device) -> dict[str, Any]:
    _verified_sha(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != FORMAL_RESUME_SCHEMA:
        raise ValueError("unexpected formal best checkpoint schema")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if tensor_state_sha256(state) != payload.get("model_state_sha256"):
        raise RuntimeError("formal best checkpoint tensor hash mismatch")
    return payload


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    found = False
    for gradient in gradients:
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("non-finite G0 encoder gradient")
            squared = squared + gradient.double().square().sum()
            found = True
    return float(torch.sqrt(squared).detach().cpu()) if found else 0.0


def _g0_evaluate(
    model: R0JointRectifiedFlow,
    bundle: ArchitectureFitDataBundle,
    bank: FormalValidationBank,
    derived: TrainDerived,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    validation = bundle.validation
    observed = validation.observed_mask
    states = validation.state
    probabilities = np.asarray(
        [
            (1.0 - derived.upper_one_jeffreys_prior)
            * derived.zero_conditional_jeffreys_prior,
            (1.0 - derived.upper_one_jeffreys_prior)
            * (1.0 - derived.zero_conditional_jeffreys_prior),
            derived.upper_one_jeffreys_prior,
        ],
        dtype=np.float64,
    )
    atom_prior_nll = float(-np.log(probabilities[states[observed]]).mean())
    interior = observed & (states == INTERIOR_STATE)
    epsilon = float(config["shared_ea"]["interior_auxiliary"]["logit_clip_epsilon"])
    values = np.clip(validation.target[interior].astype(np.float64), epsilon, 1.0 - epsilon)
    normalized = (
        np.log(values / (1.0 - values)) - derived.interior_logit_mean
    ) / derived.interior_logit_std
    location_zero = float(
        F.smooth_l1_loss(
            torch.zeros_like(torch.from_numpy(normalized)),
            torch.from_numpy(normalized),
            beta=float(config["shared_ea"]["interior_auxiliary"]["smooth_l1_beta"]),
        ).item()
    )
    trained = bank.evaluate(model, validation, stage="atom", device=device)
    audit_batch = _batch(bundle.train, np.arange(min(16, len(bundle.train))), device)
    model.zero_grad(set_to_none=True)
    losses = model.atom_loss(
        audit_batch.condition,
        states=audit_batch.state,
        observed_mask=audit_batch.observed_mask,
        location_target=audit_batch.target,
    )
    encoder_parameters = tuple(model.encoder.parameters())
    atom_gradient = _gradient_norm(losses["atom_nll"], encoder_parameters)
    location_gradient = _gradient_norm(
        losses["interior_location_smooth_l1"], encoder_parameters
    )
    gates = _read_json(_resolve_repo_path(config["gate_registry"]["path"]))[
        "G0_shared_EA_learnability"
    ]
    atom_improvement = 1.0 - trained["atom_nll"] / atom_prior_nll
    location_improvement = (
        1.0 - trained["interior_location_smooth_l1"] / location_zero
    )
    checks = {
        "atom_encoder_gradient": atom_gradient
        > float(gates["encoder_gradient_norm_from_atom_nll_min_exclusive"]),
        "location_encoder_gradient": location_gradient
        > float(gates["encoder_gradient_norm_from_location_auxiliary_min_exclusive"]),
        "atom_nll_improvement": atom_improvement
        >= float(gates["validation_atom_nll_improvement_vs_train_prior_min_relative"]),
        "location_improvement": location_improvement
        >= float(gates["validation_location_loss_improvement_vs_zero_predictor_min_relative"]),
    }
    return {
        "schema": "architecture_v1_formal_G0_v1",
        "validation_metrics": trained,
        "baselines": {
            "train_prior_constant_atom_nll": atom_prior_nll,
            "zero_predictor_location_smooth_l1": location_zero,
        },
        "relative_improvement": {
            "atom_nll": atom_improvement,
            "interior_location": location_improvement,
        },
        "encoder_gradient_norm": {
            "atom_nll": atom_gradient,
            "interior_location": location_gradient,
        },
        "upper_atom": {
            "train_exact_one": derived.train_one,
            "validation_exact_one": derived.validation_one,
            "learnable": derived.upper_one_learnable,
            "fixed_probability": derived.upper_one_jeffreys_prior,
            "success_claim_supported": derived.upper_one_learnable,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _stage_schedule(stage_spec: Mapping[str, Any], train_days: int, batch_days: int) -> dict[str, int]:
    updates_per_epoch = int(math.ceil(train_days / batch_days))
    fields = {
        "maximum_epochs": int(stage_spec["maximum_updates"]) // updates_per_epoch,
        "minimum_epochs": int(stage_spec["minimum_updates_before_early_stop"])
        // updates_per_epoch,
        "validate_every_epochs": int(stage_spec["validate_every_updates"])
        // updates_per_epoch,
    }
    if any(value < 1 for value in fields.values()):
        raise ValueError("formal stage schedule collapsed below one epoch")
    if fields["maximum_epochs"] * updates_per_epoch != int(stage_spec["maximum_updates"]):
        raise ValueError("maximum updates are not divisible by updates per epoch")
    if fields["minimum_epochs"] * updates_per_epoch != int(
        stage_spec["minimum_updates_before_early_stop"]
    ):
        raise ValueError("minimum updates are not divisible by updates per epoch")
    if fields["validate_every_epochs"] * updates_per_epoch != int(
        stage_spec["validate_every_updates"]
    ):
        raise ValueError("validation updates are not divisible by updates per epoch")
    return {**fields, "updates_per_epoch": updates_per_epoch}


def _fit_stage(
    trainer: FormalEpochTrainer,
    stage: str,
    bundle: ArchitectureFitDataBundle,
    specification: Mapping[str, Any],
    *,
    batch_days: int,
    resume_requested: bool,
) -> dict[str, Any]:
    root = trainer.output_dir / stage
    completion_path = root / "completion.json"
    best_path = root / "best.pt"
    if completion_path.is_file():
        if not resume_requested:
            raise FileExistsError(
                f"completed formal stage exists; use --resume for idempotent continuation: {root}"
            )
        completion = _read_json(completion_path)
        _verified_sha(completion_path)
        _load_best_checkpoint(best_path, trainer.model, trainer.device)
        return completion
    schedule = _stage_schedule(specification, len(bundle.train), batch_days)
    resume_stage = resume_requested and (root / "latest_safe.pt").is_file()
    gradient_kwargs: dict[str, Any] = {}
    audit_spec = specification.get("gradient_audit")
    if stage == "flow" and isinstance(audit_spec, Mapping):
        formal_config = trainer.resolved_config.get("formal_config", {})
        registry = _read_json(
            _resolve_repo_path(formal_config["gate_registry"]["path"])
        )["G1_R0_validation_stability"]["numerical_integrity"]
        gradient_kwargs = {
            "record_update_preclip_gradient_norms": True,
            "gradient_audit_epoch_start": int(
                audit_spec["formal_audit_epoch_start_1_based_inclusive"]
            ),
            "gradient_audit_epoch_end": None,
            "gradient_audit_gate": {
                "gradient_clip_fraction_max": float(
                    registry["post_warmup_gradient_clip_fraction_max"]
                ),
                "preclip_gradient_norm_p99_to_median_max": float(
                    registry["preclip_gradient_norm_p99_to_median_max"]
                ),
                "preclip_gradient_norm_max_over_all_flow_updates": float(
                    registry["preclip_gradient_norm_max_over_all_flow_updates"]
                ),
                "require_all_finite": True,
            },
        }
    return trainer.fit_stage(
        stage,  # type: ignore[arg-type]
        bundle.train,
        bundle.validation,
        max_epochs=schedule["maximum_epochs"],
        batch_days=batch_days,
        learning_rate=float(specification["learning_rate"]),
        weight_decay=float(specification["weight_decay"]),
        gradient_clip=float(specification["gradient_clip"]),
        validate_every_epochs=schedule["validate_every_epochs"],
        early_stopping_patience=int(
            specification["early_stopping_patience_validations"]
        ),
        early_stopping_minimum_delta=1e-5,
        early_stopping_relative_delta=1e-3,
        minimum_epochs_before_early_stop=schedule["minimum_epochs"],
        resume=resume_stage,
        restore_best=True,
        **gradient_kwargs,
    )


def _atomic_validation_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    if path.exists() or path.with_name(path.name + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite validation scenario archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
    )
    temporary.replace(path)
    digest = file_sha256(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


@torch.no_grad()
def _sample_validation_once(
    model: R0JointRectifiedFlow,
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
    *,
    training_seed: int,
    sampling_seed: int,
    checkpoint_sha256: str,
    config_sha256: str,
    code_sha256: str,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    sampling = config["formal_sampling"]
    members = int(sampling["members"])
    steps = int(sampling["integration_steps"])
    member_chunk = int(sampling["member_chunk"])
    day_batch = 2
    scenario_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    zero_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    total_forward_calls = 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for start in range(0, len(bundle.validation), day_batch):
        stop = min(start + day_batch, len(bundle.validation))
        condition = torch.from_numpy(
            np.ascontiguousarray(bundle.validation.condition[start:stop], dtype=np.float32)
        ).to(device)
        sampled = model.sample(
            condition,
            members=members,
            steps=steps,
            seed=int(sampling_seed + start * 1009),
            method=str(sampling["method"]),
            member_chunk=member_chunk,
        )
        if sampled.per_path_nfe != int(sampling["per_path_flow_nfe"]):
            raise RuntimeError("formal sampling NFE drifted")
        scenario_parts.append(sampled.values.cpu().numpy().astype(np.float32))
        state_parts.append(sampled.states.cpu().numpy().astype(np.int8))
        zero_parts.append(
            sampled.atom_statistics.zero_probability.cpu().numpy().astype(np.float32)
        )
        one_parts.append(
            sampled.atom_statistics.one_probability.cpu().numpy().astype(np.float32)
        )
        total_forward_calls += int(sampled.batched_forward_calls)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    scenarios = np.concatenate(scenario_parts, axis=0)
    states = np.concatenate(state_parts, axis=0)
    zero_probability = np.concatenate(zero_parts, axis=0)
    one_probability = np.concatenate(one_parts, axis=0)
    if not np.isfinite(scenarios).all() or scenarios.min() < 0.0 or scenarios.max() > 1.0:
        raise FloatingPointError("formal validation scenarios are invalid")
    if not np.all(scenarios[states == ZERO_STATE] == 0.0):
        raise RuntimeError("formal validation zero-state boundary violation")
    if not np.all(scenarios[states == ONE_STATE] == 1.0):
        raise RuntimeError("formal validation one-state boundary violation")
    metadata = {
        "schema": VALIDATION_ARCHIVE_SCHEMA,
        "candidate_id": "R0",
        "training_seed": int(training_seed),
        "sampling_seed": int(sampling_seed),
        "split_role": "validation",
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "day_batch": day_batch,
        "members": members,
        "integration_steps": steps,
        "per_path_flow_nfe": int(sampling["per_path_flow_nfe"]),
        "total_batched_flow_forward_calls": total_forward_calls,
        "member_chunk": member_chunk,
        "wall_seconds_synchronized": wall,
        "peak_memory_allocated_bytes": peak,
        "estimand": sampling["estimand"],
        "score_semantics": sampling["score_semantics"],
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    arrays = {
        "scenarios": scenarios,
        "states": states,
        "zero_probability": zero_probability,
        "one_probability": one_probability,
        "observations": bundle.validation.target.astype(np.float32),
        "observed_mask": bundle.validation.observed_mask.astype(bool),
        "raw_missing_mask": bundle.validation.raw_missing_mask.astype(bool),
        "day": bundle.validation.day.astype("datetime64[D]"),
        "zones": bundle.validation.zones.astype(np.int64),
    }
    archive_sha = _atomic_validation_npz(output_path, arrays, metadata)
    per_day = validation_per_day_metrics(
        scenarios,
        bundle.validation.target,
        bundle.validation.observed_mask,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )
    return {
        "metadata": metadata,
        "archive": str(output_path.resolve()),
        "archive_sha256": archive_sha,
        "per_day": per_day,
    }


@torch.no_grad()
def _sampling_semantics_audit(
    model: R0JointRectifiedFlow,
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sampling = config["formal_sampling"]
    condition = torch.from_numpy(
        np.ascontiguousarray(bundle.validation.condition[:1], dtype=np.float32)
    ).to(device)
    kwargs = {
        "members": int(sampling["members"]),
        "steps": int(sampling["integration_steps"]),
        "seed": int(sampling["sampling_seeds"][0]),
        "method": str(sampling["method"]),
    }
    first = model.sample(condition, member_chunk=int(sampling["member_chunk"]), **kwargs)
    replay = model.sample(condition, member_chunk=int(sampling["member_chunk"]), **kwargs)
    alternate_chunk = min(int(sampling["members"]), 2 * int(sampling["member_chunk"]))
    chunked = model.sample(condition, member_chunk=alternate_chunk, **kwargs)
    replay_equal = torch.equal(first.values, replay.values) and torch.equal(
        first.states, replay.states
    )
    chunk_equal = torch.equal(first.values, chunked.values) and torch.equal(
        first.states, chunked.states
    )
    atom_latent_zero = bool(torch.all(first.interior_latent[~first.active_mask] == 0.0))
    return {
        "same_seed_bitwise_replay": replay_equal,
        "member_chunk_bitwise_invariant": chunk_equal,
        "alternate_member_chunk": alternate_chunk,
        "atom_latent_strictly_zero": atom_latent_zero,
        "passed": bool(replay_equal and chunk_equal and atom_latent_zero),
    }


@torch.no_grad()
def _sample_validation_v2_2_chunk_audit(
    model: R0JointRectifiedFlow,
    bundle: ArchitectureFitDataBundle,
    config: Mapping[str, Any],
    *,
    training_seed: int,
    sampling_seed: int,
    checkpoint_sha256: str,
    config_sha256: str,
    code_sha256: str,
    output_dir: Path,
    device: torch.device,
    candidate_id: str = "R0",
) -> dict[str, Any]:
    """Generate the registered archive and chunk-20 audit under common RNG."""

    sampling = config["formal_sampling"]
    audit_spec = sampling["member_chunk_audit"]
    gate_spec = _read_json(_resolve_repo_path(config["gate_registry"]["path"]))[
        "G1_R0_validation_stability"
    ]["member_chunk_numerical_invariance"]
    members = int(sampling["members"])
    steps = int(sampling["integration_steps"])
    primary_chunk = int(sampling["member_chunk"])
    alternate_chunk = int(audit_spec["alternate_member_chunk"])
    atol = float(gate_spec["final_scenario_values_allclose_atol"])
    rtol = float(gate_spec["final_scenario_values_allclose_rtol"])
    day_batch = 2
    primary_parts: list[np.ndarray] = []
    alternate_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    alternate_state_parts: list[np.ndarray] = []
    zero_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    batch_audits: list[Mapping[str, Any]] = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for start in range(0, len(bundle.validation), day_batch):
        stop = min(start + day_batch, len(bundle.validation))
        condition = torch.from_numpy(
            np.ascontiguousarray(
                bundle.validation.condition[start:stop], dtype=np.float32
            )
        ).to(device)
        paired = common_random_chunk_samples(
            model,
            condition,
            members=members,
            steps=steps,
            seed=int(sampling_seed + start * 1009),
            method=str(sampling["method"]),
            primary_member_chunk=primary_chunk,
            alternate_member_chunk=alternate_chunk,
            value_allclose_atol=atol,
            value_allclose_rtol=rtol,
        )
        if paired.primary.per_path_nfe != int(sampling["per_path_flow_nfe"]):
            raise RuntimeError("formal-v2.2 sampling NFE drifted")
        primary_parts.append(paired.primary.values.cpu().numpy().astype(np.float32))
        alternate_parts.append(
            paired.alternate.values.cpu().numpy().astype(np.float32)
        )
        state_parts.append(paired.primary.states.cpu().numpy().astype(np.int8))
        alternate_state_parts.append(
            paired.alternate.states.cpu().numpy().astype(np.int8)
        )
        zero_parts.append(
            paired.primary.atom_statistics.zero_probability.cpu().numpy().astype(
                np.float32
            )
        )
        one_parts.append(
            paired.primary.atom_statistics.one_probability.cpu().numpy().astype(
                np.float32
            )
        )
        batch_audits.append(dict(paired.audit))
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))

    primary = np.concatenate(primary_parts, axis=0)
    alternate = np.concatenate(alternate_parts, axis=0)
    states = np.concatenate(state_parts, axis=0)
    alternate_states = np.concatenate(alternate_state_parts, axis=0)
    zero_probability = np.concatenate(zero_parts, axis=0)
    one_probability = np.concatenate(one_parts, axis=0)
    difference = np.abs(primary.astype(np.float64) - alternate.astype(np.float64))
    flat = difference.reshape(-1)
    distribution = {
        "count": int(flat.size),
        "minimum": float(flat.min()),
        "mean": float(flat.mean()),
        "median": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "maximum": float(flat.max()),
    }
    exact_keys = (
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
    combined_audit = {
        "schema": "architecture_v1_formal_sampling_chunk_audit_panel_v1",
        **{
            key: bool(all(item.get(key) is True for item in batch_audits))
            for key in exact_keys
        },
        "value_allclose_atol": atol,
        "value_allclose_rtol": rtol,
        "value_allclose": bool(
            np.allclose(primary, alternate, atol=atol, rtol=rtol, equal_nan=False)
        ),
        "value_absolute_difference": distribution,
        "day_count": int(len(primary)),
        "batch_count": int(len(batch_audits)),
        "batch_audits": batch_audits,
    }
    observations = bundle.validation.target.astype(np.float32)
    observed_mask = bundle.validation.observed_mask.astype(bool)
    primary_metrics = validation_per_day_metrics(
        primary,
        observations,
        observed_mask,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )
    alternate_metrics = validation_per_day_metrics(
        alternate,
        observations,
        observed_mask,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )
    proper_names = (
        "level_CRPS",
        "ramp_CRPS",
        "normalized_joint_ES",
        "lagged_increment_variogram_score",
    )
    score_deltas = {
        name: abs(
            float(np.asarray(primary_metrics[name], dtype=np.float64).mean())
            - float(np.asarray(alternate_metrics[name], dtype=np.float64).mean())
        )
        for name in proper_names
    }
    core_gate = evaluate_sampling_chunk_gate(
        combined_audit,
        score_deltas=score_deltas,
        gate={
            "value_abs_max": float(
                gate_spec["final_scenario_values_global_max_abs_difference_max"]
            ),
            "value_allclose_atol": atol,
            "value_allclose_rtol": rtol,
            "score_abs_delta_max": float(
                gate_spec["proper_score_each_aggregate_abs_difference_max"]
            ),
        },
    )
    exact_metric_names = (
        "coverage90",
        "zero_Brier",
        "one_Brier",
        "atom_state_Brier",
    )
    exact_metric_checks = {
        name: bool(np.array_equal(primary_metrics[name], alternate_metrics[name]))
        for name in exact_metric_names
    }
    passed = bool(core_gate["passed"] and all(exact_metric_checks.values()))
    metadata = {
        "schema": "architecture_v1_formal_v2_2_chunk_archive_v1",
        "candidate_id": str(candidate_id),
        "training_seed": int(training_seed),
        "sampling_seed": int(sampling_seed),
        "split_role": "validation",
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "members": members,
        "integration_steps": steps,
        "per_path_flow_nfe": int(sampling["per_path_flow_nfe"]),
        "primary_member_chunk": primary_chunk,
        "alternate_member_chunk": alternate_chunk,
        "wall_seconds_synchronized": wall,
        "peak_memory_allocated_bytes": peak,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    common_arrays = {
        "states": states,
        "zero_probability": zero_probability,
        "one_probability": one_probability,
        "observations": observations,
        "observed_mask": observed_mask,
        "raw_missing_mask": bundle.validation.raw_missing_mask.astype(bool),
        "day": bundle.validation.day.astype("datetime64[D]"),
        "zones": bundle.validation.zones.astype(np.int64),
    }
    safe_candidate = str(candidate_id).replace("/", "_").replace(" ", "_")
    primary_path = output_dir / f"{safe_candidate}_seed{training_seed}_sampling{sampling_seed}_chunk10.npz"
    alternate_path = output_dir / f"{safe_candidate}_seed{training_seed}_sampling{sampling_seed}_chunk20.npz"
    primary_sha = _atomic_validation_npz(
        primary_path, {"scenarios": primary, **common_arrays}, {**metadata, "role": "registered"}
    )
    alternate_sha = _atomic_validation_npz(
        alternate_path,
        {
            "scenarios": alternate,
            **{**common_arrays, "states": alternate_states},
        },
        {**metadata, "role": "numerical_audit"},
    )
    return {
        "schema": "architecture_v1_formal_v2_2_chunk_pair_result_v1",
        "training_seed": int(training_seed),
        "sampling_seed": int(sampling_seed),
        "primary_archive": str(primary_path.resolve()),
        "primary_archive_sha256": primary_sha,
        "alternate_archive": str(alternate_path.resolve()),
        "alternate_archive_sha256": alternate_sha,
        "audit": combined_audit,
        "proper_score_absolute_deltas": score_deltas,
        "exact_metric_checks": exact_metric_checks,
        "gate": core_gate,
        "passed": passed,
        "per_day": primary_metrics,
    }


def _post_warmup_clip_fraction(history_path: str | Path) -> float:
    payload = _read_json(history_path)
    records = payload["records"]
    warmup = max(1, int(math.floor(0.1 * len(records))))
    selected = records[warmup:]
    clipped = sum(int(record.get("gradient_clipped_updates", 0)) for record in selected)
    updates = sum(int(record.get("updates", 0)) for record in selected)
    return float(clipped / updates) if updates else 0.0


def _per_seed_g1(
    flow_completion: Mapping[str, Any],
    zero_velocity_loss: float,
    validation: Mapping[str, Any],
    semantics: Mapping[str, Any],
    shared_hash_before: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _read_json(_resolve_repo_path(config["gate_registry"]["path"]))[
        "G1_R0_validation_stability"
    ]
    quality = gate["catastrophic_quality_bounds_each_seed"]
    aggregate = validation["aggregate"]
    gradient_audit = flow_completion.get("preclip_gradient_audit")
    if isinstance(gradient_audit, Mapping):
        clip_fraction = float(gradient_audit["clip_fraction"])
        gradient_passed = bool(gradient_audit["passed"])
    else:
        clip_fraction = _post_warmup_clip_fraction(flow_completion["history"])
        gradient_passed = clip_fraction <= float(
            gate["numerical_integrity"][
                "post_warmup_gradient_clip_fraction_max"
            ]
        )
    checks = {
        "flow_improvement": float(flow_completion["best_validation_loss"])
        / zero_velocity_loss
        <= float(gate["numerical_integrity"]["best_validation_loss_relative_to_zero_velocity_max"]),
        "shared_EA_unchanged": flow_completion["shared_EA_state_sha256"]
        == shared_hash_before,
        "sampling_semantics": bool(semantics["passed"]),
        "gradient_clip_fraction": gradient_passed,
        "level_CRPS": aggregate["level_CRPS"] <= float(quality["level_CRPS_max"]),
        "ramp_CRPS": aggregate["ramp_CRPS"] <= float(quality["ramp_CRPS_max"]),
        "normalized_joint_ES": aggregate["normalized_joint_ES"]
        <= float(quality["normalized_joint_ES_max"]),
        "coverage90_lower": aggregate["coverage90"] >= float(quality["coverage90_min"]),
        "coverage90_upper": aggregate["coverage90"] <= float(quality["coverage90_max"]),
        "width90_lower": aggregate["width90"] >= float(quality["width90_min"]),
        "width90_upper": aggregate["width90"] <= float(quality["width90_max"]),
    }
    return {
        "zero_velocity_fixed_bank_loss": zero_velocity_loss,
        "best_fixed_bank_loss": float(flow_completion["best_validation_loss"]),
        "relative_flow_loss_ratio": float(flow_completion["best_validation_loss"])
        / zero_velocity_loss,
        "post_warmup_gradient_clip_fraction": clip_fraction,
        "preclip_gradient_audit": gradient_audit,
        "validation_aggregate": validation["aggregate"],
        "sampling_semantics": semantics,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _dispersion_gate(seed_reports: Mapping[int, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = _read_json(_resolve_repo_path(config["gate_registry"]["path"]))[
        "G1_R0_validation_stability"
    ]["three_seed_dispersion"]
    values = {
        metric: np.asarray(
            [seed_reports[seed]["validation_aggregate"][metric] for seed in sorted(seed_reports)],
            dtype=np.float64,
        )
        for metric in (
            "level_CRPS",
            "ramp_CRPS",
            "normalized_joint_ES",
            "coverage90",
            "zero_Brier",
            "atom_state_Brier",
        )
    }
    spreads = {metric: float(array.max() - array.min()) for metric, array in values.items()}
    joint_relative = spreads["normalized_joint_ES"] / float(
        np.median(values["normalized_joint_ES"])
    )
    checks = {
        "level_CRPS": spreads["level_CRPS"] <= float(gate["level_CRPS_max_minus_min_max"]),
        "ramp_CRPS": spreads["ramp_CRPS"] <= float(gate["ramp_CRPS_max_minus_min_max"]),
        "normalized_joint_ES": joint_relative
        <= float(gate["normalized_joint_ES_relative_max_minus_min_max"]),
        "coverage90": spreads["coverage90"] <= float(gate["coverage90_max_minus_min_max"]),
        "zero_Brier": spreads["zero_Brier"] <= float(gate["zero_Brier_max_minus_min_max"]),
        "atom_state_Brier": spreads["atom_state_Brier"]
        <= float(gate["atom_state_Brier_max_minus_min_max"]),
    }
    return {
        "values_by_metric_seed_order_0_1_2": {
            metric: array.tolist() for metric, array in values.items()
        },
        "absolute_spreads": spreads,
        "normalized_joint_ES_relative_spread": joint_relative,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def execute_training(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    device = torch.device("cuda")
    runtime = _configure_runtime(device)
    output_root = _output_root(config)
    freeze_path = output_root / "r0_training.freeze.json"
    if freeze_path.exists():
        raise RuntimeError("R0 formal training is already frozen")
    p0 = _load_v2_2_p0(config_path, config)
    bundle = build_architecture_v1_fit_data(config_path=config_path)
    preflight_path, preflight, preflight_sha = _load_preflight(
        config_path, config, bundle
    )
    batch_days = int(preflight["resolved_batch_days"])
    derived = _derive_train_quantities(bundle, config)
    if asdict(derived) != preflight["train_derived"]:
        raise RuntimeError("train-derived quantities changed after preflight")
    code = _code_manifest(config_path, config)
    bank = _make_validation_bank(
        bundle, config, evaluation_batch_days=batch_days
    )
    bank_artifact = _persist_validation_bank(bank, output_root)
    derived_path = output_root / "train_derived.json"
    if not derived_path.exists():
        _atomic_json(
            derived_path,
            {
                "schema": DERIVED_SCHEMA,
                "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
                "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
                "values": asdict(derived),
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
            },
        )
    elif _read_json(derived_path).get("values") != asdict(derived):
        raise RuntimeError("persisted train-derived quantities changed")

    model_kwargs = _model_kwargs(config, derived)
    training = config["formal_training"]
    seeds = tuple(int(seed) for seed in training["model_seeds"])
    if seeds != (0, 1, 2):
        raise RuntimeError("formal R0 seeds must remain exactly [0,1,2]")
    if not resume and (output_root / "runs").exists():
        if any((output_root / "runs").iterdir()):
            raise FileExistsError("formal run artifacts exist; use --resume")
    common_identity = {
        "formal_config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "preflight_file_sha256": preflight_sha,
        "validation_bank_file_sha256": bank_artifact["file_sha256"],
    }
    if p0 is not None:
        common_identity["formal_v2_2_P0_result_sha256"] = p0[2]
    if "formal_amendment" in config:
        common_identity["formal_amendment_file_sha256"] = config[
            "formal_amendment"
        ]["file_sha256"]
    resolved = {
        "schema": RUNNER_SCHEMA,
        "formal_config": config,
        "resolved_batch_days": batch_days,
        "train_derived": asdict(derived),
        "preflight": str(preflight_path.resolve()),
        "validation_bank": bank.manifest,
        "runtime": runtime,
    }
    if p0 is not None:
        resolved["formal_v2_2_P0"] = {
            "path": str(p0[0].resolve()),
            "file_sha256": p0[2],
            "status": p0[1]["status"],
        }
    completion_files: dict[int, Path] = {}
    seed_g1: dict[int, dict[str, Any]] = {}
    seed_outputs: dict[int, Any] = {}
    offsets = training["seed_offsets"]
    for seed in seeds:
        run_dir = output_root / "runs" / f"R0_seed{seed}"
        torch.manual_seed(int(offsets["ea_init"]) + seed)
        torch.cuda.manual_seed_all(int(offsets["ea_init"]) + seed)
        ea_model = R0JointRectifiedFlow(**model_kwargs)
        ea_trainer = FormalEpochTrainer(
            ea_model,
            run_dir,
            resolved_config=resolved,
            protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
            data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
            code_sha256=code["code_sha256"],
            validation_bank=bank,
            training_seed=seed,
            shuffle_seed=int(offsets["ea_shuffle"]) + seed,
            path_seed=int(offsets["ea_shuffle"]) + seed,
            run_id=f"architecture_v1_formal_R0_seed{seed}",
            device=device,
            extra_identity_hashes=common_identity,
            training_freeze_path=freeze_path,
        )
        ea_completion = _fit_stage(
            ea_trainer,
            "atom",
            bundle,
            training["ea_stage"],
            batch_days=batch_days,
            resume_requested=resume,
        )
        g0 = _g0_evaluate(ea_model, bundle, bank, derived, config, device)
        g0_path = run_dir / "G0_shared_EA.json"
        if not g0_path.exists():
            _atomic_json(g0_path, g0)
        elif _read_json(g0_path) != g0:
            raise RuntimeError(f"seed {seed} persisted G0 result changed")
        if not g0["passed"]:
            raise RuntimeError(
                f"seed {seed} failed preregistered G0; flow training is forbidden"
            )
        shared_hash = shared_ea_state_sha256(ea_model)

        torch.manual_seed(int(offsets["flow_init"]) + seed)
        torch.cuda.manual_seed_all(int(offsets["flow_init"]) + seed)
        flow_model = R0JointRectifiedFlow(**model_kwargs).to(device)
        flow_model.load_shared_from(ea_model, freeze=True)
        if shared_ea_state_sha256(flow_model) != shared_hash:
            raise RuntimeError("shared E/A changed while constructing the flow model")
        zero_velocity = bank.evaluate(
            flow_model, bundle.validation, stage="flow", device=device
        )["loss"]
        flow_trainer = FormalEpochTrainer(
            flow_model,
            run_dir,
            resolved_config=resolved,
            protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
            data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
            code_sha256=code["code_sha256"],
            validation_bank=bank,
            training_seed=seed,
            shuffle_seed=int(offsets["flow_shuffle"]) + seed,
            path_seed=int(offsets["flow_training_path"]) + seed,
            run_id=f"architecture_v1_formal_R0_seed{seed}",
            device=device,
            extra_identity_hashes=common_identity,
            training_freeze_path=freeze_path,
        )
        flow_completion = _fit_stage(
            flow_trainer,
            "flow",
            bundle,
            training["flow_stage"],
            batch_days=batch_days,
            resume_requested=resume,
        )
        if shared_ea_state_sha256(flow_model) != shared_hash:
            raise RuntimeError("shared E/A hash changed during flow fitting")
        checkpoint_sha = str(flow_completion["best_checkpoint_sha256"])
        replicate_metrics: list[Mapping[str, np.ndarray]] = []
        sampling_records: list[dict[str, Any]] = []
        v2_2_semantics: list[dict[str, Any]] = []
        for sampling_seed in config["formal_sampling"]["sampling_seeds"]:
            if config.get("protocol_revision") in {
                "architecture_v1_formal_v2_2_20260815",
                "architecture_v1_formal_v2_2_1_20260818",
            }:
                audit_dir = run_dir / "validation_chunk_audit"
                pair_path = audit_dir / f"R0_seed{seed}_sampling{sampling_seed}_pair.json"
                if pair_path.exists():
                    if not resume:
                        raise FileExistsError(pair_path)
                    _verified_sha(pair_path)
                    record = _read_json(pair_path)
                    for archive_key in ("primary_archive", "alternate_archive"):
                        _verified_sha(record[archive_key])
                    record["per_day"] = {
                        name: np.asarray(values, dtype=np.float64)
                        for name, values in record["per_day"].items()
                    }
                else:
                    record = _sample_validation_v2_2_chunk_audit(
                        flow_model,
                        bundle,
                        config,
                        training_seed=seed,
                        sampling_seed=int(sampling_seed),
                        checkpoint_sha256=checkpoint_sha,
                        config_sha256=flow_trainer.config_sha256,
                        code_sha256=code["code_sha256"],
                        output_dir=audit_dir,
                        device=device,
                    )
                    _atomic_json(pair_path, record)
                replicate_metrics.append(record["per_day"])
                sampling_records.append(
                    {
                        "archive": record["primary_archive"],
                        "archive_sha256": record["primary_archive_sha256"],
                        "alternate_archive": record["alternate_archive"],
                        "alternate_archive_sha256": record[
                            "alternate_archive_sha256"
                        ],
                        "sampling_seed": int(sampling_seed),
                    }
                )
                v2_2_semantics.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"per_day"}
                    }
                )
                continue
            archive = run_dir / "validation" / f"R0_seed{seed}_sampling{sampling_seed}.npz"
            if archive.exists():
                if not resume:
                    raise FileExistsError(archive)
                _verified_sha(archive)
                with np.load(archive, allow_pickle=False) as stored:
                    per_day = validation_per_day_metrics(
                        stored["scenarios"],
                        stored["observations"],
                        stored["observed_mask"],
                        zero_probability=stored["zero_probability"],
                        one_probability=stored["one_probability"],
                    )
                    metadata = json.loads(str(stored["metadata"].item()))
                record = {
                    "metadata": metadata,
                    "archive": str(archive.resolve()),
                    "archive_sha256": file_sha256(archive),
                    "per_day": per_day,
                }
            else:
                record = _sample_validation_once(
                    flow_model,
                    bundle,
                    config,
                    training_seed=seed,
                    sampling_seed=int(sampling_seed),
                    checkpoint_sha256=checkpoint_sha,
                    config_sha256=flow_trainer.config_sha256,
                    code_sha256=code["code_sha256"],
                    output_path=archive,
                    device=device,
                )
            replicate_metrics.append(record["per_day"])
            sampling_records.append(
                {
                    "archive": record["archive"],
                    "archive_sha256": record["archive_sha256"],
                    "metadata": record["metadata"],
                }
            )
        validation = aggregate_sampling_replicates(replicate_metrics)
        if v2_2_semantics:
            semantics = {
                "schema": "architecture_v1_formal_v2_2_sampling_semantics_v1",
                "required_sampling_seed_day_combinations": int(
                    len(config["formal_sampling"]["sampling_seeds"])
                    * len(bundle.validation)
                ),
                "completed_sampling_seed_day_combinations": int(
                    len(v2_2_semantics) * len(bundle.validation)
                ),
                "pair_results": v2_2_semantics,
                "passed": bool(
                    len(v2_2_semantics)
                    == len(config["formal_sampling"]["sampling_seeds"])
                    and all(item["passed"] for item in v2_2_semantics)
                ),
            }
        else:
            semantics = _sampling_semantics_audit(flow_model, bundle, config, device)
        g1 = _per_seed_g1(
            flow_completion,
            float(zero_velocity),
            validation,
            semantics,
            shared_hash,
            config,
        )
        g1_payload = {
            "schema": "architecture_v1_formal_G1_seed_v1",
            "training_seed": seed,
            "G0": g0,
            "G1": g1,
            "sampling_archives": sampling_records,
            "per_day": validation["per_day"],
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
        }
        g1_path = run_dir / "G1_validation.json"
        if not g1_path.exists():
            _atomic_json(g1_path, g1_payload)
        seed_g1[seed] = g1
        completion_files[seed] = Path(flow_completion["completion"])
        seed_outputs[seed] = {
            "atom_completion": ea_completion["completion"],
            "flow_completion": flow_completion["completion"],
            "G0": str(g0_path.resolve()),
            "G1": str(g1_path.resolve()),
        }

    dispersion = _dispersion_gate(seed_g1, config)
    all_seed_passed = all(report["passed"] for report in seed_g1.values())
    overall_passed = bool(all_seed_passed and dispersion["passed"])
    selection_plan_sha = canonical_sha256(config["selection_lock"])
    freeze = write_training_freeze(
        freeze_path,
        completion_files=completion_files,
        expected_seeds=seeds,
        candidate_id="R0",
        required_stage="flow",
        gate_config_sha256=config["gate_registry"]["file_sha256"],
        selection_plan_sha256=selection_plan_sha,
        protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
        data_bundle_sha256=bundle.manifest["fit_data_bundle_sha256"],
        code_sha256=code["code_sha256"],
    )
    final = {
        "schema": "architecture_v1_formal_R0_result_v1",
        "status": "G1_GO" if overall_passed else "G1_NO_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "R0_training_frozen": freeze,
        "per_seed": seed_outputs,
        "per_seed_gate": seed_g1,
        "three_seed_dispersion": dispersion,
        "overall_passed": overall_passed,
        "next_action": (
            "implement_T0_without_opening_selection"
            if overall_passed
            else "stop_and_audit_R0_without_opening_calibration_or_selection"
        ),
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
    }
    result_path = output_root / "R0_FORMAL_RESULT.json"
    result_sha = _atomic_json(result_path, final)
    return {**final, "path": str(result_path.resolve()), "file_sha256": result_sha}


def _dry_run(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = build_architecture_protocol(config_path, smoke=False)
    code = _code_manifest(config_path, config)
    return {
        "schema": RUNNER_SCHEMA,
        "mode": "dry_run_no_files_created_no_targets_loaded",
        "config": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "protocol_sha256": protocol.manifest["protocol_sha256"],
        "role_counts": protocol.manifest["full_date_counts"],
        "target_file_opened": protocol.manifest["predictor_calendar"][
            "target_file_opened"
        ],
        "selection_state": config["selection_lock"]["state"],
        "code_sha256": code["code_sha256"],
        "allowed_next_flags": ["--execute-preflight", "--execute-training"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=None,
        help="frozen engineering-only amendment bound to the base config hash",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute-preflight", action="store_true")
    action.add_argument("--execute-training", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exact-identity epoch checkpoints; never changes seeds/config",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path, config, _ = _load_config(args.config, args.amendment)
    if args.resume and not args.execute_training:
        raise ValueError("--resume is valid only with --execute-training")
    if args.execute_preflight:
        result = execute_preflight(config_path, config)
    elif args.execute_training:
        result = execute_training(config_path, config, resume=bool(args.resume))
    else:
        result = _dry_run(config_path, config)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
