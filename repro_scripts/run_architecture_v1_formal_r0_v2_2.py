#!/usr/bin/env python3
"""Formal-v2.2 R0 engineering gate and retained-training entry point.

The default is a predictor-only dry run.  ``--execute-p0`` is the only
mutating action currently exposed: it materializes train targets only, runs
three fresh E/A+flow pilots, persists scalar gradient traces, and discards all
weights.  Retained training remains deliberately unavailable until P0 has a
verified PASS artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture_v1.data import (  # noqa: E402
    INTERIOR_STATE,
    ONE_STATE,
    ZERO_STATE,
    ArchitectureTrainDataBundle,
    build_architecture_v1_train_data,
)
from architecture_v1.formal_training import (  # noqa: E402
    run_discarded_train_only_stage,
)
from architecture_v1.model import R0JointRectifiedFlow  # noqa: E402
from architecture_v1.protocol import (  # noqa: E402
    build_architecture_protocol,
    canonical_sha256,
    file_sha256,
)
from architecture_v1.training import ArchitectureBatch  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "repro_configs" / "architecture_v1_formal_v2_2.json"
RUNNER_SCHEMA = "architecture_v1_formal_r0_v2_2_runner_v1"
P0_SCHEMA = "architecture_v1_formal_v2_2_P0_result_v1"


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite formal-v2.2 artifact: {path}")
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
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    config = _read_json(source)
    if config.get("protocol_revision") not in {
        "architecture_v1_formal_v2_2_20260815",
        "architecture_v1_formal_v2_2_1_20260818",
    }:
        raise ValueError("runner accepts only the frozen formal-v2.2 family revisions")
    if config.get("selection_lock", {}).get("state") != "sealed":
        raise RuntimeError("selection must remain sealed")
    contract = config.get("experiment_contract", {})
    if contract.get("calibration_status") != "sealed":
        raise RuntimeError("calibration must remain sealed")
    if contract.get("formal_v2_1_status") != "G1_NO_GO_permanent":
        raise RuntimeError("formal-v2.1 No-Go lineage was altered")
    gate = config.get("gate_registry", {})
    gate_path = _resolve_repo_path(gate["path"])
    if file_sha256(gate_path) != gate.get("file_sha256"):
        raise RuntimeError("formal-v2.2 gate hash mismatch")
    if _read_json(gate_path).get("schema") != gate.get("schema"):
        raise ValueError("unexpected formal-v2.2 gate schema")
    lineage = config.get("lineage", {})
    for path_key, hash_key in (
        ("base_config", "base_config_file_sha256"),
        ("formal_v2_1_amendment", "formal_v2_1_amendment_file_sha256"),
        ("formal_v2_1_result", "formal_v2_1_result_file_sha256"),
    ):
        linked = _resolve_repo_path(lineage[path_key])
        if file_sha256(linked) != lineage[hash_key]:
            raise RuntimeError(f"formal-v2.2 lineage hash mismatch: {path_key}")
    return source, config


def _code_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "architecture_v1" / name
        for name in (
            "__init__.py",
            "atom.py",
            "data.py",
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
    records = [
        {
            "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(set(paths))
    ]
    core = {"schema": "architecture_v1_formal_v2_2_code_manifest_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


@dataclass(frozen=True)
class P0TrainDerived:
    train_observed: int
    train_zero: int
    train_interior: int
    train_one: int
    interior_logit_mean: float
    interior_logit_std: float
    upper_one_jeffreys_prior: float
    zero_conditional_jeffreys_prior: float


def _derive_train_quantities(
    bundle: ArchitectureTrainDataBundle,
    config: Mapping[str, Any],
) -> P0TrainDerived:
    train = bundle.train
    observed = train.observed_mask
    zero = int((observed & (train.state == ZERO_STATE)).sum())
    interior = int((observed & (train.state == INTERIOR_STATE)).sum())
    one = int((observed & (train.state == ONE_STATE)).sum())
    observed_count = int(observed.sum())
    minimum = int(
        config["shared_ea"]["upper_atom"]["train_positive_minimum_for_learning"]
    )
    if one >= minimum:
        raise RuntimeError(
            "train-only P0 cannot choose a contextual upper-atom branch when "
            "train support alone reaches the learning gate"
        )
    epsilon = float(
        config["shared_ea"]["interior_auxiliary"]["logit_clip_epsilon"]
    )
    values = np.clip(
        train.target[observed & (train.state == INTERIOR_STATE)].astype(np.float64),
        epsilon,
        1.0 - epsilon,
    )
    logits = np.log(values / (1.0 - values))
    mean = float(logits.mean())
    std = float(logits.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise FloatingPointError("train-only interior logit normalizer is invalid")
    return P0TrainDerived(
        train_observed=observed_count,
        train_zero=zero,
        train_interior=interior,
        train_one=one,
        interior_logit_mean=mean,
        interior_logit_std=std,
        upper_one_jeffreys_prior=(one + 0.5) / (observed_count + 1.0),
        zero_conditional_jeffreys_prior=(zero + 0.5) / (zero + interior + 1.0),
    )


def _model_kwargs(
    config: Mapping[str, Any], derived: P0TrainDerived
) -> dict[str, Any]:
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


def _configure_cuda() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal-v2.2 P0 requires a visible CUDA device")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return device, {
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


def _p0_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = _read_json(_resolve_repo_path(config["gate_registry"]["path"]))
    frozen = registry["P0_train_only_gradient_scale_preflight"]
    return {
        "gradient_clip_fraction_max": float(
            frozen["post_warmup_gradient_clip_fraction_max"]
        ),
        "preclip_gradient_norm_p99_to_median_max": float(
            frozen["preclip_gradient_norm_p99_to_median_max"]
        ),
        "preclip_gradient_norm_max_over_all_680_flow_updates": float(
            frozen["preclip_gradient_norm_max_over_all_680_flow_updates"]
        ),
        "minimum_updates": 170,
        "require_all_finite": True,
    }


def _stage_args(specification: Mapping[str, Any]) -> dict[str, float]:
    return {
        "learning_rate": float(specification["learning_rate"]),
        "weight_decay": float(specification["weight_decay"]),
        "gradient_clip": float(specification["gradient_clip"]),
    }


def _run_p0_seed(
    seed: int,
    bundle: ArchitectureTrainDataBundle,
    config: Mapping[str, Any],
    derived: P0TrainDerived,
    device: torch.device,
) -> dict[str, Any]:
    p0 = config["gradient_scale_preflight"]
    offsets = p0["seed_offsets"]
    formal = config["formal_training"]
    train_batch = ArchitectureBatch.from_split(
        bundle.train, np.arange(len(bundle.train), dtype=np.int64)
    )
    kwargs = _model_kwargs(config, derived)

    torch.manual_seed(int(offsets["ea_init"]) + seed)
    torch.cuda.manual_seed_all(int(offsets["ea_init"]) + seed)
    ea_model = R0JointRectifiedFlow(**kwargs).to(device)
    ea = run_discarded_train_only_stage(
        ea_model,
        train_batch,
        data_role="train",
        stage="atom",
        epochs=int(p0["ea_warmup"]["epochs"]),
        batch_days=int(p0["batch_days"]),
        training_seed=seed,
        shuffle_seed=int(offsets["ea_shuffle"]) + seed,
        path_seed=int(offsets["ea_shuffle"]) + seed,
        device=device,
        **_stage_args(formal["ea_stage"]),
    )

    torch.manual_seed(int(offsets["flow_init"]) + seed)
    torch.cuda.manual_seed_all(int(offsets["flow_init"]) + seed)
    flow_model = R0JointRectifiedFlow(**kwargs).to(device)
    flow_model.load_shared_from(ea_model, freeze=True)
    flow = run_discarded_train_only_stage(
        flow_model,
        train_batch,
        data_role="train",
        stage="flow",
        epochs=int(p0["flow"]["epochs"]),
        batch_days=int(p0["batch_days"]),
        training_seed=seed,
        shuffle_seed=int(offsets["flow_shuffle"]) + seed,
        path_seed=int(offsets["flow_training_path"]) + seed,
        device=device,
        audit_epoch_start=int(p0["flow"]["audit_epoch_window_1_based_inclusive"][0]),
        audit_epoch_end=int(p0["flow"]["audit_epoch_window_1_based_inclusive"][1]),
        gate=_p0_gate(config),
        **_stage_args(formal["flow_stage"]),
    )
    del flow_model, ea_model
    torch.cuda.empty_cache()
    return {
        "schema": "architecture_v1_formal_v2_2_P0_seed_v1",
        "training_seed": int(seed),
        "EA": ea,
        "flow": flow,
        "passed": bool(flow["gradient_audit"]["passed"]),
        "weights_retained": False,
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }


def execute_p0(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve_repo_path(config["output_root"])
    root = output_root / "P0_gradient_scale_preflight"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "formal-v2.2 output root is non-empty; P0 never resumes or overwrites"
        )
    device, runtime = _configure_cuda()
    bundle = build_architecture_v1_train_data(config_path=config_path)
    if bundle.materialized_roles != ("train",):
        raise RuntimeError("P0 materialized a forbidden target role")
    access = bundle.manifest["formal_train_only_target_access"]
    if access["forbidden_target_arrays_materialized"] is not False:
        raise RuntimeError("P0 target-access manifest is invalid")
    derived = _derive_train_quantities(bundle, config)
    code = _code_manifest(config_path, config)
    identity = {
        "schema": "architecture_v1_formal_v2_2_P0_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "train_only_data_bundle_sha256": bundle.manifest[
            "train_only_data_bundle_sha256"
        ],
        "code_manifest": code,
        "runtime": runtime,
        "train_derived": asdict(derived),
        "materialized_target_roles": ["train"],
        "validation_bank_constructed": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    try:
        root.mkdir(parents=True, exist_ok=False)
        identity_sha = _atomic_json(root / "identity.json", identity)
        reports: dict[str, Any] = {}
        seeds = tuple(int(value) for value in config["gradient_scale_preflight"]["model_seeds"])
        if seeds != (0, 1, 2):
            raise RuntimeError("P0 seeds must remain exactly [0,1,2]")
        for seed in seeds:
            report = _run_p0_seed(seed, bundle, config, derived, device)
            path = root / f"P0_seed{seed}.json"
            digest = _atomic_json(path, report)
            reports[str(seed)] = {
                "path": str(path.resolve()),
                "file_sha256": digest,
                "passed": bool(report["passed"]),
                "gradient_audit": report["flow"]["gradient_audit"],
            }
            if not report["passed"]:
                break
        complete = len(reports) == 3
        passed = complete and all(value["passed"] for value in reports.values())
        result = {
            "schema": P0_SCHEMA,
            "status": "P0_GO" if passed else "P0_NO_GO",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": str((root / "identity.json").resolve()),
            "identity_file_sha256": identity_sha,
            "per_seed": reports,
            "completed_all_three_seeds": complete,
            "passed": passed,
            "all_weights_discarded": True,
            "retained_training_authorized": passed,
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
            "validation_target_accessed": False,
            "r_seen_target_accessed": False,
            "final_target_accessed": False,
            "next_action": (
                "implement_and_run_retained_v2_2_from_seed0"
                if passed
                else "stop_and_freeze_v2_2_No_Go_without_moving_thresholds"
            ),
        }
        result_sha = _atomic_json(root / "P0_RESULT.json", result)
        return {**result, "file_sha256": result_sha}
    except Exception as error:
        failure = root / "P0_FAILURE.json"
        if root.exists() and not failure.exists():
            _atomic_json(
                failure,
                {
                    "schema": "architecture_v1_formal_v2_2_P0_failure_v1",
                    "status": "failed_no_retry_within_same_output_root",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "weights_retained": False,
                    "selection_target_accessed": False,
                    "calibration_target_accessed": False,
                    "validation_target_accessed": False,
                },
            )
        raise


def dry_run(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = build_architecture_protocol(config_path, smoke=False)
    return {
        "schema": RUNNER_SCHEMA,
        "mode": "dry_run_predictor_only_no_files_created_no_targets_loaded",
        "config": str(config_path.resolve()),
        "config_file_sha256": file_sha256(config_path),
        "gate_file_sha256": config["gate_registry"]["file_sha256"],
        "protocol_sha256": protocol.manifest["protocol_sha256"],
        "role_counts": protocol.manifest["full_date_counts"],
        "target_file_opened": protocol.manifest["predictor_calendar"][
            "target_file_opened"
        ],
        "P0_target_roles_if_executed": ["train"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "retained_training_implemented": False,
        "next_flag": "--execute-p0",
        "code_sha256": _code_manifest(config_path, config)["code_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-p0", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path, config = _load_config(args.config)
    result = execute_p0(config_path, config) if args.execute_p0 else dry_run(config_path, config)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
