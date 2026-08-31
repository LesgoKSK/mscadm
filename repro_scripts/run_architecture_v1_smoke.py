"""Execute the bounded architecture-v1 CPU smoke test.

This entry point is deliberately incapable of launching a formal experiment.
Without ``--execute-smoke`` it performs a read-only predictor-calendar
preflight and prints the resolved plan.  The execution path always forces the
registered smoke view, trains only R0/T0 for two updates per registered stage,
and scores only the two-day ``selection`` view.  ``r_seen`` and ``final`` are
hard-denied as modelling roles.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.data import (  # noqa: E402
    ArchitectureDataBundle,
    ArchitectureSplitData,
    build_architecture_v1_data,
)
from architecture_v1.model import (  # noqa: E402
    JointRectifiedFlowBase,
    R0JointRectifiedFlow,
    T0MemorylessRectifiedFlow,
)
from architecture_v1.protocol import (  # noqa: E402
    ArchitectureProtocol,
    DEFAULT_CONFIG_PATH,
    build_architecture_protocol,
    file_sha256,
    write_protocol_manifest,
)
from architecture_v1.training import (  # noqa: E402
    ArchitectureBatch,
    ArchitectureTrainer,
    load_checkpoint,
    shared_ea_state_sha256,
)


RUNNER_SCHEMA = "architecture_v1_smoke_runner_v1"
COMPLETION_SCHEMA = "architecture_v1_smoke_completion_v1"
FORBIDDEN_MODEL_ROLES = frozenset({"r_seen", "final"})
ALLOWED_MODEL_ROLES = frozenset({"train", "validation", "selection"})
EXPECTED_SMOKE_COUNTS = {
    "train": 8,
    "validation": 2,
    "calibration": 2,
    "selection": 2,
    "r_seen": 2,
    "final": 0,
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[D]").astype(str).tolist()
        return _canonical(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize smoke-runner value {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _canonical(value),
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
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _require_allowed_roles(roles: Sequence[str]) -> tuple[str, ...]:
    """Fail closed if a caller tries to model R-SEEN/final or another role."""

    selected = tuple(str(role) for role in roles)
    denied = sorted(set(selected).intersection(FORBIDDEN_MODEL_ROLES))
    if denied:
        raise RuntimeError(
            "architecture-v1 smoke runner refuses R-SEEN/final fitting, "
            f"generation, or scoring: {denied}"
        )
    unknown = sorted(set(selected).difference(ALLOWED_MODEL_ROLES))
    if unknown:
        raise RuntimeError(f"unregistered smoke modelling roles: {unknown}")
    return selected


def resolved_smoke_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every compute-affecting smoke value and reject larger plans."""

    smoke = source_config.get("smoke")
    if not isinstance(smoke, Mapping):
        raise RuntimeError("configuration has no registered smoke block")
    declared_counts = smoke.get("max_days_per_role")
    if declared_counts != {
        role: count for role, count in EXPECTED_SMOKE_COUNTS.items() if role != "final"
    }:
        raise RuntimeError(
            "smoke date caps drifted; formal or enlarged configurations are refused"
        )
    if list(smoke.get("model_seeds", [])) != [0]:
        raise RuntimeError("smoke runner accepts only the registered seed [0]")

    layout = source_config.get("layout")
    if layout != {
        "sample": "calendar_day",
        "zones": 10,
        "hours": 24,
        "condition_dim": 20,
    }:
        raise RuntimeError("formal/non-joint layouts are refused by the smoke runner")
    basis = source_config.get("model_common_basis")
    if not isinstance(basis, Mapping):
        raise RuntimeError("configuration has no model_common_basis")
    model_overrides = smoke.get("model_overrides")
    expected_model_overrides = {
        "encoder_dim": 32,
        "encoder_depth": 1,
        "flow_dim": 32,
        "flow_depth": 1,
        "heads": 4,
        "ff_multiplier": 2,
        "atom_hidden_dim": 32,
    }
    if model_overrides != expected_model_overrides:
        raise RuntimeError(
            "smoke model overrides drifted; formal or enlarged models are refused"
        )
    training = smoke.get("training")
    expected_training = {
        "batch_size": 2,
        "updates_per_stage": 2,
        "atom_learning_rate": 0.001,
        "flow_learning_rate": 0.001,
        "gradient_clip": 1.0,
    }
    if training != expected_training:
        raise RuntimeError(
            "smoke training block drifted; formal or enlarged schedules are refused"
        )
    sampling = smoke.get("sampling")
    expected_sampling = {
        "members": 4,
        "method": "heun",
        "integration_steps": 2,
        "per_path_flow_nfe": 3,
        "member_chunk": 2,
        "shared_sampling_seed": 20000,
        "common_random_numbers_group": "R0_T0_selection_smoke_seed20000",
        "sampling_plan_id": "selection_M4_heun2_crn_v1",
        "allocation_semantics": "balanced_shared_priority_control",
        "estimand": "finite_dependent_scenario_set",
        "score_semantics": "empirical_v_stat",
    }
    if sampling != expected_sampling:
        raise RuntimeError(
            "smoke sampling block drifted; formal or enlarged sampling is refused"
        )

    # Values are intentionally repeated here instead of inheriting formal
    # model defaults.  This dictionary is embedded in every checkpoint.
    return {
        "schema": RUNNER_SCHEMA,
        "scientific_status": "smoke_only_non_publishable",
        "force_smoke_view": True,
        "roles": {
            "fit": "train",
            "checkpoint_selection": "validation",
            "scenario_generation_and_metrics": "selection",
            "not_used": ["calibration"],
            "hard_forbidden": ["r_seen", "final"],
        },
        "date_counts": dict(EXPECTED_SMOKE_COUNTS),
        "hardware": {
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
        },
        "model": {
            "condition_dim": 20,
            "zones": 10,
            "hours": 24,
            "encoder_dim": int(model_overrides["encoder_dim"]),
            "encoder_depth": int(model_overrides["encoder_depth"]),
            "flow_dim": int(model_overrides["flow_dim"]),
            "flow_depth": int(model_overrides["flow_depth"]),
            "heads": int(model_overrides["heads"]),
            "ff_multiplier": int(model_overrides["ff_multiplier"]),
            "dropout": 0.0,
            "atom_hidden_dim": int(model_overrides["atom_hidden_dim"]),
            "atom_initial_probabilities": list(
                basis["atom_initial_probabilities"]
            ),
            "atom_shared_priority_weight": float(
                basis["atom_shared_priority_weight"]
            ),
            "logit_epsilon": float(basis["logit_epsilon"]),
        },
        "training": {
            "batch_days": int(training["batch_size"]),
            "validation_batch_days": 2,
            "atom_learning_rate": float(training["atom_learning_rate"]),
            "flow_learning_rate": float(training["flow_learning_rate"]),
            "weight_decay": 0.0,
            "gradient_clip": float(training["gradient_clip"]),
            "validate_every": 1,
            "r0_atom_steps": int(training["updates_per_stage"]),
            "r0_flow_steps": int(training["updates_per_stage"]),
            "t0_atom_steps": 0,
            "t0_flow_steps": int(training["updates_per_stage"]),
            "t0_shared_policy": "copy_and_freeze_R0_encoder_and_atom",
        },
        "sampling": {
            "role": "selection",
            "days": 2,
            "members": int(sampling["members"]),
            "steps": int(sampling["integration_steps"]),
            "method": str(sampling["method"]),
            "per_path_nfe": int(sampling["per_path_flow_nfe"]),
            "member_chunk": int(sampling["member_chunk"]),
            "estimand": str(sampling["estimand"]),
            "score_semantics": str(sampling["score_semantics"]),
            "allocation_semantics": str(sampling["allocation_semantics"]),
            "common_random_numbers_group": str(
                sampling["common_random_numbers_group"]
            ),
            "sampling_plan_id": str(sampling["sampling_plan_id"]),
        },
        "seeds": {
            "registered_model_seed": 0,
            "r0_initialization": 1000,
            "t0_initialization": 1001,
            "training": 0,
            "shared_sampling": int(sampling["shared_sampling_seed"]),
        },
    }


def _assert_smoke_protocol(protocol: ArchitectureProtocol) -> None:
    if not protocol.smoke or not bool(protocol.manifest["smoke"]["enabled"]):
        raise RuntimeError("formal protocol refused: smoke=True is mandatory")
    counts = {
        role: len(protocol.role_dates(role)) for role in EXPECTED_SMOKE_COUNTS
    }
    if counts != EXPECTED_SMOKE_COUNTS:
        raise RuntimeError(
            f"formal or drifted role counts refused: {counts} != {EXPECTED_SMOKE_COUNTS}"
        )
    if len(protocol.role_dates("final", full=True)) != 0:
        raise RuntimeError("local final dates are forbidden")
    used = _require_allowed_roles(("train", "validation", "selection"))
    r_seen = protocol.role_dates("r_seen", full=True)
    for role in used:
        active = protocol.role_dates(role)
        if np.intersect1d(active, r_seen).size:
            raise RuntimeError(f"R-SEEN contamination detected in {role}")
    for left, right in (("train", "validation"), ("train", "selection"), ("validation", "selection")):
        if np.intersect1d(protocol.role_dates(left), protocol.role_dates(right)).size:
            raise RuntimeError(f"smoke roles overlap: {left}/{right}")


def _batches(
    split: ArchitectureSplitData,
    *,
    batch_days: int,
) -> list[ArchitectureBatch]:
    if split.role not in _require_allowed_roles((split.role,)):
        raise AssertionError("role guard failed")
    if batch_days < 1:
        raise ValueError("batch_days must be positive")
    return [
        ArchitectureBatch.from_split(split, np.arange(start, stop))
        for start in range(0, len(split), batch_days)
        for stop in (min(start + batch_days, len(split)),)
    ]


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config["model"])


def _load_best(
    path: str | Path,
    model: JointRectifiedFlowBase,
    trainer: ArchitectureTrainer,
) -> None:
    load_checkpoint(
        path,
        model,
        expected_config_sha256=trainer.config_sha256,
        expected_protocol_sha256=trainer.protocol_sha256,
        expected_data_bundle_sha256=trainer.data_bundle_sha256,
        device="cpu",
    )


def _fit_models(
    bundle: ArchitectureDataBundle,
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, JointRectifiedFlowBase], dict[str, Any]]:
    _require_allowed_roles((bundle.train.role, bundle.validation.role))
    train_config = config["training"]
    train_batches = _batches(
        bundle.train, batch_days=int(train_config["batch_days"])
    )
    validation_batches = _batches(
        bundle.validation,
        batch_days=int(train_config["validation_batch_days"]),
    )
    protocol_sha = str(bundle.protocol.manifest["protocol_sha256"])
    data_sha = str(bundle.manifest["data_bundle_sha256"])

    torch.manual_seed(int(config["seeds"]["r0_initialization"]))
    r0 = R0JointRectifiedFlow(**_model_kwargs(config))
    r0_trainer = ArchitectureTrainer(
        r0,
        output_dir / "runs" / "R0_seed0",
        resolved_config=config,
        protocol_sha256=protocol_sha,
        data_bundle_sha256=data_sha,
        training_seed=int(config["seeds"]["training"]),
        run_id="architecture_v1_smoke_R0_seed0",
        device="cpu",
    )
    common_fit = {
        "weight_decay": float(train_config["weight_decay"]),
        "gradient_clip": float(train_config["gradient_clip"]),
        "validate_every": int(train_config["validate_every"]),
    }
    r0_atom = r0_trainer.fit_stage(
        "atom",
        train_batches,
        validation_batches,
        steps=int(train_config["r0_atom_steps"]),
        learning_rate=float(train_config["atom_learning_rate"]),
        **common_fit,
    )
    _load_best(r0_atom["best"], r0, r0_trainer)
    r0_flow = r0_trainer.fit_stage(
        "flow",
        train_batches,
        validation_batches,
        steps=int(train_config["r0_flow_steps"]),
        learning_rate=float(train_config["flow_learning_rate"]),
        **common_fit,
    )
    _load_best(r0_flow["best"], r0, r0_trainer)

    torch.manual_seed(int(config["seeds"]["t0_initialization"]))
    t0 = T0MemorylessRectifiedFlow(**_model_kwargs(config))
    t0.load_shared_from(r0, freeze=True)
    frozen_before = shared_ea_state_sha256(t0)
    t0_trainer = ArchitectureTrainer(
        t0,
        output_dir / "runs" / "T0_seed0",
        resolved_config=config,
        protocol_sha256=protocol_sha,
        data_bundle_sha256=data_sha,
        training_seed=int(config["seeds"]["training"]),
        run_id="architecture_v1_smoke_T0_seed0",
        device="cpu",
    )
    t0_flow = t0_trainer.fit_stage(
        "flow",
        train_batches,
        validation_batches,
        steps=int(train_config["t0_flow_steps"]),
        learning_rate=float(train_config["flow_learning_rate"]),
        **common_fit,
    )
    _load_best(t0_flow["best"], t0, t0_trainer)
    frozen_after = shared_ea_state_sha256(t0)
    if frozen_before != frozen_after:
        raise RuntimeError("T0 flow stage changed the frozen R0 E/A state")
    if shared_ea_state_sha256(r0) != frozen_after:
        raise RuntimeError("T0 E/A no longer equals the copied R0 common shell")

    report = {
        "R0": {
            "atom": r0_atom,
            "flow": r0_flow,
            "model_spec": r0.model_spec(),
            "common_shell_sha256": shared_ea_state_sha256(r0),
            "config_sha256": r0_trainer.config_sha256,
        },
        "T0": {
            "flow": t0_flow,
            "model_spec": t0.model_spec(),
            "common_shell_sha256_before_flow": frozen_before,
            "common_shell_sha256_after_flow": frozen_after,
            "common_shell_frozen_unchanged": True,
            "config_sha256": t0_trainer.config_sha256,
        },
    }
    return {"R0": r0, "T0": t0}, report


def _evaluate_selection(
    models: Mapping[str, JointRectifiedFlowBase],
    training: Mapping[str, Any],
    bundle: ArchitectureDataBundle,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Delegate archive/metric semantics to architecture_v1.evaluation."""

    _require_allowed_roles((bundle.selection.role,))
    if bundle.selection.role != "selection":
        raise RuntimeError("smoke scenarios may be generated only for selection")
    try:
        from architecture_v1.evaluation import evaluate_and_save_scenarios
    except ImportError as error:  # fail closed rather than inventing another metric API
        raise RuntimeError(
            "architecture_v1 mask-aware scenario archive API is required"
        ) from error

    sampling = config["sampling"]
    if len(bundle.selection) != int(sampling["days"]):
        raise RuntimeError("selection smoke view must contain exactly two days")
    condition = torch.from_numpy(
        np.ascontiguousarray(bundle.selection.condition, dtype=np.float32)
    )
    shared_seed = int(config["seeds"]["shared_sampling"])
    r0_statistics = models["R0"].atom_statistics(condition)
    t0_statistics = models["T0"].atom_statistics(condition)
    if not torch.equal(r0_statistics.probabilities, t0_statistics.probabilities):
        raise RuntimeError("R0/T0 copied E/A laws differ before paired sampling")
    allocation = models["R0"].atom.allocate(
        r0_statistics,
        members=int(sampling["members"]),
        seed=shared_seed + 1,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(shared_seed + 2)
    initial_noise = torch.randn(
        (
            len(bundle.selection),
            int(sampling["members"]),
            10,
            24,
        ),
        dtype=torch.float32,
        generator=noise_generator,
    )
    results: dict[str, Any] = {}
    for variant, model in models.items():
        checkpoint = training[variant]["flow"]
        shared_hash = (
            training[variant].get("common_shell_sha256")
            or training[variant]["common_shell_sha256_after_flow"]
        )
        archive_path = (
            output_dir
            / "scenarios"
            / f"{variant}_seed0_selection_smoke.npz"
        )
        evaluation = evaluate_and_save_scenarios(
            model,
            condition,
            bundle.selection.target,
            bundle.selection.observed_mask,
            bundle.selection.raw_missing_mask,
            bundle.selection.day,
            bundle.selection.zones,
            archive_path,
            members=int(sampling["members"]),
            steps=int(sampling["steps"]),
            sampling_seed=shared_seed,
            split_role="selection",
            member_chunk=int(sampling["member_chunk"]),
            config_sha256=training[variant]["config_sha256"],
            protocol_sha256=str(bundle.protocol.manifest["protocol_sha256"]),
            data_bundle_sha256=str(bundle.manifest["data_bundle_sha256"]),
            checkpoint_sha256=checkpoint["best_sha256"],
            shared_EA_state_sha256=shared_hash,
            training_seed=int(config["seeds"]["training"]),
            run_id=f"architecture_v1_smoke_{variant}_seed0",
            common_random_numbers_group=str(
                sampling["common_random_numbers_group"]
            ),
            sampling_plan_id=str(sampling["sampling_plan_id"]),
            scientific_status="smoke_only_non_publishable",
            integrator=str(sampling["method"]),
            allocation=allocation,
            initial_noise=initial_noise,
            overwrite=False,
        )
        metadata = evaluation["metadata"]
        if int(metadata["flow_nfe"]) != int(sampling["per_path_nfe"]):
            raise RuntimeError("sampler NFE disagrees with resolved smoke plan")
        for field in ("allocation_semantics", "estimand", "score_semantics"):
            if metadata[field] != sampling[field]:
                raise RuntimeError(f"scenario {field} disagrees with resolved smoke plan")
        metrics_path = (
            output_dir / "metrics" / f"{variant}_seed0_selection_smoke.json"
        )
        metrics_payload = {
            **evaluation,
            "scientific_status": "smoke_only_non_publishable",
        }
        metrics_sha = _atomic_json(metrics_path, metrics_payload)
        results[variant] = {
            "archive": str(archive_path),
            "archive_sha256": evaluation["archive_sha256"],
            "archive_manifest": str(archive_path.with_suffix(".manifest.json")),
            "metrics": str(metrics_path),
            "metrics_sha256": metrics_sha,
            "summary": evaluation["summary"],
            "sampling": {
                "per_path_nfe": int(metadata["flow_nfe"]),
                "batched_forward_calls": int(metadata["batched_flow_forward_calls"]),
                "wall_seconds": float(metadata["sampling_wall_seconds"]),
                "common_random_numbers_group": sampling[
                    "common_random_numbers_group"
                ],
                "sampling_plan_id": sampling["sampling_plan_id"],
            },
        }
    return results


def _preflight(
    *, config_path: Path, data_dir: Path | None
) -> tuple[ArchitectureProtocol, dict[str, Any]]:
    protocol = build_architecture_protocol(
        config_path,
        data_dir=data_dir,
        smoke=True,
    )
    _assert_smoke_protocol(protocol)
    config = resolved_smoke_config(protocol.config)
    return protocol, config


def dry_run(
    *, config_path: Path, data_dir: Path | None, output_dir: Path
) -> dict[str, Any]:
    protocol, config = _preflight(config_path=config_path, data_dir=data_dir)
    return {
        "schema": RUNNER_SCHEMA,
        "mode": "dry_run",
        "executed": False,
        "message": "read-only preflight passed; add --execute-smoke to run",
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "protocol_sha256": protocol.manifest["protocol_sha256"],
        "active_date_counts": protocol.manifest["smoke"]["date_counts"],
        "resolved_smoke_config": config,
        "would_write": [
            "protocol manifest + SHA256",
            "R0/T0 validation-selected checkpoints",
            "selection scenario archives + mask-aware metrics",
            "completion manifest",
        ],
    }


def execute_smoke(
    *, config_path: Path, data_dir: Path | None, output_dir: Path
) -> dict[str, Any]:
    protocol, config = _preflight(config_path=config_path, data_dir=data_dir)
    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite a non-empty smoke run directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "SMOKE_ONLY.json"
    _atomic_json(
        marker,
        {
            "schema": RUNNER_SCHEMA,
            "status": "running",
            "scientific_status": "smoke_only_non_publishable",
            "formal_training_authorized": False,
            "r_seen_or_final_use_authorized": False,
            "resolved_smoke_config": config,
        },
    )
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_python_random = random.getstate()
    previous_numpy_random = np.random.get_state()
    previous_torch_random = torch.random.get_rng_state()
    try:
        torch.set_num_threads(int(config["hardware"]["torch_num_threads"]))
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        random.seed(int(config["seeds"]["registered_model_seed"]))
        np.random.seed(int(config["seeds"]["registered_model_seed"]))
        torch.manual_seed(int(config["seeds"]["registered_model_seed"]))

        bundle = build_architecture_v1_data(
            data_dir,
            config_path=config_path,
            smoke=True,
        )
        _assert_smoke_protocol(bundle.protocol)
        if bundle.protocol.manifest["protocol_sha256"] != protocol.manifest[
            "protocol_sha256"
        ]:
            raise RuntimeError("protocol drifted between preflight and target access")
        write_protocol_manifest(
            bundle.protocol,
            destination / "architecture_v1.protocol.manifest.json",
        )
        data_manifest_sha = _atomic_json(
            destination / "architecture_v1.data.manifest.json", bundle.manifest
        )
        models, training = _fit_models(bundle, destination, config)
        evaluation = _evaluate_selection(models, training, bundle, destination, config)
        result = {
            "schema": COMPLETION_SCHEMA,
            "status": "complete",
            "scientific_status": "smoke_only_non_publishable",
            "formal_training_performed": False,
            "device": "cpu",
            "dtype": "float32",
            "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
            "data_bundle_sha256": bundle.manifest["data_bundle_sha256"],
            "data_manifest_file_sha256": data_manifest_sha,
            "roles_used": ["train", "validation", "selection"],
            "roles_never_used": ["calibration", "r_seen", "final"],
            "training": training,
            "evaluation": evaluation,
            "resolved_smoke_config": config,
        }
        completion_sha = _atomic_json(destination / "completion.json", result)
        result["completion_file_sha256"] = completion_sha
        _atomic_json(
            marker,
            {
                "schema": RUNNER_SCHEMA,
                "status": "complete",
                "scientific_status": "smoke_only_non_publishable",
                "formal_training_authorized": False,
                "r_seen_or_final_use_authorized": False,
                "completion": "completion.json",
                "completion_file_sha256": completion_sha,
                "resolved_smoke_config": config,
            },
        )
        return result
    except Exception as error:
        _atomic_json(
            marker,
            {
                "schema": RUNNER_SCHEMA,
                "status": "failed",
                "scientific_status": "smoke_only_non_publishable",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "formal_training_authorized": False,
                "r_seen_or_final_use_authorized": False,
                "resolved_smoke_config": config,
            },
        )
        raise
    finally:
        torch.random.set_rng_state(previous_torch_random)
        np.random.set_state(previous_numpy_random)
        random.setstate(previous_python_random)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute the bounded architecture-v1 CPU smoke test."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "architecture_v1" / "smoke_v1_seed0",
    )
    parser.add_argument(
        "--execute-smoke",
        action="store_true",
        help="execute the fixed CPU smoke plan; omission is a read-only dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.execute_smoke:
        result = execute_smoke(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    else:
        result = dry_run(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(_canonical(result), ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
