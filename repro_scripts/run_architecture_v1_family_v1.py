#!/usr/bin/env python3
"""Execute the frozen family-v1 train-only P0 preflight.

P0 is deliberately disposable.  It materialises train targets only, performs
50 optimizer updates for F0 and D0, audits an exact checkpoint-resume fork,
and deletes every weight-bearing checkpoint before writing the result.
Validation/calibration/selection/R-SEEN/final targets are never loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
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
P0_SCHEMA = "architecture_v1_family_v1_P0_result_v1"
P0_FAMILY_SCHEMA = "architecture_v1_family_v1_P0_family_v1"

from architecture_v1.atom import INTERIOR_STATE, ONE_STATE, ZERO_STATE
from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.family_diffusion import FamilyEMATrainer, MaskedJointDDPM
from architecture_v1.model import R0JointRectifiedFlow, T0StableSourceRectifiedFlow
from architecture_v1.training import (
    ArchitectureBatch,
    canonical_sha256,
    file_sha256,
    parameter_manifest,
    shared_ea_state_sha256,
    tensor_state_sha256,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
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
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


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
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _verified(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != str(expected):
        raise RuntimeError(f"frozen file SHA256 mismatch: {path}")
    return actual


def _stable_seed(*parts: object) -> int:
    encoded = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (
        2**63 - 1
    )


def _configure_cuda() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("family-v1 P0 requires a visible CUDA device")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "amp": False,
        "tf32": False,
    }


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(path)
    if config.get("schema") != "architecture_v1_family_v1":
        raise ValueError("unexpected family-v1 schema")
    if config.get("status") != "frozen_before_family_v1_implementation_smoke_or_training":
        raise RuntimeError("family-v1 protocol is not frozen at the P0 boundary")
    sidecar = path.with_name(path.name + ".sha256")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError("family-v1 config sidecar is malformed")
    _verified(path, pieces[0])
    lineage = config["lineage"]
    for key in ("v3_2_result", "v3_2_freeze", "shared_EA_checkpoint"):
        _verified(ROOT / lineage[key], lineage[f"{key}_sha256"])
    result = _read(ROOT / lineage["v3_2_result"])
    if result.get("status") != lineage["v3_2_required_status"]:
        raise RuntimeError("v3.2 did not authorize the family comparison")
    model_config_path = Path(result["formal_config"])
    if not model_config_path.is_absolute():
        model_config_path = ROOT / model_config_path
    _verified(model_config_path, result["formal_config_sha256"])
    model_config = _read(model_config_path)
    roles = config["role_access"]
    if roles["p0_allowed_target_roles"] != ["train"]:
        raise RuntimeError("P0 target-role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    return config, model_config


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        DATA_CONFIG,
        ROOT / "architecture_v1" / "atom.py",
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "tests" / "test_architecture_v1_family_diffusion.py",
        ROOT / "tests" / "test_architecture_v1_family_v1_protocol.py",
    ]
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    core = {"schema": "architecture_v1_family_v1_P0_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _model_kwargs(
    family_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    common = family_config["common_model"]
    inherited = model_config["common_model"]
    kwargs = {
        "condition_dim": int(common["condition_dim"]),
        "zones": int(common["zones"]),
        "hours": int(common["hours"]),
        "encoder_dim": int(common["encoder_dim"]),
        "encoder_depth": int(common["encoder_depth"]),
        "flow_dim": int(common["transport_dim"]),
        "flow_depth": int(common["transport_depth"]),
        "heads": int(common["heads"]),
        "ff_multiplier": int(common["ff_multiplier"]),
        "dropout": float(common["dropout"]),
        "atom_hidden_dim": int(common["atom_hidden_dim"]),
        "atom_initial_probabilities": tuple(
            float(value) for value in inherited["atom_initial_probabilities"]
        ),
        "atom_fixed_one_probability": float(
            inherited["atom_fixed_one_probability"]
        ),
        "atom_shared_priority_weight": float(
            inherited["atom_shared_priority_weight"]
        ),
        "atom_location_auxiliary_weight": float(
            inherited["atom_location_auxiliary_weight"]
        ),
        "atom_location_mean": float(inherited["atom_location_mean"]),
        "atom_location_std": float(inherited["atom_location_std"]),
        "atom_location_smooth_l1_beta": float(
            inherited["atom_location_smooth_l1_beta"]
        ),
        "stable_source_rho_max": float(inherited["stable_source_rho_max"]),
        "logit_epsilon": float(common["logit_epsilon"]),
    }
    if kwargs["atom_location_mean"] != float(
        family_config["common_latent_contract"][
            "train_only_location_mean_is_not_a_transport_normalizer"
        ]
    ):
        raise RuntimeError("inherited atom location mean differs from family-v1")
    if kwargs["atom_location_std"] != float(
        family_config["common_latent_contract"][
            "train_only_location_std_is_not_a_transport_normalizer"
        ]
    ):
        raise RuntimeError("inherited atom location std differs from family-v1")
    return kwargs


def _load_shared_ea(
    config: Mapping[str, Any], kwargs: Mapping[str, Any], device: torch.device
) -> R0JointRectifiedFlow:
    checkpoint = ROOT / config["lineage"]["shared_EA_checkpoint"]
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    shared = R0JointRectifiedFlow(**kwargs).to(device)
    shared.load_state_dict(payload["model_state_dict"], strict=True)
    state = {name: value.detach().cpu() for name, value in shared.state_dict().items()}
    if tensor_state_sha256(state) != payload["model_state_sha256"]:
        raise RuntimeError("shared E/A source checkpoint tensor hash mismatch")
    if shared_ea_state_sha256(shared) != config["lineage"]["shared_EA_state_sha256"]:
        raise RuntimeError("shared E/A state identity mismatch")
    shared.freeze_shared(True)
    return shared


def _take_batch(
    train: Any, indices: Sequence[int] | np.ndarray, device: torch.device
) -> ArchitectureBatch:
    return ArchitectureBatch.from_split(train, indices).to(device)


def _batch_plan(length: int, *, updates: int, batch_days: int, seed: int) -> list[np.ndarray]:
    if length < 1 or updates < 1 or batch_days < 1:
        raise ValueError("invalid P0 batch-plan dimensions")
    plan: list[np.ndarray] = []
    epoch = 0
    while len(plan) < updates:
        order = np.random.default_rng(int(seed) + epoch).permutation(length)
        for start in range(0, length, batch_days):
            plan.append(np.ascontiguousarray(order[start : start + batch_days], dtype=np.int64))
            if len(plan) == updates:
                break
        epoch += 1
    return plan


def _plan_sha256(plan: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in plan:
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _nested_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(
            first.detach().cpu(), second.detach().cpu()
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(_nested_equal(a, b) for a, b in zip(first, second))
        )
    return bool(first == second)


def _model_state_hash(model: torch.nn.Module) -> str:
    return tensor_state_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )


def _finite_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not bool(np.isfinite(array).all()):
        raise FloatingPointError("P0 scalar trace is empty or non-finite")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def _counted_step(
    trainer: FamilyEMATrainer, batch: ArchitectureBatch
) -> tuple[dict[str, float], int]:
    calls = 0

    def count(*_args: Any) -> None:
        nonlocal calls
        calls += 1

    hook = trainer.model.flow.register_forward_hook(count)
    try:
        metrics = trainer.train_step(batch)
    finally:
        hook.remove()
    if calls != 1:
        raise RuntimeError(f"one optimizer update made {calls} network calls")
    return metrics, calls


def _fixed_train_proxy_bank(
    trainer: FamilyEMATrainer,
    batch: ArchitectureBatch,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    noise_seeds = [int(value) for value in training["validation_bank_noise_seeds"]]
    time_seeds = [
        int(value) for value in training["validation_bank_time_or_timestep_seeds"]
    ]
    if len(noise_seeds) != len(time_seeds) or len(noise_seeds) != int(
        training["validation_bank_replicates"]
    ):
        raise RuntimeError("fixed-bank seed registry is inconsistent")
    combined = [
        _stable_seed("family-v1-P0-train-proxy-bank", noise, time)
        for noise, time in zip(noise_seeds, time_seeds)
    ]
    online_before = _model_state_hash(trainer.model)
    first = [trainer.evaluate_loss([batch], seed=seed) for seed in combined]
    second = [trainer.evaluate_loss([batch], seed=seed) for seed in combined]
    deterministic = first == second
    online_restored = _model_state_hash(trainer.model) == online_before
    finite = all(
        math.isfinite(float(value))
        for record in first
        for value in record.values()
    )
    return {
        "role": "train_proxy_only",
        "validation_targets_loaded": False,
        "noise_seeds": noise_seeds,
        "time_or_timestep_seeds": time_seeds,
        "combined_deterministic_seeds": combined,
        "first_pass": first,
        "second_pass": second,
        "bitwise_scalar_replay": deterministic,
        "online_weights_restored_after_EMA_scope": online_restored,
        "all_finite": finite,
        "passed": deterministic and online_restored and finite,
    }


@torch.no_grad()
def _sampling_audit(
    trainer: FamilyEMATrainer,
    diffusion: MaskedJointDDPM,
    condition: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    family = trainer.family
    members = int(config["evaluation"]["members"])
    seed = int(config["evaluation"]["sampling_seeds"][0])
    primary_chunk, alternate_chunk = 10, 20
    with trainer.ema_weights() as model:
        encoded = model.encode_condition(condition)
        statistics = model.atom(encoded)
        allocation = model.atom.allocate(statistics, members=members, seed=seed + 1)
        generator = torch.Generator(device=condition.device)
        generator.manual_seed(seed + 2)
        initial_noise = torch.randn(
            allocation.states.shape,
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )

        def sample(chunk: int):
            if family == "F0":
                return model.sample(
                    condition,
                    members=members,
                    steps=16,
                    seed=seed,
                    method="heun",
                    member_chunk=chunk,
                    allocation=allocation,
                    initial_noise=initial_noise,
                )
            return diffusion.sample_ddim(
                model,
                condition,
                members=members,
                steps=31,
                eta=0.0,
                seed=seed,
                member_chunk=chunk,
                allocation=allocation,
                initial_noise=initial_noise,
            )

        primary = sample(primary_chunk)
        replay = sample(primary_chunk)
        alternate = sample(alternate_chunk)
    replay_exact = (
        torch.equal(primary.values, replay.values)
        and torch.equal(primary.states, replay.states)
        and torch.equal(primary.interior_latent, replay.interior_latent)
    )
    difference = (primary.values - alternate.values).abs()
    max_difference = float(difference.max().cpu())
    chunk_close = bool(
        torch.allclose(
            primary.values, alternate.values, atol=5e-7, rtol=1e-6
        )
    ) and max_difference <= 1e-6
    state_exact = torch.equal(primary.states, alternate.states)
    active_exact = torch.equal(primary.active_mask, alternate.active_mask)
    inactive_max = (
        float(primary.interior_latent[~primary.active_mask].abs().max().cpu())
        if bool((~primary.active_mask).any())
        else 0.0
    )
    zero_exact = bool((primary.values[primary.states == ZERO_STATE] == 0.0).all())
    one_exact = bool((primary.values[primary.states == ONE_STATE] == 1.0).all())
    finite = bool(torch.isfinite(primary.values).all()) and bool(
        torch.isfinite(primary.interior_latent).all()
    )
    expected_nfe = 31
    nfe_exact = primary.per_path_nfe == expected_nfe
    return {
        "members": members,
        "sampling_seed": seed,
        "primary_member_chunk": primary_chunk,
        "alternate_member_chunk": alternate_chunk,
        "per_path_nfe": primary.per_path_nfe,
        "expected_per_path_nfe": expected_nfe,
        "primary_batched_forward_calls": primary.batched_forward_calls,
        "same_seed_replay_bitwise_exact": replay_exact,
        "member_chunk_states_bitwise_exact": state_exact,
        "member_chunk_active_mask_bitwise_exact": active_exact,
        "member_chunk_values_allclose": chunk_close,
        "member_chunk_values_max_abs_difference": max_difference,
        "inactive_latent_max_abs": inactive_max,
        "zero_atom_values_exact": zero_exact,
        "one_atom_values_exact": one_exact,
        "all_finite": finite,
        "nfe_exact": nfe_exact,
        "passed": all(
            (
                replay_exact,
                state_exact,
                active_exact,
                chunk_close,
                inactive_max == 0.0,
                zero_exact,
                one_exact,
                finite,
                nfe_exact,
            )
        ),
    }


def _run_family(
    family: str,
    *,
    shared: R0JointRectifiedFlow,
    kwargs: Mapping[str, Any],
    bundle: Any,
    plan: Sequence[np.ndarray],
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    pilot_seed = int(config["P0_preflight"]["pilot_seed"])
    initialization_seed = 12000 + pilot_seed
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    model = T0StableSourceRectifiedFlow(**kwargs).to(device)
    model.load_shared_from(shared, freeze=True)
    initial_hash = _model_state_hash(model)
    shared_before = shared_ea_state_sha256(model)
    diffusion = MaskedJointDDPM(
        timesteps=int(config["D0_diffusion"]["training_timesteps"]),
        cosine_offset=float(config["D0_diffusion"]["cosine_offset"]),
        beta_min=float(config["D0_diffusion"]["beta_clip"][0]),
        beta_max=float(config["D0_diffusion"]["beta_clip"][1]),
    ).to(device)
    trainer = FamilyEMATrainer(
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
    if model.parameter_count() != int(config["common_model"]["total_parameters_expected"]):
        raise RuntimeError("formal model parameter count differs from family-v1")
    if sum(parameter.numel() for parameter in diffusion.parameters()) != 0:
        raise RuntimeError("diffusion wrapper introduced trainable parameters")

    print(f"[P0] {family}: initialized; starting 50 train-only updates", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["fresh_seed_protocol"]["common_path_noise_seed"]))
    torch.cuda.manual_seed_all(
        int(config["fresh_seed_protocol"]["common_path_noise_seed"])
    )
    metrics: list[dict[str, float]] = []
    network_calls = 0
    resume_report: dict[str, Any] | None = None
    checkpoint = output_root / family / "resume_audit.pt"
    weight_files_deleted = False
    started = time.perf_counter()
    for update_number, indices in enumerate(plan, start=1):
        batch = _take_batch(bundle.train, indices, device)
        if update_number == 26:
            checkpoint_identity = {
                **dict(identity),
                "family": family,
                "checkpoint_after_optimizer_updates": 25,
            }
            trainer.save_checkpoint(checkpoint, identity=checkpoint_identity)
            original_metrics, calls = _counted_step(trainer, batch)
            network_calls += calls
            original_model_hash = _model_state_hash(trainer.model)
            original_ema_hash = trainer.ema.tensor_sha256()
            original_optimizer = trainer.optimizer.state_dict()

            torch.manual_seed(initialization_seed + 999)
            torch.cuda.manual_seed_all(initialization_seed + 999)
            resumed_model = T0StableSourceRectifiedFlow(**kwargs).to(device)
            resumed = FamilyEMATrainer(
                resumed_model,
                family=family,
                diffusion=MaskedJointDDPM(
                    timesteps=int(config["D0_diffusion"]["training_timesteps"]),
                    cosine_offset=float(config["D0_diffusion"]["cosine_offset"]),
                    beta_min=float(config["D0_diffusion"]["beta_clip"][0]),
                    beta_max=float(config["D0_diffusion"]["beta_clip"][1]),
                ).to(device),
                learning_rate=float(training["learning_rate"]),
                betas=tuple(float(value) for value in training["betas"]),
                eps=float(training["eps"]),
                weight_decay=float(training["weight_decay"]),
                gradient_clip=float(training["gradient_clip"]),
                ema_decay=float(config["common_EMA_amendment"]["decay"]),
            )
            resumed.load_checkpoint(
                checkpoint, expected_identity=checkpoint_identity, restore_rng=True
            )
            resumed_metrics, calls = _counted_step(resumed, batch)
            network_calls += calls
            model_exact = _model_state_hash(resumed.model) == original_model_hash
            ema_exact = resumed.ema.tensor_sha256() == original_ema_hash
            optimizer_exact = _nested_equal(
                resumed.optimizer.state_dict(), original_optimizer
            )
            metrics_exact = all(
                resumed_metrics[key] == original_metrics[key]
                for key in original_metrics
                if key != "wall_seconds"
            )
            resume_report = {
                "checkpoint_after_updates": 25,
                "replayed_update": 26,
                "online_model_bitwise_exact": model_exact,
                "EMA_bitwise_exact": ema_exact,
                "optimizer_bitwise_exact": optimizer_exact,
                "metrics_except_wall_time_exact": metrics_exact,
                "RNG_restored": metrics_exact and model_exact,
                "passed": model_exact and ema_exact and optimizer_exact and metrics_exact,
            }
            del trainer, model
            trainer = resumed
            model = resumed_model
            metrics.append(resumed_metrics)
            checkpoint.unlink()
            checkpoint.with_name(checkpoint.name + ".sha256").unlink()
            weight_files_deleted = True
            if not resume_report["passed"]:
                raise RuntimeError(f"{family} checkpoint-resume audit failed")
            print(
                f"[P0] {family}: checkpoint/RNG resume audit passed at update 26",
                flush=True,
            )
        else:
            update_metrics, calls = _counted_step(trainer, batch)
            network_calls += calls
            metrics.append(update_metrics)
        if update_number % 10 == 0 or update_number == 50:
            print(
                f"[P0] {family}: update {update_number}/50; "
                f"loss={metrics[-1]['loss']:.6g}; "
                f"grad={metrics[-1]['gradient_norm']:.6g}",
                flush=True,
            )
    if len(metrics) != 50 or trainer.optimizer_updates != 50:
        raise RuntimeError(f"{family} did not complete exactly 50 retained-sequence updates")
    if resume_report is None or not weight_files_deleted:
        raise RuntimeError(f"{family} resume audit was not completed and discarded")

    bank_batch = _take_batch(bundle.train, plan[0], device)
    bank_report = _fixed_train_proxy_bank(trainer, bank_batch, config)
    print(f"[P0] {family}: fixed train-proxy random bank audited", flush=True)
    sample_condition = _take_batch(bundle.train, plan[0][:1], device).condition
    sampling_report = _sampling_audit(
        trainer, trainer.diffusion or diffusion, sample_condition, config
    )
    print(f"[P0] {family}: replay/chunk/NFE sampling audit completed", flush=True)
    torch.cuda.synchronize(device)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    memory_limit = int(4.5 * 1024**3)
    losses = [record["loss"] for record in metrics]
    gradients = [record["gradient_norm"] for record in metrics]
    all_metrics_finite = all(
        math.isfinite(float(value))
        for record in metrics
        for value in record.values()
    )
    shared_unchanged = shared_ea_state_sha256(model) == shared_before
    weights_remaining = list((output_root / family).glob("*.pt")) + list(
        (output_root / family).glob("*.pth")
    )
    report = {
        "schema": P0_FAMILY_SCHEMA,
        "family": family,
        "pilot_seed": pilot_seed,
        "initialization_seed": initialization_seed,
        "initial_model_tensor_sha256": initial_hash,
        "parameter_manifest": parameter_manifest(model),
        "diffusion_trainable_parameters": 0,
        "optimizer_updates": trainer.optimizer_updates,
        "extra_resume_probe_updates_not_in_retained_sequence": 1,
        "network_forward_calls_during_updates_including_resume_probe": network_calls,
        "every_update_one_network_forward": network_calls == 51,
        "loss": _finite_summary(losses),
        "preclip_gradient_norm": _finite_summary(gradients),
        "gradient_clip_fraction": float(
            np.mean(np.asarray(gradients) > float(training["gradient_clip"]))
        ),
        "first_update": metrics[0],
        "last_update": metrics[-1],
        "all_train_metrics_finite": all_metrics_finite,
        "shared_EA_before_sha256": shared_before,
        "shared_EA_after_sha256": shared_ea_state_sha256(model),
        "shared_EA_unchanged": shared_unchanged,
        "fixed_train_proxy_bank": bank_report,
        "checkpoint_resume": resume_report,
        "sampling": sampling_report,
        "peak_GPU_memory_allocated_bytes": peak_memory,
        "peak_GPU_memory_limit_bytes": memory_limit,
        "peak_GPU_memory_below_4_5_GiB": peak_memory < memory_limit,
        "wall_seconds": float(time.perf_counter() - started),
        "weights_retained": False,
        "temporary_checkpoint_deleted": weight_files_deleted,
        "weight_files_remaining": [str(path) for path in weights_remaining],
    }
    report["passed"] = all(
        (
            report["optimizer_updates"] == 50,
            report["every_update_one_network_forward"],
            report["all_train_metrics_finite"],
            report["shared_EA_unchanged"],
            bank_report["passed"],
            resume_report["passed"],
            sampling_report["passed"],
            report["peak_GPU_memory_below_4_5_GiB"],
            not weights_remaining,
        )
    )
    print(
        f"[P0] {family}: {'PASS' if report['passed'] else 'FAIL'}; "
        f"peak={peak_memory / 1024**3:.3f} GiB; "
        f"wall={report['wall_seconds']:.1f}s",
        flush=True,
    )
    del trainer, model, diffusion
    torch.cuda.empty_cache()
    return report


def execute_p0(config_path: Path) -> dict[str, Any]:
    config, model_config = _load_config(config_path)
    p0 = config["P0_preflight"]
    if int(p0["discarded_train_updates_per_family"]) != 50:
        raise RuntimeError("family-v1 P0 update count changed")
    output_root = ROOT / config["output_root"] / "P0_preflight"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("family-v1 P0 output is non-empty; refusing overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        device, runtime = _configure_cuda()
        print(
            f"[P0] runtime: {runtime['device_name']}; "
            f"VRAM={runtime['total_memory_bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )
        bundle = build_architecture_v1_train_data(config_path=DATA_CONFIG)
        if bundle.materialized_roles != ("train",):
            raise RuntimeError("P0 materialized a forbidden target role")
        access = bundle.manifest["formal_train_only_target_access"]
        if access["materialized_roles"] != ["train"]:
            raise RuntimeError("train-only target-access manifest changed")
        if access["forbidden_target_arrays_materialized"] is not False:
            raise RuntimeError("P0 materialized forbidden target arrays")
        if access["validation_bank_constructed"] is not False:
            raise RuntimeError("P0 constructed a validation-target bank")
        if bundle.protocol.manifest["protocol_sha256"] != config["lineage"]["protocol_sha256"]:
            raise RuntimeError("P0 data protocol identity mismatch")
        print(
            f"[P0] train-only data loaded: {len(bundle.train)} days; "
            "forbidden target arrays materialized=false",
            flush=True,
        )
        if bundle.manifest["train_only_data_bundle_sha256"] != config["lineage"]["fit_data_bundle_sha256"]:
            # The train-only facade has a deliberately different manifest from
            # fit-data, but its raw train split must match the registered split.
            fit_sha = bundle.manifest["data_audit"]["split_array_sha256"]["train"]
        else:
            fit_sha = bundle.manifest["train_only_data_bundle_sha256"]

        code = _code_manifest(config_path)
        identity = {
            "schema": "architecture_v1_family_v1_P0_identity_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path.resolve()),
            "config_file_sha256": file_sha256(config_path),
            "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
            "train_only_data_bundle_sha256": bundle.manifest[
                "train_only_data_bundle_sha256"
            ],
            "train_split_array_sha256": fit_sha,
            "code_manifest": code,
            "runtime": runtime,
            "materialized_target_roles": ["train"],
            "validation_target_accessed": False,
            "calibration_target_accessed": False,
            "selection_target_accessed": False,
            "r_seen_target_accessed": False,
            "final_target_accessed": False,
            "selection_state": "sealed",
            "calibration_state": "sealed",
        }
        identity_sha = _atomic_json(output_root / "identity.json", identity)
        kwargs = _model_kwargs(config, model_config)
        shared = _load_shared_ea(config, kwargs, device)
        plan = _batch_plan(
            len(bundle.train),
            updates=50,
            batch_days=int(config["training"]["batch_calendar_days"]),
            seed=int(config["fresh_seed_protocol"]["common_epoch_shuffle_seed"]),
        )
        common = {
            "batch_plan_sha256": _plan_sha256(plan),
            "train_days": int(len(bundle.train)),
            "batch_days": int(config["training"]["batch_calendar_days"]),
            "updates": 50,
            "shared_EA_state_sha256": shared_ea_state_sha256(shared),
        }
        reports: dict[str, Any] = {}
        report_files: dict[str, Any] = {}
        for family in ("F0", "D0"):
            report = _run_family(
                family,
                shared=shared,
                kwargs=kwargs,
                bundle=bundle,
                plan=plan,
                config=config,
                identity={
                    "config_sha256": file_sha256(config_path),
                    "code_sha256": code["code_sha256"],
                    "data_sha256": bundle.manifest["train_only_data_bundle_sha256"],
                    "batch_plan_sha256": common["batch_plan_sha256"],
                },
                output_root=output_root,
                device=device,
            )
            family_path = output_root / family / "P0_FAMILY_RESULT.json"
            digest = _atomic_json(family_path, report)
            reports[family] = report
            report_files[family] = {
                "path": str(family_path.resolve()),
                "sha256": digest,
                "passed": report["passed"],
            }
            print(f"[P0] {family}: audit JSON committed", flush=True)
        initial_match = (
            reports["F0"]["initial_model_tensor_sha256"]
            == reports["D0"]["initial_model_tensor_sha256"]
        )
        all_weights = list(output_root.rglob("*.pt")) + list(output_root.rglob("*.pth"))
        passed = all(report["passed"] for report in reports.values()) and initial_match and not all_weights
        result = {
            "schema": P0_SCHEMA,
            "status": "P0_GO" if passed else "P0_NO_GO",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "identity": str((output_root / "identity.json").resolve()),
            "identity_sha256": identity_sha,
            "common": common,
            "family_reports": report_files,
            "F0_D0_initial_tensor_sha256_match": initial_match,
            "all_weights_discarded": not all_weights,
            "weight_files_remaining": [str(path) for path in all_weights],
            "materialized_target_roles": ["train"],
            "validation_target_accessed": False,
            "calibration_target_accessed": False,
            "selection_target_accessed": False,
            "r_seen_target_accessed": False,
            "final_target_accessed": False,
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "retained_training_authorized": passed,
            "next_action": (
                "implement_family_v1_formal_training_runner"
                if passed
                else "P0_No_Go_new_revision_required_no_formal_training"
            ),
        }
        result_sha = _atomic_json(output_root / "P0_RESULT.json", result)
        return {**result, "file_sha256": result_sha}
    except Exception as error:
        failure = output_root / "P0_FAILURE.json"
        if not failure.exists():
            _atomic_json(
                failure,
                {
                    "schema": "architecture_v1_family_v1_P0_failure_v1",
                    "status": "P0_execution_failed_no_formal_training_authorized",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "all_weights_discarded": not bool(
                        list(output_root.rglob("*.pt"))
                        + list(output_root.rglob("*.pth"))
                    ),
                    "validation_target_accessed": False,
                    "calibration_target_accessed": False,
                    "selection_target_accessed": False,
                    "r_seen_target_accessed": False,
                    "final_target_accessed": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-p0", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not args.execute_p0:
        config, _ = _load_config(config_path)
        print(
            json.dumps(
                {
                    "schema": "architecture_v1_family_v1_P0_dry_run_v1",
                    "config": str(config_path),
                    "config_sha256": file_sha256(config_path),
                    "target_roles_if_executed": config["role_access"][
                        "p0_allowed_target_roles"
                    ],
                    "updates_per_family": config["P0_preflight"][
                        "discarded_train_updates_per_family"
                    ],
                    "weights_retained": False,
                    "next_flag": "--execute-p0",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = execute_p0(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
