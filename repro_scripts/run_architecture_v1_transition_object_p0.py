#!/usr/bin/env python3
"""Run the frozen TGO-v1 seven-path discard-after P0.

The default dry-run is target free.  ``--execute-p0`` alone materializes the
existing train role, replays the frozen G0-A nuisance construction, and uses
only outer-fold-0 outer-train rows for seven paired 50-update runs.  Model,
optimizer, EMA, and resume payloads remain in memory and are discarded before
the JSON result is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    ROOT / "repro_configs" / "architecture_v1_transition_object_probe.json"
)
RESULT_SCHEMA = "architecture_v1_transition_object_p0_result_v1"
RUNNER_PATH = Path(__file__).resolve()

from architecture_v1.atom import INTERIOR_STATE
from architecture_v1.family_diffusion import ddim_timestep_grid
from architecture_v1.g0b_tiny_denoiser import (
    TinyEMA,
    module_state_sha256,
    tensor_mapping_sha256,
)
from architecture_v1.transition_atom import (
    TransitionAtomNuisance,
    train_only_atom_contract,
)
from architecture_v1.transition_object import (
    MaskConditionedOperator,
    ddim_direct_x0_step,
    fit_level_rms,
    fit_operator_scale,
    matched_noise_level_forward,
    transition_native_forward,
    wrong_adjacency_audit,
)
from architecture_v1.transition_probe import (
    PATH_IDS,
    TransitionDenoisingSystem,
    sample_transition_ddim,
    transition_denoising_loss,
)
from repro_scripts.run_architecture_v1_g0_b_tiny_denoiser import (
    _baseline_schedule,
    _replay_nuisance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verified_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"invalid SHA256 sidecar: {sidecar}")
    actual = _sha256(path)
    if actual != fields[0]:
        raise RuntimeError(f"SHA256 sidecar mismatch: {path}")
    return actual


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
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


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
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    sidecar_temporary.replace(sidecar)
    return digest


def _torch_blob(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _semantic_state_sha256(value: Any) -> str:
    """Hash nested state by value, independent of torch archive metadata.

    ``torch.save`` archives may assign different storage identifiers to two
    numerically identical CUDA optimizer states.  Those identifiers are an
    implementation detail, so they must not be part of the resume gate.  This
    digest records container structure, scalar values, tensor metadata, and
    canonical CPU tensor bytes instead.
    """

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            array = tensor.numpy()
            digest.update(b"tensor\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            ordered = sorted(
                item.items(),
                key=lambda pair: (type(pair[0]).__name__, repr(pair[0])),
            )
            digest.update(str(len(ordered)).encode("ascii"))
            digest.update(b"\0")
            for key, nested in ordered:
                update(key)
                update(nested)
            return
        if isinstance(item, tuple):
            digest.update(b"tuple\0")
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for nested in item:
                update(nested)
            return
        if isinstance(item, list):
            digest.update(b"list\0")
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for nested in item:
                update(nested)
            return
        if item is None:
            digest.update(b"none\0")
            return
        if isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")
            return
        raise TypeError(f"unsupported semantic-state value: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _configure_device(
    requested: str, *, allow_cpu: bool
) -> tuple[torch.device, dict[str, Any]]:
    if requested == "cuda":
        if not torch.cuda.is_available():
            if not allow_cpu:
                raise RuntimeError("CUDA requested but unavailable; pass --allow-cpu explicitly")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif requested == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU execution requires --allow-cpu")
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cuda or cpu")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    metadata: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": True,
    }
    if device.type == "cuda":
        metadata.update(
            {
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(device),
                "device_total_memory_bytes": int(
                    torch.cuda.get_device_properties(device).total_memory
                ),
            }
        )
    return device, metadata


def _validate_config(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    digest = _verified_sidecar(config_path)
    if config.get("schema") != "architecture_v1_transition_object_probe_v1":
        raise RuntimeError("unexpected TGO-v1 config schema")
    if config.get("probe_id") != "TGO_V1":
        raise RuntimeError("unexpected TGO-v1 probe identity")
    if tuple(config["paired_training_matrix"]["paths"]) != PATH_IDS:
        raise RuntimeError("config path order differs from code registry")
    matrix = config["paired_training_matrix"]
    expected_runs = (
        len(matrix["outer_folds"])
        * len(matrix["model_seeds"])
        * len(matrix["paths"])
    )
    if int(matrix["retained_runs"]) != expected_runs or expected_runs != 84:
        raise RuntimeError("TGO-v1 retained matrix no longer contains 84 runs")
    p0 = config["P0"]
    if int(p0["total_discarded_updates"]) != len(PATH_IDS) * int(
        p0["discarded_updates_per_path"]
    ):
        raise RuntimeError("TGO-v1 P0 update accounting drifted")
    lineage = config["lineage"]
    for key, hash_key in (
        ("data_protocol_config", "data_protocol_config_sha256"),
        ("g0_a_config", "g0_a_config_sha256"),
        ("g0_a_result", "g0_a_result_sha256"),
        ("g0_b_config", "g0_b_config_sha256"),
        ("g0_b_result", "g0_b_result_sha256"),
    ):
        path = ROOT / lineage[key]
        if _sha256(path) != lineage[hash_key]:
            raise RuntimeError(f"TGO-v1 lineage hash drifted: {key}")
    return {"config_sha256": digest, "paths": list(PATH_IDS), "retained_runs": 84}


def _random_bank(
    *,
    seed: int,
    updates: int,
    batch_size: int,
    train_days: int,
    timesteps: int,
) -> tuple[dict[str, torch.Tensor], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bank = {
        "day_index": torch.randint(
            train_days,
            (updates, batch_size),
            generator=generator,
            dtype=torch.long,
        ),
        "timestep": torch.randint(
            timesteps,
            (updates, batch_size),
            generator=generator,
            dtype=torch.long,
        ),
        "noise": torch.randn(
            updates,
            batch_size,
            10,
            24,
            generator=generator,
            dtype=torch.float32,
        ),
    }
    return bank, tensor_mapping_sha256(bank)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["torch_cuda"]]
        )


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
    )


def _ema_hash(ema: TinyEMA) -> str:
    return tensor_mapping_sha256(ema.shadow)


def _checkpoint_blob(
    system: TransitionDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    *,
    next_update: int,
) -> bytes:
    return _torch_blob(
        {
            "system": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "rng": _rng_state(),
            "next_update": int(next_update),
        }
    )


def _restore_checkpoint(
    blob: bytes,
    path_id: str,
    model_seed: int,
    device: torch.device,
    training: Mapping[str, Any],
) -> tuple[
    TransitionDenoisingSystem,
    torch.optim.Optimizer,
    TinyEMA,
    int,
]:
    try:
        payload = torch.load(
            io.BytesIO(blob), map_location=device, weights_only=False
        )
    except TypeError:  # Older PyTorch releases do not expose weights_only.
        payload = torch.load(io.BytesIO(blob), map_location=device)
    system = TransitionDenoisingSystem(path_id, model_seed=model_seed).to(device)
    system.load_state_dict(payload["system"])
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    ema = TinyEMA(system, decay=float(training["EMA_decay"]))
    ema.load_state_dict(payload["ema"], system)
    _set_rng_state(payload["rng"])
    return system, optimizer, ema, int(payload["next_update"])


def _batch(
    tensors: Mapping[str, torch.Tensor],
    bank: Mapping[str, torch.Tensor],
    update: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = bank["day_index"][update].to(device)
    return (
        tensors["level"].index_select(0, index),
        tensors["active"].index_select(0, index),
        tensors["condition"].index_select(0, index),
        bank["timestep"][update].to(device),
        bank["noise"][update].to(device),
    )


def _one_update(
    system: TransitionDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    tensors: Mapping[str, torch.Tensor],
    bank: Mapping[str, torch.Tensor],
    update: int,
    device: torch.device,
    alpha_bar: torch.Tensor,
    true_operator: MaskConditionedOperator,
    wrong_operator: MaskConditionedOperator,
    dct_operator: MaskConditionedOperator,
    *,
    gradient_clip: float,
) -> tuple[float, float]:
    level, active, condition, timestep, noise = _batch(
        tensors, bank, update, device
    )
    optimizer.zero_grad(set_to_none=True)
    sample = transition_denoising_loss(
        system,
        level,
        active,
        condition,
        timestep,
        noise,
        alpha_bar,
        true_operator=true_operator,
        wrong_operator=wrong_operator,
        dct_operator=dct_operator,
    )
    sample.loss.backward()
    for name, parameter in system.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"registered trainable parameter has no gradient: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient: {name}")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        system.parameters(), float(gradient_clip)
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("TGO-v1 gradient norm is non-finite")
    optimizer.step()
    if not _optimizer_finite(optimizer):
        raise FloatingPointError("TGO-v1 optimizer state is non-finite")
    if not all(bool(torch.isfinite(value).all()) for value in system.parameters()):
        raise FloatingPointError("TGO-v1 model parameter is non-finite")
    ema.update(system)
    return float(sample.loss.detach().cpu()), float(gradient_norm.detach().cpu())


def _operator_audit(
    level_double: torch.Tensor,
    active_cpu: torch.Tensor,
    alpha_bar_numpy: np.ndarray,
    operators: Mapping[str, MaskConditionedOperator],
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    masked = torch.where(active_cpu, level_double, torch.zeros_like(level_double))
    for name, operator in operators.items():
        coefficient = operator.transform(level_double, active_cpu)
        reconstructed = operator.inverse(coefficient, active_cpu)
        error64 = float(torch.max(torch.abs(reconstructed - masked)))
        value32 = level_double.float()
        reconstructed32 = operator.inverse(
            operator.transform(value32, active_cpu), active_cpu
        )
        masked32 = torch.where(active_cpu, value32, torch.zeros_like(value32))
        error32 = float(torch.max(torch.abs(reconstructed32 - masked32)))
        audits[name] = {
            "scale": operator.scale,
            "inverse_max_abs_error_FP64": error64,
            "inverse_max_abs_error_FP32": error32,
        }
        if error64 > 1e-10 or error32 > 1e-5:
            raise RuntimeError(f"{name} inverse audit failed")

    contaminated = level_double.clone()
    contaminated[~active_cpu] = 1e6
    clean_transition = operators["transition_true"].transform(
        level_double, active_cpu
    )
    contaminated_transition = operators["transition_true"].transform(
        contaminated, active_cpu
    )
    inactive_influence = float(
        torch.max(torch.abs(clean_transition - contaminated_transition))
    )
    if inactive_influence != 0.0:
        raise RuntimeError("inactive cells influence true-transition coefficients")

    adjacency = wrong_adjacency_audit(active_cpu)
    if adjacency.eligible_edges == 0 or adjacency.retained_true_edge_fraction > 0.30:
        raise RuntimeError("wrong-adjacency edge-destruction audit failed")

    sample_level = level_double[:4]
    sample_active = active_cpu[:4]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(63004)
    noise = torch.randn(sample_level.shape, generator=generator, dtype=torch.float64)
    alpha_schedule = torch.from_numpy(alpha_bar_numpy).double()
    selected_alpha = alpha_schedule[
        torch.tensor([13, 71, 149, 249], dtype=torch.long)
    ]
    noisy_transition = transition_native_forward(
        sample_level,
        sample_active,
        selected_alpha,
        noise,
        operators["transition_true"],
    )
    pulled = operators["transition_true"].inverse(
        noisy_transition, sample_active
    )
    matched = matched_noise_level_forward(
        sample_level,
        sample_active,
        selected_alpha,
        noise,
        operators["transition_true"],
    )
    forward_equivalence = float(torch.max(torch.abs(pulled - matched)))
    if forward_equivalence > 1e-10:
        raise RuntimeError("matched-noise forward identity failed")

    grid = ddim_timestep_grid(len(alpha_schedule), 31).flip(0)
    current_index = int(grid[0])
    endpoint_alpha = alpha_schedule[current_index].expand(len(sample_level))
    native = transition_native_forward(
        sample_level,
        sample_active,
        endpoint_alpha,
        noise,
        operators["transition_true"],
    )
    level = operators["transition_true"].inverse(native, sample_active)
    clean_native = operators["transition_true"].transform(
        sample_level, sample_active
    )
    maximum_ddim_error = 0.0
    for index, current_tensor in enumerate(grid):
        current = alpha_schedule[int(current_tensor)].expand(len(sample_level))
        if index + 1 < len(grid):
            previous = alpha_schedule[int(grid[index + 1])].expand(len(sample_level))
            native = ddim_direct_x0_step(
                native,
                clean_native,
                sample_active,
                current,
                previous,
            )
            level = ddim_direct_x0_step(
                level,
                torch.where(
                    sample_active, sample_level, torch.zeros_like(sample_level)
                ),
                sample_active,
                current,
                previous,
            )
        else:
            native = clean_native
            level = torch.where(
                sample_active, sample_level, torch.zeros_like(sample_level)
            )
        pulled_step = operators["transition_true"].inverse(native, sample_active)
        maximum_ddim_error = max(
            maximum_ddim_error,
            float(torch.max(torch.abs(pulled_step - level))),
        )
    if maximum_ddim_error > 1e-9:
        raise RuntimeError("full-grid DDIM pullback identity failed")

    return {
        "per_operator": audits,
        "inactive_value_max_influence": inactive_influence,
        "wrong_adjacency": {
            "eligible_segments": adjacency.eligible_segments,
            "eligible_edges": adjacency.eligible_edges,
            "retained_true_edges": adjacency.retained_true_edges,
            "retained_true_edge_fraction": adjacency.retained_true_edge_fraction,
        },
        "matched_noise_forward_max_abs_error": forward_equivalence,
        "oracle_full_grid_DDIM_pullback_max_abs_error": maximum_ddim_error,
    }


def _atom_semantic_audit(
    condition: torch.Tensor,
    observation: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    contract = train_only_atom_contract(observation, observed_mask)
    torch.manual_seed(int(seed))
    model = TransitionAtomNuisance(
        fixed_one_probability=(
            None
            if contract["fixed_one_probability"] is None
            else float(contract["fixed_one_probability"])
        ),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    )
    model.eval()
    diagnostic_condition = condition[:2]
    statistics, allocation = model.allocate(
        diagnostic_condition, members=11, seed=61001
    )
    replays = [
        model.allocate(diagnostic_condition, members=11, seed=61001)[1]
        for _path in PATH_IDS
    ]
    if any(not torch.equal(item.states, allocation.states) for item in replays):
        raise RuntimeError("atom allocation is not common across paths")
    if any(
        not torch.equal(item.analytic_probabilities, allocation.analytic_probabilities)
        for item in replays
    ):
        raise RuntimeError("atom probabilities are not common across paths")
    if not torch.equal(allocation.active_mask, allocation.states == INTERIOR_STATE):
        raise RuntimeError("atom active-mask semantics drifted")
    return {
        "contract": contract,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "probability_sha256": tensor_mapping_sha256(
            {"probabilities": statistics.probabilities}
        ),
        "state_sha256": tensor_mapping_sha256({"states": allocation.states}),
        "paths_with_exact_common_allocation": len(replays),
        "target_state_argument_used_during_allocate": False,
        "classification": "untrained_semantic_P0_only_not_retained_atom_nuisance",
    }


def dry_run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    identity = _validate_config(config, config_path)
    return {
        "schema": RESULT_SCHEMA,
        "mode": "target_free_dry_run",
        "probe_id": config["probe_id"],
        **identity,
        "P0": config["P0"],
        "training": config["common_denoiser"],
        "randomness": config["randomness_contract"],
        "target_arrays_materialized": False,
        "weights_written": False,
    }


def execute_p0(
    config_path: Path,
    *,
    device_name: str,
    allow_cpu: bool,
    output_path: Path | None,
) -> tuple[dict[str, Any], Path, str]:
    config = _read_json(config_path)
    identity = _validate_config(config, config_path)
    p0 = config["P0"]
    training = config["common_denoiser"]
    if output_path is None:
        output_path = ROOT / config["output_root"] / "TGO_V1_P0_RESULT.json"
    if output_path.exists() or output_path.with_name(output_path.name + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite existing P0 result: {output_path}")
    device, device_metadata = _configure_device(
        device_name, allow_cpu=allow_cpu
    )

    g0_b_path = ROOT / config["lineage"]["g0_b_config"]
    g0_b_config = _read_json(g0_b_path)
    bundle, _groups, folds, nuisance_replay, _subset_records = _replay_nuisance(
        g0_b_config
    )
    fold_index = int(p0["outer_fold"])
    fold = folds[fold_index]
    if fold.fold != fold_index:
        raise RuntimeError("outer-fold identity drifted")

    residual_double = torch.from_numpy(fold.residual_train).double()
    active_cpu = torch.from_numpy(fold.active_train).bool()
    level_rms = fit_level_rms(residual_double, active_cpu)
    level_double = residual_double / level_rms
    true_scale = fit_operator_scale(
        level_double, active_cpu, kind="transition_true"
    )
    wrong_scale = fit_operator_scale(
        level_double, active_cpu, kind="transition_wrong"
    )
    dct_scale = fit_operator_scale(
        level_double, active_cpu, kind="orthogonal_dct"
    )
    operators = {
        "transition_true": MaskConditionedOperator(
            "transition_true", scale=true_scale
        ),
        "transition_wrong": MaskConditionedOperator(
            "transition_wrong", scale=wrong_scale
        ),
        "orthogonal_dct": MaskConditionedOperator(
            "orthogonal_dct", scale=dct_scale
        ),
    }
    alpha_numpy, schedule_audit = _baseline_schedule(g0_b_config)
    operator_audit = _operator_audit(
        level_double, active_cpu, alpha_numpy, operators
    )

    train_tensors = {
        "level": level_double.float().to(device),
        "active": active_cpu.to(device),
        "condition": torch.from_numpy(fold.condition_train).float().to(device),
    }
    alpha_bar = torch.from_numpy(alpha_numpy).float().to(device)
    updates = int(p0["discarded_updates_per_path"])
    batch_size = int(training["batch_calendar_days"])
    bank, bank_hash = _random_bank(
        seed=int(config["randomness_contract"]["P0_bank_seed"]),
        updates=updates,
        batch_size=batch_size,
        train_days=len(fold.outer_train),
        timesteps=len(alpha_bar),
    )

    systems = {
        path_id: TransitionDenoisingSystem(
            path_id, model_seed=int(p0["model_seed"])
        ).to(device)
        for path_id in PATH_IDS
    }
    initial_hashes = {
        path_id: module_state_sha256(system)
        for path_id, system in systems.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("seven paths do not share the same initialization")

    path_results: dict[str, Any] = {}
    maximum_peak = 0
    for path_id in PATH_IDS:
        system = systems.pop(path_id)
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=float(training["learning_rate"]),
            betas=tuple(float(value) for value in training["betas"]),
            eps=float(training["eps"]),
            weight_decay=float(training["weight_decay"]),
        )
        ema = TinyEMA(system, decay=float(training["EMA_decay"]))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        gradient_norms: list[float] = []
        checkpoint: bytes | None = None
        resume_audit: dict[str, Any] | None = None
        for update in range(updates):
            loss, gradient_norm = _one_update(
                system,
                optimizer,
                ema,
                train_tensors,
                bank,
                update,
                device,
                alpha_bar,
                operators["transition_true"],
                operators["transition_wrong"],
                operators["orthogonal_dct"],
                gradient_clip=float(training["gradient_clip"]),
            )
            losses.append(loss)
            gradient_norms.append(gradient_norm)
            if update == 24:
                checkpoint = _checkpoint_blob(
                    system, optimizer, ema, next_update=25
                )
            elif update == 25:
                if checkpoint is None:
                    raise RuntimeError("resume checkpoint was not constructed")
                reference = {
                    "loss": loss,
                    "model": module_state_sha256(system),
                    "optimizer": _semantic_state_sha256(
                        optimizer.state_dict()
                    ),
                    "ema": _ema_hash(ema),
                }
                replay_system, replay_optimizer, replay_ema, next_update = (
                    _restore_checkpoint(
                        checkpoint,
                        path_id,
                        int(p0["model_seed"]),
                        device,
                        training,
                    )
                )
                if next_update != update:
                    raise RuntimeError("resume update index drifted")
                replay_loss, _ = _one_update(
                    replay_system,
                    replay_optimizer,
                    replay_ema,
                    train_tensors,
                    bank,
                    update,
                    device,
                    alpha_bar,
                    operators["transition_true"],
                    operators["transition_wrong"],
                    operators["orthogonal_dct"],
                    gradient_clip=float(training["gradient_clip"]),
                )
                replay = {
                    "loss": replay_loss,
                    "model": module_state_sha256(replay_system),
                    "optimizer": _semantic_state_sha256(
                        replay_optimizer.state_dict()
                    ),
                    "ema": _ema_hash(replay_ema),
                }
                if replay != reference:
                    mismatched = {
                        key: {"reference": reference[key], "replay": replay[key]}
                        for key in reference
                        if reference[key] != replay[key]
                    }
                    raise RuntimeError(
                        f"checkpoint resume mismatch for {path_id}: {mismatched}"
                    )
                resume_audit = {
                    "checkpoint_after_updates": 25,
                    "replayed_update_one_based": 26,
                    "exact": True,
                    "checkpoint_retained": False,
                }
                del replay_system, replay_optimizer, replay_ema
                checkpoint = None

        if resume_audit is None or checkpoint is not None:
            raise RuntimeError(f"resume audit incomplete for {path_id}")
        diagnostic_index = bank["day_index"][0][:2].to(device)
        diagnostic_noise = bank["noise"][0][:2].to(device)
        sample_first = sample_transition_ddim(
            system,
            train_tensors["condition"].index_select(0, diagnostic_index),
            train_tensors["active"].index_select(0, diagnostic_index),
            diagnostic_noise,
            alpha_bar,
            true_operator=operators["transition_true"],
            wrong_operator=operators["transition_wrong"],
            dct_operator=operators["orthogonal_dct"],
            steps=5,
        )
        sample_second = sample_transition_ddim(
            system,
            train_tensors["condition"].index_select(0, diagnostic_index),
            train_tensors["active"].index_select(0, diagnostic_index),
            diagnostic_noise,
            alpha_bar,
            true_operator=operators["transition_true"],
            wrong_operator=operators["transition_wrong"],
            dct_operator=operators["orthogonal_dct"],
            steps=5,
        )
        if not torch.equal(sample_first.level, sample_second.level):
            raise RuntimeError(f"sampler replay mismatch for {path_id}")
        active_diagnostic = train_tensors["active"].index_select(
            0, diagnostic_index
        )
        if bool((sample_first.level[~active_diagnostic] != 0.0).any()):
            raise RuntimeError(f"inactive sampler value moved for {path_id}")
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        maximum_peak = max(maximum_peak, peak)
        path_results[path_id] = {
            "updates": updates,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_max": max(losses),
            "gradient_norm_max": max(gradient_norms),
            "final_model_sha256_before_discard": module_state_sha256(system),
            "final_EMA_sha256_before_discard": _ema_hash(ema),
            "optimizer_state_finite": _optimizer_finite(optimizer),
            "resume": resume_audit,
            "sampler_replay_exact": True,
            "sampler_finite": bool(torch.isfinite(sample_first.level).all()),
            "sampler_inactive_max_abs": (
                float(sample_first.level[~active_diagnostic].abs().max())
                if bool((~active_diagnostic).any())
                else 0.0
            ),
            "peak_memory_allocated_bytes": peak,
            "weights_retained": False,
        }
        del system, optimizer, ema, sample_first, sample_second
        if device.type == "cuda":
            torch.cuda.empty_cache()

    memory_limit = int(float(p0["maximum_peak_GPU_memory_GiB"]) * 1024**3)
    if device.type == "cuda" and maximum_peak > memory_limit:
        raise RuntimeError("TGO-v1 P0 exceeded the registered GPU-memory limit")

    outer_train = torch.from_numpy(fold.outer_train).long()
    observation = torch.from_numpy(bundle.train.target).float().index_select(
        0, outer_train
    )
    observed_mask = torch.from_numpy(bundle.train.observed_mask).bool().index_select(
        0, outer_train
    )
    raw_condition = torch.from_numpy(bundle.train.condition).float().index_select(
        0, outer_train
    )
    atom_audit = _atom_semantic_audit(
        raw_condition,
        observation,
        observed_mask,
        seed=51000 + fold_index,
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": "TGO_V1_P0_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        **identity,
        "code_sha256": {
            "transition_object": _sha256(
                ROOT / "architecture_v1" / "transition_object.py"
            ),
            "transition_probe": _sha256(
                ROOT / "architecture_v1" / "transition_probe.py"
            ),
            "transition_atom": _sha256(
                ROOT / "architecture_v1" / "transition_atom.py"
            ),
            "runner": _sha256(RUNNER_PATH),
        },
        "role_access": {
            "materialized_roles": list(bundle.materialized_roles),
            "training_update_target_scope": "fold_0_outer_train_only",
            "outer_test_TGO_metrics_constructed": False,
            "validation_bank_constructed": False,
        },
        "device": device_metadata,
        "nuisance_replay": nuisance_replay,
        "fold": {
            "outer_fold": fold_index,
            "outer_train_days": int(len(fold.outer_train)),
            "outer_test_days_not_used_for_TGO_metrics": int(len(fold.outer_test)),
            "level_RMS": level_rms,
        },
        "schedule": schedule_audit,
        "operator_audit": operator_audit,
        "atom_semantic_audit": atom_audit,
        "pairing": {
            "random_bank_seed": int(
                config["randomness_contract"]["P0_bank_seed"]
            ),
            "random_bank_sha256": bank_hash,
            "common_initial_model_sha256": next(iter(initial_hashes.values())),
            "all_seven_initial_model_hashes_equal": True,
            "all_paths_updates": updates,
            "total_optimizer_updates": updates * len(PATH_IDS),
        },
        "paths": path_results,
        "hard_gates": {
            "operator_inverse": True,
            "inactive_isolation": True,
            "wrong_adjacency_destruction": True,
            "matched_noise_equivalence": True,
            "full_grid_DDIM_equivalence": True,
            "common_initialization_and_random_bank": True,
            "finite_loss_gradient_optimizer_EMA_sampler": True,
            "checkpoint_resume_exact_all_paths": True,
            "common_atom_semantics": True,
            "no_held_out_TGO_metric": True,
            "no_weight_or_checkpoint_retained": True,
            "GPU_memory_within_limit": (
                maximum_peak <= memory_limit if device.type == "cuda" else None
            ),
        },
        "maximum_peak_memory_allocated_bytes": maximum_peak,
        "artifacts": {
            "JSON_and_SHA256_only": True,
            "weights_written": False,
            "checkpoint_written": False,
        },
        "interpretation": "P0 validates execution only and provides no path ranking or scientific utility evidence",
    }
    digest = _atomic_json(output_path, result)
    return result, output_path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-p0", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not args.execute_p0:
        print(json.dumps(dry_run(config_path), indent=2, sort_keys=True))
        return 0
    result, path, digest = execute_p0(
        config_path,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
        output_path=(args.output.resolve() if args.output is not None else None),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "path": str(path),
                "sha256": digest,
                "device": result["device"]["resolved"],
                "total_optimizer_updates": result["pairing"][
                    "total_optimizer_updates"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
