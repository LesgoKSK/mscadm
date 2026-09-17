#!/usr/bin/env python3
"""Evaluate all frozen G0-B final-EMA checkpoints on one common outer bank.

The default mode is target-free and verifies the 324-of-324 training freeze.
Only ``--execute-evaluation`` materializes the train role and uses each outer
fold's held-out subset.  Validation, calibration, selection, R-SEEN, and final
targets are never constructed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"
EVALUATION_ROOT_NAME = "formal_evaluation"
RESULT_SCHEMA = "architecture_v1_g0_b_tiny_denoiser_result_v1"
DAILY_SCHEMA = "architecture_v1_g0_b_tiny_denoiser_daily_arrays_v1"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "architecture_v1" / "g0b_evaluation.py"

from architecture_v1.g0b_evaluation import (
    CONTRAST_IDS,
    ENDPOINTS,
    adjudicate,
    balanced_reconstruction_risk,
    learning_curve_aulc,
    month_cluster_max_t_bands,
    relative_benefit,
    sample_reconstruction_metrics,
)
from architecture_v1.g0b_tiny_denoiser import (
    PATH_IDS,
    ModeProjectorBank,
    TinyDenoisingSystem,
    tensor_mapping_sha256,
    tiny_denoising_loss,
)
from architecture_v1.protocol import date_list_sha256
from repro_scripts import run_architecture_v1_g0_b_tiny_denoiser as p0_runner
from repro_scripts import run_architecture_v1_g0_b_tiny_denoiser_formal as formal


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype(str).tolist()
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        if isinstance(value, np.datetime64):
            return str(value)
        return value.item()
    if isinstance(value, Path):
        return str(value)
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
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _training_freeze_audit(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    p0_hashes = p0_runner._validate_config(config, config_path)
    if tuple(config["inference"]["confirmatory_contrasts"]) != CONTRAST_IDS:
        raise RuntimeError("formal contrast registry drifted")
    formal_root = ROOT / config["output_root"] / formal.FORMAL_ROOT_NAME
    freeze_path = formal_root / config["planned_implementation_files"]["training_freeze"]
    freeze_sha = formal._verified_sidecar(freeze_path)
    freeze = _read_json(freeze_path)
    if (
        freeze.get("schema") != formal.FREEZE_SCHEMA
        or freeze.get("status")
        != "G0_B_TINY_DENOISER_324_OF_324_TRAINING_FROZEN"
        or freeze.get("training_closed") is not True
        or freeze.get("outer_test_reconstruction_evaluation_authorized") is not True
        or freeze.get("outer_test_reconstruction_evaluation_performed") is not False
        or int(freeze.get("expected_runs", -1)) != 324
        or int(freeze.get("completed_runs", -1)) != 324
    ):
        raise RuntimeError("G0-B training freeze does not authorize evaluation")
    identity_path = formal_root / "training.identity.json"
    identity_sha = formal._verified_sidecar(identity_path)
    if identity_sha != freeze["global_training_identity_sha256"]:
        raise RuntimeError("training identity/freeze hash mismatch")
    matrix = formal._matrix(config)
    registry = freeze["final_EMA_checkpoint_sha256_by_run"]
    if set(registry) != {spec.key for spec in matrix}:
        raise RuntimeError("training freeze checkpoint registry is incomplete")
    for spec in matrix:
        run_root = formal._run_root(formal_root, spec)
        completion_path = run_root / "run.json"
        completion_sha = formal._verified_sidecar(completion_path)
        completion = _read_json(completion_path)
        final_path = run_root / "final_ema.pt"
        actual_sha = formal._verified_sidecar(final_path)
        if (
            completion.get("run_key") != spec.key
            or completion.get("status") != "complete"
            or completion.get("final_EMA_checkpoint_sha256") != actual_sha
            or registry[spec.key] != actual_sha
            or completion.get("outer_test_reconstruction_evaluation_performed")
            is not False
        ):
            raise RuntimeError(f"frozen run identity drifted: {spec.key}")
        if not completion_sha:
            raise RuntimeError("unreachable empty completion digest")
    if list(formal_root.rglob("resume.pt")):
        raise RuntimeError("training freeze contains a residual resume checkpoint")
    if list(formal_root.rglob("failure.json")):
        raise RuntimeError("training freeze contains a failure record")
    return {
        "config_sha256": p0_hashes["config"],
        "training_freeze_path": str(freeze_path.resolve()),
        "training_freeze_sha256": freeze_sha,
        "training_identity_path": str(identity_path.resolve()),
        "training_identity_sha256": identity_sha,
        "verified_runs": len(matrix),
        "verified_final_EMA_checkpoints": len(registry),
        "matrix": matrix,
        "formal_root": formal_root,
        "freeze": freeze,
    }


def dry_run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    audit = _training_freeze_audit(config, config_path)
    return {
        "schema": "architecture_v1_g0_b_tiny_evaluation_dry_run_v1",
        "mode": "target_free_no_evaluation_bank_constructed",
        "config_sha256": audit["config_sha256"],
        "training_freeze_sha256": audit["training_freeze_sha256"],
        "verified_runs": audit["verified_runs"],
        "verified_final_EMA_checkpoints": audit[
            "verified_final_EMA_checkpoints"
        ],
        "target_roles_materialized": [],
        "evaluation_noise_seed_formula": "34300 + outer_fold",
        "evaluation_noise_shape_order": "outer_test_day,timestep,replicate,site,hour",
        "timesteps": list(config["evaluation_bank"]["timesteps"]),
        "noise_replicates": int(
            config["evaluation_bank"]["gaussian_noise_replicates_per_day_timestep"]
        ),
        "confirmatory_contrasts": list(CONTRAST_IDS),
        "bootstrap_replicates": int(config["inference"]["bootstrap_replicates"]),
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "requires_execute_evaluation": True,
    }


def _noise_banks(
    config: Mapping[str, Any], folds: Sequence[Any]
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    timesteps = tuple(int(value) for value in config["evaluation_bank"]["timesteps"])
    replicates = int(
        config["evaluation_bank"]["gaussian_noise_replicates_per_day_timestep"]
    )
    seed_root = int(config["evaluation_bank"]["evaluation_noise_seed_root"])
    banks: dict[int, torch.Tensor] = {}
    per_fold: dict[str, Any] = {}
    for fold in folds:
        generator = torch.Generator(device="cpu")
        seed = seed_root + int(fold.fold)
        generator.manual_seed(seed)
        value = torch.randn(
            len(fold.outer_test),
            len(timesteps),
            replicates,
            10,
            24,
            generator=generator,
            dtype=torch.float32,
        )
        banks[int(fold.fold)] = value
        per_fold[str(fold.fold)] = {
            "seed": seed,
            "shape": list(value.shape),
            "sha256": tensor_mapping_sha256({"noise": value}),
        }
    return banks, {
        "seed_root": seed_root,
        "seed_formula": "evaluation_noise_seed_root + outer_fold",
        "shape_order": "outer_test_day,timestep,replicate,site,hour",
        "timesteps": list(timesteps),
        "replicates": replicates,
        "per_fold": per_fold,
        "global_sha256": tensor_mapping_sha256(
            {f"fold_{fold}": value for fold, value in banks.items()}
        ),
    }


def _load_ema_system(
    config: Mapping[str, Any],
    spec: formal.RunSpec,
    formal_root: Path,
    freeze_registry: Mapping[str, str],
    device: torch.device,
) -> TinyDenoisingSystem:
    run_root = formal._run_root(formal_root, spec)
    completion = _read_json(run_root / "run.json")
    checkpoint = run_root / "final_ema.pt"
    if formal._verified_sidecar(checkpoint) != freeze_registry[spec.key]:
        raise RuntimeError(f"checkpoint/freeze hash mismatch: {spec.key}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != formal.FINAL_EMA_SCHEMA
        or payload.get("run_identity_sha256")
        != completion["run_identity_sha256"]
        or payload.get("EMA_system_state_sha256")
        != completion["final_EMA_system_state_sha256"]
        or int(payload.get("completed_updates", -1)) != 1024
        or int(payload.get("EMA_updates", -1)) != 1024
        or payload.get("outer_test_reconstruction_evaluation_performed") is not False
    ):
        raise RuntimeError(f"final EMA payload drifted: {spec.key}")
    state = payload["EMA_system_state"]
    if tensor_mapping_sha256(state) != payload["EMA_system_state_sha256"]:
        raise RuntimeError(f"final EMA tensor hash drifted: {spec.key}")
    system = TinyDenoisingSystem(spec.path, model_seed=spec.model_seed).to(device)
    system.load_state_dict(state, strict=True)
    system.eval()
    return system


@torch.inference_mode()
def _evaluate_one(
    *,
    system: TinyDenoisingSystem,
    fold: Any,
    groups: Sequence[Any],
    noise_bank: torch.Tensor,
    timesteps: Sequence[int],
    baseline_alpha_numpy: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    shuffle_permutations: Sequence[np.ndarray],
    batch_size: int = 1024,
) -> tuple[dict[str, np.ndarray], float]:
    days = len(fold.outer_test)
    timestep_count = len(timesteps)
    replicates = noise_bank.shape[2]
    expected_shape = (days, timestep_count, replicates, 10, 24)
    if tuple(noise_bank.shape) != expected_shape:
        raise RuntimeError("evaluation noise bank shape drifted")
    residual = torch.as_tensor(fold.residual_test, dtype=torch.float32, device=device)
    active = torch.as_tensor(fold.active_test, dtype=torch.bool, device=device)
    condition = torch.as_tensor(
        fold.condition_test, dtype=torch.float32, device=device
    )
    variance = torch.as_tensor(
        fold.variance_test, dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(fold.weights_test, dtype=torch.float32, device=device)
    effective = torch.as_tensor(
        fold.effective_test, dtype=torch.float32, device=device
    )
    fixed = torch.as_tensor(fold.fixed_variance, dtype=torch.float32, device=device)
    baseline_alpha = torch.as_tensor(
        baseline_alpha_numpy, dtype=torch.float32, device=device
    )
    projectors = ModeProjectorBank(groups).to(device)
    outer_train_second_moment = float(
        np.square(fold.residual_train[fold.active_train]).mean()
    )
    if not np.isfinite(outer_train_second_moment) or outer_train_second_moment <= 0:
        raise RuntimeError("outer-train active residual second moment is invalid")

    bank_samples = days * timestep_count * replicates
    day_numpy = np.repeat(np.arange(days, dtype=np.int64), timestep_count * replicates)
    timestep_numpy = np.tile(
        np.repeat(np.asarray(timesteps, dtype=np.int64), replicates), days
    )
    noise_flat = noise_bank.reshape(bank_samples, 10, 24)
    permutations: tuple[np.ndarray | None, ...]
    if system.path_id == "PA_SHUFFLE":
        permutations = tuple(np.asarray(value, dtype=np.int64) for value in shuffle_permutations)
        if len(permutations) != 4:
            raise RuntimeError("PA_SHUFFLE evaluation requires four derangements")
    else:
        permutations = (None,)

    names = (*ENDPOINTS, "six_group_MSE", "gaussian_reference_risk")
    result = {
        name: np.zeros((days, 6), dtype=np.float64)
        if name == "six_group_MSE"
        else np.zeros(days, dtype=np.float64)
        for name in names
    }
    maximum_budget_error = 0.0
    for permutation in permutations:
        sample_values = {
            name: np.empty((bank_samples, 6), dtype=np.float64)
            if name == "six_group_MSE"
            else np.empty(bank_samples, dtype=np.float64)
            for name in names
        }
        permutation_tensor = (
            torch.as_tensor(permutation, dtype=torch.long, device=device)
            if permutation is not None
            else None
        )
        for start in range(0, bank_samples, batch_size):
            stop = min(start + batch_size, bank_samples)
            day = torch.as_tensor(
                day_numpy[start:stop], dtype=torch.long, device=device
            )
            timestep = torch.as_tensor(
                timestep_numpy[start:stop], dtype=torch.long, device=device
            )
            shuffle_proxy = (
                variance[permutation_tensor[day]]
                if permutation_tensor is not None
                else None
            )
            sample = tiny_denoising_loss(
                system,
                projectors,
                residual[day],
                active[day],
                condition[day],
                variance[day],
                weights[day],
                fixed,
                timestep,
                noise_flat[start:stop].to(device),
                baseline_alpha,
                shuffle_proxy=shuffle_proxy,
                eta=float(config["shared_proxy_allocation"]["eta"]),
            )
            metrics = sample_reconstruction_metrics(
                sample.prediction,
                residual[day],
                active[day],
                projectors,
                effective[day],
                outer_train_second_moment=outer_train_second_moment,
            )
            reference = torch.sum(
                weights[day]
                * variance[day]
                * torch.exp(-2.0 * sample.schedule.information),
                dim=1,
            )
            allocated = torch.sum(
                weights[day] * sample.schedule.information, dim=1
            )
            maximum_budget_error = max(
                maximum_budget_error,
                float(
                    torch.max(
                        torch.abs(allocated - sample.schedule.baseline_budget)
                    ).detach().cpu()
                ),
            )
            if not bool(torch.isfinite(reference).all()) or bool((reference <= 0).any()):
                raise FloatingPointError("Gaussian reference risk became invalid")
            for name in ENDPOINTS:
                sample_values[name][start:stop] = (
                    metrics[name].detach().cpu().numpy().astype(np.float64)
                )
            sample_values["six_group_MSE"][start:stop] = (
                metrics["six_group_MSE"].detach().cpu().numpy().astype(np.float64)
            )
            sample_values["gaussian_reference_risk"][start:stop] = (
                reference.detach().cpu().numpy().astype(np.float64)
            )
        for name, value in sample_values.items():
            trailing = (6,) if name == "six_group_MSE" else ()
            reshaped = value.reshape(days, timestep_count, replicates, *trailing)
            result[name] += reshaped.mean(axis=(1, 2)) / len(permutations)

    dimension_weighted = np.sum(
        fold.weights_test * result["six_group_MSE"], axis=1
    )
    result["oracle_efficiency_ratio"] = (
        dimension_weighted / result["gaussian_reference_risk"]
    )
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("held-out reconstruction metrics became non-finite")
    if any(np.any(value < 0.0) for value in result.values()):
        raise FloatingPointError("held-out reconstruction metric became negative")
    return result, maximum_budget_error


def _contrast_contributions(
    *,
    balanced: np.ndarray,
    aulc: np.ndarray,
    oracle: np.ndarray,
    endpoints: np.ndarray,
    path_index: Mapping[str, int],
    middle_fraction: int,
) -> np.ndarray:
    iid = path_index["IID"]
    fixed = path_index["FIXED_BAND"]
    cw = path_index["CW_GROUP"]
    mulan = path_index["MULAN_LITE"]
    pa = path_index["PA_RWF"]
    shuffle = path_index["PA_SHUFFLE"]
    values = (
        relative_benefit(balanced[:, iid, middle_fraction], balanced[:, pa, middle_fraction]),
        relative_benefit(balanced[:, fixed, middle_fraction], balanced[:, pa, middle_fraction]),
        relative_benefit(balanced[:, cw, middle_fraction], balanced[:, pa, middle_fraction]),
        relative_benefit(balanced[:, shuffle, middle_fraction], balanced[:, pa, middle_fraction]),
        relative_benefit(balanced[:, mulan, middle_fraction], balanced[:, pa, middle_fraction]),
        relative_benefit(aulc[:, mulan], aulc[:, pa]),
        relative_benefit(oracle[:, iid, middle_fraction], oracle[:, pa, middle_fraction]),
        relative_benefit(
            endpoints[:, iid, middle_fraction, 0],
            endpoints[:, pa, middle_fraction, 0],
        ),
        relative_benefit(
            endpoints[:, iid, middle_fraction, 1],
            endpoints[:, pa, middle_fraction, 1],
        ),
        relative_benefit(
            endpoints[:, iid, middle_fraction, 2],
            endpoints[:, pa, middle_fraction, 2],
        ),
        relative_benefit(balanced[:, iid, middle_fraction], balanced[:, fixed, middle_fraction]),
        relative_benefit(balanced[:, iid, middle_fraction], balanced[:, cw, middle_fraction]),
        relative_benefit(balanced[:, iid, middle_fraction], balanced[:, mulan, middle_fraction]),
    )
    return np.column_stack(values)


def _aggregate_table(
    *,
    paths: Sequence[str],
    fractions: Sequence[float],
    endpoints: np.ndarray,
    group: np.ndarray,
    gaussian: np.ndarray,
    oracle: np.ndarray,
    balanced: np.ndarray,
    aulc: np.ndarray,
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for path_index, path in enumerate(paths):
        rows = []
        for fraction_index, fraction in enumerate(fractions):
            rows.append(
                {
                    "fraction": float(fraction),
                    **{
                        name: float(endpoints[:, path_index, fraction_index, endpoint].mean())
                        for endpoint, name in enumerate(ENDPOINTS)
                    },
                    "six_group_MSE": group[:, path_index, fraction_index].mean(axis=0),
                    "gaussian_reference_risk": float(
                        gaussian[:, path_index, fraction_index].mean()
                    ),
                    "oracle_efficiency_ratio": float(
                        oracle[:, path_index, fraction_index].mean()
                    ),
                    "balanced_reconstruction_risk": float(
                        balanced[:, path_index, fraction_index].mean()
                    ),
                }
            )
        table[path] = {
            "by_fraction": rows,
            "learning_curve_AULC": float(aulc[:, path_index].mean()),
        }
    return table


def execute_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    *,
    device_name: str = "auto",
    allow_cpu: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    freeze_audit = _training_freeze_audit(config, config_path)
    output_root = ROOT / config["output_root"] / EVALUATION_ROOT_NAME
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("formal evaluation output is non-empty; overwrite is forbidden")
    output_root.mkdir(parents=True, exist_ok=True)
    device, runtime = formal._configure_device(device_name, allow_cpu=allow_cpu)
    started = datetime.now(timezone.utc)

    # This is the first target-materializing operation in the evaluation path.
    bundle, groups, folds, nuisance_audit, subset_replay = p0_runner._replay_nuisance(
        config
    )
    if len(bundle.train) != 267 or tuple(bundle.materialized_roles) != ("train",):
        raise RuntimeError("formal evaluation materialized an unexpected target role")
    if max(nuisance_audit["maximum_absolute_errors"].values()) > 1e-10:
        raise RuntimeError("G0-A nuisance replay exceeded its frozen tolerance")
    baseline_alpha, baseline_identity = p0_runner._baseline_schedule(config)
    schedule_audit, shuffle_registry = p0_runner._pretraining_schedule_audits(
        config, folds, baseline_alpha, bundle.train.day
    )
    noise_banks, noise_identity = _noise_banks(config, folds)
    timesteps = tuple(int(value) for value in config["evaluation_bank"]["timesteps"])
    paths = tuple(PATH_IDS)
    seeds = tuple(int(value) for value in config["training"]["model_seeds"])
    fractions = tuple(float(value) for value in config["learning_curve"]["fractions"])
    path_index = {value: index for index, value in enumerate(paths)}
    seed_index = {value: index for index, value in enumerate(seeds)}
    fraction_index = {value: index for index, value in enumerate(fractions)}
    shape = (len(bundle.train), len(paths), len(seeds), len(fractions))
    raw = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (*ENDPOINTS, "gaussian_reference_risk", "oracle_efficiency_ratio")
    }
    group_raw = np.full((*shape, 6), np.nan, dtype=np.float64)
    fold_by_day = np.full(len(bundle.train), -1, dtype=np.int64)
    maximum_float32_budget_error = 0.0
    matrix: Sequence[formal.RunSpec] = freeze_audit["matrix"]
    formal_root: Path = freeze_audit["formal_root"]
    freeze_registry = freeze_audit["freeze"]["final_EMA_checkpoint_sha256_by_run"]
    for run_number, spec in enumerate(matrix, start=1):
        fold = folds[spec.outer_fold]
        fold_by_day[fold.outer_test] = spec.outer_fold
        system = _load_ema_system(
            config, spec, formal_root, freeze_registry, device
        )
        values, budget_error = _evaluate_one(
            system=system,
            fold=fold,
            groups=groups,
            noise_bank=noise_banks[spec.outer_fold],
            timesteps=timesteps,
            baseline_alpha_numpy=baseline_alpha,
            config=config,
            device=device,
            shuffle_permutations=shuffle_registry[
                f"fold_{spec.outer_fold}_evaluation"
            ],
        )
        maximum_float32_budget_error = max(
            maximum_float32_budget_error, budget_error
        )
        index = (
            fold.outer_test,
            path_index[spec.path],
            seed_index[spec.model_seed],
            fraction_index[spec.fraction],
        )
        for name in raw:
            raw[name][index] = values[name]
        group_raw[index] = values["six_group_MSE"]
        del system
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if run_number % 12 == 0 or run_number == len(matrix):
            print(
                f"evaluated {run_number}/{len(matrix)} frozen EMA runs",
                file=sys.stderr,
                flush=True,
            )

    if np.any(fold_by_day < 0):
        raise RuntimeError("not every out-of-fold day received a fold identity")
    if not all(np.isfinite(value).all() for value in (*raw.values(), group_raw)):
        raise FloatingPointError("formal evaluation matrix is incomplete or non-finite")
    if len(np.unique(bundle.train.day)) != 267 or not np.all(
        np.diff(bundle.train.day.astype("datetime64[D]").astype(np.int64)) > 0
    ):
        raise RuntimeError("out-of-fold calendar-day registry is not unique chronological")

    endpoint_by_seed = np.stack([raw[name] for name in ENDPOINTS], axis=-1)
    iid = path_index["IID"]
    iid_denominators = endpoint_by_seed[:, iid].mean(axis=(0, 1))
    # Reorder to [day,path,seed,fraction,endpoint], whose last two axes are
    # exactly the normalization contract accepted by the metric function.
    balanced_by_seed = balanced_reconstruction_risk(
        endpoint_by_seed, iid_denominators
    )
    endpoints = endpoint_by_seed.mean(axis=2)
    group = group_raw.mean(axis=2)
    gaussian = raw["gaussian_reference_risk"].mean(axis=2)
    oracle = raw["oracle_efficiency_ratio"].mean(axis=2)
    balanced = balanced_by_seed.mean(axis=2)
    aulc = learning_curve_aulc(balanced, fractions)
    middle = fractions.index(float(config["learning_curve"]["primary_fraction"]))
    contributions = _contrast_contributions(
        balanced=balanced,
        aulc=aulc,
        oracle=oracle,
        endpoints=endpoints,
        path_index=path_index,
        middle_fraction=middle,
    )
    inference = config["inference"]
    bands = month_cluster_max_t_bands(
        contributions,
        bundle.train.day,
        repetitions=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
        confidence=float(inference["confidence_level"]),
    )
    contrast_table: dict[str, dict[str, float]] = {}
    for index, name in enumerate(CONTRAST_IDS):
        contrast_table[name] = {
            "estimate": float(bands["estimate"][index]),
            "standard_error": float(bands["standard_error"][index]),
            "simultaneous_low": float(bands["simultaneous_low"][index]),
            "simultaneous_high": float(bands["simultaneous_high"][index]),
        }

    pa = path_index["PA_RWF"]
    fold_effects = {
        str(fold): float(
            relative_benefit(
                balanced[fold_by_day == fold, iid, middle],
                balanced[fold_by_day == fold, pa, middle],
            ).mean()
        )
        for fold in range(6)
    }
    seed_effects = {
        str(seed): float(
            relative_benefit(
                balanced_by_seed[:, iid, seed_index[seed], middle],
                balanced_by_seed[:, pa, seed_index[seed], middle],
            ).mean()
        )
        for seed in seeds
    }
    positive_folds = sum(value > 0.0 for value in fold_effects.values())
    positive_seeds = sum(value > 0.0 for value in seed_effects.values())
    decision = adjudicate(
        contrast_table,
        positive_outer_folds=positive_folds,
        positive_model_seeds=positive_seeds,
        technical_eligibility=True,
        thresholds=config["go_no_go"],
    )

    optional_contribution = relative_benefit(
        balanced[:, iid, fraction_index[1.0]],
        balanced[:, pa, middle],
    )
    optional_band = month_cluster_max_t_bands(
        optional_contribution[:, None],
        bundle.train.day,
        repetitions=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
        confidence=float(inference["confidence_level"]),
    )
    optional = {
        "contrast": "PA_RWF_fraction_0.5_vs_IID_fraction_1.0",
        "noninferiority_margin": 0.01,
        "estimate": float(optional_band["estimate"][0]),
        "lower": float(optional_band["simultaneous_low"][0]),
        "upper": float(optional_band["simultaneous_high"][0]),
        "noninferior": bool(optional_band["simultaneous_low"][0] >= -0.01),
        "required_for_GO": False,
    }

    daily_path = output_root / "G0_B_TINY_DAILY_METRICS.npz"
    daily_sha = _atomic_npz(
        daily_path,
        {
            "schema": np.asarray(DAILY_SCHEMA),
            "days": bundle.train.day.astype("datetime64[D]"),
            "outer_fold": fold_by_day,
            "paths": np.asarray(paths),
            "model_seeds": np.asarray(seeds, dtype=np.int64),
            "fractions": np.asarray(fractions, dtype=np.float64),
            "cell_MSE_by_seed": raw["cell_MSE"],
            "increment_MSE_by_seed": raw["increment_MSE"],
            "joint_day_normalized_SSE_by_seed": raw[
                "joint_day_normalized_SSE"
            ],
            "six_group_MSE_by_seed": group_raw,
            "gaussian_reference_risk_by_seed": raw[
                "gaussian_reference_risk"
            ],
            "oracle_efficiency_ratio_by_seed": raw["oracle_efficiency_ratio"],
            "IID_endpoint_denominators": iid_denominators,
            "balanced_reconstruction_risk_by_seed": balanced_by_seed,
            "balanced_reconstruction_risk": balanced,
            "learning_curve_AULC": aulc,
            "confirmatory_daily_contributions": contributions,
        },
    )
    finished = datetime.now(timezone.utc)
    result = {
        "schema": RESULT_SCHEMA,
        "status": decision["status"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "evidence_scope": (
            "267 train-role outer-held-out calendar days; tiny-denoiser reconstruction "
            "utility only, not validation, scenario generation, or family selection"
        ),
        "identity": {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": freeze_audit["config_sha256"],
            "training_freeze_path": freeze_audit["training_freeze_path"],
            "training_freeze_sha256": freeze_audit["training_freeze_sha256"],
            "training_identity_sha256": freeze_audit["training_identity_sha256"],
            "evaluation_module_sha256": _sha256(MODULE_PATH),
            "evaluation_runner_sha256": _sha256(RUNNER_PATH),
            "repository_execution_head": _git_head(),
            "daily_metrics_path": str(daily_path.resolve()),
            "daily_metrics_sha256": daily_sha,
        },
        "target_access": {
            "materialized_roles": ["train"],
            "evaluated_role": "train_outer_held_out",
            "validation_accessed": False,
            "calibration_accessed": False,
            "selection_accessed": False,
            "r_seen_accessed": False,
            "final_accessed": False,
            "selection_state": "sealed",
            "calibration_state": "sealed",
        },
        "runtime": runtime,
        "technical_audit": {
            "training_freeze_verified": True,
            "verified_final_EMA_checkpoints": freeze_audit[
                "verified_final_EMA_checkpoints"
            ],
            "out_of_fold_days": len(bundle.train),
            "outer_folds": len(folds),
            "model_runs_evaluated": len(matrix),
            "all_18_training_subset_hashes_replayed": len(subset_replay) == 18,
            "nuisance_replay": nuisance_audit,
            "baseline_schedule": baseline_identity,
            "float64_schedule_audit": schedule_audit,
            "maximum_float32_evaluation_budget_abs_error": maximum_float32_budget_error,
            "all_metrics_finite": True,
            "common_noise_bank_verified": True,
            "no_checkpoint_selection": True,
        },
        "evaluation_bank": noise_identity,
        "aggregation": {
            "within_day": (
                "equal mean over 16 timesteps and four noise replicates; "
                "PA_SHUFFLE also equal-means four registered evaluation derangements"
            ),
            "model_seed": "equal mean over seeds 3,4,5 before formal contrasts",
            "fold": "concatenate 267 unique out-of-fold calendar days",
            "IID_endpoint_denominators_by_fraction": {
                str(fraction): {
                    name: float(iid_denominators[index, endpoint])
                    for endpoint, name in enumerate(ENDPOINTS)
                }
                for index, fraction in enumerate(fractions)
            },
        },
        "aggregate_metrics": _aggregate_table(
            paths=paths,
            fractions=fractions,
            endpoints=endpoints,
            group=group,
            gaussian=gaussian,
            oracle=oracle,
            balanced=balanced,
            aulc=aulc,
        ),
        "inference": {
            "method": "paired calendar-month cluster bootstrap with max-T simultaneous bands",
            "unit": "calendar_day",
            "clusters": int(bands["month_clusters"]),
            "replicates": int(bands["repetitions"]),
            "seed": int(bands["seed"]),
            "confidence": float(bands["confidence"]),
            "max_T_critical_value": float(bands["critical_value"]),
            "contrast_order": list(CONTRAST_IDS),
            "contrasts": contrast_table,
        },
        "stability": {
            "PA_vs_IID_balanced_risk_relative_improvement_by_outer_fold": fold_effects,
            "positive_outer_folds": positive_folds,
            "PA_vs_IID_balanced_risk_relative_improvement_by_model_seed": seed_effects,
            "positive_model_seeds": positive_seeds,
        },
        "optional_data_equivalence": optional,
        "decision": decision,
        "authorization": (
            config["go_no_go"]["GO_authorizes"]
            if decision["status"] == config["go_no_go"]["GO_status"]
            else "no full-model authorization; selection, calibration, and final remain sealed"
        ),
        "full_model_trained": False,
        "scenario_generation_performed": False,
        "flow_vs_diffusion_family_selection_performed": False,
    }
    result_path = output_root / config["planned_implementation_files"]["formal_result"]
    result_sha = _atomic_json(result_path, result)
    return {
        "status": result["status"],
        "all_GO_gates_passed": decision["all_GO_gates_passed"],
        "result_path": str(result_path.resolve()),
        "result_sha256": result_sha,
        "daily_metrics_sha256": daily_sha,
        "out_of_fold_days": len(bundle.train),
        "model_runs_evaluated": len(matrix),
        "outer_test_reconstruction_evaluation_performed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-evaluation", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow the formal bank to run on CPU when CUDA is unavailable",
    )
    arguments = parser.parse_args()
    if arguments.allow_cpu and not arguments.execute_evaluation:
        parser.error("--allow-cpu is valid only with --execute-evaluation")
    result = (
        execute_evaluation(
            arguments.config,
            device_name=arguments.device,
            allow_cpu=arguments.allow_cpu,
        )
        if arguments.execute_evaluation
        else dry_run(arguments.config)
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
