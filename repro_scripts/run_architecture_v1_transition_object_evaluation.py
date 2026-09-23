#!/usr/bin/env python3
"""Common physical-scale sampling and frozen evaluation for TGO-v1.

The default mode is target free.  ``--execute-sampling`` is authorized only
after the exact 84/84 training freeze.  It generates every registered
fold/seed/path/sampling-seed archive with common atom allocations and native
noise, writes per-day physical metrics, freezes all 336 archives, and only
then performs the preregistered calendar-month cluster inference.
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
    ROOT / "repro_configs" / "architecture_v1_transition_object_evaluation_v1_3.json"
)
RUNNER_PATH = Path(__file__).resolve()
IDENTITY_SCHEMA = "architecture_v1_tgo_v1_evaluation_identity_v1_3"
ARCHIVE_COMPLETION_SCHEMA = "architecture_v1_tgo_v1_archive_completion_v1_3"
SAMPLING_PROGRESS_SCHEMA = "architecture_v1_tgo_v1_sampling_progress_v1_3"
SAMPLING_FREEZE_SCHEMA = "architecture_v1_tgo_v1_sampling_freeze_v1_3"
RESULT_SCHEMA = "architecture_v1_tgo_v1_formal_result_v1_3"
COMPLETE_CASE_EXCLUDED_DAY = np.datetime64("2013-12-31", "D")

from architecture_v1.g0_predictability import (
    continuous_logit_target,
    fit_cellwise_ridge,
)
from architecture_v1.g0b_evaluation import month_cluster_max_t_bands
from architecture_v1.g0b_tiny_denoiser import tensor_mapping_sha256
from architecture_v1.protocol import date_list_sha256
from architecture_v1.transition_atom import TransitionAtomNuisance
from architecture_v1.transition_evaluation import (
    ARCHIVE_MANIFEST_SCHEMA,
    ARCHIVE_SCHEMA,
    METRIC_NAMES,
    fit_train_only_ramp_thresholds,
    load_transition_archive,
    transition_per_day_metrics,
    validate_transition_archive_arrays,
    write_transition_archive,
)
from architecture_v1.transition_object import (
    MaskConditionedOperator,
    fit_level_rms,
    fit_operator_scale,
)
from architecture_v1.transition_probe import (
    PATH_IDS,
    TransitionDenoisingSystem,
    sample_transition_ddim,
)
from repro_scripts import run_architecture_v1_transition_object_formal as training
from repro_scripts.run_architecture_v1_g0_b_tiny_denoiser import (
    _baseline_schedule,
    _read_json as _read_g0_json,
    _replay_nuisance,
)
from repro_scripts.run_architecture_v1_g0_b_tiny_denoiser_formal import (
    _array_sha256,
    _atomic_json,
    _canonical_sha256,
    _read_json,
    _sha256,
    _verified_sidecar,
)


@dataclass(frozen=True)
class ArchiveSpec:
    outer_fold: int
    model_seed: int
    path_id: str
    sampling_seed: int

    @property
    def run_key(self) -> str:
        return f"fold{self.outer_fold}__seed{self.model_seed}__{self.path_id}"

    @property
    def key(self) -> str:
        return f"{self.run_key}__sampling{self.sampling_seed}"

    def manifest(self) -> dict[str, Any]:
        return {
            "archive_key": self.key,
            "run_key": self.run_key,
            "outer_fold": self.outer_fold,
            "model_seed": self.model_seed,
            "path_id": self.path_id,
            "sampling_seed": self.sampling_seed,
        }


@dataclass(frozen=True)
class FoldContext:
    fold: Any
    level_rms: float
    operators: Mapping[str, MaskConditionedOperator]
    mean_test: np.ndarray
    ramp_thresholds: Mapping[str, float | int]
    condition_test: torch.Tensor
    observation: np.ndarray
    observed_mask: np.ndarray
    raw_missing_mask: np.ndarray
    day: np.ndarray
    zones: np.ndarray
    atom_model: TransitionAtomNuisance
    atom_checkpoint_sha256: str


@dataclass(frozen=True)
class CommonSamplingGroup:
    states: torch.Tensor
    active_mask: torch.Tensor
    zero_probability: np.ndarray
    one_probability: np.ndarray
    native_epsilon: torch.Tensor
    allocation_sha256: str
    probability_sha256: str
    epsilon_sha256: str


def _matrix(config: Mapping[str, Any]) -> tuple[ArchiveSpec, ...]:
    sampling = config["sampling"]
    specs = tuple(
        ArchiveSpec(int(fold), int(seed), str(path_id), int(sampling_seed))
        for fold in sampling["outer_folds"]
        for seed in sampling["model_seeds"]
        for path_id in sampling["paths"]
        for sampling_seed in sampling["sampling_seeds"]
    )
    if tuple(sampling["paths"]) != PATH_IDS:
        raise RuntimeError("TGO-v1 evaluation path registry drifted")
    if len(specs) != 336 or int(sampling["expected_archives"]) != len(specs):
        raise RuntimeError("TGO-v1 sampling matrix must contain 336 archives")
    if len({spec.key for spec in specs}) != len(specs):
        raise RuntimeError("TGO-v1 archive keys are not unique")
    return specs


def _validate_config(config: Mapping[str, Any], config_path: Path) -> dict[str, str]:
    digest = _verified_sidecar(config_path)
    if config.get("schema") != "architecture_v1_transition_object_evaluation_v1_3":
        raise RuntimeError("unexpected TGO-v1 evaluation schema")
    if config.get("status") != "frozen_before_physical_sampling":
        raise RuntimeError("TGO-v1 evaluation config is not frozen")
    paths = config["lineage"]
    for key, hash_key in (
        ("probe_config", "probe_config_sha256"),
        ("training_freeze", "training_freeze_sha256"),
        ("prior_evaluation_config", "prior_evaluation_config_sha256"),
        ("prior_technical_no_go", "prior_technical_no_go_sha256"),
        ("prior_v1_1_evaluation_config", "prior_v1_1_evaluation_config_sha256"),
        ("prior_v1_1_technical_no_go", "prior_v1_1_technical_no_go_sha256"),
        ("prior_v1_2_evaluation_config", "prior_v1_2_evaluation_config_sha256"),
        ("prior_v1_2_technical_no_go", "prior_v1_2_technical_no_go_sha256"),
    ):
        path = ROOT / paths[key]
        if _sha256(path) != paths[hash_key]:
            raise RuntimeError(f"TGO-v1 evaluation lineage drifted: {key}")
    prior = _read_json(ROOT / paths["prior_technical_no_go"])
    if (
        prior.get("status") != "TGO_V1_EVALUATION_FP32_ARCHIVE_CONTRACT_NO_GO"
        or prior.get("completed_archives_before_stop") != 24
        or prior.get("formal_inference_performed") is not False
        or prior.get("evaluation_config_sha256")
        != paths["prior_evaluation_config_sha256"]
    ):
        raise RuntimeError("TGO-v1.1 prior technical No-Go lineage drifted")
    prior_v1_1 = _read_json(ROOT / paths["prior_v1_1_technical_no_go"])
    if (
        prior_v1_1.get("status") != "TGO_V1_1_EVALUATION_FP64_INTERIOR_SATURATION_NO_GO"
        or prior_v1_1.get("completed_archives_before_stop") != 137
        or prior_v1_1.get("formal_inference_performed") is not False
        or prior_v1_1.get("evaluation_config_sha256")
        != paths["prior_v1_1_evaluation_config_sha256"]
    ):
        raise RuntimeError("TGO-v1.2 prior technical No-Go lineage drifted")
    prior_v1_2 = _read_json(ROOT / paths["prior_v1_2_technical_no_go"])
    if (
        prior_v1_2.get("status") != "TGO_V1_2_EVALUATION_UNDEFINED_LATE_HORIZON_METRIC_NO_GO"
        or prior_v1_2.get("completed_archives_before_stop") != 280
        or prior_v1_2.get("formal_inference_performed") is not False
        or prior_v1_2.get("evaluation_config_sha256")
        != paths["prior_v1_2_evaluation_config_sha256"]
    ):
        raise RuntimeError("TGO-v1.3 prior technical No-Go lineage drifted")
    code = {
        "transition_object": _sha256(ROOT / "architecture_v1" / "transition_object.py"),
        "transition_probe": _sha256(ROOT / "architecture_v1" / "transition_probe.py"),
        "transition_atom": _sha256(ROOT / "architecture_v1" / "transition_atom.py"),
        "transition_evaluation": _sha256(
            ROOT / "architecture_v1" / "transition_evaluation.py"
        ),
        "training_runner": _sha256(
            ROOT / "repro_scripts" / "run_architecture_v1_transition_object_formal.py"
        ),
        "evaluation_runner": _sha256(RUNNER_PATH),
    }
    if code != config["code_sha256"]:
        raise RuntimeError("TGO-v1 evaluation code changed after freeze")
    if tuple(config["metrics"]["registry"]) != METRIC_NAMES:
        raise RuntimeError("TGO-v1 metric registry drifted")
    if tuple(config["inference"]["endpoint_registry"]) != _endpoint_registry():
        raise RuntimeError("TGO-v1 inference endpoint registry drifted")
    sampling = config["sampling"]
    expected_sampling = {
        "outer_folds": list(range(6)),
        "model_seeds": [3, 4],
        "paths": list(PATH_IDS),
        "sampling_seeds": [61000, 61001, 61002, 61003],
        "expected_archives": 336,
        "members": 100,
        "member_chunk": 25,
        "DDIM_steps": 31,
        "eta": 0.0,
        "archive_dtype": "float64",
        "interior_encoding": "nextafter_only_if_finite_fp64_sigmoid_rounds_to_boundary",
        "post_hoc_clipping": "machine_nextafter_only",
    }
    if any(sampling.get(key) != value for key, value in expected_sampling.items()):
        raise RuntimeError("TGO-v1 physical sampling contract drifted")
    bootstrap = config["inference"]["bootstrap"]
    if bootstrap != {
        "cluster": "calendar_month",
        "confidence": 0.95,
        "repetitions": 10000,
        "seed": 62000,
        "simultaneous_method": "two_sided_joint_max_abs_studentized_T",
    }:
        raise RuntimeError("TGO-v1 bootstrap contract drifted")
    complete_case = config["inference"]["complete_case"]
    if complete_case != {
        "excluded_calendar_days": ["2013-12-31"],
        "expected_archive_calendar_days": 267,
        "expected_inference_calendar_days": 266,
        "rule": "exclude the same target-mask-defined day from every one of the 28 paired inference endpoints",
    }:
        raise RuntimeError("TGO-v1.3 complete-case contract drifted")
    _matrix(config)
    return {"config_sha256": digest, **code}


def _validate_training_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / config["lineage"]["training_freeze"]
    digest = _verified_sidecar(path)
    freeze = _read_json(path)
    if (
        freeze.get("schema") != training.FREEZE_SCHEMA
        or freeze.get("status") != "TGO_V1_84_OF_84_TRAINING_FROZEN"
        or int(freeze.get("completed_atom_models", -1)) != 6
        or int(freeze.get("completed_denoiser_runs", -1)) != 84
        or freeze.get("all_cross_path_pairing_checks_passed") is not True
        or freeze.get("only_final_EMA_checkpoints_retained") is not True
        or freeze.get("common_physical_sampling_authorized") is not True
        or freeze.get("outer_test_TGO_metrics_constructed") is not False
        or freeze.get("scenario_generation_performed") is not False
    ):
        raise RuntimeError("TGO-v1 training freeze does not authorize sampling")
    return {"path": str(path.resolve()), "sha256": digest, "payload": freeze}


def _configure_device(
    device_name: str, *, allow_cpu: bool
) -> tuple[torch.device, dict[str, Any]]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("formal TGO-v1 sampling on CPU requires --allow-cpu")
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
        "deterministic_algorithms": True,
        "automatic_mixed_precision": False,
    }


def _archive_root(evaluation_root: Path, spec: ArchiveSpec) -> Path:
    return (
        evaluation_root
        / "archives"
        / f"fold{spec.outer_fold}"
        / f"seed{spec.model_seed}"
        / spec.path_id
        / f"sampling{spec.sampling_seed}"
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _mean_test_prediction(
    *,
    bundle: Any,
    fold: Any,
    g0_config: Mapping[str, Any],
    g0_result: Mapping[str, Any],
) -> np.ndarray:
    latent, active = continuous_logit_target(
        bundle.train.target,
        bundle.train.state,
        bundle.train.observed_mask,
        epsilon=float(g0_config["continuous_target"]["logit_epsilon"]),
    )
    selected = {
        int(row["outer_fold"]): row
        for row in g0_result["selected_hyperparameters"]
    }[int(fold.fold)]
    model = fit_cellwise_ridge(
        bundle.train.raw_condition,
        latent,
        active,
        fold.outer_train,
        alpha=float(selected["mean_ridge_alpha"]),
        minimum_observations=int(
            g0_config["conditional_mean"]["minimum_fit_observations_per_cell"]
        ),
    )
    prediction = model.predict(bundle.train.raw_condition[fold.outer_test])
    reconstructed = np.where(
        fold.active_test,
        latent[fold.outer_test] - prediction,
        0.0,
    )
    error = float(np.max(np.abs(reconstructed - fold.residual_test)))
    if error > 1e-10:
        raise RuntimeError("TGO-v1 mean replay differs from frozen fold residual")
    return np.ascontiguousarray(prediction, dtype=np.float32)


def _load_atom_model(
    *,
    formal_root: Path,
    fold: int,
    expected_sha: str,
    device: torch.device,
) -> tuple[TransitionAtomNuisance, str]:
    path = formal_root / "atoms" / f"fold{fold}" / "final_ema.pt"
    digest = _verified_sidecar(path)
    if digest != expected_sha:
        raise RuntimeError(f"shared atom checkpoint hash drifted: fold {fold}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != training.ATOM_FINAL_SCHEMA
        or int(payload.get("completed_updates", -1)) != 4080
        or int(payload.get("EMA_updates", -1)) != 4080
        or tensor_mapping_sha256(payload["EMA_system_state"])
        != payload.get("EMA_system_state_sha256")
    ):
        raise RuntimeError(f"shared atom final EMA is invalid: fold {fold}")
    contract = payload["identity"]["train_only_contract"]
    model = TransitionAtomNuisance(
        fixed_one_probability=(
            None
            if contract["fixed_one_probability"] is None
            else float(contract["fixed_one_probability"])
        ),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    ).to(device)
    model.load_state_dict(payload["EMA_system_state"], strict=True)
    model.eval()
    return model, digest


def _build_fold_context(
    *,
    config: Mapping[str, Any],
    bundle: Any,
    fold: Any,
    g0_config: Mapping[str, Any],
    g0_result: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    formal_root: Path,
    device: torch.device,
) -> FoldContext:
    residual = torch.from_numpy(fold.residual_train).double()
    active = torch.from_numpy(fold.active_train).bool()
    level_rms = fit_level_rms(residual, active)
    level = residual / level_rms
    scales = {
        kind: fit_operator_scale(level, active, kind=kind)
        for kind in (
            "transition_true",
            "transition_wrong",
            "orthogonal_dct",
        )
    }
    operators = {
        kind: MaskConditionedOperator(kind, scale=value)
        for kind, value in scales.items()
    }
    observation_train = bundle.train.target[fold.outer_train]
    observed_train = bundle.train.observed_mask[fold.outer_train]
    threshold_config = config["metrics"]["ramp_thresholds"]
    thresholds = fit_train_only_ramp_thresholds(
        observation_train,
        observed_train,
        upper_quantile=float(threshold_config["upper_quantile"]),
        lower_quantile=float(threshold_config["lower_quantile"]),
    )
    atom_model, atom_sha = _load_atom_model(
        formal_root=formal_root,
        fold=int(fold.fold),
        expected_sha=training_freeze[
            "atom_final_EMA_checkpoint_sha256_by_fold"
        ][str(fold.fold)],
        device=device,
    )
    return FoldContext(
        fold=fold,
        level_rms=level_rms,
        operators=operators,
        mean_test=_mean_test_prediction(
            bundle=bundle,
            fold=fold,
            g0_config=g0_config,
            g0_result=g0_result,
        ),
        ramp_thresholds=thresholds,
        condition_test=torch.from_numpy(fold.condition_test).float().to(device),
        observation=np.ascontiguousarray(
            bundle.train.target[fold.outer_test], dtype=np.float32
        ),
        observed_mask=np.ascontiguousarray(
            bundle.train.observed_mask[fold.outer_test], dtype=bool
        ),
        raw_missing_mask=np.ascontiguousarray(
            bundle.train.raw_missing_mask[fold.outer_test], dtype=bool
        ),
        day=np.asarray(bundle.train.day[fold.outer_test], dtype="datetime64[D]"),
        zones=np.asarray(bundle.train.zones, dtype=np.int64),
        atom_model=atom_model,
        atom_checkpoint_sha256=atom_sha,
    )


@torch.no_grad()
def _common_sampling_group(
    context: FoldContext,
    *,
    bundle: Any,
    sampling_seed: int,
    members: int,
    device: torch.device,
) -> CommonSamplingGroup:
    raw_condition = torch.from_numpy(
        np.ascontiguousarray(
            bundle.train.condition[context.fold.outer_test], dtype=np.float32
        )
    ).to(device)
    statistics, allocation = context.atom_model.allocate(
        raw_condition,
        members=members,
        seed=int(sampling_seed) + 1,
    )
    states = allocation.states.detach().cpu()
    active = allocation.active_mask.detach().cpu()
    probabilities = statistics.probabilities.detach().cpu()
    probability_sha = tensor_mapping_sha256({"probabilities": probabilities})
    allocation_sha = tensor_mapping_sha256(
        {"states": states, "active_mask": active}
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sampling_seed) + 2)
    epsilon = torch.randn(
        (len(context.day), members, 10, 24),
        generator=generator,
        dtype=torch.float32,
    )
    epsilon_sha = tensor_mapping_sha256({"native_epsilon": epsilon})
    return CommonSamplingGroup(
        states=states,
        active_mask=active,
        zero_probability=probabilities[..., 0].numpy(),
        one_probability=probabilities[..., 2].numpy(),
        native_epsilon=epsilon,
        allocation_sha256=allocation_sha,
        probability_sha256=probability_sha,
        epsilon_sha256=epsilon_sha,
    )


def _load_denoiser(
    *,
    formal_root: Path,
    spec: ArchiveSpec,
    expected_sha: str,
    context: FoldContext,
    device: torch.device,
) -> tuple[TransitionDenoisingSystem, str]:
    path = (
        formal_root
        / "runs"
        / f"fold{spec.outer_fold}"
        / f"seed{spec.model_seed}"
        / spec.path_id
        / "final_ema.pt"
    )
    digest = _verified_sidecar(path)
    if digest != expected_sha:
        raise RuntimeError(f"denoiser checkpoint hash drifted: {spec.run_key}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != training.RUN_FINAL_SCHEMA
        or int(payload.get("completed_updates", -1)) != 1024
        or int(payload.get("EMA_updates", -1)) != 1024
        or tensor_mapping_sha256(payload["EMA_system_state"])
        != payload.get("EMA_system_state_sha256")
    ):
        raise RuntimeError(f"denoiser final EMA is invalid: {spec.run_key}")
    identity = payload["identity"]
    if (
        identity["path_id"] != spec.path_id
        or int(identity["outer_fold"]) != spec.outer_fold
        or int(identity["model_seed"]) != spec.model_seed
        or abs(float(identity["fold_training_data"]["level_RMS"]) - context.level_rms)
        > 1e-12
    ):
        raise RuntimeError(f"denoiser/fold sampling identity drifted: {spec.run_key}")
    for kind, operator in context.operators.items():
        if abs(
            float(identity["fold_training_data"]["operator_scales"][kind])
            - operator.scale
        ) > 1e-12:
            raise RuntimeError(f"operator scale drifted: {spec.run_key}/{kind}")
    system = TransitionDenoisingSystem(
        spec.path_id, model_seed=spec.model_seed
    ).to(device)
    system.load_state_dict(payload["EMA_system_state"], strict=True)
    system.eval()
    return system, digest


def _metric_path(root: Path) -> Path:
    return root / "metrics.npz"


def _write_metrics(
    path: Path,
    *,
    day: np.ndarray,
    metrics: Mapping[str, np.ndarray],
) -> str:
    if tuple(metrics) != METRIC_NAMES:
        raise RuntimeError("TGO-v1 metric write registry drifted")
    dates = np.asarray(day, dtype="datetime64[D]")
    late_valid = np.isfinite(metrics["late_horizon_level_CRPS"])
    if not np.array_equal(~late_valid, dates == COMPLETE_CASE_EXCLUDED_DAY):
        raise RuntimeError("TGO-v1.3 late-horizon valid-day contract drifted")
    return _atomic_npz(
        path,
        day=dates,
        late_horizon_valid=np.asarray(late_valid, dtype=bool),
        **{
            name: np.ascontiguousarray(metrics[name], dtype=np.float64)
            for name in METRIC_NAMES
        },
    )


def _load_metrics(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    digest = _verified_sidecar(path)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"day", "late_horizon_valid", *METRIC_NAMES}:
            raise RuntimeError("TGO-v1 metric artifact schema drifted")
        day = np.asarray(stored["day"], dtype="datetime64[D]")
        late_valid = np.asarray(stored["late_horizon_valid"], dtype=bool)
        metrics = {
            name: np.asarray(stored[name], dtype=np.float64)
            for name in METRIC_NAMES
        }
    if any(value.shape != day.shape for value in metrics.values()):
        raise RuntimeError("TGO-v1 metric/day arrays do not align")
    if late_valid.shape != day.shape or not np.array_equal(
        late_valid, day != COMPLETE_CASE_EXCLUDED_DAY
    ):
        raise RuntimeError("TGO-v1.3 late-horizon valid-day artifact drifted")
    if not np.array_equal(
        np.isfinite(metrics["late_horizon_level_CRPS"]), late_valid
    ):
        raise RuntimeError("TGO-v1.3 late-horizon missingness artifact drifted")
    if any(not np.isfinite(value).all() for name, value in metrics.items()
           if name != "late_horizon_level_CRPS"):
        raise RuntimeError("TGO-v1 metric artifact contains non-finite values")
    return day, metrics, digest


def _completion_payload(
    *,
    spec: ArchiveSpec,
    archive_path: Path,
    archive_sha: str,
    metric_path: Path,
    metric_sha: str,
    metadata: Mapping[str, Any],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    return {
        "schema": ARCHIVE_COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **spec.manifest(),
        "scenario_archive": str(archive_path.resolve()),
        "scenario_archive_sha256": archive_sha,
        "metric_artifact": str(metric_path.resolve()),
        "metric_artifact_sha256": metric_sha,
        "metadata": dict(metadata),
        "train_only_ramp_thresholds": dict(thresholds),
        "metric_registry": list(METRIC_NAMES),
        "target_role": "train_outer_test_only",
        "target_state_argument_used_for_sampling": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }


def _recover_or_validate_completion(
    root: Path,
    *,
    spec: ArchiveSpec,
    context: FoldContext,
) -> dict[str, Any] | None:
    completion_path = root / "run.json"
    archive_path = root / "scenarios.npz"
    manifest_path = archive_path.with_suffix(".manifest.json")
    metric_path = _metric_path(root)
    if not any(
        path.exists()
        for path in (completion_path, archive_path, manifest_path, metric_path)
    ):
        return None
    if manifest_path.exists() and not archive_path.exists():
        raise RuntimeError(f"manifest exists without scenario archive: {spec.key}")
    if archive_path.exists() and not manifest_path.exists():
        with np.load(archive_path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            arrays = validate_transition_archive_arrays(
                scenarios=stored["scenarios"],
                observations=stored["observations"],
                observed_mask=stored["observed_mask"],
                raw_missing_mask=stored["raw_missing_mask"],
                states=stored["states"],
                zero_probability=stored["zero_probability"],
                one_probability=stored["one_probability"],
                day=stored["day"],
                zones=stored["zones"],
            )
        digest = _sha256(archive_path)
        _atomic_json(
            manifest_path,
            {
                "schema": ARCHIVE_MANIFEST_SCHEMA,
                "archive": archive_path.name,
                "sha256": digest,
                "bytes": int(archive_path.stat().st_size),
                "days": int(len(arrays["day"])),
                "members": int(arrays["scenarios"].shape[1]),
                "metadata": metadata,
            },
        )

    def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
        expected = {
            "observations": context.observation,
            "observed_mask": context.observed_mask,
            "raw_missing_mask": context.raw_missing_mask,
            "day": context.day,
            "zones": context.zones,
        }
        if any(
            not np.array_equal(arrays[name], value)
            for name, value in expected.items()
        ):
            raise RuntimeError(f"TGO-v1 archive heldout arrays drifted: {spec.key}")

    if completion_path.exists():
        _verified_sidecar(completion_path)
        record = _read_json(completion_path)
        if (
            record.get("schema") != ARCHIVE_COMPLETION_SCHEMA
            or record.get("status") != "complete"
            or record.get("archive_key") != spec.key
            or record.get("target_state_argument_used_for_sampling") is not False
        ):
            raise RuntimeError(f"TGO-v1 archive completion drifted: {spec.key}")
        arrays, metadata, archive_sha = load_transition_archive(archive_path)
        validate_arrays(arrays)
        day, _metrics, metric_sha = _load_metrics(metric_path)
        if (
            archive_sha != record["scenario_archive_sha256"]
            or metric_sha != record["metric_artifact_sha256"]
            or metadata != record["metadata"]
            or Path(record["scenario_archive"]) != archive_path.resolve()
            or Path(record["metric_artifact"]) != metric_path.resolve()
            or not np.array_equal(day, context.day)
        ):
            raise RuntimeError(f"TGO-v1 completed archive hash drifted: {spec.key}")
        return record
    if not archive_path.exists():
        raise RuntimeError(f"metric exists without scenario archive: {spec.key}")
    arrays, metadata, archive_sha = load_transition_archive(archive_path)
    validate_arrays(arrays)
    if metric_path.exists():
        day, _metrics, metric_sha = _load_metrics(metric_path)
        if not np.array_equal(day, context.day):
            raise RuntimeError(f"uncommitted metric days drifted: {spec.key}")
    else:
        metrics = transition_per_day_metrics(
            arrays["scenarios"],
            arrays["observations"],
            arrays["observed_mask"],
            zero_probability=arrays["zero_probability"],
            one_probability=arrays["one_probability"],
            ramp_thresholds=context.ramp_thresholds,
        )
        metric_sha = _write_metrics(
            metric_path, day=context.day, metrics=metrics
        )
    record = _completion_payload(
        spec=spec,
        archive_path=archive_path,
        archive_sha=archive_sha,
        metric_path=metric_path,
        metric_sha=metric_sha,
        metadata=metadata,
        thresholds=context.ramp_thresholds,
    )
    _atomic_json(completion_path, record)
    return record


@torch.no_grad()
def _decode_strict_interior_sigmoid(
    logit: torch.Tensor,
    states: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Encode finite latent interiors at the nearest FP64 interior values."""

    if logit.dtype != torch.float64 or states.shape != logit.shape:
        raise ValueError("FP64 logit and atom states must have identical shapes")
    if not bool(torch.isfinite(logit).all()):
        raise FloatingPointError("non-finite physical decode logit")
    if bool(((states < 0) | (states > 2)).any()):
        raise ValueError("sampled atom state lies outside {0,1,2}")
    interior = torch.sigmoid(logit)
    active = states == 1
    saturated_zero = active & (interior == 0.0)
    saturated_one = active & (interior == 1.0)
    lower = torch.nextafter(logit.new_zeros(()), logit.new_ones(()))
    upper = torch.nextafter(logit.new_ones(()), logit.new_zeros(()))
    if not bool((lower > 0.0) & (upper < 1.0)):
        raise RuntimeError("FP64 nextafter did not return strict interior values")
    decoded = torch.where(
        states == 0,
        torch.zeros_like(interior),
        torch.where(states == 2, torch.ones_like(interior), interior),
    )
    decoded = torch.where(saturated_zero, lower, decoded)
    decoded = torch.where(saturated_one, upper, decoded)
    if bool((active & ((decoded <= 0.0) | (decoded >= 1.0))).any()):
        raise FloatingPointError("strict interior encoding failed")
    return decoded, {
        "zero_count": int(saturated_zero.sum()),
        "one_count": int(saturated_one.sum()),
    }
@torch.no_grad()
def _sample_one(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    training_freeze_sha: str,
    formal_root: Path,
    evaluation_root: Path,
    spec: ArchiveSpec,
    context: FoldContext,
    group: CommonSamplingGroup,
    expected_checkpoint_sha: str,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    root = _archive_root(evaluation_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    complete = _recover_or_validate_completion(
        root, spec=spec, context=context
    )
    if complete is not None:
        return complete
    archive_path = root / "scenarios.npz"
    metric_path = _metric_path(root)
    system, checkpoint_sha = _load_denoiser(
        formal_root=formal_root,
        spec=spec,
        expected_sha=expected_checkpoint_sha,
        context=context,
        device=device,
    )
    sampling = config["sampling"]
    members = int(sampling["members"])
    chunk_size = int(sampling["member_chunk"])
    steps = int(sampling["DDIM_steps"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    chunks: list[torch.Tensor] = []
    saturation_counts = {"zero_count": 0, "one_count": 0}
    for start in range(0, members, chunk_size):
        stop = min(start + chunk_size, members)
        width = stop - start
        active = group.active_mask[:, start:stop].reshape(-1, 10, 24).to(device)
        noise = group.native_epsilon[:, start:stop].reshape(-1, 10, 24).to(device)
        condition = (
            context.condition_test[:, None]
            .expand(-1, width, -1)
            .reshape(-1, 47)
        )
        sampled = sample_transition_ddim(
            system,
            condition,
            active,
            noise,
            alpha_bar,
            true_operator=context.operators["transition_true"],
            wrong_operator=context.operators["transition_wrong"],
            dct_operator=context.operators["orthogonal_dct"],
            steps=steps,
        )
        level = sampled.level.reshape(len(context.day), width, 10, 24)
        residual = level.double() * float(context.level_rms)
        mean = torch.from_numpy(context.mean_test).to(device=device, dtype=torch.float64)
        states = group.states[:, start:stop].to(device)
        decoded, adjustment = _decode_strict_interior_sigmoid(mean[:, None] + residual, states)
        for name in saturation_counts:
            saturation_counts[name] += adjustment[name]
        if not bool(torch.isfinite(decoded).all()):
            raise FloatingPointError(f"non-finite decoded scenario: {spec.key}")
        if bool(((states == 1) & ((decoded <= 0.0) | (decoded >= 1.0))).any()):
            raise FloatingPointError(
                f"interior decode reached an exact boundary without clipping: {spec.key}"
            )
        chunks.append(decoded.cpu())
    scenarios = torch.cat(chunks, dim=1).numpy()
    wall_seconds = float(time.perf_counter() - started)
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    states_numpy = group.states.numpy()
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "config_sha256": config_sha,
        "training_freeze_sha256": training_freeze_sha,
        "atom_checkpoint_sha256": context.atom_checkpoint_sha256,
        "denoiser_checkpoint_sha256": checkpoint_sha,
        "outer_fold": spec.outer_fold,
        "model_seed": spec.model_seed,
        "path_id": spec.path_id,
        "sampling_seed": spec.sampling_seed,
        "members": members,
        "DDIM_steps": steps,
        "member_chunk": chunk_size,
        "atom_allocation_sha256": group.allocation_sha256,
        "atom_probability_sha256": group.probability_sha256,
        "native_epsilon_sha256": group.epsilon_sha256,
        "target_state_argument_used_for_sampling": False,
        "sampling_wall_seconds": wall_seconds,
        "peak_memory_allocated_bytes": peak,
        "day_sha256": date_list_sha256(context.day),
        "common_random_group": f"fold{spec.outer_fold}__sampling{spec.sampling_seed}",
        "decode": "outer_train_mean_plus_generated_residual_then_sigmoid_and_exact_atoms",
        "archive_dtype": "float64",
        "interior_encoding": "nextafter_only_if_finite_fp64_sigmoid_rounds_to_boundary",
        "interior_fp64_saturation_zero_count": saturation_counts["zero_count"],
        "interior_fp64_saturation_one_count": saturation_counts["one_count"],
        "post_hoc_clipping": "machine_nextafter_only",
    }
    archive_sha, _manifest_path = write_transition_archive(
        archive_path,
        arrays={
            "scenarios": scenarios,
            "observations": context.observation,
            "observed_mask": context.observed_mask,
            "raw_missing_mask": context.raw_missing_mask,
            "states": states_numpy,
            "zero_probability": group.zero_probability,
            "one_probability": group.one_probability,
            "day": context.day,
            "zones": context.zones,
        },
        metadata=metadata,
    )
    metrics = transition_per_day_metrics(
        scenarios,
        context.observation,
        context.observed_mask,
        zero_probability=group.zero_probability,
        one_probability=group.one_probability,
        ramp_thresholds=context.ramp_thresholds,
    )
    metric_sha = _write_metrics(metric_path, day=context.day, metrics=metrics)
    record = _completion_payload(
        spec=spec,
        archive_path=archive_path,
        archive_sha=archive_sha,
        metric_path=metric_path,
        metric_sha=metric_sha,
        metadata=metadata,
        thresholds=context.ramp_thresholds,
    )
    _atomic_json(root / "run.json", record)
    del system, scenarios, chunks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _completion_records(
    evaluation_root: Path, matrix: Sequence[ArchiveSpec]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for spec in matrix:
        path = _archive_root(evaluation_root, spec) / "run.json"
        if not path.exists():
            continue
        _verified_sidecar(path)
        record = _read_json(path)
        if (
            record.get("schema") != ARCHIVE_COMPLETION_SCHEMA
            or record.get("status") != "complete"
            or record.get("archive_key") != spec.key
        ):
            raise RuntimeError(f"invalid TGO-v1 archive completion: {spec.key}")
        records[spec.key] = record
    return records


def _write_progress(
    evaluation_root: Path,
    matrix: Sequence[ArchiveSpec],
    *,
    selected_this_invocation: Sequence[str],
) -> dict[str, Any]:
    records = _completion_records(evaluation_root, matrix)
    pending = [spec.key for spec in matrix if spec.key not in records]
    value = {
        "schema": SAMPLING_PROGRESS_SCHEMA,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_archives": len(matrix),
        "completed_archives": len(records),
        "remaining_archives": len(pending),
        "selected_this_invocation": list(selected_this_invocation),
        "pending_archive_keys": pending,
        "sampling_frozen": (evaluation_root / "sampling.freeze.json").exists(),
        "formal_result_written": (evaluation_root / "TGO_V1_RESULT.json").exists(),
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_action": (
            "complete_remaining_common_sampling"
            if pending
            else "freeze_sampling_and_execute_registered_inference"
        ),
    }
    _atomic_json(evaluation_root / "sampling.progress.json", value)
    return value


def _try_sampling_freeze(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    evaluation_root: Path,
    matrix: Sequence[ArchiveSpec],
    identity_sha: str,
) -> dict[str, Any] | None:
    freeze_path = evaluation_root / "sampling.freeze.json"
    if freeze_path.exists():
        _verified_sidecar(freeze_path)
        return _read_json(freeze_path)
    records = _completion_records(evaluation_root, matrix)
    if len(records) != len(matrix):
        return None
    archive_hashes: dict[str, str] = {}
    metric_hashes: dict[str, str] = {}
    common: dict[tuple[int, int], dict[str, set[str]]] = {}
    adjustment_totals = {"zero_count": 0, "one_count": 0}
    adjustment_by_path = {
        path_id: {"zero_count": 0, "one_count": 0} for path_id in PATH_IDS
    }
    total_interior_values = 0
    for spec in matrix:
        record = records[spec.key]
        metadata = record["metadata"]
        if (
            metadata["config_sha256"] != config_sha
            or metadata["training_freeze_sha256"]
            != config["lineage"]["training_freeze_sha256"]
            or int(metadata["outer_fold"]) != spec.outer_fold
            or int(metadata["model_seed"]) != spec.model_seed
            or metadata["path_id"] != spec.path_id
            or int(metadata["sampling_seed"]) != spec.sampling_seed
            or metadata["target_state_argument_used_for_sampling"] is not False
            or metadata["post_hoc_clipping"] != "machine_nextafter_only"
            or metadata["archive_dtype"] != "float64"
            or metadata["interior_encoding"] != "nextafter_only_if_finite_fp64_sigmoid_rounds_to_boundary"
        ):
            raise RuntimeError(f"TGO-v1 archive metadata drifted: {spec.key}")
        archive_path = Path(record["scenario_archive"])
        _arrays, loaded_metadata, archive_sha = load_transition_archive(archive_path)
        _day, _metrics, metric_sha = _load_metrics(Path(record["metric_artifact"]))
        if (
            loaded_metadata != metadata
            or archive_sha != record["scenario_archive_sha256"]
            or metric_sha != record["metric_artifact_sha256"]
        ):
            raise RuntimeError(f"TGO-v1 archive freeze hash drifted: {spec.key}")
        archive_hashes[spec.key] = archive_sha
        metric_hashes[spec.key] = metric_sha
        counts = {
            name: metadata.get(f"interior_fp64_saturation_{name}")
            for name in adjustment_totals
        }
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise RuntimeError(f"invalid interior encoding count: {spec.key}")
        active_count = int((_arrays["states"] == 1).sum())
        if sum(counts.values()) > active_count:
            raise RuntimeError(f"interior encoding count exceeds active cells: {spec.key}")
        total_interior_values += active_count
        for name, value in counts.items():
            adjustment_totals[name] += value
            adjustment_by_path[spec.path_id][name] += value
        group = common.setdefault(
            (spec.outer_fold, spec.sampling_seed),
            {"allocation": set(), "probability": set(), "epsilon": set()},
        )
        group["allocation"].add(metadata["atom_allocation_sha256"])
        group["probability"].add(metadata["atom_probability_sha256"])
        group["epsilon"].add(metadata["native_epsilon_sha256"])
    if len(common) != 24 or any(
        len(values) != 1 for group in common.values() for values in group.values()
    ):
        raise RuntimeError("TGO-v1 cross-path common sampling identity failed")
    freeze = {
        "schema": SAMPLING_FREEZE_SCHEMA,
        "status": "TGO_V1_336_OF_336_SAMPLING_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_config_sha256": config_sha,
        "evaluation_identity_sha256": identity_sha,
        "expected_archives": len(matrix),
        "completed_archives": len(records),
        "archive_calendar_days": 267,
        "inference_complete_case_calendar_days": 266,
        "common_fold_sampling_groups": len(common),
        "common_allocation_probability_and_epsilon_exact": True,
        "archive_sha256_by_key": archive_hashes,
        "metric_sha256_by_key": metric_hashes,
        "machine_precision_adjustments": {
            "zero_count": adjustment_totals["zero_count"],
            "one_count": adjustment_totals["one_count"],
            "total_interior_values": total_interior_values,
            "by_path": adjustment_by_path,
        },
        "target_state_argument_used_for_sampling": False,
        "post_hoc_clipping": "machine_nextafter_only",
        "calendar_day_is_inference_unit": True,
        "finite_members_are_inference_replicates": False,
        "formal_inference_authorized": True,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    _atomic_json(freeze_path, freeze)
    return freeze


def _daily_benefit(
    reference: np.ndarray, candidate: np.ndarray, *, relative: bool
) -> np.ndarray:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired benefit arrays do not align")
    result = left - right
    if relative:
        denominator = float(left.mean())
        if denominator <= 0.0:
            raise ValueError("relative benefit reference mean is not positive")
        result = result / denominator
    return result


def _daily_harm(
    candidate: np.ndarray, reference: np.ndarray, *, relative: bool
) -> np.ndarray:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired harm arrays do not align")
    result = left - right
    if relative:
        denominator = float(right.mean())
        if denominator <= 0.0:
            raise ValueError("relative harm reference mean is not positive")
        result = result / denominator
    return result


def _complete_case_mask(
    days: np.ndarray,
    metrics: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Apply one target-mask-defined date exclusion to every paired path."""

    dates = np.asarray(days, dtype="datetime64[D]")
    if dates.ndim != 1 or len(np.unique(dates)) != len(dates):
        raise ValueError("complete-case calendar days must be unique")
    excluded = dates == COMPLETE_CASE_EXCLUDED_DAY
    if int(excluded.sum()) != 1 or tuple(metrics) != PATH_IDS:
        raise RuntimeError("TGO-v1.3 complete-case registry drifted")
    valid = ~excluded
    for path_id in PATH_IDS:
        late = np.asarray(metrics[path_id]["late_horizon_level_CRPS"])
        if late.shape != dates.shape or not np.array_equal(np.isfinite(late), valid):
            raise RuntimeError(
                f"TGO-v1.3 late-horizon valid days drifted: {path_id}"
            )
    return valid
def _aggregate_metrics(
    *,
    matrix: Sequence[ArchiveSpec],
    evaluation_root: Path,
) -> tuple[
    np.ndarray,
    dict[str, dict[str, np.ndarray]],
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, dict[str, dict[str, np.ndarray]]],
]:
    loaded: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for spec in matrix:
        day, metrics, _sha = _load_metrics(_metric_path(_archive_root(evaluation_root, spec)))
        loaded[spec.key] = (day, metrics)
    combined: dict[str, dict[str, list[np.ndarray]]] = {
        path_id: {name: [] for name in METRIC_NAMES} for path_id in PATH_IDS
    }
    fold_metrics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    seed_parts: dict[int, dict[str, dict[str, list[np.ndarray]]]] = {
        seed: {
            path_id: {name: [] for name in METRIC_NAMES}
            for path_id in PATH_IDS
        }
        for seed in (3, 4)
    }
    days: list[np.ndarray] = []
    for fold in range(6):
        reference_spec = ArchiveSpec(fold, 3, PATH_IDS[0], 61000)
        fold_day = loaded[reference_spec.key][0]
        fold_specs = [spec for spec in matrix if spec.outer_fold == fold]
        if any(
            not np.array_equal(loaded[spec.key][0], fold_day)
            for spec in fold_specs
        ):
            raise RuntimeError(f"TGO-v1 archive calendar days drifted: fold {fold}")
        days.append(fold_day)
        fold_metrics[fold] = {}
        for path_id in PATH_IDS:
            replicates = [
                loaded[ArchiveSpec(fold, seed, path_id, sampling_seed).key][1]
                for seed in (3, 4)
                for sampling_seed in (61000, 61001, 61002, 61003)
            ]
            averaged = {
                name: np.mean(
                    np.stack([item[name] for item in replicates], axis=0), axis=0
                )
                for name in METRIC_NAMES
            }
            fold_metrics[fold][path_id] = averaged
            for name in METRIC_NAMES:
                combined[path_id][name].append(averaged[name])
            for seed in (3, 4):
                seed_replicates = [
                    loaded[
                        ArchiveSpec(fold, seed, path_id, sampling_seed).key
                    ][1]
                    for sampling_seed in (61000, 61001, 61002, 61003)
                ]
                for name in METRIC_NAMES:
                    seed_parts[seed][path_id][name].append(
                        np.mean(
                            np.stack(
                                [item[name] for item in seed_replicates], axis=0
                            ),
                            axis=0,
                        )
                    )
    all_days = np.concatenate(days)
    if len(all_days) != 267 or len(np.unique(all_days)) != 267:
        raise RuntimeError("TGO-v1 outer-held-out calendar coverage drifted")
    order = np.argsort(all_days)
    all_days = all_days[order]
    combined_final = {
        path_id: {
            name: np.concatenate(parts)[order]
            for name, parts in metric_parts.items()
        }
        for path_id, metric_parts in combined.items()
    }
    seed_final = {
        seed: {
            path_id: {
                name: np.concatenate(parts)[order]
                for name, parts in metric_parts.items()
            }
            for path_id, metric_parts in path_parts.items()
        }
        for seed, path_parts in seed_parts.items()
    }
    valid = _complete_case_mask(all_days, combined_final)
    combined_final = {
        path_id: {name: value[valid] for name, value in scores.items()}
        for path_id, scores in combined_final.items()
    }
    seed_final = {
        seed: {
            path_id: {name: value[valid] for name, value in scores.items()}
            for path_id, scores in path_scores.items()
        }
        for seed, path_scores in seed_final.items()
    }
    for fold in range(6):
        fold_valid = days[fold] != COMPLETE_CASE_EXCLUDED_DAY
        fold_metrics[fold] = {
            path_id: {
                name: value[fold_valid] for name, value in scores.items()
            }
            for path_id, scores in fold_metrics[fold].items()
        }
    all_days = all_days[valid]
    if len(all_days) != 266:
        raise RuntimeError("TGO-v1.3 expected exactly 266 complete-case days")
    return all_days, combined_final, fold_metrics, seed_final


def _endpoint_registry() -> tuple[str, ...]:
    """Return the frozen ordered family of jointly controlled endpoints."""

    primary = (
        ("ramp_CRPS", "absolute_benefit"),
        ("lagged_increment_variogram_score", "relative_benefit"),
    )
    values: list[str] = []
    for path_id in PATH_IDS[1:]:
        values.extend(
            f"{path_id}_vs_LEVEL_IID__{metric}__{scale}"
            for metric, scale in primary
        )
    for control in (
        "LEVEL_MATCHED_METRIC",
        "LEVEL_MATCHED_NOISE",
        "LEVEL_MATCHED_BOTH",
        "ORTHOGONAL_DCT",
        "TRANSITION_WRONG",
    ):
        values.extend(
            f"TRANSITION_TRUE_vs_{control}__{metric}__{scale}"
            for metric, scale in primary
        )
    values.extend(
        (
            "TRANSITION_TRUE_vs_LEVEL_IID__level_CRPS__absolute_harm",
            "TRANSITION_TRUE_vs_LEVEL_IID__normalized_joint_ES__relative_harm",
            "TRANSITION_TRUE_vs_LEVEL_IID__coverage90__absolute_difference",
            "TRANSITION_TRUE_vs_LEVEL_IID__width90__relative_harm",
            "TRANSITION_TRUE_vs_LEVEL_IID__daily_mean_power_CRPS__relative_harm",
            "TRANSITION_TRUE_vs_LEVEL_IID__late_horizon_level_CRPS__absolute_harm",
        )
    )
    result = tuple(values)
    if len(result) != 28 or len(set(result)) != len(result):
        raise RuntimeError("TGO-v1 inference endpoint registry must contain 28 entries")
    return result


def _inference_contributions(
    metrics: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Construct paired daily endpoint contributions in registry order."""

    if tuple(metrics) != PATH_IDS:
        raise ValueError("TGO-v1 path metric registry or order drifted")
    iid = metrics["LEVEL_IID"]
    true = metrics["TRANSITION_TRUE"]
    values: dict[str, np.ndarray] = {}
    for path_id in PATH_IDS[1:]:
        candidate = metrics[path_id]
        values[
            f"{path_id}_vs_LEVEL_IID__ramp_CRPS__absolute_benefit"
        ] = _daily_benefit(iid["ramp_CRPS"], candidate["ramp_CRPS"], relative=False)
        values[
            f"{path_id}_vs_LEVEL_IID__lagged_increment_variogram_score__relative_benefit"
        ] = _daily_benefit(
            iid["lagged_increment_variogram_score"],
            candidate["lagged_increment_variogram_score"],
            relative=True,
        )
    for control in (
        "LEVEL_MATCHED_METRIC",
        "LEVEL_MATCHED_NOISE",
        "LEVEL_MATCHED_BOTH",
        "ORTHOGONAL_DCT",
        "TRANSITION_WRONG",
    ):
        values[
            f"TRANSITION_TRUE_vs_{control}__ramp_CRPS__absolute_benefit"
        ] = _daily_benefit(
            metrics[control]["ramp_CRPS"], true["ramp_CRPS"], relative=False
        )
        values[
            f"TRANSITION_TRUE_vs_{control}__lagged_increment_variogram_score__relative_benefit"
        ] = _daily_benefit(
            metrics[control]["lagged_increment_variogram_score"],
            true["lagged_increment_variogram_score"],
            relative=True,
        )
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__level_CRPS__absolute_harm"
    ] = _daily_harm(true["level_CRPS"], iid["level_CRPS"], relative=False)
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__normalized_joint_ES__relative_harm"
    ] = _daily_harm(
        true["normalized_joint_ES"], iid["normalized_joint_ES"], relative=True
    )
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__coverage90__absolute_difference"
    ] = np.asarray(true["coverage90"] - iid["coverage90"], dtype=np.float64)
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__width90__relative_harm"
    ] = _daily_harm(true["width90"], iid["width90"], relative=True)
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__daily_mean_power_CRPS__relative_harm"
    ] = _daily_harm(
        true["daily_mean_power_CRPS"],
        iid["daily_mean_power_CRPS"],
        relative=True,
    )
    values[
        "TRANSITION_TRUE_vs_LEVEL_IID__late_horizon_level_CRPS__absolute_harm"
    ] = _daily_harm(
        true["late_horizon_level_CRPS"],
        iid["late_horizon_level_CRPS"],
        relative=False,
    )
    if tuple(values) != _endpoint_registry():
        raise RuntimeError("TGO-v1 inference contribution order drifted")
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1 or any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("TGO-v1 endpoint contributions are invalid")
    return values


def _gate(passed: bool, *, value: Any, requirement: str) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "requirement": requirement}


def _adjudicate(
    contrast_bands: Mapping[str, Mapping[str, float]],
    *,
    positive_outer_folds: int,
    positive_model_seeds: int,
    technical_eligibility: bool,
    margins: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen TGO-v1 decision tree to simultaneous bands."""

    if tuple(contrast_bands) != _endpoint_registry():
        raise ValueError("TGO-v1 contrast registry or order drifted")

    def name(left: str, metric: str, scale: str, right: str = "LEVEL_IID") -> str:
        return f"{left}_vs_{right}__{metric}__{scale}"

    def estimate(endpoint: str) -> float:
        return float(contrast_bands[endpoint]["estimate"])

    def low(endpoint: str) -> float:
        return float(contrast_bands[endpoint]["simultaneous_low"])

    def high(endpoint: str) -> float:
        return float(contrast_bands[endpoint]["simultaneous_high"])

    ramp_minimum = float(margins["ramp_CRPS_absolute_improvement_min"])
    lag_minimum = float(margins["lagged_variogram_relative_improvement_min"])
    ramp_ni = float(margins["attribution_ramp_CRPS_absolute_noninferiority_margin"])
    lag_ni = float(margins["attribution_lagged_variogram_relative_noninferiority_margin"])

    def primary_vs_iid(path_id: str) -> tuple[bool, bool]:
        ramp = name(path_id, "ramp_CRPS", "absolute_benefit")
        lag = name(
            path_id,
            "lagged_increment_variogram_score",
            "relative_benefit",
        )
        return (
            estimate(ramp) >= ramp_minimum and low(ramp) > 0.0,
            estimate(lag) >= lag_minimum and low(lag) > 0.0,
        )

    def attribution(control: str, *, require_both: bool) -> tuple[bool, dict[str, bool]]:
        ramp = name(
            "TRANSITION_TRUE", "ramp_CRPS", "absolute_benefit", control
        )
        lag = name(
            "TRANSITION_TRUE",
            "lagged_increment_variogram_score",
            "relative_benefit",
            control,
        )
        superior = {"ramp": low(ramp) > 0.0, "lag": low(lag) > 0.0}
        noninferior = {"ramp": low(ramp) >= -ramp_ni, "lag": low(lag) >= -lag_ni}
        passed = (
            all(superior.values())
            if require_both
            else any(superior.values()) and all(noninferior.values())
        )
        return passed, {
            "ramp_superior": superior["ramp"],
            "lag_superior": superior["lag"],
            "ramp_noninferior": noninferior["ramp"],
            "lag_noninferior": noninferior["lag"],
        }

    true_primary_parts = primary_vs_iid("TRANSITION_TRUE")
    true_primary = all(true_primary_parts)
    control_primary = {
        path_id: all(primary_vs_iid(path_id))
        for path_id in (
            "LEVEL_MATCHED_METRIC",
            "LEVEL_MATCHED_NOISE",
            "LEVEL_MATCHED_BOTH",
            "ORTHOGONAL_DCT",
            "TRANSITION_WRONG",
        )
    }
    wrong_pass, wrong_detail = attribution("TRANSITION_WRONG", require_both=True)
    dct_pass, dct_detail = attribution("ORTHOGONAL_DCT", require_both=False)
    both_pass, both_detail = attribution("LEVEL_MATCHED_BOTH", require_both=False)
    metric_pass, metric_detail = attribution("LEVEL_MATCHED_METRIC", require_both=False)
    noise_pass, noise_detail = attribution("LEVEL_MATCHED_NOISE", require_both=False)

    safety_specs = (
        (
            "level_CRPS",
            name("TRANSITION_TRUE", "level_CRPS", "absolute_harm"),
            high(name("TRANSITION_TRUE", "level_CRPS", "absolute_harm"))
            <= float(margins["level_CRPS_absolute_noninferiority_margin"]),
            f"simultaneous upper <= {margins['level_CRPS_absolute_noninferiority_margin']}",
        ),
        (
            "normalized_joint_ES",
            name("TRANSITION_TRUE", "normalized_joint_ES", "relative_harm"),
            high(name("TRANSITION_TRUE", "normalized_joint_ES", "relative_harm"))
            <= float(margins["normalized_joint_ES_relative_noninferiority_margin"]),
            f"simultaneous upper <= {margins['normalized_joint_ES_relative_noninferiority_margin']}",
        ),
        (
            "coverage90",
            name("TRANSITION_TRUE", "coverage90", "absolute_difference"),
            max(
                abs(low(name("TRANSITION_TRUE", "coverage90", "absolute_difference"))),
                abs(high(name("TRANSITION_TRUE", "coverage90", "absolute_difference"))),
            )
            <= float(margins["coverage90_absolute_difference_max"]),
            f"maximum absolute simultaneous band endpoint <= {margins['coverage90_absolute_difference_max']}",
        ),
        (
            "width90",
            name("TRANSITION_TRUE", "width90", "relative_harm"),
            high(name("TRANSITION_TRUE", "width90", "relative_harm"))
            <= float(margins["width90_relative_increase_max"]),
            f"simultaneous upper <= {margins['width90_relative_increase_max']}",
        ),
        (
            "daily_mean_power_CRPS",
            name("TRANSITION_TRUE", "daily_mean_power_CRPS", "relative_harm"),
            high(name("TRANSITION_TRUE", "daily_mean_power_CRPS", "relative_harm"))
            <= float(margins["daily_mean_power_CRPS_relative_noninferiority_margin"]),
            f"simultaneous upper <= {margins['daily_mean_power_CRPS_relative_noninferiority_margin']}",
        ),
        (
            "late_horizon_level_CRPS",
            name("TRANSITION_TRUE", "late_horizon_level_CRPS", "absolute_harm"),
            high(name("TRANSITION_TRUE", "late_horizon_level_CRPS", "absolute_harm"))
            <= float(margins["late_horizon_level_CRPS_absolute_noninferiority_margin"]),
            f"simultaneous upper <= {margins['late_horizon_level_CRPS_absolute_noninferiority_margin']}",
        ),
    )
    safety_gates = {
        label: _gate(
            passed,
            value={
                "estimate": estimate(endpoint),
                "simultaneous_low": low(endpoint),
                "simultaneous_high": high(endpoint),
            },
            requirement=requirement,
        )
        for label, endpoint, passed, requirement in safety_specs
    }
    safety_pass = all(item["passed"] for item in safety_gates.values())
    fold_pass = positive_outer_folds >= int(margins["minimum_positive_outer_folds"])
    seed_pass = positive_model_seeds >= int(margins["minimum_positive_model_seeds"])
    gates = {
        "technical_eligibility": _gate(
            technical_eligibility, value=technical_eligibility, requirement="true"
        ),
        "TRANSITION_TRUE_vs_LEVEL_IID_ramp": _gate(
            true_primary_parts[0],
            value=estimate(name("TRANSITION_TRUE", "ramp_CRPS", "absolute_benefit")),
            requirement=f"estimate >= {ramp_minimum} and simultaneous lower > 0",
        ),
        "TRANSITION_TRUE_vs_LEVEL_IID_lagged_variogram": _gate(
            true_primary_parts[1],
            value=estimate(
                name(
                    "TRANSITION_TRUE",
                    "lagged_increment_variogram_score",
                    "relative_benefit",
                )
            ),
            requirement=f"estimate >= {lag_minimum} and simultaneous lower > 0",
        ),
        "true_adjacency_vs_wrong_adjacency": _gate(
            wrong_pass,
            value=wrong_detail,
            requirement="both primary simultaneous lower bounds > 0",
        ),
        "true_transition_vs_general_DCT_basis": _gate(
            dct_pass,
            value=dct_detail,
            requirement=(
                "at least one primary lower > 0 and the other lower >= "
                f"(-{ramp_ni} absolute ramp, -{lag_ni} relative lag)"
            ),
        ),
        "true_transition_vs_level_matched_both": _gate(
            both_pass,
            value=both_detail,
            requirement=(
                "at least one primary lower > 0 and the other lower >= "
                f"(-{ramp_ni} absolute ramp, -{lag_ni} relative lag)"
            ),
        ),
        "distribution_safety": _gate(
            safety_pass,
            value=safety_gates,
            requirement="all six simultaneous safety limits pass",
        ),
        "fold_stability": _gate(
            fold_pass,
            value=int(positive_outer_folds),
            requirement=f">= {margins['minimum_positive_outer_folds']} of 6 folds",
        ),
        "seed_stability": _gate(
            seed_pass,
            value=int(positive_model_seeds),
            requirement=f">= {margins['minimum_positive_model_seeds']} of 2 seeds",
        ),
    }
    all_go = all(item["passed"] for item in gates.values())

    if not technical_eligibility:
        status = "TGO_V1_TECHNICAL_NO_GO"
        reason = "sampling identity, finite-value, or atom hard checks failed"
    elif not true_primary:
        if control_primary["LEVEL_MATCHED_METRIC"]:
            status = "DYNAMIC_METRIC_ONLY"
            reason = "the matched dynamic metric cleared IID while the candidate did not"
        elif control_primary["LEVEL_MATCHED_NOISE"] or control_primary["LEVEL_MATCHED_BOTH"]:
            status = "NOISE_GEOMETRY_ONLY"
            reason = "a matched-noise control cleared IID while the candidate did not"
        elif control_primary["ORTHOGONAL_DCT"]:
            status = "GENERAL_BASIS_ONLY"
            reason = "the general orthogonal basis cleared IID while the candidate did not"
        else:
            status = "TGO_V1_NO_GO"
            reason = "the candidate failed one or both practical primary gates"
    elif not wrong_pass:
        status = "ADJACENCY_NOT_IDENTIFIED"
        reason = "true adjacency did not beat wrong adjacency on both primary metrics"
    elif not dct_pass:
        status = "GENERAL_BASIS_ONLY"
        reason = "true transition did not separate from a general orthogonal basis"
    elif not both_pass:
        status = "NOISE_GEOMETRY_ONLY"
        reason = "true transition did not separate from the matched metric-and-noise control"
    elif not safety_pass:
        status = "DISTRIBUTION_DAMAGE_NO_GO"
        reason = "dynamic gates passed but at least one distribution-safety limit failed"
    elif not (fold_pass and seed_pass):
        status = "TGO_V1_NO_GO"
        reason = "dynamic effects did not satisfy the frozen fold/seed stability gates"
    elif all_go:
        status = "TGO_V1_ATTRIBUTED_GO"
        reason = "all registered technical, dynamic, attribution, safety, and stability gates passed"
    else:
        status = "TGO_V1_NO_GO"
        reason = "the frozen gate conjunction was not satisfied"

    return {
        "status": status,
        "reason": reason,
        "all_GO_gates_passed": bool(all_go),
        "gates": gates,
        "safety_gates": safety_gates,
        "control_primary_vs_IID": control_primary,
        "additional_attribution_diagnostics": {
            "TRANSITION_TRUE_vs_LEVEL_MATCHED_METRIC": {
                "passed_one_superior_other_noninferior": metric_pass,
                **metric_detail,
            },
            "TRANSITION_TRUE_vs_LEVEL_MATCHED_NOISE": {
                "passed_one_superior_other_noninferior": noise_pass,
                **noise_detail,
            },
        },
    }


def _stability_counts(
    fold_metrics: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    seed_metrics: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for fold in range(6):
        iid = fold_metrics[fold]["LEVEL_IID"]
        true = fold_metrics[fold]["TRANSITION_TRUE"]
        ramp = float(np.mean(iid["ramp_CRPS"] - true["ramp_CRPS"]))
        lag = float(
            np.mean(
                iid["lagged_increment_variogram_score"]
                - true["lagged_increment_variogram_score"]
            )
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "ramp_CRPS_absolute_benefit": ramp,
                "lagged_variogram_absolute_benefit": lag,
                "both_positive": ramp > 0.0 and lag > 0.0,
            }
        )
    seed_rows: list[dict[str, Any]] = []
    for seed in (3, 4):
        iid = seed_metrics[seed]["LEVEL_IID"]
        true = seed_metrics[seed]["TRANSITION_TRUE"]
        ramp = float(np.mean(iid["ramp_CRPS"] - true["ramp_CRPS"]))
        lag = float(
            np.mean(
                iid["lagged_increment_variogram_score"]
                - true["lagged_increment_variogram_score"]
            )
        )
        seed_rows.append(
            {
                "model_seed": seed,
                "ramp_CRPS_absolute_benefit": ramp,
                "lagged_variogram_absolute_benefit": lag,
                "both_positive": ramp > 0.0 and lag > 0.0,
            }
        )
    return (
        sum(bool(row["both_positive"]) for row in fold_rows),
        sum(bool(row["both_positive"]) for row in seed_rows),
        fold_rows,
        seed_rows,
    )


def _without_created(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_utc"}


def _evaluation_identity(
    *,
    config_path: Path,
    config_sha: str,
    code: Mapping[str, str],
    training_freeze: Mapping[str, Any],
    bundle: Any,
    nuisance_replay: Mapping[str, Any],
    runtime: Mapping[str, Any],
    matrix: Sequence[ArchiveSpec],
) -> dict[str, Any]:
    return {
        "schema": IDENTITY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "code_sha256": dict(code),
        "training_freeze": {
            "path": training_freeze["path"],
            "sha256": training_freeze["sha256"],
            "status": training_freeze["payload"]["status"],
        },
        "runtime": dict(runtime),
        "data_identity": {
            "materialized_target_roles": list(bundle.materialized_roles),
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
            {"archives": [spec.manifest() for spec in matrix]}
        ),
        "expected_archives": len(matrix),
        "inference_endpoint_registry": list(_endpoint_registry()),
        "outer_heldout_role": "cross_fitted_outer_test_within_train_only",
        "validation_selection_calibration_and_final_roles_materialized": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }


def _prepare_evaluation_identity(
    evaluation_root: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> str:
    path = evaluation_root / "evaluation.identity.json"
    if path.exists():
        if not resume:
            raise FileExistsError("formal evaluation identity exists; use --resume")
        _verified_sidecar(path)
        previous = _read_json(path)
        if _without_created(previous) != _without_created(identity):
            raise RuntimeError("TGO-v1 evaluation identity drifted")
        return _sha256(path)
    if resume and evaluation_root.exists() and any(evaluation_root.iterdir()):
        raise RuntimeError("cannot resume non-empty evaluation root without identity")
    return _atomic_json(path, identity)


def _execute_inference(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    evaluation_root: Path,
    matrix: Sequence[ArchiveSpec],
    sampling_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = evaluation_root / "TGO_V1_RESULT.json"
    if result_path.exists():
        _verified_sidecar(result_path)
        return _read_json(result_path)
    if sampling_freeze.get("formal_inference_authorized") is not True:
        raise RuntimeError("TGO-v1 sampling freeze does not authorize inference")
    days, metrics, fold_metrics, seed_metrics = _aggregate_metrics(
        matrix=matrix, evaluation_root=evaluation_root
    )
    contributions = _inference_contributions(metrics)
    registry = _endpoint_registry()
    if tuple(config["inference"]["endpoint_registry"]) != registry:
        raise RuntimeError("frozen TGO-v1 endpoint registry drifted")
    contribution_matrix = np.column_stack([contributions[name] for name in registry])
    bootstrap = config["inference"]["bootstrap"]
    raw_bands = month_cluster_max_t_bands(
        contribution_matrix,
        days,
        repetitions=int(bootstrap["repetitions"]),
        seed=int(bootstrap["seed"]),
        confidence=float(bootstrap["confidence"]),
    )
    bands = {
        endpoint: {
            "estimate": float(raw_bands["estimate"][index]),
            "standard_error": float(raw_bands["standard_error"][index]),
            "simultaneous_low": float(raw_bands["simultaneous_low"][index]),
            "simultaneous_high": float(raw_bands["simultaneous_high"][index]),
        }
        for index, endpoint in enumerate(registry)
    }
    positive_folds, positive_seeds, fold_rows, seed_rows = _stability_counts(
        fold_metrics, seed_metrics
    )
    decision = _adjudicate(
        bands,
        positive_outer_folds=positive_folds,
        positive_model_seeds=positive_seeds,
        technical_eligibility=(
            sampling_freeze["common_allocation_probability_and_epsilon_exact"] is True
            and sampling_freeze["target_state_argument_used_for_sampling"] is False
            and sampling_freeze["post_hoc_clipping"] == "machine_nextafter_only"
        ),
        margins=config["decision"]["margins"],
    )
    contribution_path = evaluation_root / "formal_daily_contributions.npz"
    contribution_sha = _atomic_npz(
        contribution_path,
        day=days,
        **{name: contributions[name] for name in registry},
    )
    metric_means = {
        path_id: {
            name: float(np.mean(metrics[path_id][name])) for name in METRIC_NAMES
        }
        for path_id in PATH_IDS
    }
    sampling_freeze_path = evaluation_root / "sampling.freeze.json"
    sampling_freeze_sha = _verified_sidecar(sampling_freeze_path)
    result = {
        "schema": RESULT_SCHEMA,
        "status": decision["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": decision["reason"],
        "evaluation_config_sha256": config_sha,
        "sampling_freeze": str(sampling_freeze_path.resolve()),
        "sampling_freeze_sha256": sampling_freeze_sha,
        "complete_scenario_archives": len(matrix),
        "calendar_days": int(len(days)),
        "archive_calendar_days": 267,
        "excluded_calendar_days": ["2013-12-31"],
        "model_seeds_averaged_within_day": 2,
        "sampling_seeds_averaged_within_model_seed_and_day": 4,
        "predictive_members_per_archive": int(config["sampling"]["members"]),
        "machine_precision_adjustments": sampling_freeze["machine_precision_adjustments"],
        "path_metric_means": metric_means,
        "inference": {
            key: value
            for key, value in raw_bands.items()
            if key not in {"estimate", "standard_error", "simultaneous_low", "simultaneous_high"}
        },
        "contrast_bands": bands,
        "positive_outer_folds": positive_folds,
        "positive_model_seeds": positive_seeds,
        "fold_stability": fold_rows,
        "seed_stability": seed_rows,
        "decision": decision,
        "formal_daily_contributions": str(contribution_path.resolve()),
        "formal_daily_contributions_sha256": contribution_sha,
        "secondary_metrics_status": "descriptive_not_used_for_promotion",
        "reconstruction_diagnostics_status": "not_implemented_not_used_for_sampling_label",
        "role_access": {
            "materialized_target_roles": ["train"],
            "evaluated_subset": "six cross-fitted outer-test blocks within train",
            "validation_materialized": False,
            "selection_materialized": False,
            "calibration_materialized": False,
            "r_seen_materialized": False,
            "external_final_materialized": False,
        },
        "claim_boundary": (
            "This probe adjudicates transition-object utility and attribution only; "
            "it does not establish paper novelty, external-test performance, or an "
            "automatic Diffusion-over-Flow choice."
        ),
    }
    _atomic_json(result_path, result)
    return result


def dry_run(config_path: Path, *, include_archive_keys: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _validate_config(config, config_path)
    training_freeze = _validate_training_freeze(config)
    matrix = _matrix(config)
    evaluation_root = ROOT / config["output_root"]
    completed = (
        len(_completion_records(evaluation_root, matrix))
        if evaluation_root.exists()
        else 0
    )
    return {
        "schema": "architecture_v1_tgo_v1_evaluation_dry_run_v1",
        "mode": "target_free_no_files_created",
        "config_sha256": code["config_sha256"],
        "training_freeze_status": training_freeze["payload"]["status"],
        "expected_archives": len(matrix),
        "completed_archives": completed,
        "remaining_archives": len(matrix) - completed,
        "matrix_dimensions": {
            "outer_folds": 6,
            "model_seeds": 2,
            "paths": 7,
            "sampling_seeds": 4,
        },
        "inference_endpoints": len(_endpoint_registry()),
        "archive_keys": (
            [spec.key for spec in matrix]
            if include_archive_keys
            else "use --list-archive-keys"
        ),
        "outer_heldout_targets_materialized": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "CPU_execution_requires_allow_cpu": True,
        "next_flag": "--execute-sampling",
    }


def execute_sampling(
    config_path: Path,
    *,
    resume: bool,
    device_name: str,
    allow_cpu: bool,
    archive_keys: Sequence[str],
    max_archives: int | None,
) -> dict[str, Any]:
    config = _read_json(config_path)
    code = _validate_config(config, config_path)
    training_freeze = _validate_training_freeze(config)
    matrix = _matrix(config)
    by_key = {spec.key: spec for spec in matrix}
    unknown = sorted(set(archive_keys) - set(by_key))
    if unknown:
        raise ValueError(f"unknown TGO-v1 archive keys: {unknown}")
    if len(archive_keys) != len(set(archive_keys)):
        raise ValueError("--archive-key contains duplicates")
    if max_archives is not None and max_archives < 1:
        raise ValueError("--max-archives must be positive")
    evaluation_root = ROOT / config["output_root"]
    result_path = evaluation_root / "TGO_V1_RESULT.json"
    if result_path.exists():
        _verified_sidecar(result_path)
        return _read_json(result_path)
    if evaluation_root.exists() and any(evaluation_root.iterdir()) and not resume:
        raise FileExistsError("formal evaluation output is non-empty; use --resume")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    device, runtime = _configure_device(device_name, allow_cpu=allow_cpu)
    probe_config = _read_json(ROOT / config["lineage"]["probe_config"])
    g0b_config = _read_json(ROOT / probe_config["lineage"]["g0_b_config"])
    g0_config = _read_json(ROOT / probe_config["lineage"]["g0_a_config"])
    g0_result = _read_json(ROOT / probe_config["lineage"]["g0_a_result"])
    bundle, _groups, folds, nuisance_replay, _subsets = _replay_nuisance(g0b_config)
    identity = _evaluation_identity(
        config_path=config_path,
        config_sha=code["config_sha256"],
        code=code,
        training_freeze=training_freeze,
        bundle=bundle,
        nuisance_replay=nuisance_replay,
        runtime=runtime,
        matrix=matrix,
    )
    identity_sha = _prepare_evaluation_identity(
        evaluation_root, identity, resume=resume
    )
    formal_root = (ROOT / config["lineage"]["training_freeze"]).parent
    alpha_numpy, _schedule = _baseline_schedule(g0b_config)
    alpha_bar = torch.from_numpy(alpha_numpy).float().to(device)
    already = _completion_records(evaluation_root, matrix)
    selected = [by_key[key] for key in archive_keys] if archive_keys else list(matrix)
    selected = [spec for spec in selected if spec.key not in already]
    if max_archives is not None:
        selected = selected[:max_archives]
    contexts: dict[int, FoldContext] = {}
    groups: dict[tuple[int, int], CommonSamplingGroup] = {}
    completed_this_invocation: list[str] = []
    checkpoint_hashes = training_freeze["payload"][
        "denoiser_final_EMA_checkpoint_sha256_by_run"
    ]
    for spec in selected:
        if spec.outer_fold not in contexts:
            contexts[spec.outer_fold] = _build_fold_context(
                config=config,
                bundle=bundle,
                fold=folds[spec.outer_fold],
                g0_config=g0_config,
                g0_result=g0_result,
                training_freeze=training_freeze["payload"],
                formal_root=formal_root,
                device=device,
            )
        context = contexts[spec.outer_fold]
        group_key = (spec.outer_fold, spec.sampling_seed)
        if group_key not in groups:
            groups[group_key] = _common_sampling_group(
                context,
                bundle=bundle,
                sampling_seed=spec.sampling_seed,
                members=int(config["sampling"]["members"]),
                device=device,
            )
        _sample_one(
            config=config,
            config_sha=code["config_sha256"],
            training_freeze_sha=training_freeze["sha256"],
            formal_root=formal_root,
            evaluation_root=evaluation_root,
            spec=spec,
            context=context,
            group=groups[group_key],
            expected_checkpoint_sha=checkpoint_hashes[spec.run_key],
            alpha_bar=alpha_bar,
            device=device,
        )
        completed_this_invocation.append(spec.key)
        _write_progress(
            evaluation_root,
            matrix,
            selected_this_invocation=completed_this_invocation,
        )
    progress = _write_progress(
        evaluation_root,
        matrix,
        selected_this_invocation=completed_this_invocation,
    )
    sampling_freeze = _try_sampling_freeze(
        config=config,
        config_sha=code["config_sha256"],
        evaluation_root=evaluation_root,
        matrix=matrix,
        identity_sha=identity_sha,
    )
    result = (
        _execute_inference(
            config=config,
            config_sha=code["config_sha256"],
            evaluation_root=evaluation_root,
            matrix=matrix,
            sampling_freeze=sampling_freeze,
        )
        if sampling_freeze is not None
        else None
    )
    if result is not None:
        _write_progress(
            evaluation_root,
            matrix,
            selected_this_invocation=completed_this_invocation,
        )
    return {
        "schema": "architecture_v1_tgo_v1_evaluation_invocation_result_v1",
        "status": (
            result["status"]
            if result is not None
            else "TGO_V1_COMMON_SAMPLING_PARTIAL"
        ),
        "completed_this_invocation": len(completed_this_invocation),
        "completed_archives_total": progress["completed_archives"],
        "remaining_archives": progress["remaining_archives"],
        "sampling_freeze_written": sampling_freeze is not None,
        "formal_result_written": result is not None,
        "formal_result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute-sampling", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--archive-key", action="append", default=[])
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--list-archive-keys", action="store_true")
    arguments = parser.parse_args()
    if arguments.resume and not arguments.execute_sampling:
        parser.error("--resume requires --execute-sampling")
    if arguments.allow_cpu and not arguments.execute_sampling:
        parser.error("--allow-cpu is valid only with --execute-sampling")
    if arguments.archive_key and not arguments.execute_sampling:
        parser.error("--archive-key requires --execute-sampling")
    if arguments.max_archives is not None and not arguments.execute_sampling:
        parser.error("--max-archives requires --execute-sampling")
    config_path = arguments.config.resolve()
    value = (
        execute_sampling(
            config_path,
            resume=arguments.resume,
            device_name=arguments.device,
            allow_cpu=arguments.allow_cpu,
            archive_keys=arguments.archive_key,
            max_archives=arguments.max_archives,
        )
        if arguments.execute_sampling
        else dry_run(config_path, include_archive_keys=arguments.list_archive_keys)
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
