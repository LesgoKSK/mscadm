#!/usr/bin/env python3
"""Run the frozen architecture-v1 G0-B tiny-denoiser Probe.

The default command is target-free and prints the frozen plan.  The
``--execute-p0`` path alone materializes train targets, replays all six G0-A
nuisance folds, then performs six paired 50-update discarded-weight runs.
It never constructs a validation bank or reads another target role.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"
)
RESULT_SCHEMA = "architecture_v1_g0_b_tiny_denoiser_p0_result_v1"
MODULE_PATH = ROOT / "architecture_v1" / "g0b_tiny_denoiser.py"
RUNNER_PATH = Path(__file__).resolve()

from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.family_diffusion import MaskedJointDDPM
from architecture_v1.g0_predictability import (
    ModeGroup,
    build_mode_groups,
    chronological_folds,
    continuous_logit_target,
    cross_fitted_cellwise_predictions,
    derangement,
    fit_cellwise_ridge,
    fit_log_variance_glm,
    mask_nuisance_features,
    masked_band_energies,
    nwp_summary_features,
    select_cellwise_alpha,
    select_variance_alpha,
)
from architecture_v1.g0b_allocation import (
    gaussian_information,
    reserve_then_water_fill,
    weighted_information_budget,
)
from architecture_v1.g0b_tiny_denoiser import (
    MULAN_LITE_PARAMETERS,
    PATH_IDS,
    TINY_DENOISER_PARAMETERS,
    ConditionStandardizer,
    ModeProjectorBank,
    TinyDenoisingSystem,
    TinyEMA,
    module_state_sha256,
    proxy_reserve_then_water_fill,
    rank_weighted_fixed_variance,
    tensor_mapping_sha256,
    tiny_denoising_loss,
)
from architecture_v1.protocol import date_list_sha256


@dataclass(frozen=True)
class FoldNuisance:
    fold: int
    outer_train: np.ndarray
    outer_test: np.ndarray
    residual_train: np.ndarray
    residual_test: np.ndarray
    active_train: np.ndarray
    active_test: np.ndarray
    variance_train: np.ndarray
    variance_test: np.ndarray
    effective_train: np.ndarray
    effective_test: np.ndarray
    condition_train: np.ndarray
    condition_test: np.ndarray
    fixed_variance: np.ndarray

    @property
    def weights_train(self) -> np.ndarray:
        return self.effective_train / self.effective_train.sum(axis=1, keepdims=True)

    @property
    def weights_test(self) -> np.ndarray:
        return self.effective_test / self.effective_test.sum(axis=1, keepdims=True)


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
        raise FileNotFoundError(f"frozen artifact or sidecar is missing: {path}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
    digest = _sha256(path)
    if pieces[0] != digest:
        raise RuntimeError(f"frozen SHA256 mismatch: {path}")
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


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _validate_config(config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    digest = _verified_sidecar(path)
    if config.get("schema") != "architecture_v1_g0_b_tiny_denoiser_utility_v1":
        raise ValueError("unexpected G0-B tiny-denoiser config schema")
    if config.get("status") != (
        "frozen_before_any_G0_B_target_aware_P0_or_tiny_denoiser_training"
    ):
        raise RuntimeError("G0-B tiny-denoiser config is not frozen")
    if tuple(path_spec["id"] for path_spec in config["paths"]) != PATH_IDS:
        raise RuntimeError("six-path registry drifted")
    if config["role_access"]["P0_allowed_target_roles"] != ["train"]:
        raise RuntimeError("P0 target role must be train only")
    if set(config["role_access"]["forbidden_target_roles"]) != {
        "validation",
        "calibration",
        "selection",
        "r_seen",
        "final",
    }:
        raise RuntimeError("forbidden target-role registry drifted")
    p0 = config["P0_preflight"]
    if (
        int(p0["paths"]) != 6
        or int(p0["discarded_updates_per_path"]) != 50
        or p0["retained_weights"] is not False
        or int(p0["pilot_outer_fold"]) != 0
        or float(p0["pilot_fraction"]) != 0.25
        or int(p0["pilot_model_seed"]) != 3
    ):
        raise RuntimeError("P0 execution contract drifted")
    training = config["training"]
    if (
        int(training["batch_calendar_days"]) != 16
        or float(training["learning_rate"]) != 3e-4
        or float(training["weight_decay"]) != 1e-4
        or float(training["gradient_clip_norm"]) != 1.0
        or float(training["EMA_decay"]) != 0.999
        or training["automatic_mixed_precision"] is not False
        or training["deterministic_algorithms"] is not True
    ):
        raise RuntimeError("P0 optimizer/determinism contract drifted")

    lineage = config["lineage"]
    checked: dict[str, str] = {"config": digest}
    for key, hash_key in (
        ("data_protocol_config", "data_protocol_config_sha256"),
        ("g0_a_config", "g0_a_config_sha256"),
        ("g0_a_result", "g0_a_result_sha256"),
        ("g0_b0_config", "g0_b0_config_sha256"),
        ("g0_b0_result", "g0_b0_result_sha256"),
    ):
        artifact = ROOT / lineage[key]
        # The architecture-v1 data protocol predates the mandatory sidecar
        # convention; its frozen digest is nevertheless checked byte-for-byte.
        actual = (
            _verified_sidecar(artifact)
            if artifact.with_name(artifact.name + ".sha256").is_file()
            else _sha256(artifact)
        )
        if actual != lineage[hash_key]:
            raise RuntimeError(f"lineage artifact drifted: {key}")
        checked[key] = actual
    for relative, expected, label in (
        (
            ROOT / "architecture_v1" / "g0_predictability.py",
            lineage["g0_predictability_module_sha256"],
            "g0_predictability_module",
        ),
        (
            ROOT / "architecture_v1" / "g0b_allocation.py",
            lineage["g0_b0_allocation_module_sha256"],
            "g0_b0_allocation_module",
        ),
    ):
        actual = _sha256(relative)
        if actual != expected:
            raise RuntimeError(f"lineage code drifted: {label}")
        checked[label] = actual
    g0_a = _read_json(ROOT / lineage["g0_a_result"])
    g0_b0 = _read_json(ROOT / lineage["g0_b0_result"])
    if g0_a.get("status") != lineage["g0_a_required_status"]:
        raise RuntimeError("G0-A prerequisite is not GO")
    if g0_b0.get("status") != lineage["g0_b0_required_status"]:
        raise RuntimeError("G0-B0 prerequisite is not GO")
    base = str(lineage["repository_base_commit"])
    if not _git_is_ancestor(base):
        raise RuntimeError("registered repository base is not an ancestor of HEAD")
    checked["repository_base_commit"] = base
    checked["repository_execution_head"] = _git_head()
    checked["implementation_module"] = _sha256(MODULE_PATH)
    checked["runner"] = _sha256(RUNNER_PATH)
    return checked


def _mode_groups(g0_a_config: Mapping[str, Any]) -> tuple[ModeGroup, ...]:
    temporal = g0_a_config["mode_registry"]["temporal"]
    ranges = {
        label: temporal[label]["indices_half_open"]
        for label in ("low", "mid", "high")
    }
    return build_mode_groups(
        ranges, g0_a_config["mode_registry"]["ordered_groups"]
    )


def _nested_subset_indices(
    days: np.ndarray,
    outer_train: np.ndarray,
    *,
    fold: int,
    fraction: float,
    seed_root: int,
) -> np.ndarray:
    selected_days = days[outer_train].astype("datetime64[D]")
    tag = "architecture_v1_g0_b_tiny_v1"
    ranked = sorted(
        range(len(outer_train)),
        key=lambda index: (
            hashlib.sha256(
                f"{tag}|{seed_root + fold}|{selected_days[index]}".encode("ascii")
            ).digest(),
            str(selected_days[index]),
        ),
    )
    count = (
        len(outer_train)
        if fraction == 1.0
        else math.ceil(float(fraction) * len(outer_train))
    )
    local = np.asarray(ranked[:count], dtype=np.int64)
    return local[np.argsort(selected_days[local], kind="mergesort")]


def _verify_all_subset_hashes(
    config: Mapping[str, Any], days: np.ndarray, outer_folds: Sequence[np.ndarray]
) -> list[dict[str, Any]]:
    curve = config["learning_curve"]
    all_indices = np.arange(len(days), dtype=np.int64)
    records: list[dict[str, Any]] = []
    for fold, outer_test in enumerate(outer_folds):
        outer_train = np.setdiff1d(all_indices, outer_test, assume_unique=True)
        previous: set[int] = set()
        registry = curve["subset_registry"][str(fold)]
        for fraction in curve["fractions"]:
            local = _nested_subset_indices(
                days,
                outer_train,
                fold=fold,
                fraction=float(fraction),
                seed_root=int(curve["subset_seed_root"]),
            )
            global_indices = outer_train[local]
            actual = date_list_sha256(days[global_indices])
            frozen = registry[f"fraction_{fraction}"]
            if len(global_indices) != int(frozen["days"]):
                raise RuntimeError("nested subset day count drifted")
            if actual != frozen["date_sha256"]:
                raise RuntimeError("nested subset date hash drifted")
            current = set(global_indices.tolist())
            if not previous.issubset(current):
                raise RuntimeError("learning-curve subsets are not nested")
            previous = current
            records.append(
                {
                    "outer_fold": fold,
                    "fraction": float(fraction),
                    "days": int(len(global_indices)),
                    "date_sha256": actual,
                }
            )
    if len(records) != 18:
        raise RuntimeError("expected 18 registered fold/fraction subsets")
    return records


def _replay_nuisance(
    config: Mapping[str, Any],
) -> tuple[
    Any,
    tuple[ModeGroup, ...],
    list[FoldNuisance],
    dict[str, Any],
    list[dict[str, Any]],
]:
    lineage = config["lineage"]
    bundle = build_architecture_v1_train_data(
        config_path=ROOT / lineage["data_protocol_config"]
    )
    if bundle.materialized_roles != ("train",):
        raise RuntimeError("P0 materialized a non-train target role")
    access = bundle.manifest["formal_train_only_target_access"]
    if (
        access["materialized_roles"] != ["train"]
        or access["forbidden_target_arrays_materialized"] is not False
        or access["validation_bank_constructed"] is not False
    ):
        raise RuntimeError("train-only loader boundary drifted")
    expected_identity = {
        "train_days": int(lineage["expected_train_days"]),
        "train_date_sha256": lineage["expected_train_date_sha256"],
        "train_split_array_sha256": lineage["expected_train_split_array_sha256"],
        "train_only_data_bundle_sha256": lineage[
            "expected_train_only_data_bundle_sha256"
        ],
    }
    actual_identity = {
        "train_days": int(len(bundle.train)),
        "train_date_sha256": access["materialized_date_sha256"],
        "train_split_array_sha256": bundle.manifest["data_audit"][
            "split_array_sha256"
        ]["train"],
        "train_only_data_bundle_sha256": bundle.manifest[
            "train_only_data_bundle_sha256"
        ],
    }
    if actual_identity != expected_identity:
        raise RuntimeError("G0-B train-only data identity drifted")

    train = bundle.train
    g0_config = _read_json(ROOT / lineage["g0_a_config"])
    evidence = _read_json(ROOT / lineage["g0_a_result"])
    groups = _mode_groups(g0_config)
    names = [group.name for group in groups]
    latent, active = continuous_logit_target(
        train.target,
        train.state,
        train.observed_mask,
        epsilon=float(g0_config["continuous_target"]["logit_epsilon"]),
    )
    registry = g0_config["mode_registry"]
    _, effective_all = masked_band_energies(
        np.zeros_like(latent),
        active,
        groups,
        minimum_effective_rank=float(registry["minimum_effective_rank"]),
        energy_floor=float(registry["energy_floor"]),
    )
    mask_features = mask_nuisance_features(active, effective_all, groups)
    nwp_features, _ = nwp_summary_features(train.raw_condition, groups)
    outer_count = int(g0_config["cross_fitting"]["outer_folds"])
    inner_count = int(g0_config["cross_fitting"]["inner_folds"])
    all_indices = np.arange(len(train), dtype=np.int64)
    outer_folds = chronological_folds(all_indices, outer_count)
    subset_records = _verify_all_subset_hashes(config, train.day, outer_folds)
    frozen_by_day = {str(row["day"]): row for row in evidence["per_day"]}
    frozen_hyper = {
        int(row["outer_fold"]): row for row in evidence["selected_hyperparameters"]
    }
    mean_contract = g0_config["conditional_mean"]
    variance_contract = g0_config["uncertainty_models"]
    variance_clip = tuple(
        float(value) for value in variance_contract["prediction_variance_clip"]
    )
    maximum_errors = {
        "energy": 0.0,
        "effective_rank": 0.0,
        "N1_variance": 0.0,
        "selected_hyperparameter": 0.0,
    }
    folds: list[FoldNuisance] = []

    for fold, outer_test in enumerate(outer_folds):
        outer_train = np.setdiff1d(all_indices, outer_test, assume_unique=True)
        inner_global = chronological_folds(outer_train, inner_count)
        mean_alpha, _ = select_cellwise_alpha(
            train.raw_condition,
            latent,
            active,
            outer_train,
            inner_global,
            mean_contract["shared_ridge_alpha_grid"],
            minimum_observations=int(
                mean_contract["minimum_fit_observations_per_cell"]
            ),
        )
        nested_prediction = cross_fitted_cellwise_predictions(
            train.raw_condition,
            latent,
            active,
            outer_train,
            inner_global,
            alpha=mean_alpha,
            minimum_observations=int(
                mean_contract["minimum_fit_observations_per_cell"]
            ),
        )
        mean_model = fit_cellwise_ridge(
            train.raw_condition,
            latent,
            active,
            outer_train,
            alpha=mean_alpha,
            minimum_observations=int(
                mean_contract["minimum_fit_observations_per_cell"]
            ),
        )
        test_prediction = mean_model.predict(train.raw_condition[outer_test])
        residual_train = np.where(
            active[outer_train],
            latent[outer_train] - nested_prediction[outer_train],
            0.0,
        )
        residual_test = np.where(
            active[outer_test], latent[outer_test] - test_prediction, 0.0
        )
        energy_train, effective_train = masked_band_energies(
            residual_train,
            active[outer_train],
            groups,
            minimum_effective_rank=float(registry["minimum_effective_rank"]),
            energy_floor=float(registry["energy_floor"]),
        )
        energy_test, effective_test = masked_band_energies(
            residual_test,
            active[outer_test],
            groups,
            minimum_effective_rank=float(registry["minimum_effective_rank"]),
            energy_floor=float(registry["energy_floor"]),
        )
        train_mask_x = mask_features[outer_train]
        test_mask_x = mask_features[outer_test]
        train_nwp_x = np.concatenate(
            [train_mask_x, nwp_features[outer_train]], axis=1
        )
        test_nwp_x = np.concatenate(
            [test_mask_x, nwp_features[outer_test]], axis=1
        )
        inner_local = chronological_folds(np.arange(len(outer_train)), inner_count)
        mask_alpha, _ = select_variance_alpha(
            train_mask_x,
            energy_train,
            effective_train,
            inner_local,
            variance_contract["ridge_alpha_grid"],
            variance_clip=variance_clip,
            maximum_iterations=int(
                variance_contract["maximum_newton_iterations"]
            ),
            tolerance=float(variance_contract["newton_tolerance"]),
        )
        nwp_alpha, _ = select_variance_alpha(
            train_nwp_x,
            energy_train,
            effective_train,
            inner_local,
            variance_contract["ridge_alpha_grid"],
            variance_clip=variance_clip,
            maximum_iterations=int(
                variance_contract["maximum_newton_iterations"]
            ),
            tolerance=float(variance_contract["newton_tolerance"]),
        )
        variance_train = np.empty((len(outer_train), 6), dtype=np.float64)
        variance_test = np.empty((len(outer_test), 6), dtype=np.float64)
        for band in range(6):
            model = fit_log_variance_glm(
                train_nwp_x,
                energy_train[:, band],
                effective_train[:, band],
                alpha=nwp_alpha,
                variance_clip=variance_clip,
                maximum_iterations=int(
                    variance_contract["maximum_newton_iterations"]
                ),
                tolerance=float(variance_contract["newton_tolerance"]),
            )
            variance_train[:, band] = model.predict(train_nwp_x)
            variance_test[:, band] = model.predict(test_nwp_x)

        selected = frozen_hyper[fold]
        for key, actual in (
            ("mean_ridge_alpha", mean_alpha),
            ("mask_variance_ridge_alpha", mask_alpha),
            ("nwp_variance_ridge_alpha", nwp_alpha),
        ):
            error = abs(float(actual) - float(selected[key]))
            maximum_errors["selected_hyperparameter"] = max(
                maximum_errors["selected_hyperparameter"], error
            )
        frozen_energy = np.asarray(
            [
                [frozen_by_day[str(day)]["energy"][name] for name in names]
                for day in train.day[outer_test].astype(str)
            ],
            dtype=np.float64,
        )
        frozen_effective = np.asarray(
            [
                [
                    frozen_by_day[str(day)]["effective_rank"][name]
                    for name in names
                ]
                for day in train.day[outer_test].astype(str)
            ],
            dtype=np.float64,
        )
        frozen_variance = np.asarray(
            [
                [frozen_by_day[str(day)]["N1_variance"][name] for name in names]
                for day in train.day[outer_test].astype(str)
            ],
            dtype=np.float64,
        )
        maximum_errors["energy"] = max(
            maximum_errors["energy"],
            float(np.max(np.abs(energy_test - frozen_energy))),
        )
        maximum_errors["effective_rank"] = max(
            maximum_errors["effective_rank"],
            float(np.max(np.abs(effective_test - frozen_effective))),
        )
        maximum_errors["N1_variance"] = max(
            maximum_errors["N1_variance"],
            float(np.max(np.abs(variance_test - frozen_variance))),
        )
        expected_fold_hash = config["outer_cross_fitting"][
            "outer_test_date_sha256"
        ][str(fold)]
        if date_list_sha256(train.day[outer_test]) != expected_fold_hash:
            raise RuntimeError("outer-test fold date identity drifted")

        raw_condition_train = np.concatenate(
            [train_nwp_x, np.log(variance_train)], axis=1
        )
        raw_condition_test = np.concatenate(
            [test_nwp_x, np.log(variance_test)], axis=1
        )
        standardizer = ConditionStandardizer.fit(raw_condition_train)
        fixed_variance = rank_weighted_fixed_variance(
            variance_train, effective_train
        )
        folds.append(
            FoldNuisance(
                fold=fold,
                outer_train=outer_train,
                outer_test=outer_test,
                residual_train=residual_train,
                residual_test=residual_test,
                active_train=active[outer_train],
                active_test=active[outer_test],
                variance_train=variance_train,
                variance_test=variance_test,
                effective_train=effective_train,
                effective_test=effective_test,
                condition_train=standardizer.transform(raw_condition_train),
                condition_test=standardizer.transform(raw_condition_test),
                fixed_variance=fixed_variance,
            )
        )

    tolerance = 1e-10
    if any(value > tolerance for value in maximum_errors.values()):
        raise RuntimeError(f"G0-A nuisance replay exceeded tolerance: {maximum_errors}")
    return (
        bundle,
        groups,
        folds,
        {
            "tolerance": tolerance,
            "maximum_absolute_errors": maximum_errors,
            "folds_replayed": len(folds),
            "outer_test_days_replayed": int(sum(len(fold.outer_test) for fold in folds)),
            "selected_hyperparameters_match": True,
        },
        subset_records,
    )


def _baseline_schedule(config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    spec = config["baseline_information_contract"]
    diffusion = MaskedJointDDPM(
        timesteps=int(spec["timesteps"]),
        cosine_offset=float(spec["cosine_offset"]),
        beta_min=float(spec["beta_min"]),
        beta_max=float(spec["beta_max"]),
    )
    tensor = diffusion.alpha_bar.detach().cpu().numpy()
    if tensor.dtype != np.float32:
        raise RuntimeError("baseline alpha_bar dtype drifted")
    digest = hashlib.sha256(
        tensor.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    if digest != spec["alpha_bar_float32_sha256"]:
        raise RuntimeError("baseline cosine schedule hash drifted")
    return tensor.astype(np.float64), {
        "timesteps": int(len(tensor)),
        "float32_sha256": digest,
        "clean_endpoint_alpha_bar": float(tensor[0]),
        "noisy_endpoint_alpha_bar": float(tensor[-1]),
    }


def _projector_audit(
    groups: Sequence[ModeGroup], folds: Sequence[FoldNuisance]
) -> dict[str, Any]:
    symmetric = 0.0
    idempotent = 0.0
    pairwise = 0.0
    for group in groups:
        symmetric = max(
            symmetric,
            float(np.max(np.abs(group.spatial_projector.T - group.spatial_projector))),
            float(np.max(np.abs(group.temporal_projector.T - group.temporal_projector))),
        )
        idempotent = max(
            idempotent,
            float(
                np.max(
                    np.abs(
                        group.spatial_projector @ group.spatial_projector
                        - group.spatial_projector
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        group.temporal_projector @ group.temporal_projector
                        - group.temporal_projector
                    )
                )
            ),
        )
    for left in range(6):
        for right in range(left + 1, 6):
            spatial_product = groups[left].spatial_projector @ groups[right].spatial_projector
            temporal_product = groups[right].temporal_projector @ groups[left].temporal_projector
            pairwise = max(
                pairwise,
                float(np.max(np.abs(spatial_product)))
                * float(np.max(np.abs(temporal_product))),
            )

    def project(value: np.ndarray, group: ModeGroup) -> np.ndarray:
        return np.einsum(
            "ij,djh,hl->dil",
            group.spatial_projector,
            value,
            group.temporal_projector,
            optimize=True,
        )

    residual = folds[0].residual_train
    parts = np.stack([project(residual, group) for group in groups], axis=1)
    reconstruction = parts.sum(axis=1)
    partition_error = float(np.max(np.abs(reconstruction - residual)))
    variance = folds[0].variance_train
    whitened = np.sum(parts / np.sqrt(variance)[:, :, None, None], axis=1)
    white_parts = np.stack([project(whitened, group) for group in groups], axis=1)
    unwhitened = np.sum(
        white_parts * np.sqrt(variance)[:, :, None, None], axis=1
    )
    active = folds[0].active_train
    cw_error = float(np.max(np.abs(unwhitened[active] - residual[active])))
    tolerance = 1e-10
    if max(symmetric, idempotent, pairwise, partition_error, cw_error) > tolerance:
        raise RuntimeError("projector or CW inverse audit failed")
    return {
        "tolerance": tolerance,
        "symmetry_max_abs_error": symmetric,
        "idempotence_max_abs_error": idempotent,
        "pairwise_orthogonality_max_abs_error": pairwise,
        "partition_identity_max_abs_error": partition_error,
        "CW_active_inverse_max_abs_error": cw_error,
    }


def _schedule_audit(
    path_id: str,
    true_variance: np.ndarray,
    weights: np.ndarray,
    baseline_alpha: np.ndarray,
    config: Mapping[str, Any],
    *,
    proxy_variance: np.ndarray | None = None,
) -> dict[str, Any]:
    true = np.asarray(true_variance, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if true.shape != weight.shape or true.ndim != 2 or true.shape[1] != 6:
        raise ValueError("schedule audit inputs must have shape [day,6]")
    if proxy_variance is not None:
        proxy = np.asarray(proxy_variance, dtype=np.float64)
        if proxy.shape != true.shape:
            raise ValueError("schedule proxy shape drifted")
    else:
        proxy = None
    baseline_snr = baseline_alpha / (1.0 - baseline_alpha)
    maximum_budget_error = 0.0
    maximum_forward_snr_delta = -np.inf
    all_finite = True
    all_positive = True
    previous_snr: np.ndarray | None = None
    clean_alpha: np.ndarray | None = None
    noisy_alpha: np.ndarray | None = None
    eta = float(config["shared_proxy_allocation"]["eta"])
    for step, ratio in enumerate(baseline_snr):
        iid_information = gaussian_information(true, float(ratio))
        budget = weighted_information_budget(iid_information, weight)
        if path_id == "IID":
            information = iid_information
            snr = np.full_like(true, float(ratio))
            alpha = np.full_like(true, float(baseline_alpha[step]))
        elif path_id == "CW_GROUP":
            information = np.broadcast_to(budget[:, None], true.shape)
            scalar_snr = np.expm1(2.0 * budget)
            snr = np.broadcast_to(scalar_snr[:, None], true.shape)
            alpha = snr / (1.0 + snr)
        else:
            if proxy is None:
                raise ValueError(f"{path_id} requires a schedule proxy")
            allocated = proxy_reserve_then_water_fill(
                true,
                weight,
                proxy,
                float(ratio),
                eta=eta,
                tolerance=float(
                    config["baseline_information_contract"][
                        "budget_absolute_tolerance"
                    ]
                ),
            )
            information = allocated.information
            snr = allocated.snr
            alpha = allocated.alpha_bar
        error = weighted_information_budget(information, weight) - budget
        maximum_budget_error = max(
            maximum_budget_error, float(np.max(np.abs(error)))
        )
        all_finite = all_finite and all(
            np.isfinite(value).all() for value in (information, snr, alpha)
        )
        all_positive = all_positive and bool(np.all(snr > 0.0)) and bool(
            np.all(alpha > 0.0)
        )
        if previous_snr is not None:
            maximum_forward_snr_delta = max(
                maximum_forward_snr_delta,
                float(np.max(snr - previous_snr)),
            )
        previous_snr = np.asarray(snr)
        if step == 0:
            clean_alpha = np.asarray(alpha)
        if step == len(baseline_snr) - 1:
            noisy_alpha = np.asarray(alpha)
    assert clean_alpha is not None and noisy_alpha is not None
    contract = config["baseline_information_contract"]
    summary = {
        "path": path_id,
        "days": int(len(true)),
        "timesteps": int(len(baseline_alpha)),
        "maximum_budget_abs_error": maximum_budget_error,
        "maximum_forward_snr_delta": maximum_forward_snr_delta,
        "all_finite": bool(all_finite),
        "all_snr_and_alpha_strictly_positive": bool(all_positive),
        "clean_endpoint_alpha_bar_min": float(clean_alpha.min()),
        "noisy_endpoint_alpha_bar_max": float(noisy_alpha.max()),
    }
    if maximum_budget_error > float(contract["budget_absolute_tolerance"]):
        raise RuntimeError(f"{path_id} changed the daily information budget")
    if maximum_forward_snr_delta > float(
        contract["schedule_monotonicity_tolerance"]
    ):
        raise RuntimeError(f"{path_id} schedule is not monotone")
    if not all_finite or not all_positive:
        raise RuntimeError(f"{path_id} schedule is non-finite or non-positive")
    if float(clean_alpha.min()) < float(contract["clean_endpoint_minimum_alpha_bar"]):
        raise RuntimeError(f"{path_id} clean endpoint is too noisy")
    if float(noisy_alpha.max()) > float(contract["noisy_endpoint_maximum_alpha_bar"]):
        raise RuntimeError(f"{path_id} noisy endpoint is too clean")
    return summary


def _registered_derangements(
    size: int, *, fold: int, partition: str, seed_root: int, count: int
) -> tuple[np.ndarray, ...]:
    if partition not in {"training", "evaluation"}:
        raise ValueError("unknown shuffle partition")
    offset = 0 if partition == "training" else count
    return tuple(
        derangement(size, seed_root + fold * (2 * count) + offset + index)
        for index in range(count)
    )


def _pretraining_schedule_audits(
    config: Mapping[str, Any],
    folds: Sequence[FoldNuisance],
    baseline_alpha: np.ndarray,
    train_days: np.ndarray,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, ...]]]:
    summaries: dict[str, list[dict[str, Any]]] = {
        path: [] for path in PATH_IDS
    }
    shuffle_registry: dict[str, tuple[np.ndarray, ...]] = {}
    shuffle_spec = next(path for path in config["paths"] if path["id"] == "PA_SHUFFLE")
    seed_root = int(shuffle_spec["derangement_seed_root"])
    count = int(shuffle_spec["training_derangements"])
    curve = config["learning_curve"]
    for fold_data in folds:
        fold = fold_data.fold
        complete_true = np.concatenate(
            [fold_data.variance_train, fold_data.variance_test], axis=0
        )
        complete_weights = np.concatenate(
            [fold_data.weights_train, fold_data.weights_test], axis=0
        )
        fixed_proxy = np.broadcast_to(
            fold_data.fixed_variance[None], complete_true.shape
        )
        for path_id, proxy in (
            ("IID", None),
            ("CW_GROUP", None),
            ("FIXED_BAND", fixed_proxy),
            ("MULAN_LITE", fixed_proxy),
            ("PA_RWF", complete_true),
        ):
            summaries[path_id].append(
                _schedule_audit(
                    path_id,
                    complete_true,
                    complete_weights,
                    baseline_alpha,
                    config,
                    proxy_variance=proxy,
                )
            )

        local = _nested_subset_indices(
            train_days,
            fold_data.outer_train,
            fold=fold,
            fraction=0.25,
            seed_root=int(curve["subset_seed_root"]),
        )
        train_permutations = _registered_derangements(
            len(local),
            fold=fold,
            partition="training",
            seed_root=seed_root,
            count=count,
        )
        evaluation_permutations = _registered_derangements(
            len(fold_data.outer_test),
            fold=fold,
            partition="evaluation",
            seed_root=seed_root,
            count=count,
        )
        shuffle_registry[f"fold_{fold}_training"] = train_permutations
        shuffle_registry[f"fold_{fold}_evaluation"] = evaluation_permutations
        for partition, true, weight, permutations in (
            (
                "training",
                fold_data.variance_train[local],
                fold_data.weights_train[local],
                train_permutations,
            ),
            (
                "evaluation",
                fold_data.variance_test,
                fold_data.weights_test,
                evaluation_permutations,
            ),
        ):
            for permutation_index, permutation in enumerate(permutations):
                if bool(np.any(permutation == np.arange(len(permutation)))):
                    raise RuntimeError("PA_SHUFFLE derangement has a fixed point")
                item = _schedule_audit(
                    "PA_SHUFFLE",
                    true,
                    weight,
                    baseline_alpha,
                    config,
                    proxy_variance=true[permutation],
                )
                item["outer_fold"] = fold
                item["partition"] = partition
                item["permutation"] = permutation_index
                summaries["PA_SHUFFLE"].append(item)

    pa_error = 0.0
    for fold_data in folds:
        true = np.concatenate(
            [fold_data.variance_train, fold_data.variance_test], axis=0
        )
        weight = np.concatenate(
            [fold_data.weights_train, fold_data.weights_test], axis=0
        )
        for alpha in baseline_alpha:
            ratio = float(alpha / (1.0 - alpha))
            reference = reserve_then_water_fill(
                true,
                weight,
                ratio,
                eta=float(config["shared_proxy_allocation"]["eta"]),
                tolerance=float(
                    config["baseline_information_contract"][
                        "budget_absolute_tolerance"
                    ]
                ),
            )
            replay = proxy_reserve_then_water_fill(
                true,
                weight,
                true,
                ratio,
                eta=float(config["shared_proxy_allocation"]["eta"]),
                tolerance=float(
                    config["baseline_information_contract"][
                        "budget_absolute_tolerance"
                    ]
                ),
            )
            pa_error = max(
                pa_error,
                float(np.max(np.abs(reference.alpha_bar - replay.alpha_bar))),
                float(np.max(np.abs(reference.information - replay.information))),
            )
    if pa_error > 1e-12:
        raise RuntimeError("PA_RWF no longer exactly replays G0-B0 eta=0.5")
    result = {
        "all_six_folds_audited": True,
        "fixed_paths": summaries,
        "PA_RWF_G0_B0_max_abs_error": pa_error,
        "shuffle_derangement_seed_formula": (
            "seed_root + outer_fold*(2*count) + partition_offset + permutation; "
            "partition_offset=0 training,count evaluation"
        ),
        "shuffle_fixed_points": 0,
    }
    return result, shuffle_registry


def _random_bank(
    *, seed: int, updates: int, batch_size: int, subset_size: int, timesteps: int
) -> tuple[dict[str, torch.Tensor], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bank = {
        "day_index": torch.randint(
            subset_size, (updates, batch_size), generator=generator, dtype=torch.long
        ),
        "timestep": torch.randint(
            timesteps, (updates, batch_size), generator=generator, dtype=torch.long
        ),
        "noise": torch.randn(
            updates, batch_size, 10, 24, generator=generator, dtype=torch.float32
        ),
    }
    return bank, tensor_mapping_sha256(bank)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(
            left, right
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def _tree_sha256(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
    return True


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
    # A CUDA ``map_location`` also moves the serialized CPU RNG tensor unless
    # we normalize it explicitly.  PyTorch's CPU and CUDA RNG restore APIs
    # both require CPU ByteTensors even when the model checkpoint is loaded
    # onto a CUDA device.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["torch_cuda"]]
        )


def _batch_tensors(
    subset: Mapping[str, torch.Tensor],
    random_bank: Mapping[str, torch.Tensor],
    update: int,
    device: torch.device,
    *,
    shuffle_permutations: Sequence[torch.Tensor] | None,
) -> dict[str, torch.Tensor | None]:
    local_index_cpu = random_bank["day_index"][update]
    local_index = local_index_cpu.to(device)
    result: dict[str, torch.Tensor | None] = {
        "residual": subset["residual"][local_index],
        "active": subset["active"][local_index],
        "condition": subset["condition"][local_index],
        "variance": subset["variance"][local_index],
        "weights": subset["weights"][local_index],
        "timestep": random_bank["timestep"][update].to(device),
        "noise": random_bank["noise"][update].to(device),
        "shuffle_proxy": None,
    }
    if shuffle_permutations is not None:
        permutation = shuffle_permutations[update % len(shuffle_permutations)]
        wrong_index = permutation[local_index]
        result["shuffle_proxy"] = subset["variance"][wrong_index]
    return result


def _training_update(
    system: TinyDenoisingSystem,
    optimizer: torch.optim.Optimizer,
    ema: TinyEMA,
    projectors: ModeProjectorBank,
    subset: Mapping[str, torch.Tensor],
    random_bank: Mapping[str, torch.Tensor],
    update: int,
    device: torch.device,
    baseline_alpha: torch.Tensor,
    fixed_variance: torch.Tensor,
    config: Mapping[str, Any],
    *,
    shuffle_permutations: Sequence[torch.Tensor] | None,
) -> tuple[float, float]:
    batch = _batch_tensors(
        subset,
        random_bank,
        update,
        device,
        shuffle_permutations=shuffle_permutations,
    )
    optimizer.zero_grad(set_to_none=True)
    sample = tiny_denoising_loss(
        system,
        projectors,
        batch["residual"],
        batch["active"],
        batch["condition"],
        batch["variance"],
        batch["weights"],
        fixed_variance,
        batch["timestep"],
        batch["noise"],
        baseline_alpha,
        shuffle_proxy=batch["shuffle_proxy"],
        eta=float(config["shared_proxy_allocation"]["eta"]),
    )
    sample.loss.backward()
    for name, parameter in system.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"registered trainable parameter has no gradient: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"gradient is non-finite: {name}")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        system.parameters(), float(config["training"]["gradient_clip_norm"])
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("gradient norm is non-finite")
    optimizer.step()
    if not _optimizer_finite(optimizer):
        raise FloatingPointError("optimizer state is non-finite")
    ema.update(system)
    return float(sample.loss.detach().cpu()), float(gradient_norm.detach().cpu())


def _mulan_proxy(
    system: TinyDenoisingSystem,
    condition: torch.Tensor,
    fixed_variance: torch.Tensor,
) -> np.ndarray:
    if system.scheduler is None:
        raise RuntimeError("MuLAN-lite schedule head is absent")
    with torch.no_grad():
        return (
            system.scheduler(condition, fixed_variance)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def _run_paired_training(
    config: Mapping[str, Any],
    fold: FoldNuisance,
    train_days: np.ndarray,
    groups: Sequence[ModeGroup],
    baseline_alpha_numpy: np.ndarray,
    shuffle_registry: Mapping[str, tuple[np.ndarray, ...]],
    device: torch.device,
    config_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    p0 = config["P0_preflight"]
    training = config["training"]
    updates = int(p0["discarded_updates_per_path"])
    batch_size = int(training["batch_calendar_days"])
    fraction = float(p0["pilot_fraction"])
    model_seed = int(p0["pilot_model_seed"])
    local = _nested_subset_indices(
        train_days,
        fold.outer_train,
        fold=fold.fold,
        fraction=fraction,
        seed_root=int(config["learning_curve"]["subset_seed_root"]),
    )
    selected_days = train_days[fold.outer_train[local]]
    expected_hash = config["learning_curve"]["subset_registry"][str(fold.fold)][
        f"fraction_{fraction}"
    ]["date_sha256"]
    if date_list_sha256(selected_days) != expected_hash:
        raise RuntimeError("P0 subset hash drifted")
    weights = fold.weights_train[local]
    subset = {
        "residual": torch.as_tensor(
            fold.residual_train[local], dtype=torch.float32, device=device
        ),
        "active": torch.as_tensor(
            fold.active_train[local], dtype=torch.bool, device=device
        ),
        "condition": torch.as_tensor(
            fold.condition_train[local], dtype=torch.float32, device=device
        ),
        "variance": torch.as_tensor(
            fold.variance_train[local], dtype=torch.float32, device=device
        ),
        "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
    }
    fixed_variance = torch.as_tensor(
        fold.fixed_variance, dtype=torch.float32, device=device
    )
    baseline_alpha = torch.as_tensor(
        baseline_alpha_numpy, dtype=torch.float32, device=device
    )
    projectors = ModeProjectorBank(groups).to(device)
    fraction_index = list(config["learning_curve"]["fractions"]).index(fraction)
    random_seed = (
        int(training["training_random_seed_root"])
        + fold.fold * 10_000
        + fraction_index * 1_000
        + model_seed
    )
    random_bank, random_bank_sha = _random_bank(
        seed=random_seed,
        updates=updates,
        batch_size=batch_size,
        subset_size=len(local),
        timesteps=len(baseline_alpha),
    )
    random_bank_device = {
        "day_index": random_bank["day_index"],
        "timestep": random_bank["timestep"],
        "noise": random_bank["noise"],
    }
    permutation_numpy = shuffle_registry[f"fold_{fold.fold}_training"]
    shuffle_permutations = tuple(
        torch.as_tensor(value, dtype=torch.long, device=device)
        for value in permutation_numpy
    )
    results: list[dict[str, Any]] = []
    initial_hashes: dict[str, str] = {}
    resume_audit: dict[str, Any] | None = None
    learned_schedule_audits: list[dict[str, Any]] = []

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    for path_id in PATH_IDS:
        system = TinyDenoisingSystem(path_id, model_seed=model_seed).to(device)
        base_hash = module_state_sha256(system.denoiser)
        initial_hashes[path_id] = base_hash
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        ema = TinyEMA(system, decay=float(training["EMA_decay"]))
        parameter_count = sum(parameter.numel() for parameter in system.parameters())
        expected_count = TINY_DENOISER_PARAMETERS + (
            MULAN_LITE_PARAMETERS if path_id == "MULAN_LITE" else 0
        )
        if parameter_count != expected_count:
            raise RuntimeError(f"{path_id} parameter count drifted")
        if path_id == "MULAN_LITE":
            proxy_zero = _mulan_proxy(system, subset["condition"], fixed_variance)
            fixed_float32 = (
                fixed_variance.detach().cpu().numpy().astype(np.float64)
            )
            fixed_proxy = np.broadcast_to(fixed_float32[None], proxy_zero.shape)
            if not np.array_equal(proxy_zero, fixed_proxy):
                raise RuntimeError("MuLAN-lite does not initialize exactly at FIXED_BAND")
            initial_audit = _schedule_audit(
                "MULAN_LITE",
                fold.variance_train[local],
                weights,
                baseline_alpha_numpy,
                config,
                proxy_variance=proxy_zero,
            )
            initial_audit["after_update"] = 0
            learned_schedule_audits.append(initial_audit)

        losses: list[float] = []
        gradient_norms: list[float] = []
        path_shuffle = shuffle_permutations if path_id == "PA_SHUFFLE" else None
        with tempfile.TemporaryDirectory(prefix="g0b_p0_resume_", dir="/tmp") as temp:
            checkpoint_path = Path(temp) / "discarded_resume_checkpoint.pt"
            for update in range(updates):
                loss, gradient_norm = _training_update(
                    system,
                    optimizer,
                    ema,
                    projectors,
                    subset,
                    random_bank_device,
                    update,
                    device,
                    baseline_alpha,
                    fixed_variance,
                    config,
                    shuffle_permutations=path_shuffle,
                )
                losses.append(loss)
                gradient_norms.append(gradient_norm)
                if path_id == "MULAN_LITE":
                    proxy = _mulan_proxy(system, subset["condition"], fixed_variance)
                    audit = _schedule_audit(
                        "MULAN_LITE",
                        fold.variance_train[local],
                        weights,
                        baseline_alpha_numpy,
                        config,
                        proxy_variance=proxy,
                    )
                    audit["after_update"] = update + 1
                    learned_schedule_audits.append(audit)

                if path_id == "PA_RWF" and update == 24:
                    checkpoint = {
                        "schema": "architecture_v1_g0_b_tiny_p0_resume_v1",
                        "config_sha256": config_sha,
                        "random_bank_sha256": random_bank_sha,
                        "path": path_id,
                        "completed_updates": update + 1,
                        "system": system.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(),
                        "rng": _rng_state(),
                    }
                    torch.save(checkpoint, checkpoint_path)
                if path_id == "PA_RWF" and update == 25:
                    main_system = {
                        key: value.detach().cpu().clone()
                        for key, value in system.state_dict().items()
                    }
                    main_optimizer = optimizer.state_dict()
                    main_ema = ema.state_dict()
                    main_rng = _rng_state()
                    loaded = torch.load(
                        checkpoint_path, map_location=device, weights_only=False
                    )
                    resumed_system = TinyDenoisingSystem(
                        path_id, model_seed=model_seed
                    ).to(device)
                    resumed_optimizer = torch.optim.AdamW(
                        resumed_system.parameters(),
                        lr=float(training["learning_rate"]),
                        weight_decay=float(training["weight_decay"]),
                    )
                    resumed_ema = TinyEMA(
                        resumed_system, decay=float(training["EMA_decay"])
                    )
                    if (
                        loaded["config_sha256"] != config_sha
                        or loaded["random_bank_sha256"] != random_bank_sha
                        or loaded["completed_updates"] != 25
                    ):
                        raise RuntimeError("resume checkpoint identity drifted")
                    resumed_system.load_state_dict(loaded["system"])
                    resumed_optimizer.load_state_dict(loaded["optimizer"])
                    resumed_ema.load_state_dict(loaded["ema"], resumed_system)
                    _set_rng_state(loaded["rng"])
                    resumed_loss, _ = _training_update(
                        resumed_system,
                        resumed_optimizer,
                        resumed_ema,
                        projectors,
                        subset,
                        random_bank_device,
                        update,
                        device,
                        baseline_alpha,
                        fixed_variance,
                        config,
                        shuffle_permutations=None,
                    )
                    exact = (
                        resumed_loss == loss
                        and _tree_equal(resumed_system.state_dict(), main_system)
                        and _tree_equal(resumed_optimizer.state_dict(), main_optimizer)
                        and _tree_equal(resumed_ema.state_dict(), main_ema)
                    )
                    if not exact:
                        raise RuntimeError("checkpoint resume did not reproduce update 26")
                    resume_audit = {
                        "path": path_id,
                        "checkpoint_after_updates": 25,
                        "replayed_update": 26,
                        "loss_exact": resumed_loss == loss,
                        "model_optimizer_EMA_exact": exact,
                        "temporary_checkpoint_deleted": False,
                    }
                    del resumed_ema, resumed_optimizer, resumed_system, loaded
                    _set_rng_state(main_rng)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if path_id == "PA_RWF" and resume_audit is not None:
                resume_audit["temporary_checkpoint_deleted"] = not checkpoint_path.exists()

        diagnostic_batch = _batch_tensors(
            subset,
            random_bank_device,
            updates - 1,
            device,
            shuffle_permutations=path_shuffle,
        )
        with torch.no_grad():
            diagnostic = tiny_denoising_loss(
                system,
                projectors,
                diagnostic_batch["residual"],
                diagnostic_batch["active"],
                diagnostic_batch["condition"],
                diagnostic_batch["variance"],
                diagnostic_batch["weights"],
                fixed_variance,
                diagnostic_batch["timestep"],
                diagnostic_batch["noise"],
                baseline_alpha,
                shuffle_proxy=diagnostic_batch["shuffle_proxy"],
                eta=float(config["shared_proxy_allocation"]["eta"]),
            )
        diagnostic_loss = float(diagnostic.loss.detach().cpu())
        if not np.isfinite(diagnostic_loss):
            raise FloatingPointError("train-subset diagnostic loss is non-finite")
        results.append(
            {
                "path": path_id,
                "updates_completed": updates,
                "calendar_day_exposures": updates * batch_size,
                "parameters": parameter_count,
                "base_initialization_sha256": base_hash,
                "common_random_bank_sha256": random_bank_sha,
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "loss_min": min(losses),
                "loss_max": max(losses),
                "gradient_norm_max_before_clipping": max(gradient_norms),
                "train_subset_diagnostic_loss": diagnostic_loss,
                "final_online_state_sha256_before_discard": module_state_sha256(system),
                "final_EMA_state_sha256_before_discard": tensor_mapping_sha256(
                    ema.shadow
                ),
                "final_optimizer_state_sha256_before_discard": _tree_sha256(
                    optimizer.state_dict()
                ),
                "finite": True,
                "weights_discarded": True,
            }
        )
        del diagnostic, ema, optimizer, system
        if device.type == "cuda":
            torch.cuda.empty_cache()

    unique_initial = set(initial_hashes.values())
    if len(unique_initial) != 1:
        raise RuntimeError("base tiny-denoiser initialization is not paired")
    if resume_audit is None or not resume_audit["temporary_checkpoint_deleted"]:
        raise RuntimeError("resume audit did not delete its temporary checkpoint")
    if len(learned_schedule_audits) != updates + 1:
        raise RuntimeError("MuLAN-lite was not schedule-audited after every P0 update")
    paired = {
        "outer_fold": fold.fold,
        "fraction": fraction,
        "model_seed": model_seed,
        "subset_days": int(len(local)),
        "subset_date_sha256": date_list_sha256(selected_days),
        "training_random_seed": random_seed,
        "random_bank_sha256": random_bank_sha,
        "unique_base_initialization_hashes": len(unique_initial),
        "base_initialization_sha256": next(iter(unique_initial)),
        "all_paths_share_random_bank": all(
            row["common_random_bank_sha256"] == random_bank_sha for row in results
        ),
        "MuLAN_schedule_audits": len(learned_schedule_audits),
        "MuLAN_schedule_worst_budget_abs_error": max(
            row["maximum_budget_abs_error"] for row in learned_schedule_audits
        ),
        "MuLAN_schedule_worst_forward_snr_delta": max(
            row["maximum_forward_snr_delta"] for row in learned_schedule_audits
        ),
    }
    return results, paired, resume_audit


def run_p0(
    config_path: Path, output_root: Path, *, device_name: str
) -> tuple[dict[str, Any], str]:
    config = _read_json(config_path)
    lineage_hashes = _validate_config(config, config_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            "P0 output directory is non-empty; frozen protocol forbids overwrite/resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(timezone.utc)
    bundle, groups, folds, nuisance_audit, subset_registry = _replay_nuisance(config)
    baseline_alpha, baseline_identity = _baseline_schedule(config)
    projector_audit = _projector_audit(groups, folds)
    schedule_audit, shuffle_registry = _pretraining_schedule_audits(
        config, folds, baseline_alpha, bundle.train.day
    )
    path_runs, paired_audit, resume_audit = _run_paired_training(
        config,
        folds[int(config["P0_preflight"]["pilot_outer_fold"])],
        bundle.train.day,
        groups,
        baseline_alpha,
        shuffle_registry,
        device,
        lineage_hashes["config"],
    )
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    maximum_gpu_bytes = int(
        float(config["P0_preflight"]["maximum_peak_GPU_memory_GiB"])
        * (1024**3)
    )
    gpu_within_limit = peak_gpu_bytes <= maximum_gpu_bytes
    if not gpu_within_limit:
        raise RuntimeError("P0 exceeded its frozen peak GPU-memory limit")
    if len(path_runs) != 6 or sum(row["updates_completed"] for row in path_runs) != 300:
        raise RuntimeError("P0 path/update count is incomplete")
    weight_files = sorted(
        str(path.relative_to(output_root))
        for suffix in ("*.pt", "*.pth")
        for path in output_root.rglob(suffix)
    )
    if weight_files:
        raise RuntimeError(f"P0 retained forbidden weight files: {weight_files}")
    finished = datetime.now(timezone.utc)
    hard_gates = {
        "artifact_and_code_hashes_match": True,
        "train_is_only_materialized_target_role": True,
        "all_fold_and_subset_hashes_match": len(subset_registry) == 18,
        "G0_A_nuisance_replay_within_1e_10": all(
            value <= 1e-10
            for value in nuisance_audit["maximum_absolute_errors"].values()
        ),
        "projector_and_CW_inverse_checks_pass": True,
        "six_path_budget_finite_positive_monotone_checks_pass": True,
        "PA_RWF_exactly_replays_G0_B0": schedule_audit[
            "PA_RWF_G0_B0_max_abs_error"
        ]
        <= 1e-12,
        "paired_initialization_and_random_bank": (
            paired_audit["unique_base_initialization_hashes"] == 1
            and paired_audit["all_paths_share_random_bank"]
        ),
        "MuLAN_registered_parameter_overhead_and_initial_identity": (
            next(row for row in path_runs if row["path"] == "MULAN_LITE")[
                "parameters"
            ]
            == TINY_DENOISER_PARAMETERS + MULAN_LITE_PARAMETERS
        ),
        "shuffle_zero_fixed_points_and_proxy_only": schedule_audit[
            "shuffle_fixed_points"
        ]
        == 0,
        "all_300_updates_and_diagnostic_batches_finite": all(
            row["finite"] for row in path_runs
        ),
        "checkpoint_resume_exact": bool(
            resume_audit["model_optimizer_EMA_exact"]
            and resume_audit["temporary_checkpoint_deleted"]
        ),
        "peak_GPU_memory_within_2_GiB": gpu_within_limit,
        "no_P0_weight_file_retained": not weight_files,
    }
    all_gates = all(hard_gates.values())
    status = (
        config["P0_preflight"]["P0_GO_status"]
        if all_gates
        else config["P0_preflight"]["failure"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "all_hard_gates_passed": all_gates,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": lineage_hashes["config"],
        "implementation_identity": lineage_hashes,
        "evidence_scope": "discarded-weight train-only engineering P0; no held-out reconstruction evidence",
        "target_roles_materialized": list(bundle.materialized_roles),
        "forbidden_target_arrays_materialized": False,
        "validation_bank_constructed": False,
        "checkpoint_selection": "none",
        "device": {
            "requested": device_name,
            "executed": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime_exercised": device.type == "cuda",
            "peak_GPU_memory_bytes": peak_gpu_bytes,
            "peak_GPU_memory_GiB": peak_gpu_bytes / float(1024**3),
            "maximum_allowed_GPU_memory_GiB": float(
                config["P0_preflight"]["maximum_peak_GPU_memory_GiB"]
            ),
            "interpretation": (
                "CUDA peak allocation was directly measured"
                if device.type == "cuda"
                else "CPU fallback allocated zero GPU bytes; this verifies the cap for this run, not future CUDA allocator behavior"
            ),
        },
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
        "subset_registry_replay": subset_registry,
        "nuisance_replay": nuisance_audit,
        "baseline_schedule": baseline_identity,
        "projector_audit": projector_audit,
        "schedule_audit": schedule_audit,
        "paired_training_audit": paired_audit,
        "resume_audit": resume_audit,
        "path_runs": path_runs,
        "P0_runs": len(path_runs),
        "P0_updates_total": sum(row["updates_completed"] for row in path_runs),
        "weights_retained": False,
        "weight_files_after_run": weight_files,
        "formal_outer_test_evaluation_performed": False,
        "hard_gates": hard_gates,
        "authorization": (
            "P0 GO permits a separately explicit 324-run retained training execution; "
            "it does not evaluate or select a scientific winner"
        ),
    }
    result_path = output_root / config["planned_implementation_files"]["P0_result"]
    result_sha = _atomic_json(result_path, result)
    forbidden_after = [
        path
        for suffix in ("*.pt", "*.pth")
        for path in output_root.rglob(suffix)
    ]
    if forbidden_after:
        raise RuntimeError("P0 output acquired a forbidden weight after result write")
    return result, result_sha


def _dry_run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    hashes = _validate_config(config, config_path)
    p0 = config["P0_preflight"]
    return {
        "schema": "architecture_v1_g0_b_tiny_denoiser_dry_run_v1",
        "config_sha256": hashes["config"],
        "target_roles_materialized": [],
        "paths": list(PATH_IDS),
        "planned_runs": int(p0["paths"]),
        "updates_per_path": int(p0["discarded_updates_per_path"]),
        "retained_weights": bool(p0["retained_weights"]),
        "requires_execute_p0": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--execute-p0",
        action="store_true",
        help="materialize train targets and run six discarded 50-update paths",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = _read_json(config_path)
    if not arguments.execute_p0:
        print(json.dumps(_dry_run(config_path), indent=2, sort_keys=True))
        return
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root is not None
        else (ROOT / config["output_root"]).resolve()
    )
    result, digest = run_p0(
        config_path, output_root, device_name=arguments.device
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_hard_gates_passed": result["all_hard_gates_passed"],
                "P0_runs": result["P0_runs"],
                "P0_updates_total": result["P0_updates_total"],
                "device": result["device"]["executed"],
                "result_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
