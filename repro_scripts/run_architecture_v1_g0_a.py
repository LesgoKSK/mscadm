#!/usr/bin/env python3
"""Run the frozen train-only G0-A NWP predictability audit.

Without ``--execute`` this command only prints the frozen plan and never opens
target files.  The executing path uses ``build_architecture_v1_train_data``;
its bundle type cannot expose validation, calibration, selection, R-SEEN, or
final targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_a_predictability.json"
RESULT_SCHEMA = "architecture_v1_g0_a_predictability_result_v1"
ANALYSIS_MODULE = ROOT / "architecture_v1" / "g0_predictability.py"

from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.g0_predictability import (
    build_mode_groups,
    chronological_folds,
    continuous_logit_target,
    cross_fitted_cellwise_predictions,
    derangement,
    fit_cellwise_ridge,
    fit_log_variance_glm,
    fit_static_variance,
    gaussian_variance_score,
    masked_band_energies,
    masked_mse,
    mask_nuisance_features,
    month_cluster_bootstrap,
    nwp_summary_features,
    select_cellwise_alpha,
    select_variance_alpha,
    spearman_correlation,
)
from architecture_v1.protocol import date_list_sha256


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
        raise FileNotFoundError(f"frozen file or SHA256 sidecar is missing: {path}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
    actual = _sha256(path)
    if actual != pieces[0]:
        raise RuntimeError(f"frozen SHA256 mismatch: {path}")
    return actual


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


def _mode_groups(config: Mapping[str, Any]):
    registry = config["mode_registry"]
    temporal = registry["temporal"]
    ranges = {
        label: temporal[label]["indices_half_open"]
        for label in ("low", "mid", "high")
    }
    return build_mode_groups(ranges, registry["ordered_groups"])


def _fold_manifest(days: np.ndarray, outer_folds, inner_by_outer) -> dict[str, Any]:
    return {
        "outer": [
            {
                "fold": index,
                "days": [str(value) for value in days[test].astype("datetime64[D]")],
                "date_sha256": date_list_sha256(days[test]),
                "count": int(len(test)),
                "inner_training_folds": [
                    {
                        "fold": inner_index,
                        "date_sha256": date_list_sha256(days[inner]),
                        "count": int(len(inner)),
                    }
                    for inner_index, inner in enumerate(inner_by_outer[index])
                ],
            }
            for index, test in enumerate(outer_folds)
        ]
    }


def _calibration(
    energies: np.ndarray,
    predicted_variance: np.ndarray,
    group_names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for band, name in enumerate(group_names):
        energy = energies[:, band]
        predicted = predicted_variance[:, band]
        log_energy = np.log(energy)
        log_prediction = np.log(predicted)
        design = np.column_stack([np.ones(len(predicted)), log_prediction])
        coefficient = np.linalg.lstsq(design, log_energy, rcond=None)[0]
        order = np.argsort(predicted, kind="mergesort")
        quintiles = []
        for index, block in enumerate(np.array_split(order, 5), start=1):
            quintiles.append(
                {
                    "quintile": index,
                    "count": int(len(block)),
                    "predicted_variance_mean": float(predicted[block].mean()),
                    "observed_energy_mean": float(energy[block].mean()),
                }
            )
        result[name] = {
            "log_energy_intercept": float(coefficient[0]),
            "log_energy_calibration_slope": float(coefficient[1]),
            "spearman": spearman_correlation(predicted, energy),
            "quintiles": quintiles,
        }
    return result


def _validate_config(config: Mapping[str, Any], config_path: Path) -> str:
    digest = _verified_sidecar(config_path)
    if config.get("schema") != "architecture_v1_g0_a_predictability_audit_v1":
        raise ValueError("unexpected G0-A config schema")
    if config.get("status") != "frozen_before_any_G0_A_target_aware_execution":
        raise RuntimeError("G0-A config is not frozen for execution")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train"]:
        raise RuntimeError("G0-A may only access train targets")
    if set(roles["forbidden_target_roles"]) != {
        "validation",
        "calibration",
        "selection",
        "r_seen",
        "final",
    }:
        raise RuntimeError("G0-A forbidden-role registry drifted")
    lineage_path = ROOT / config["lineage"]["data_protocol_config"]
    if _sha256(lineage_path) != config["lineage"]["data_protocol_config_sha256"]:
        raise RuntimeError("G0-A data-protocol lineage drifted")
    return digest


def run(config_path: Path, output_root: Path) -> tuple[dict[str, Any], str]:
    config = _read_json(config_path)
    config_sha = _validate_config(config, config_path)
    started = datetime.now(timezone.utc)
    lineage = config["lineage"]
    bundle = build_architecture_v1_train_data(
        config_path=ROOT / lineage["data_protocol_config"]
    )
    if bundle.materialized_roles != ("train",):
        raise RuntimeError("G0-A loader materialized a non-train target role")
    access = bundle.manifest["formal_train_only_target_access"]
    if access["materialized_roles"] != ["train"]:
        raise RuntimeError("train-only access manifest drifted")
    if access["forbidden_target_arrays_materialized"] is not False:
        raise RuntimeError("a forbidden target array was materialized")
    if access["validation_bank_constructed"] is not False:
        raise RuntimeError("G0-A must not construct a validation bank")
    train = bundle.train
    actual_identity = {
        "train_days": int(len(train)),
        "train_date_sha256": access["materialized_date_sha256"],
        "train_split_array_sha256": bundle.manifest["data_audit"][
            "split_array_sha256"
        ]["train"],
        "train_only_data_bundle_sha256": bundle.manifest[
            "train_only_data_bundle_sha256"
        ],
    }
    expected_identity = {
        "train_days": int(lineage["expected_train_days"]),
        "train_date_sha256": lineage["expected_train_date_sha256"],
        "train_split_array_sha256": lineage["expected_train_split_array_sha256"],
        "train_only_data_bundle_sha256": lineage[
            "expected_train_only_data_bundle_sha256"
        ],
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"G0-A train-only identity drifted: {actual_identity} != {expected_identity}"
        )

    groups = _mode_groups(config)
    group_names = [group.name for group in groups]
    continuous = config["continuous_target"]
    latent, active = continuous_logit_target(
        train.target,
        train.state,
        train.observed_mask,
        epsilon=float(continuous["logit_epsilon"]),
    )
    registry = config["mode_registry"]
    zero_residual = np.zeros_like(latent)
    _, effective_all = masked_band_energies(
        zero_residual,
        active,
        groups,
        minimum_effective_rank=float(registry["minimum_effective_rank"]),
        energy_floor=float(registry["energy_floor"]),
    )
    mask_features = mask_nuisance_features(active, effective_all, groups)
    nwp_features, nwp_feature_names = nwp_summary_features(
        train.raw_condition, groups
    )
    if nwp_features.shape[1] != config["uncertainty_features"]["nwp_feature_count"]:
        raise RuntimeError("NWP feature-count contract drifted")
    if mask_features.shape[1] != config["uncertainty_features"][
        "mask_feature_count"
    ]:
        raise RuntimeError("mask feature-count contract drifted")

    day_count = len(train)
    band_count = len(groups)
    outer_count = int(config["cross_fitting"]["outer_folds"])
    inner_count = int(config["cross_fitting"]["inner_folds"])
    outer_folds = chronological_folds(np.arange(day_count), outer_count)
    inner_by_outer = []
    for outer_test in outer_folds:
        outer_train = np.setdiff1d(
            np.arange(day_count, dtype=np.int64), outer_test, assume_unique=True
        )
        inner_by_outer.append(chronological_folds(outer_train, inner_count))

    energies_oof = np.full((day_count, band_count), np.nan, dtype=np.float64)
    effective_oof = np.full_like(energies_oof, np.nan)
    static_variance = np.full_like(energies_oof, np.nan)
    mask_variance = np.full_like(energies_oof, np.nan)
    nwp_variance = np.full_like(energies_oof, np.nan)
    shuffle_count = int(config["shuffle_control"]["permutations"])
    shuffle_variance = np.full(
        (shuffle_count, day_count, band_count), np.nan, dtype=np.float64
    )
    outer_id = np.full(day_count, -1, dtype=np.int64)
    selected_hyperparameters: list[dict[str, Any]] = []
    outer_mean_mse: list[float] = []

    mean_contract = config["conditional_mean"]
    variance_contract = config["uncertainty_models"]
    variance_clip = tuple(float(x) for x in variance_contract["prediction_variance_clip"])
    if len(variance_clip) != 2:
        raise ValueError("variance clip must contain two values")

    all_indices = np.arange(day_count, dtype=np.int64)
    for outer_index, outer_test in enumerate(outer_folds):
        outer_train = np.setdiff1d(all_indices, outer_test, assume_unique=True)
        inner_global = inner_by_outer[outer_index]
        mean_alpha, mean_cv = select_cellwise_alpha(
            train.raw_condition,
            latent,
            active,
            outer_train,
            inner_global,
            mean_contract["shared_ridge_alpha_grid"],
            minimum_observations=int(mean_contract["minimum_fit_observations_per_cell"]),
        )
        nested_prediction = cross_fitted_cellwise_predictions(
            train.raw_condition,
            latent,
            active,
            outer_train,
            inner_global,
            alpha=mean_alpha,
            minimum_observations=int(mean_contract["minimum_fit_observations_per_cell"]),
        )
        outer_mean_model = fit_cellwise_ridge(
            train.raw_condition,
            latent,
            active,
            outer_train,
            alpha=mean_alpha,
            minimum_observations=int(mean_contract["minimum_fit_observations_per_cell"]),
        )
        test_prediction = outer_mean_model.predict(train.raw_condition[outer_test])
        test_sse, test_count = masked_mse(
            test_prediction, latent[outer_test], active[outer_test]
        )
        outer_mean_mse.append(test_sse / test_count)

        train_residual = np.where(
            active[outer_train],
            latent[outer_train] - nested_prediction[outer_train],
            0.0,
        )
        test_residual = np.where(
            active[outer_test], latent[outer_test] - test_prediction, 0.0
        )
        train_energy, train_effective = masked_band_energies(
            train_residual,
            active[outer_train],
            groups,
            minimum_effective_rank=float(registry["minimum_effective_rank"]),
            energy_floor=float(registry["energy_floor"]),
        )
        test_energy, test_effective = masked_band_energies(
            test_residual,
            active[outer_test],
            groups,
            minimum_effective_rank=float(registry["minimum_effective_rank"]),
            energy_floor=float(registry["energy_floor"]),
        )
        if not np.allclose(test_effective, effective_all[outer_test], atol=1e-12):
            raise RuntimeError("effective rank depends on residual values")
        energies_oof[outer_test] = test_energy
        effective_oof[outer_test] = test_effective
        outer_id[outer_test] = outer_index

        train_mask_x = mask_features[outer_train]
        test_mask_x = mask_features[outer_test]
        train_nwp_x = np.concatenate(
            [train_mask_x, nwp_features[outer_train]], axis=1
        )
        test_nwp_x = np.concatenate(
            [test_mask_x, nwp_features[outer_test]], axis=1
        )
        inner_local = chronological_folds(np.arange(len(outer_train)), inner_count)
        mask_alpha, mask_cv = select_variance_alpha(
            train_mask_x,
            train_energy,
            train_effective,
            inner_local,
            variance_contract["ridge_alpha_grid"],
            variance_clip=variance_clip,
            maximum_iterations=int(variance_contract["maximum_newton_iterations"]),
            tolerance=float(variance_contract["newton_tolerance"]),
        )
        nwp_alpha, nwp_cv = select_variance_alpha(
            train_nwp_x,
            train_energy,
            train_effective,
            inner_local,
            variance_contract["ridge_alpha_grid"],
            variance_clip=variance_clip,
            maximum_iterations=int(variance_contract["maximum_newton_iterations"]),
            tolerance=float(variance_contract["newton_tolerance"]),
        )
        fitted_nwp_models = []
        for band in range(band_count):
            constant = fit_static_variance(
                train_energy[:, band],
                train_effective[:, band],
                variance_clip=variance_clip,
            )
            static_variance[outer_test, band] = constant
            mask_model = fit_log_variance_glm(
                train_mask_x,
                train_energy[:, band],
                train_effective[:, band],
                alpha=mask_alpha,
                variance_clip=variance_clip,
                maximum_iterations=int(variance_contract["maximum_newton_iterations"]),
                tolerance=float(variance_contract["newton_tolerance"]),
            )
            nwp_model = fit_log_variance_glm(
                train_nwp_x,
                train_energy[:, band],
                train_effective[:, band],
                alpha=nwp_alpha,
                variance_clip=variance_clip,
                maximum_iterations=int(variance_contract["maximum_newton_iterations"]),
                tolerance=float(variance_contract["newton_tolerance"]),
            )
            mask_variance[outer_test, band] = mask_model.predict(test_mask_x)
            nwp_variance[outer_test, band] = nwp_model.predict(test_nwp_x)
            fitted_nwp_models.append(nwp_model)

        seed_root = int(config["shuffle_control"]["seed_root"])
        for permutation_index in range(shuffle_count):
            permutation = derangement(
                len(outer_test), seed_root + permutation_index * outer_count + outer_index
            )
            shuffled_x = np.concatenate(
                [test_mask_x, nwp_features[outer_test][permutation]], axis=1
            )
            for band, model in enumerate(fitted_nwp_models):
                shuffle_variance[permutation_index, outer_test, band] = model.predict(
                    shuffled_x
                )

        selected_hyperparameters.append(
            {
                "outer_fold": outer_index,
                "mean_ridge_alpha": mean_alpha,
                "mask_variance_ridge_alpha": mask_alpha,
                "nwp_variance_ridge_alpha": nwp_alpha,
                "mean_inner_cv_mse": mean_cv,
                "mask_inner_cv_score": mask_cv,
                "nwp_inner_cv_score": nwp_cv,
            }
        )

    for name, value in (
        ("energies_oof", energies_oof),
        ("effective_oof", effective_oof),
        ("static_variance", static_variance),
        ("mask_variance", mask_variance),
        ("nwp_variance", nwp_variance),
        ("shuffle_variance", shuffle_variance),
    ):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} is incomplete or non-finite")
    if not np.array_equal(np.sort(np.unique(outer_id)), np.arange(outer_count)):
        raise RuntimeError("outer-fold result coverage is incomplete")

    static_score = gaussian_variance_score(energies_oof, static_variance)
    mask_score = gaussian_variance_score(energies_oof, mask_variance)
    nwp_score = gaussian_variance_score(energies_oof, nwp_variance)
    shuffle_score = gaussian_variance_score(
        np.broadcast_to(energies_oof, shuffle_variance.shape), shuffle_variance
    )
    primary_day = (mask_score - nwp_score).mean(axis=1)
    static_day = (static_score - nwp_score).mean(axis=1)
    primary_bootstrap = month_cluster_bootstrap(
        primary_day,
        train.day,
        replicates=int(config["inference"]["bootstrap_replicates"]),
        seed=int(config["inference"]["bootstrap_seed"]),
        confidence_level=float(config["inference"]["confidence_level"]),
    )
    band_improvement = (mask_score - nwp_score).mean(axis=0)
    fold_improvement = np.asarray(
        [primary_day[outer_id == index].mean() for index in range(outer_count)]
    )
    saturated_score = np.log(energies_oof) + 1.0
    reducible = float((mask_score - saturated_score).mean())
    if not np.isfinite(reducible) or reducible <= 0.0:
        raise RuntimeError("mask reference has no positive reducible deviance")
    deviance_explained = float(primary_day.mean() / reducible)
    shuffle_improvement = (mask_score[None] - shuffle_score).mean(axis=(1, 2))
    shuffle_quantile = float(
        np.quantile(
            shuffle_improvement,
            float(config["inference"]["shuffle_reference_quantile"]),
        )
    )

    positive_groups = [
        group_names[index]
        for index, value in enumerate(band_improvement)
        if value > 0.0
    ]
    positive_spatial = {name.split("_")[1] for name in positive_groups}
    positive_temporal = {name.split("_")[0] for name in positive_groups}
    gates_config = config["go_no_go"]
    gates = {
        "primary_cluster_bootstrap_lower_bound_strictly_positive": bool(
            primary_bootstrap["lower"] > 0.0
        ),
        "minimum_deviance_explained": bool(
            deviance_explained >= float(gates_config["minimum_deviance_explained"])
        ),
        "minimum_positive_outer_folds": bool(
            int(np.sum(fold_improvement > 0.0))
            >= int(gates_config["minimum_positive_outer_folds"])
        ),
        "minimum_positive_mode_groups": bool(
            len(positive_groups) >= int(gates_config["minimum_positive_mode_groups"])
        ),
        "required_spatial_coverage": set(gates_config["required_spatial_coverage"])
        .issubset(positive_spatial),
        "minimum_supported_temporal_bands": bool(
            len(positive_temporal)
            >= int(gates_config["minimum_supported_temporal_bands"])
        ),
        "N1_must_beat_S0_on_macro_score": bool(static_day.mean() > 0.0),
        "correct_NWP_improvement_must_exceed_shuffle_quantile": bool(
            primary_day.mean() > shuffle_quantile
        ),
        "finite_and_provenance_checks": True,
    }
    passed = all(gates.values())
    status = gates_config["GO_status"] if passed else gates_config["NO_GO_status"]

    per_day = []
    for day_index in range(day_count):
        per_day.append(
            {
                "day": str(train.day[day_index].astype("datetime64[D]")),
                "outer_fold": int(outer_id[day_index]),
                "energy": dict(zip(group_names, energies_oof[day_index].tolist())),
                "effective_rank": dict(
                    zip(group_names, effective_oof[day_index].tolist())
                ),
                "S0_variance": dict(
                    zip(group_names, static_variance[day_index].tolist())
                ),
                "M0_variance": dict(
                    zip(group_names, mask_variance[day_index].tolist())
                ),
                "N1_variance": dict(
                    zip(group_names, nwp_variance[day_index].tolist())
                ),
                "primary_macro_improvement": float(primary_day[day_index]),
                "static_macro_improvement": float(static_day[day_index]),
            }
        )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha,
        "implementation_identity": {
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "analysis_module": {
                "path": str(ANALYSIS_MODULE.relative_to(ROOT)),
                "sha256": _sha256(ANALYSIS_MODULE),
            },
        },
        "evidence_scope": "train-only nested cross-fitted prerequisite audit",
        "authorized_next_step": (
            "design_and_separately_freeze_G0_B"
            if passed
            else "stop_predictability_aligned_diffusion_hypothesis"
        ),
        "target_roles_materialized": ["train"],
        "forbidden_target_arrays_materialized": False,
        "validation_bank_constructed": False,
        "data_identity": actual_identity,
        "sample_audit": {
            "observed_fraction": float(train.observed_mask.mean()),
            "interior_fraction_of_observed": float(active.sum() / train.observed_mask.sum()),
            "active_cells_per_day_min": int(active.reshape(day_count, -1).sum(axis=1).min()),
            "active_cells_per_day_median": float(
                np.median(active.reshape(day_count, -1).sum(axis=1))
            ),
            "active_cells_per_day_max": int(active.reshape(day_count, -1).sum(axis=1).max()),
            "minimum_effective_rank_by_group": dict(
                zip(group_names, effective_oof.min(axis=0).tolist())
            ),
        },
        "mode_groups": [
            {
                "name": group.name,
                "spatial": group.spatial_label,
                "temporal": group.temporal_label,
                "full_rank": group.full_rank,
            }
            for group in groups
        ],
        "nwp_feature_names": list(nwp_feature_names),
        "fold_manifest": _fold_manifest(train.day, outer_folds, inner_by_outer),
        "selected_hyperparameters": selected_hyperparameters,
        "outer_conditional_mean_logit_mse": outer_mean_mse,
        "scores": {
            "S0_static_macro": float(static_score.mean()),
            "M0_mask_macro": float(mask_score.mean()),
            "N1_nwp_macro": float(nwp_score.mean()),
            "primary_M0_minus_N1": float(primary_day.mean()),
            "static_S0_minus_N1": float(static_day.mean()),
            "primary_month_cluster_bootstrap": primary_bootstrap,
            "reducible_deviance_M0_minus_saturated": reducible,
            "deviance_explained": deviance_explained,
            "outer_fold_primary_improvement": fold_improvement.tolist(),
            "positive_outer_folds": int(np.sum(fold_improvement > 0.0)),
            "band_primary_improvement": dict(
                zip(group_names, band_improvement.tolist())
            ),
            "positive_mode_groups": positive_groups,
            "supported_spatial_groups": sorted(positive_spatial),
            "supported_temporal_bands": sorted(positive_temporal),
            "shuffle_improvements": shuffle_improvement.tolist(),
            "shuffle_improvement_quantile_95": shuffle_quantile,
        },
        "calibration": _calibration(energies_oof, nwp_variance, group_names),
        "gates": gates,
        "all_gates_passed": passed,
        "per_day": per_day,
        "retained_learned_weights": False,
    }
    result_path = output_root / "G0_A_RESULT.json"
    result_sha = _atomic_json(result_path, result)
    return result, result_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the frozen train-only target-aware audit",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    config_sha = _validate_config(config, config_path)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (ROOT / config["output_root"]).resolve()
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_TARGET_ACCESS",
                    "config": str(config_path),
                    "config_sha256": config_sha,
                    "allowed_target_roles": config["role_access"][
                        "allowed_target_roles"
                    ],
                    "outer_folds": config["cross_fitting"]["outer_folds"],
                    "mode_groups": config["mode_registry"]["ordered_groups"],
                    "output_root": str(output_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result, result_sha = run(config_path, output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_passed": result["all_gates_passed"],
                "primary_M0_minus_N1": result["scores"]["primary_M0_minus_N1"],
                "bootstrap_lower": result["scores"][
                    "primary_month_cluster_bootstrap"
                ]["lower"],
                "deviance_explained": result["scores"]["deviance_explained"],
                "result": str(output_root / "G0_A_RESULT.json"),
                "result_sha256": result_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
