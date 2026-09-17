#!/usr/bin/env python3
"""Run the frozen schedule-only G0-B0 allocation feasibility audit.

The default dry run verifies only the frozen config.  ``--execute`` reads the
already-frozen G0-A result artifact, never a raw dataset, and audits the
reserve-then-water-fill allocation over the registered cosine-250 grid.  No
denoiser, optimizer, checkpoint, or target array exists in this runner.
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

DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b0_schedule.json"
RESULT_SCHEMA = "architecture_v1_g0_b0_schedule_result_v1"
ALLOCATION_MODULE = ROOT / "architecture_v1" / "g0b_allocation.py"
SCHEDULE_MODULE = ROOT / "architecture_v1" / "family_diffusion.py"

from architecture_v1.family_diffusion import MaskedJointDDPM
from architecture_v1.g0b_allocation import (
    gaussian_posterior_variance,
    reserve_then_water_fill,
)


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
        raise RuntimeError(f"frozen artifact SHA256 mismatch: {path}")
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


def _validate_config(config: Mapping[str, Any], path: Path) -> str:
    digest = _verified_sidecar(path)
    if config.get("schema") != "architecture_v1_g0_b0_schedule_audit_v1":
        raise ValueError("unexpected G0-B0 config schema")
    if config.get("status") != (
        "frozen_after_documented_schedule_exploration_before_executable_G0_B0_audit"
    ):
        raise RuntimeError("G0-B0 config does not disclose its exploration status")
    if config.get("evidence_class") != (
        "schedule_level_engineering_feasibility_not_independent_scientific_confirmation"
    ):
        raise RuntimeError("G0-B0 evidence boundary drifted")
    if float(config["allocation"]["primary_eta"]) != 0.5:
        raise RuntimeError("registered primary eta drifted")
    if config["allocation"]["primary_eta_must_not_be_tuned"] is not True:
        raise RuntimeError("primary eta must be outcome-independent")
    if list(config["sensitivity"]["eta_values"]) != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("registered eta sensitivity grid drifted")
    forbidden = set(config["not_authorized"])
    required = {
        "a G0-B denoiser implementation or training run",
        "raw train target access",
        "validation, calibration, selection, R-SEEN, or final target access",
        "selection of eta from reconstruction outcomes",
    }
    if not required.issubset(forbidden):
        raise RuntimeError("G0-B0 forbidden-action registry drifted")
    access = config["input_access"]
    if access["raw_dataset_loader_imported"] is not False:
        raise RuntimeError("G0-B0 cannot import a raw dataset loader")
    if access["raw_target_arrays_materialized"] is not False:
        raise RuntimeError("G0-B0 cannot materialize raw targets")
    if access["learned_weights_loaded_or_retained"] is not False:
        raise RuntimeError("G0-B0 cannot load or retain fitted weights")
    return digest


def _date_sha256(days: list[str]) -> str:
    parsed = np.asarray(days, dtype="datetime64[D]")
    if parsed.ndim != 1 or len(parsed) == 0:
        raise ValueError("G0-A evidence dates must be a non-empty vector")
    if len(np.unique(parsed)) != len(parsed):
        raise ValueError("G0-A evidence dates contain duplicates")
    ordered = np.sort(parsed).astype(str).tolist()
    if ordered != [str(value) for value in parsed.astype(str)]:
        raise ValueError("G0-A per-day evidence is not chronologically sorted")
    return hashlib.sha256("\n".join(ordered).encode("ascii")).hexdigest()


def _load_g0_a_evidence(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str], np.ndarray]:
    lineage = config["lineage"]
    path = ROOT / lineage["g0_a_result"]
    digest = _verified_sidecar(path)
    if digest != lineage["g0_a_result_sha256"]:
        raise RuntimeError("G0-A result identity drifted")
    evidence = _read_json(path)
    if evidence.get("schema") != lineage["g0_a_result_schema"]:
        raise RuntimeError("G0-A result schema drifted")
    if evidence.get("status") != lineage["g0_a_required_status"]:
        raise RuntimeError("G0-A did not carry the required GO status")
    if evidence.get("config_sha256") != lineage["g0_a_config_sha256"]:
        raise RuntimeError("G0-A config identity drifted")
    if evidence.get("all_gates_passed") is not True:
        raise RuntimeError("G0-A result does not pass all frozen gates")
    if evidence.get("target_roles_materialized") != ["train"]:
        raise RuntimeError("G0-A evidence materialized a non-train target role")
    if evidence.get("forbidden_target_arrays_materialized") is not False:
        raise RuntimeError("G0-A evidence materialized a forbidden target")
    if evidence.get("validation_bank_constructed") is not False:
        raise RuntimeError("G0-A evidence constructed a validation bank")
    if evidence.get("retained_learned_weights") is not False:
        raise RuntimeError("G0-A evidence retained learned weights")

    identity = evidence["data_identity"]
    expected_identity = {
        "train_days": int(lineage["expected_train_days"]),
        "train_date_sha256": lineage["expected_train_date_sha256"],
        "train_split_array_sha256": lineage["expected_train_split_array_sha256"],
        "train_only_data_bundle_sha256": lineage[
            "expected_train_only_data_bundle_sha256"
        ],
    }
    if identity != expected_identity:
        raise RuntimeError("G0-A train-only data identity drifted")

    records = evidence.get("per_day")
    if not isinstance(records, list) or len(records) != expected_identity["train_days"]:
        raise RuntimeError("G0-A per-day evidence count drifted")
    names = list(config["mode_registry"]["ordered_groups"])
    dates = [str(record["day"]) for record in records]
    if _date_sha256(dates) != expected_identity["train_date_sha256"]:
        raise RuntimeError("G0-A per-day date identity drifted")
    outer_folds = np.asarray([int(record["outer_fold"]) for record in records])
    if set(outer_folds.tolist()) != set(range(6)):
        raise RuntimeError("G0-A outer-fold labels drifted")

    variance = np.asarray(
        [[record["N1_variance"][name] for name in names] for record in records],
        dtype=np.float64,
    )
    effective_rank = np.asarray(
        [[record["effective_rank"][name] for name in names] for record in records],
        dtype=np.float64,
    )
    if variance.shape != (expected_identity["train_days"], len(names)):
        raise RuntimeError("G0-A variance evidence shape drifted")
    if effective_rank.shape != variance.shape:
        raise RuntimeError("G0-A effective-rank evidence shape drifted")
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise FloatingPointError("G0-A variance evidence is invalid")
    if not np.isfinite(effective_rank).all() or np.any(effective_rank <= 0.0):
        raise FloatingPointError("G0-A effective-rank evidence is invalid")
    weights = effective_rank / effective_rank.sum(axis=1, keepdims=True)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("G0-B0 information weights do not sum to one")
    return evidence, variance, effective_rank, dates, outer_folds


def _registered_baseline_schedule(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = config["baseline_schedule"]
    diffusion = MaskedJointDDPM(
        timesteps=int(spec["timesteps"]),
        cosine_offset=float(spec["cosine_offset"]),
        beta_min=float(spec["beta_min"]),
        beta_max=float(spec["beta_max"]),
    )
    tensor = diffusion.alpha_bar.detach().cpu().numpy()
    if tensor.dtype != np.float32:
        raise RuntimeError("registered baseline alpha_bar is no longer float32")
    little_endian = tensor.astype("<f4", copy=False)
    digest = hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()
    if digest != spec["alpha_bar_raw_little_endian_float32_sha256"]:
        raise RuntimeError("registered baseline schedule identity drifted")
    alpha_bar = tensor.astype(np.float64)
    if float(alpha_bar[0]) != float(spec["expected_clean_endpoint_alpha_bar"]):
        raise RuntimeError("baseline clean endpoint drifted")
    if float(alpha_bar[-1]) != float(spec["expected_noisy_endpoint_alpha_bar"]):
        raise RuntimeError("baseline noisy endpoint drifted")
    snr = alpha_bar / (1.0 - alpha_bar)
    log_snr = np.log(snr)
    expected_range = np.asarray(spec["expected_log_snr_range"], dtype=np.float64)
    actual_range = np.asarray([log_snr.min(), log_snr.max()])
    if not np.allclose(actual_range, expected_range, atol=1e-12, rtol=0.0):
        raise RuntimeError("baseline log-SNR range drifted")
    return alpha_bar, snr, {
        "timesteps": int(len(alpha_bar)),
        "alpha_bar_sha256": digest,
        "clean_endpoint_alpha_bar": float(alpha_bar[0]),
        "noisy_endpoint_alpha_bar": float(alpha_bar[-1]),
        "log_snr_min": float(log_snr.min()),
        "log_snr_max": float(log_snr.max()),
    }


def _quantile_map(values: np.ndarray, probabilities: list[float]) -> dict[str, float]:
    labels = {0.0: "min", 0.01: "q01", 0.5: "q50", 0.99: "q99", 1.0: "max"}
    result = np.quantile(values, probabilities)
    return {
        labels.get(float(probability), f"q{probability:g}"): float(value)
        for probability, value in zip(probabilities, result, strict=True)
    }


def _eta_audit(
    variance: np.ndarray,
    weights: np.ndarray,
    baseline_alpha_bar: np.ndarray,
    baseline_snr: np.ndarray,
    *,
    eta: float,
    budget_tolerance: float,
    monotonicity_tolerance: float,
    quantiles: list[float],
    mode_names: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    snr_steps: list[np.ndarray] = []
    alpha_steps: list[np.ndarray] = []
    information_steps: list[np.ndarray] = []
    baseline_information_steps: list[np.ndarray] = []
    budget_errors: list[np.ndarray] = []
    for ratio in baseline_snr:
        allocation = reserve_then_water_fill(
            variance,
            weights,
            float(ratio),
            eta=eta,
            tolerance=budget_tolerance,
        )
        snr_steps.append(allocation.snr)
        alpha_steps.append(allocation.alpha_bar)
        information_steps.append(allocation.information)
        baseline_information_steps.append(allocation.baseline_information)
        budget_errors.append(allocation.budget_error)

    snr = np.stack(snr_steps, axis=1)
    alpha_bar = np.stack(alpha_steps, axis=1)
    information = np.stack(information_steps, axis=1)
    baseline_information = np.stack(baseline_information_steps, axis=1)
    errors = np.stack(budget_errors, axis=1)
    forward_delta = np.diff(snr, axis=1)
    violations = forward_delta > monotonicity_tolerance
    positive = snr > 0.0
    baseline_log_snr = np.log(baseline_snr)[None, :, None]
    finite_shift = np.log(snr[positive]) - np.broadcast_to(
        baseline_log_snr, snr.shape
    )[positive]
    retention = information / baseline_information

    iid_posterior = gaussian_posterior_variance(
        variance[:, None, :], baseline_snr[None, :, None]
    )
    allocated_posterior = variance[:, None, :] * np.exp(-2.0 * information)
    iid_risk = np.sum(weights[:, None, :] * iid_posterior, axis=2)
    allocated_risk = np.sum(weights[:, None, :] * allocated_posterior, axis=2)
    risk_ratio = allocated_risk / iid_risk

    per_mode: dict[str, Any] = {}
    for index, name in enumerate(mode_names):
        mode_positive = positive[:, :, index]
        mode_shift = (
            np.log(snr[:, :, index][mode_positive])
            - np.broadcast_to(np.log(baseline_snr)[None, :], snr[:, :, index].shape)[
                mode_positive
            ]
        )
        per_mode[name] = {
            "positive_snr_fraction": float(mode_positive.mean()),
            "information_retention_min": float(retention[:, :, index].min()),
            "information_retention_median": float(
                np.median(retention[:, :, index])
            ),
            "information_retention_max": float(retention[:, :, index].max()),
            "finite_log_snr_shift": _quantile_map(mode_shift, quantiles),
        }

    summary = {
        "eta": float(eta),
        "budget_max_abs_error": float(np.max(np.abs(errors))),
        "budget_mean_abs_error": float(np.mean(np.abs(errors))),
        "monotonicity_comparisons": int(violations.size),
        "monotonicity_violations": int(violations.sum()),
        "maximum_forward_snr_delta": float(np.max(forward_delta)),
        "all_snr_finite": bool(np.isfinite(snr).all()),
        "all_snr_strictly_positive": bool(positive.all()),
        "nonpositive_snr_count": int((~positive).sum()),
        "information_retention_min": float(retention.min()),
        "information_retention_median": float(np.median(retention)),
        "information_retention_max": float(retention.max()),
        "finite_log_snr_shift": _quantile_map(finite_shift, quantiles),
        "finite_log_snr_shift_count": int(len(finite_shift)),
        "nonfinite_log_snr_shift_count": int(snr.size - len(finite_shift)),
        "clean_endpoint_alpha_bar_min": float(alpha_bar[:, 0, :].min()),
        "clean_endpoint_alpha_bar_max": float(alpha_bar[:, 0, :].max()),
        "noisy_endpoint_alpha_bar_min": float(alpha_bar[:, -1, :].min()),
        "noisy_endpoint_alpha_bar_max": float(alpha_bar[:, -1, :].max()),
        "gaussian_absolute_posterior_risk_ratio_to_IID": _quantile_map(
            risk_ratio, quantiles
        ),
        "per_mode": per_mode,
    }
    arrays = {
        "snr": snr,
        "alpha_bar": alpha_bar,
        "information": information,
        "baseline_information": baseline_information,
        "retention": retention,
    }
    return summary, arrays


def _equal_variance_identity(
    config: Mapping[str, Any],
    baseline_alpha_bar: np.ndarray,
    baseline_snr: np.ndarray,
) -> dict[str, Any]:
    ranks = np.asarray(config["mode_registry"]["expected_nominal_ranks"], dtype=float)
    weights = (ranks / ranks.sum())[None, :]
    maximum = 0.0
    cases: list[dict[str, Any]] = []
    for variance_value in config["audit"]["synthetic_equal_variance_grid"]:
        variance = np.full((1, len(ranks)), float(variance_value), dtype=np.float64)
        for eta in config["sensitivity"]["eta_values"]:
            case_maximum = 0.0
            for step, ratio in enumerate(baseline_snr):
                allocation = reserve_then_water_fill(
                    variance,
                    weights,
                    float(ratio),
                    eta=float(eta),
                    tolerance=float(config["audit"]["budget_absolute_tolerance"]),
                )
                error = float(
                    np.max(np.abs(allocation.alpha_bar - baseline_alpha_bar[step]))
                )
                case_maximum = max(case_maximum, error)
            maximum = max(maximum, case_maximum)
            cases.append(
                {
                    "variance": float(variance_value),
                    "eta": float(eta),
                    "alpha_bar_max_abs_error": case_maximum,
                }
            )
    return {"maximum_alpha_bar_abs_error": maximum, "cases": cases}


def run(config_path: Path, output_root: Path) -> tuple[dict[str, Any], str]:
    config = _read_json(config_path)
    config_sha = _validate_config(config, config_path)
    started = datetime.now(timezone.utc)
    evidence, variance, effective_rank, dates, outer_folds = _load_g0_a_evidence(
        config
    )
    weights = effective_rank / effective_rank.sum(axis=1, keepdims=True)
    baseline_alpha_bar, baseline_snr, schedule_identity = (
        _registered_baseline_schedule(config)
    )

    audit = config["audit"]
    mode_names = list(config["mode_registry"]["ordered_groups"])
    sensitivity: dict[str, Any] = {}
    arrays_by_eta: dict[float, dict[str, np.ndarray]] = {}
    for eta_value in config["sensitivity"]["eta_values"]:
        eta = float(eta_value)
        summary, arrays = _eta_audit(
            variance,
            weights,
            baseline_alpha_bar,
            baseline_snr,
            eta=eta,
            budget_tolerance=float(audit["budget_absolute_tolerance"]),
            monotonicity_tolerance=float(audit["monotonicity_absolute_tolerance"]),
            quantiles=[float(value) for value in audit["log_snr_shift_quantiles"]],
            mode_names=mode_names,
        )
        sensitivity[f"eta_{eta:g}"] = summary
        arrays_by_eta[eta] = arrays

    primary_eta = float(config["allocation"]["primary_eta"])
    primary = sensitivity[f"eta_{primary_eta:g}"]
    primary_arrays = arrays_by_eta[primary_eta]
    eta_one_arrays = arrays_by_eta[1.0]
    eta_one_identity_error = float(
        np.max(
            np.abs(
                eta_one_arrays["alpha_bar"]
                - baseline_alpha_bar[None, :, None]
            )
        )
    )
    equal_variance = _equal_variance_identity(
        config, baseline_alpha_bar, baseline_snr
    )

    allowed_shift = audit["primary_allowed_log_snr_shift_range"]
    primary_shift = primary["finite_log_snr_shift"]
    gates = {
        "g0_a_evidence_identity_and_access_boundary": True,
        "baseline_schedule_identity": True,
        "primary_budget_max_abs_error_at_most": primary[
            "budget_max_abs_error"
        ]
        <= float(config["go_no_go"]["primary_budget_max_abs_error_at_most"]),
        "primary_monotonicity_violations": primary["monotonicity_violations"]
        == int(config["go_no_go"]["primary_monotonicity_violations"]),
        "primary_all_snr_finite_and_strictly_positive": bool(
            primary["all_snr_finite"] and primary["all_snr_strictly_positive"]
        ),
        "primary_minimum_information_retention_ratio_at_least": primary[
            "information_retention_min"
        ]
        + float(audit["identity_absolute_tolerance"])
        >= float(audit["primary_required_minimum_information_retention_ratio"]),
        "primary_log_snr_shift_within_registered_range": (
            primary_shift["min"] >= float(allowed_shift[0])
            and primary_shift["max"] <= float(allowed_shift[1])
        ),
        "primary_clean_and_noisy_endpoints_within_registered_bounds": (
            primary["clean_endpoint_alpha_bar_min"]
            >= float(audit["primary_required_clean_endpoint_minimum_alpha_bar"])
            and primary["noisy_endpoint_alpha_bar_max"]
            <= float(audit["primary_required_noisy_endpoint_maximum_alpha_bar"])
        ),
        "eta_1_exact_IID_identity": eta_one_identity_error
        <= float(audit["identity_absolute_tolerance"]),
        "synthetic_equal_variance_IID_identity": equal_variance[
            "maximum_alpha_bar_abs_error"
        ]
        <= float(audit["identity_absolute_tolerance"]),
    }
    all_gates_passed = all(bool(value) for value in gates.values())
    expected_comparisons = int(audit["expected_day_step_mode_comparisons"])
    if primary["monotonicity_comparisons"] != expected_comparisons:
        raise RuntimeError("registered monotonicity comparison count drifted")

    fold_counts = {
        str(index): int(np.sum(outer_folds == index)) for index in range(6)
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": (
            config["go_no_go"]["GO_status"]
            if all_gates_passed
            else config["go_no_go"]["NO_GO_status"]
        ),
        "all_gates_passed": all_gates_passed,
        "authorized_next_step": (
            "freeze_a_separate_G0_B_tiny_denoiser_protocol_only"
            if all_gates_passed
            else "no_G0_B_runner_until_a_new_schedule_contract_is_frozen"
        ),
        "evidence_class": config["evidence_class"],
        "exploration_disclosed": True,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "implementation_identity": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "allocation_module": str(ALLOCATION_MODULE.resolve()),
            "allocation_module_sha256": _sha256(ALLOCATION_MODULE),
            "schedule_module": str(SCHEDULE_MODULE.resolve()),
            "schedule_module_sha256": _sha256(SCHEDULE_MODULE),
        },
        "input_access": {
            "g0_a_result_path": str(
                (ROOT / config["lineage"]["g0_a_result"]).resolve()
            ),
            "g0_a_result_sha256": config["lineage"]["g0_a_result_sha256"],
            "consumed_g0_a_fields": config["input_access"][
                "consumed_g0_a_fields"
            ],
            "raw_dataset_loader_imported_by_runner": False,
            "raw_target_arrays_materialized": False,
            "learned_weights_loaded_or_retained": False,
            "denoiser_or_optimizer_constructed": False,
        },
        "g0_a_identity": {
            "schema": evidence["schema"],
            "status": evidence["status"],
            "config_sha256": evidence["config_sha256"],
            "data_identity": evidence["data_identity"],
        },
        "sample_audit": {
            "days": int(len(dates)),
            "date_sha256": _date_sha256(dates),
            "outer_fold_day_counts": fold_counts,
            "mode_names": mode_names,
            "variance_min": float(variance.min()),
            "variance_median": float(np.median(variance)),
            "variance_max": float(variance.max()),
            "effective_rank_min": float(effective_rank.min()),
            "effective_rank_median": float(np.median(effective_rank)),
            "effective_rank_max": float(effective_rank.max()),
            "weight_row_sum_max_abs_error": float(
                np.max(np.abs(weights.sum(axis=1) - 1.0))
            ),
        },
        "schedule_identity": schedule_identity,
        "primary": primary,
        "primary_eta": primary_eta,
        "eta_1_IID_identity_alpha_bar_max_abs_error": eta_one_identity_error,
        "synthetic_equal_variance_identity": equal_variance,
        "sensitivity": sensitivity,
        "gates": gates,
        "retained_learned_weights": False,
        "raw_targets_materialized": False,
        "denoiser_trained": False,
    }
    output_path = output_root / "G0_B0_SCHEDULE_RESULT.json"
    result_sha = _atomic_json(output_path, result)
    return result, result_sha


def _dry_run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    digest = _validate_config(config, config_path)
    return {
        "status": "DRY_RUN_NO_EVIDENCE_OR_TARGET_ACCESS",
        "config": str(config_path.resolve()),
        "config_sha256": digest,
        "evidence_class": config["evidence_class"],
        "g0_a_result_expected_sha256": config["lineage"]["g0_a_result_sha256"],
        "primary_eta": float(config["allocation"]["primary_eta"]),
        "sensitivity_eta": config["sensitivity"]["eta_values"],
        "denoiser_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="read the frozen G0-A result and execute the schedule-only audit",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    if not arguments.execute:
        print(json.dumps(_dry_run(config_path), indent=2, sort_keys=True))
        return
    config = _read_json(config_path)
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root is not None
        else (ROOT / config["output_root"]).resolve()
    )
    result, result_sha = run(config_path, output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_passed": result["all_gates_passed"],
                "primary_eta": result["primary_eta"],
                "result": str((output_root / "G0_B0_SCHEDULE_RESULT.json")),
                "result_sha256": result_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
