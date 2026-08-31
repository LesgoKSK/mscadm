"""Run the frozen RAHC-CR-MS-CADM calibration experiment.

Stages are intentionally separable.  ``cv`` never opens the test archives;
``finalize`` requires the validation selection artifact before applying any
chosen model to test.  Long stages are resumable at the candidate/method level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch

from rahc.empirical import HourEmpiricalCalibrator
from rahc.evaluation import overall_scores, selection_record
from rahc.group_metrics import GroupingProtocol, group_interval_metrics, summarize_group_metrics
from rahc.validation import fold_table, grouped_day_folds
from rahc.v2_calibration import RAHCalibratorV2, rank_audit


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "repro_configs" / "rahc.json"
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "rahc_cr_mscadm"
SOURCE_ROOT = WORKSPACE / "outputs" / "cr_mscadm" / "scenarios"
SEEDS = (0, 1, 2)
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANT_SPECS: dict[str, tuple[str, tuple[float, ...]]] = {
    "C1_global": ("global", (0.03,)),
    "C2_hour": ("hour", (0.03,)),
    "C3_hour_zone": ("hour_zone", (0.03,)),
    "C4_hour_regime": ("hour_regime", (0.03,)),
    "C5_additive": ("additive", (0.03,)),
    "C6_RAHC": ("full", (0.003, 0.01, 0.03, 0.1)),
    "C7_no_shrink": ("full_no_shrink", (0.0,)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _archive_path(seed: int, split: str, suffix: str = "raw") -> Path:
    return SOURCE_ROOT / f"full_seed{seed}_{split}_{suffix}.npz"


def load_archives(split: str, suffix: str = "raw") -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for seed in SEEDS:
        path = _archive_path(seed, split, suffix)
        with np.load(path, allow_pickle=False) as archive:
            result.append({key: archive[key] for key in archive.files})
    reference = result[0]
    for archive in result[1:]:
        for key in ("observations", "zone", "day"):
            if not np.array_equal(reference[key], archive[key]):
                raise AssertionError(f"seed archives are not aligned for {split}/{key}")
    return result


def _save_scenario_archive(
    path: Path,
    scenarios: np.ndarray,
    reference: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        scenarios=np.asarray(scenarios, dtype=np.float32),
        observations=reference["observations"],
        zone=reference["zone"],
        day=reference["day"],
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(path)


def _input_audit(split: str, archives: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    records = []
    for seed, archive in zip(SEEDS, archives):
        path = _archive_path(seed, split)
        values = archive["scenarios"]
        records.append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    day = archives[0]["day"].astype("datetime64[D]")
    zone = archives[0]["zone"].astype(int)
    unique_days, counts = np.unique(day, return_counts=True)
    return {
        "split": split,
        "archives": records,
        "cases": int(len(day)),
        "unique_calendar_days": int(len(unique_days)),
        "calendar_days": [str(value) for value in unique_days],
        "zones": sorted(np.unique(zone).tolist()),
        "zone_days_per_calendar_day": sorted(np.unique(counts).tolist()),
        "unique_zone_day_pairs": int(len(set(zip(zone.tolist(), day.astype(str).tolist())))),
    }


def prepare_validation(output: Path) -> None:
    validation = load_archives("validation")
    day = validation[0]["day"]
    assignments = grouped_day_folds(day, folds=5, seed=20260718)
    table = pd.DataFrame(fold_table(day, assignments)).sort_values("day")
    protocol_dir = output / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(protocol_dir / "fold_assignments.csv", index=False)
    raw_stack = np.stack([item["scenarios"] for item in validation])
    zone = validation[0]["zone"]
    full_grouping = GroupingProtocol.fit(raw_stack, zone)
    write_json(protocol_dir / "grouping_thresholds.json", full_grouping.to_dict())
    group_codes = {
        family: np.empty((len(day), 24), dtype=np.int16)
        for family in ("zone", "hour", "wind_quintile", "spread_quintile", "zone_x_spread")
    }
    for fold in range(5):
        train = assignments != fold
        held = ~train
        fold_protocol = GroupingProtocol.fit(raw_stack[:, train], zone[train])
        assigned = fold_protocol.assign(raw_stack[:, held], zone[held])
        for family, values in assigned.items():
            group_codes[family][held] = values
    np.savez_compressed(protocol_dir / "crossfit_group_assignments.npz", **group_codes)
    audit = {
        "protocol_config": str(CONFIG_PATH.resolve()),
        "protocol_sha256": sha256(CONFIG_PATH),
        "validation": _input_audit("validation", validation),
        "folds": {
            str(fold): {
                "calendar_days": int(len(np.unique(day[assignments == fold]))),
                "zone_days": int(np.sum(assignments == fold)),
                "cells": int(np.sum(assignments == fold) * 24),
            }
            for fold in range(5)
        },
        "test_opened_during_stage": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(protocol_dir / "validation_input_audit.json", audit)
    print(json.dumps({"stage": "prepare", "folds": audit["folds"]}), flush=True)


def _load_cv_protocol(output: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame = pd.read_csv(output / "protocol" / "fold_assignments.csv")
    date_to_fold = {
        np.datetime64(day, "D"): int(fold) for day, fold in zip(frame["day"], frame["fold"])
    }
    validation = load_archives("validation")
    day = validation[0]["day"].astype("datetime64[D]")
    assignments = np.asarray([date_to_fold[value] for value in day], dtype=np.int64)
    with np.load(output / "protocol" / "crossfit_group_assignments.npz") as archive:
        groups = {key: archive[key] for key in archive.files}
    return assignments, groups


def _candidate_name(method: str, regularization: float) -> str:
    return f"{method}_reg{regularization:.6g}".replace(".", "p")


def _save_oof(path: Path, arrays: list[np.ndarray], reference: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seed0=arrays[0],
        seed1=arrays[1],
        seed2=arrays[2],
        observations=reference["observations"],
        zone=reference["zone"],
        day=reference["day"],
        metadata=np.asarray(json.dumps(metadata)),
    )


def _score_strengths(
    method: str,
    variant: str,
    regularization: float,
    oof: np.ndarray,
    observations: np.ndarray,
    groups: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    for index, strength in enumerate(STRENGTHS):
        metrics = selection_record(list(oof[index]), observations, groups)
        records.append(
            {
                "method": method,
                "variant": variant,
                "regularization": float(regularization),
                "strength": float(strength),
                **metrics,
            }
        )
    selected = min(
        range(len(records)),
        key=lambda idx: (
            records[idx]["objective"], records[idx]["strength"], -records[idx]["regularization"]
        ),
    )
    return records, selected


def _crossfit_empirical(
    validation: list[dict[str, np.ndarray]], assignments: np.ndarray
) -> np.ndarray:
    shape = validation[0]["scenarios"].shape
    oof = np.empty((len(STRENGTHS), len(SEEDS), *shape), dtype=np.float32)
    for fold in range(5):
        train = assignments != fold
        held = ~train
        for seed, archive in enumerate(validation):
            calibrator = HourEmpiricalCalibrator.fit(
                archive["scenarios"][train], archive["observations"][train]
            )
            for strength_index, strength in enumerate(STRENGTHS):
                if strength == 0.0:
                    oof[strength_index, seed, held] = archive["scenarios"][held]
                else:
                    oof[strength_index, seed, held] = calibrator.transform(
                        archive["scenarios"][held], strength=strength, tail_rule="bounded"
                    )
        print(json.dumps({"candidate": "C0_empirical", "fold": fold}), flush=True)
    return oof


def _crossfit_model(
    validation: list[dict[str, np.ndarray]],
    assignments: np.ndarray,
    *,
    method: str,
    variant: str,
    regularization: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    shape = validation[0]["scenarios"].shape
    oof = np.empty((len(STRENGTHS), len(SEEDS), *shape), dtype=np.float32)
    fit_summaries: list[dict[str, Any]] = []
    for fold in range(5):
        train = assignments != fold
        held = ~train
        pooled_scenarios = np.concatenate([archive["scenarios"][train] for archive in validation])
        pooled_observations = np.concatenate([archive["observations"][train] for archive in validation])
        pooled_zone = np.concatenate([archive["zone"][train] for archive in validation])
        calibrator = RAHCalibratorV2.fit(
            pooled_scenarios,
            pooled_observations,
            pooled_zone,
            variant=variant,
            regularization=regularization,
            epochs=160,
            learning_rate=0.03,
            seed=20260718 + fold,
        )
        fit_summaries.append({"fold": fold, **calibrator.fit_summary.to_dict()})
        for seed, archive in enumerate(validation):
            for strength_index, strength in enumerate(STRENGTHS):
                if strength == 0.0:
                    oof[strength_index, seed, held] = archive["scenarios"][held]
                else:
                    oof[strength_index, seed, held] = calibrator.transform(
                        archive["scenarios"][held],
                        archive["zone"][held],
                        strength=strength,
                        tail_rule="bounded",
                    )
        print(
            json.dumps(
                {
                    "candidate": method,
                    "variant": variant,
                    "regularization": regularization,
                    "fold": fold,
                    "fit_seconds": calibrator.fit_summary.seconds,
                    "final_nll": calibrator.fit_summary.final_nll,
                }
            ),
            flush=True,
        )
    return oof, fit_summaries


def cross_validate(output: Path) -> None:
    if not (output / "protocol" / "fold_assignments.csv").exists():
        prepare_validation(output)
    validation = load_archives("validation")
    assignments, groups = _load_cv_protocol(output)
    cv_dir = output / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    candidate_oof_paths: dict[tuple[str, float], Path] = {}

    empirical_path = cv_dir / "candidate_C0_empirical.npz"
    empirical_records_path = cv_dir / "candidate_C0_empirical.json"
    if empirical_path.exists() and empirical_records_path.exists():
        with np.load(empirical_path) as archive:
            empirical_selected = [archive[f"seed{seed}"] for seed in SEEDS]
        payload = json.loads(empirical_records_path.read_text(encoding="utf-8"))
        records = payload["records"]
    else:
        empirical_oof = _crossfit_empirical(validation, assignments)
        records, selected_index = _score_strengths(
            "C0_empirical", "hour_empirical", 0.0, empirical_oof,
            validation[0]["observations"], groups
        )
        empirical_selected = [empirical_oof[selected_index, seed] for seed in SEEDS]
        _save_oof(
            empirical_path,
            empirical_selected,
            validation[0],
            {"selected_strength": STRENGTHS[selected_index], "tail_rule": "bounded"},
        )
        write_json(empirical_records_path, {"records": records})
    all_records.extend(records)
    candidate_oof_paths[("C0_empirical", 0.0)] = empirical_path

    for method, (variant, regularizations) in VARIANT_SPECS.items():
        for regularization in regularizations:
            name = _candidate_name(method, regularization)
            selected_path = cv_dir / f"candidate_{name}.npz"
            record_path = cv_dir / f"candidate_{name}.json"
            if selected_path.exists() and record_path.exists():
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                records = payload["records"]
            else:
                oof, summaries = _crossfit_model(
                    validation,
                    assignments,
                    method=method,
                    variant=variant,
                    regularization=regularization,
                )
                records, selected_index = _score_strengths(
                    method,
                    variant,
                    regularization,
                    oof,
                    validation[0]["observations"],
                    groups,
                )
                selected_arrays = [oof[selected_index, seed] for seed in SEEDS]
                _save_oof(
                    selected_path,
                    selected_arrays,
                    validation[0],
                    {
                        "method": method,
                        "variant": variant,
                        "regularization": regularization,
                        "selected_strength_within_candidate": STRENGTHS[selected_index],
                    },
                )
                write_json(record_path, {"records": records, "fit_summaries": summaries})
            all_records.extend(records)
            candidate_oof_paths[(method, float(regularization))] = selected_path

    selection = pd.DataFrame(all_records).sort_values(
        ["method", "objective", "strength", "regularization"], ascending=[True, True, True, False]
    )
    selection.to_csv(cv_dir / "selection_table.csv", index=False)
    selected_configs: dict[str, dict[str, Any]] = {}
    oof_dir = output / "scenarios" / "validation_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for method in selection["method"].unique():
        rows = selection.loc[selection["method"] == method]
        chosen = rows.sort_values(
            ["objective", "strength", "regularization"], ascending=[True, True, False]
        ).iloc[0].to_dict()
        selected_configs[method] = {
            key: (float(value) if isinstance(value, (np.floating, float)) else value)
            for key, value in chosen.items()
            if key in {
                "variant", "regularization", "strength", "objective", "mean_seed_CRPS",
                "mean_seed_global_ACE_50_80_90", "mean_seed_family_equal_conditional_ACE90"
            }
        }
        source = candidate_oof_paths[(method, float(chosen["regularization"]))]
        with np.load(source, allow_pickle=False) as archive:
            arrays = [archive[f"seed{seed}"] for seed in SEEDS]
        _save_oof(
            oof_dir / f"{method}.npz",
            arrays,
            validation[0],
            {"selected_configuration": selected_configs[method]},
        )
    write_json(
        cv_dir / "selected_configs.json",
        {
            "selection_split": "validation OOF only",
            "test_opened_during_stage": False,
            "shared_across_model_seeds": True,
            "tail_rule": "bounded",
            "configs": selected_configs,
        },
    )
    print(json.dumps({"stage": "cv_complete", "selected": selected_configs}), flush=True)


def finalize_test(output: Path) -> None:
    selected_path = output / "cv" / "selected_configs.json"
    if not selected_path.exists():
        raise FileNotFoundError("run validation-only CV before finalize")
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    selected = selected_payload["configs"]
    validation = load_archives("validation")
    test = load_archives("test")
    validation_days = np.unique(validation[0]["day"].astype("datetime64[D]"))
    test_days = np.unique(test[0]["day"].astype("datetime64[D]"))
    if np.intersect1d(validation_days, test_days).size:
        raise AssertionError("validation/test calendar dates overlap")
    protocol_dir = output / "protocol"
    existing_audit = json.loads((protocol_dir / "validation_input_audit.json").read_text(encoding="utf-8"))
    existing_audit["test"] = _input_audit("test", test)
    existing_audit["validation_test_day_overlap"] = 0
    existing_audit["test_opened_after_selection_artifact"] = True
    write_json(protocol_dir / "input_audit.json", existing_audit)

    scenario_dir = output / "scenarios" / "test"
    model_dir = output / "calibrators"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_records: dict[str, Any] = {}

    c0_config = selected["C0_empirical"]
    for seed, (val_archive, test_archive) in enumerate(zip(validation, test)):
        destination = scenario_dir / f"C0_empirical_seed{seed}.npz"
        calibrator = HourEmpiricalCalibrator.fit(val_archive["scenarios"], val_archive["observations"])
        np.savez_compressed(
            model_dir / f"C0_empirical_seed{seed}.npz",
            sorted_pits=calibrator.sorted_pits,
            strength=float(c0_config["strength"]),
            tail_rule=np.asarray("bounded"),
        )
        if not destination.exists():
            calibrated = calibrator.transform(
                test_archive["scenarios"],
                strength=float(c0_config["strength"]),
                tail_rule="bounded",
            )
            _save_scenario_archive(
                destination,
                calibrated,
                test_archive,
                {
                    "method": "C0_empirical",
                    "seed": seed,
                    "selection": c0_config,
                    "split": "test_once",
                    "source_sha256": sha256(_archive_path(seed, "test")),
                },
            )

    pooled_scenarios = np.concatenate([archive["scenarios"] for archive in validation])
    pooled_observations = np.concatenate([archive["observations"] for archive in validation])
    pooled_zone = np.concatenate([archive["zone"] for archive in validation])
    for method, (variant, _) in VARIANT_SPECS.items():
        config = selected[method]
        checkpoint = model_dir / f"{method}.pt"
        if checkpoint.exists():
            calibrator, _ = RAHCalibratorV2.load(
                checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            calibrator = RAHCalibratorV2.fit(
                pooled_scenarios,
                pooled_observations,
                pooled_zone,
                variant=variant,
                regularization=float(config["regularization"]),
                epochs=160,
                learning_rate=0.03,
                seed=20260718,
            )
            calibrator.save(
                checkpoint,
                metadata={"method": method, "selection": config, "pooled_model_seeds": list(SEEDS)},
            )
        fit_records[method] = calibrator.fit_summary.to_dict()
        for seed, test_archive in enumerate(test):
            destination = scenario_dir / f"{method}_seed{seed}.npz"
            if not destination.exists():
                calibrated = calibrator.transform(
                    test_archive["scenarios"],
                    test_archive["zone"],
                    strength=float(config["strength"]),
                    tail_rule="bounded",
                )
                _save_scenario_archive(
                    destination,
                    calibrated,
                    test_archive,
                    {
                        "method": method,
                        "seed": seed,
                        "selection": config,
                        "tail_rule": "bounded",
                        "split": "test_once",
                        "source_sha256": sha256(_archive_path(seed, "test")),
                        "calibrator_sha256": sha256(checkpoint),
                    },
                )
        if method == "C6_RAHC":
            for tail_rule in ("linear", "clamp"):
                for seed, test_archive in enumerate(test):
                    destination = scenario_dir / f"C6_RAHC_{tail_rule}_seed{seed}.npz"
                    if not destination.exists():
                        calibrated = calibrator.transform(
                            test_archive["scenarios"],
                            test_archive["zone"],
                            strength=float(config["strength"]),
                            tail_rule=tail_rule,
                        )
                        _save_scenario_archive(
                            destination,
                            calibrated,
                            test_archive,
                            {
                                "method": f"C6_RAHC_{tail_rule}",
                                "seed": seed,
                                "selection": config,
                                "tail_rule": tail_rule,
                                "split": "test_once_sensitivity",
                            },
                        )
    write_json(model_dir / "fit_summaries.json", fit_records)
    print(json.dumps({"stage": "finalize_complete", "methods": list(selected)}), flush=True)


def _load_scenario_file(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive["scenarios"]


def evaluate(output: Path) -> None:
    test = load_archives("test")
    validation = load_archives("validation")
    protocol = GroupingProtocol.fit(
        np.stack([archive["scenarios"] for archive in validation]), validation[0]["zone"]
    )
    common_test_raw = np.stack([archive["scenarios"] for archive in test])
    scenario_dir = output / "scenarios" / "test"
    methods: dict[str, list[np.ndarray]] = {
        "raw": [archive["scenarios"] for archive in test],
        "legacy_G0": [
            _load_scenario_file(_archive_path(seed, "test", "calibrated")) for seed in SEEDS
        ],
    }
    for method in ("C0_empirical", *VARIANT_SPECS.keys()):
        methods[method] = [
            _load_scenario_file(scenario_dir / f"{method}_seed{seed}.npz") for seed in SEEDS
        ]
    for tail_rule in ("linear", "clamp"):
        name = f"C6_RAHC_{tail_rule}"
        methods[name] = [
            _load_scenario_file(scenario_dir / f"{name}_seed{seed}.npz") for seed in SEEDS
        ]
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    overall_records: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    group_summary_records: list[dict[str, Any]] = []
    rank_records: list[dict[str, Any]] = []
    for method, arrays in methods.items():
        for seed, values in enumerate(arrays):
            scores = overall_scores(values, test[seed]["observations"])
            overall_records.append({"method": method, "seed": seed, **scores})
            group = group_interval_metrics(
                values,
                test[seed]["observations"],
                test[seed]["zone"],
                protocol,
                grouping_raw_scenarios=common_test_raw,
            )
            group.insert(0, "seed", seed)
            group.insert(0, "method", method)
            group_frames.append(group)
            family, summary = summarize_group_metrics(group)
            family.insert(0, "seed", seed)
            family.insert(0, "method", method)
            family.to_csv(metrics_dir / f"family_summary_{method}_seed{seed}.csv", index=False)
            group_summary_records.append({"method": method, "seed": seed, **summary})
            if method != "raw":
                audit = rank_audit(test[seed]["scenarios"], values)
                rank_records.append({"method": method, "seed": seed, **audit})
            print(
                json.dumps(
                    {"stage": "evaluate", "method": method, "seed": seed,
                     "CRPS": scores["CRPS"], "coverage_90": scores["coverage_90"]}
                ),
                flush=True,
            )
    overall = pd.DataFrame(overall_records)
    overall.to_csv(metrics_dir / "overall_metrics.csv", index=False)
    seed_summary = overall.groupby("method").agg(["mean", "std"])
    seed_summary.columns = [f"{metric}_{stat}" for metric, stat in seed_summary.columns]
    seed_summary.reset_index().to_csv(metrics_dir / "overall_seed_summary.csv", index=False)
    pd.concat(group_frames, ignore_index=True).to_csv(metrics_dir / "group_metrics.csv", index=False)
    pd.DataFrame(group_summary_records).to_csv(metrics_dir / "group_summary.csv", index=False)
    pd.DataFrame(rank_records).to_csv(metrics_dir / "copula_rank_audit.csv", index=False)
    write_json(
        metrics_dir / "evaluation_scope.json",
        {
            "status": "exploratory reused-test evaluation",
            "test_thresholds": "frozen from aligned validation raw forecasts",
            "seed_handling": "three model seeds separately scored; same observations are not independent",
            "copula_claim": "stable tie-break temporal member-rank template within each zone-day only",
            "methods": list(methods),
        },
    )
    print(json.dumps({"stage": "evaluation_complete", "methods": list(methods)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAHC-CR-MS-CADM experiment")
    parser.add_argument(
        "stage", choices=("prepare", "cv", "finalize", "evaluate", "all"), nargs="?", default="all"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in {"prepare", "all"}:
        prepare_validation(output)
    if args.stage in {"cv", "all"}:
        cross_validate(output)
    if args.stage in {"finalize", "all"}:
        finalize_test(output)
    if args.stage in {"evaluate", "all"}:
        evaluate(output)


if __name__ == "__main__":
    main()
