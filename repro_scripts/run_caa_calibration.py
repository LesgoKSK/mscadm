"""Leakage-closed CAA calibration, locking, final fitting, and test application.

The three phases in this runner form a hard state machine:

``select-and-lock``
    Reads only the nine outer-calibration raw archives, performs strict nested
    OOF candidate generation, aggregates the three outer replications as nine
    model replicates over the same 50 calendar dates, and atomically freezes a
    single shared method lock.
``fit-final``
    Validates that lock and fits calibration-only final calibrators.  Test
    archive paths are neither constructed nor inspected in this phase.
``apply-test``
    Validates the lock and every final-calibrator content hash before any of
    the nine test archives are opened, then applies the frozen configurations.

This file deliberately imports :mod:`caa_rahc.candidates_nested`, not the
early prototype candidate runner.  The nested implementation closes the
upstream C0 dependency inside every gate-training fold.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from caa_rahc.aggregate import aggregate_outer_records
from caa_rahc.baselines import (
    crossfit_legacy_linear,
    fit_c0,
    transform_c0,
)
from caa_rahc.candidates_nested import (
    FAMILIES,
    CandidateGenerationResult,
    CandidateGrid,
    generate_oof_candidates,
)
from caa_rahc.features import build_features
from caa_rahc.gate import FittedTailGate, fit_tail_gate
from caa_rahc.hinge_tail import atom_aware_logit_hinge_tail
from caa_rahc.metrics import analytic_atom_scores
from caa_rahc.selection import CandidateRecord, METRIC_KEYS, select_candidates
from caa_rahc.structural_atom import StructuralZeroModel, blend_zero_probability
from cr_mscadm.calibration import CopulaPITCalibrator
from rahc.group_metrics import GroupingProtocol
from rahc.v2_calibration import RAHCalibratorV2, rank_audit
from repro_scripts import run_caa_outer as outer_runner


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = outer_runner.DEFAULT_CONFIG
SELECTION_DIR_NAME = "calibration"
FINAL_DIR_NAME = "calibrators"
TEST_OUTPUT_DIR_NAME = "scenarios"

# A1 is a frozen ablation, not a searched hyperparameter family.
A1_PROTOCOL = {
    "variant": "full",
    "regularization": 0.1,
    "strength": 1.0,
    "epochs": 160,
    "learning_rate": 0.03,
    "tail_rule": "linear",
}

ANALYSIS_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *outer_runner.SELECTION_CODE_PATHS,
            "caa_rahc/candidates_nested.py",
            "caa_rahc/aggregate.py",
            "rahc/group_metrics.py",
            "rahc/evaluation.py",
            "rahc/v2_features.py",
            "rahc/v2_model.py",
            "repro_scripts/run_caa_calibration.py",
        )
    )
)

LOCK_SELECTION_KEYS = ("main_A4", "A1", "A2", "A3", "A5", "A6")
APPLIED_FAMILIES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
AUTHORITATIVE_MODULE = "caa_rahc.candidates_nested.generate_oof_candidates"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{field} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{field} must be UTC")
    return parsed


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file blocks atomic write: {temporary}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file blocks atomic write: {temporary}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file blocks atomic write: {temporary}")
    columns = sorted({str(key) for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                    )
                    for key in columns
                }
            )
    temporary.replace(path)


def analysis_code_fingerprint() -> dict[str, Any]:
    """Hash every implementation file capable of changing the lock decision."""

    records: list[dict[str, Any]] = []
    for relative in ANALYSIS_CODE_PATHS:
        path = WORKSPACE / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered analysis code is missing: {path}")
        records.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": outer_runner.sha256_file(path),
            }
        )
    return {
        "algorithm": "sha256",
        "combined_sha256": outer_runner.canonical_json_sha256(records),
        "files": records,
    }


@dataclass(frozen=True)
class ExperimentContext:
    config_path: Path
    config: dict[str, Any]
    manifest: dict[str, Any]
    root: Path
    bundles: dict[int, Any]


@dataclass(frozen=True)
class RawOuterInputs:
    outer: int
    split: str
    scenarios_by_seed: list[np.ndarray]
    observations: np.ndarray
    zone: np.ndarray
    day: np.ndarray
    metadata_by_seed: dict[int, dict[str, Any]]
    audit_by_seed: dict[int, dict[str, Any]]
    archive_sha256_by_seed: dict[int, str]


@dataclass(frozen=True)
class SelectionRun:
    groupings: dict[int, dict[str, Any]]
    grouping_sha256: dict[int, str]
    outer_results: dict[int, CandidateGenerationResult]
    aggregated_records: dict[str, list[CandidateRecord]]
    selector_audits: dict[str, dict[str, Any]]
    selected: dict[str, dict[str, Any]]
    atom_score_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class LoadedFinalOuter:
    outer: int
    c0_by_seed: dict[int, CopulaPITCalibrator]
    structural_atom: StructuralZeroModel
    regularized_gate: FittedTailGate
    no_shrink_gate: FittedTailGate
    legacy_a1: RAHCalibratorV2
    grouping: GroupingProtocol
    manifest: dict[str, Any]


def load_context(config: str | Path = DEFAULT_CONFIG) -> ExperimentContext:
    config_path, experiment = outer_runner.load_experiment_config(config)
    manifest = outer_runner.load_frozen_manifest(config_path, experiment)
    root = outer_runner.output_root(experiment)
    bundles: dict[int, Any] = {}
    for outer in experiment["outer_splits"]:
        index = int(outer)
        bundle = outer_runner.build_nested_gefcom2014(
            outer_runner.data_root(experiment), outer=index
        )
        outer_runner.verify_bundle_against_manifest(bundle, manifest, index)
        bundles[index] = bundle
    return ExperimentContext(config_path, experiment, manifest, root, bundles)


def _expected_archive_metadata(
    context: ExperimentContext, *, outer: int, seed: int, split: str
) -> dict[str, Any]:
    bundle = context.bundles[outer]
    sampling = context.config["sampling"]
    checkpoint = outer_runner.run_directory(context.root, outer, seed) / "final.pt"
    checkpoint_audit = outer_runner.validate_checkpoint(
        checkpoint,
        config_path=context.config_path,
        config=context.config,
        manifest=context.manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
    )
    return {
        "model": "cr_mscadm",
        "variant": "full",
        "training_seed": int(seed),
        "sampling_seed": outer_runner.sampling_seed(outer, seed, split),
        "split": split,
        "outer": int(outer),
        "scenarios": int(sampling["scenarios"]),
        "sampling_steps": int(sampling["steps"]),
        "sampler": "ddim",
        "eta": float(sampling["eta"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
        "split_date_sha256": bundle.protocol["date_sha256"][split],
        "protocol_sha256": bundle.protocol["protocol_sha256"],
        "config_sha256": outer_runner.sha256_file(context.config_path),
        "code_sha256": context.manifest["code"]["combined_sha256"],
    }


def _load_metadata_scalar(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise RuntimeError("archive metadata must be a scalar JSON string")
    parsed = json.loads(str(value.item()))
    if not isinstance(parsed, dict):
        raise RuntimeError("archive metadata JSON must contain an object")
    return parsed


def load_raw_inputs(
    context: ExperimentContext, *, outer: int, split: str
) -> RawOuterInputs:
    """Strictly validate and load one outer's three raw archives.

    Callers choose the split explicitly.  Consequently ``select-and-lock``
    and ``fit-final`` never construct a test path by accident.
    """

    if split not in {"calibration", "test"}:
        raise ValueError("raw split must be calibration or test")
    bundle = context.bundles[int(outer)]
    split_data = getattr(bundle, split)
    scenarios: list[np.ndarray] = []
    metadata_by_seed: dict[int, dict[str, Any]] = {}
    audit_by_seed: dict[int, dict[str, Any]] = {}
    archive_hashes: dict[int, str] = {}
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for configured_seed in context.config["model_seeds"]:
        seed = int(configured_seed)
        archive = outer_runner.scenario_archive_path(
            context.root, int(outer), seed, split
        )
        audit = outer_runner.validate_archive(
            archive,
            split_data=split_data,
            expected_metadata=_expected_archive_metadata(
                context, outer=int(outer), seed=seed, split=split
            ),
            scenarios_count=int(context.config["sampling"]["scenarios"]),
        )
        with np.load(archive, allow_pickle=False) as stored:
            values = np.asarray(stored["scenarios"]).copy()
            observations = np.asarray(stored["observations"]).copy()
            zone = np.asarray(stored["zone"]).copy()
            day = np.asarray(stored["day"]).copy()
            metadata = _load_metadata_scalar(stored["metadata"])
        aligned = (observations, zone, day)
        if reference is None:
            reference = aligned
        elif any(not np.array_equal(left, right) for left, right in zip(reference, aligned)):
            raise RuntimeError(f"outer{outer} {split} raw archives are not seed-aligned")
        scenarios.append(values)
        metadata_by_seed[seed] = metadata
        audit_by_seed[seed] = audit
        archive_hashes[seed] = outer_runner.sha256_file(archive)
    assert reference is not None
    return RawOuterInputs(
        outer=int(outer),
        split=split,
        scenarios_by_seed=scenarios,
        observations=reference[0],
        zone=reference[1],
        day=reference[2],
        metadata_by_seed=metadata_by_seed,
        audit_by_seed=audit_by_seed,
        archive_sha256_by_seed=archive_hashes,
    )


def candidate_grid_from_config(config: Mapping[str, Any], *, device: str) -> CandidateGrid:
    caa = config["caa"]
    baseline = config["baseline"]
    if caa.get("width_cap_reference") != "atom_only":
        raise ValueError("frozen CAA width_cap_reference must be atom_only")
    return CandidateGrid(
        atom_strengths=tuple(float(value) for value in caa["atom_strengths"]),
        tail_strengths=tuple(float(value) for value in caa["tail_strengths"]),
        maximum_logit_shifts=tuple(
            float(value) for value in caa["maximum_logit_shifts"]
        ),
        maximum_width_increases=tuple(
            float(value) for value in caa["maximum_width_increases"]
        ),
        atom_regularization_c=float(caa["atom_regularization_c"]),
        zero_model_maximum_iterations=int(caa.get("zero_model_maximum_iterations", 1000)),
        anchor=float(caa["anchor"]),
        maximum_lambda=float(caa["maximum_lambda"]),
        gate_regularization=float(caa["gate_regularization"]),
        gate_no_shrink_regularization=float(caa["gate_no_shrink_regularization"]),
        gate_steps=int(caa["gate_steps"]),
        gate_batch_cells=int(caa["gate_batch_cells"]),
        gate_learning_rate=float(caa["gate_learning_rate"]),
        gate_device=str(device),
        folds=int(baseline["folds"]),
        fold_seed=int(baseline["fold_seed"]),
        c0_strength=float(baseline["calibration_strength"]),
    )


def _selector_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    constraints = config["constraints"]
    return {
        "bootstrap_replicates": int(constraints["bootstrap_replicates"]),
        "bootstrap_seed": int(constraints["bootstrap_seed"]),
        "upper_probability": float(constraints["upper_probability"]),
        "crps_relative_margin": float(constraints["crps_relative_degradation_u95"]),
        "secondary_relative_margin": float(
            constraints["secondary_relative_degradation_u95"]
        ),
        "width_absolute_margin": float(constraints["width_absolute_increase_u95"]),
    }


def _record_payload(record: CandidateRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "selection_policy": record.selection_policy,
        "metrics": {
            metric: np.asarray(record.metrics[metric], dtype=np.float64).tolist()
            for metric in METRIC_KEYS
        },
        "conditional_ace90": np.asarray(record.conditional_ace90).tolist(),
        "gate_maximum": float(record.gate_maximum),
        "width_cap": float(record.width_cap),
        "shrinkage": float(record.shrinkage),
        "metadata": _jsonable(record.metadata),
    }


def _aggregate_catalog(
    outer_results: Mapping[int, CandidateGenerationResult],
) -> dict[str, list[CandidateRecord]]:
    aggregated: dict[str, list[CandidateRecord]] = {}
    for family in FAMILIES:
        records_by_outer = {
            int(outer): result.records_for_family(family)
            for outer, result in sorted(outer_results.items())
        }
        aggregated[family] = aggregate_outer_records(records_by_outer)
        for record in aggregated[family]:
            for metric in METRIC_KEYS:
                values = np.asarray(record.metrics[metric])
                if values.shape != (9, 50):
                    raise RuntimeError(
                        f"aggregated {family}/{record.name}.{metric} must retain "
                        f"the frozen [9,50] replicate/date axes, found {values.shape}"
                    )
    return aggregated


def _find_config_across_outers(
    outer_results: Mapping[int, CandidateGenerationResult],
    *,
    family: str,
    selected_name: str,
) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for outer, result in sorted(outer_results.items()):
        matches = [
            record
            for record in result.records_for_family(family)
            if record.name == selected_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"outer{outer} does not contain exactly one {family}/{selected_name} record"
            )
        config = matches[0].metadata.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError(f"{family}/{selected_name} lacks a frozen config")
        configs.append(dict(config))
    reference = configs[0]
    if any(outer_runner.canonical_json_sha256(value) != outer_runner.canonical_json_sha256(reference) for value in configs[1:]):
        raise RuntimeError(f"selected {family}/{selected_name} config differs across outers")
    return _jsonable(reference)


def _baseline_locked_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "definition": "exact A0",
        "method": "per-seed full-calibration CopulaPITCalibrator",
        "strength": float(config["baseline"]["calibration_strength"]),
        "tail_rule": "linear",
        "fallback_contract": "bytewise/equality exact A0 array, no CAA transform call",
    }


def _selected_configs_payload(
    config: Mapping[str, Any], selected: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_config = _baseline_locked_config(config)
    entries: dict[str, dict[str, Any]] = {
        "A0": {
            "selected": "A0",
            "fallback": False,
            "config": baseline_config,
            "selected_config": baseline_config,
            "selected_config_sha256": outer_runner.canonical_json_sha256(baseline_config),
        }
    }
    for family in APPLIED_FAMILIES[1:]:
        key = "main_A4" if family == "A4" else family
        decision = selected[key]
        locked_config = dict(decision["config"])
        entries[family] = {
            "selected": str(decision["selected"]),
            "fallback": bool(decision["fallback"]),
            "config": locked_config,
            "selected_config": locked_config,
            "selected_config_sha256": outer_runner.canonical_json_sha256(
                locked_config
            ),
        }
    return entries


def _selection_entry(
    *,
    family: str,
    selector_audit: Mapping[str, Any],
    decision_key: str,
    outer_results: Mapping[int, CandidateGenerationResult],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    decision = selector_audit[decision_key]
    selected_name = str(decision["selected"])
    fallback = bool(decision["fallback"])
    if fallback:
        if selected_name != "A0":
            raise RuntimeError("fallback decision is not exact A0")
        locked_config = _baseline_locked_config(config)
    else:
        locked_config = _find_config_across_outers(
            outer_results, family=family, selected_name=selected_name
        )
    return {
        "family": family,
        "selected": selected_name,
        "fallback": fallback,
        "config": locked_config,
        "decision_key": decision_key,
    }


def _atom_score_rows(
    outer: int,
    result: CandidateGenerationResult,
    bundle: Any,
) -> list[dict[str, Any]]:
    model = result.zero_model
    rows: list[dict[str, Any]] = []
    for split_name, split in (("train", bundle.train), ("calibration", bundle.calibration)):
        pi0 = model.predict_zero(split.condition)
        pi1 = (1.0 - pi0) * float(model.upper_conditional_probability)
        score = analytic_atom_scores(pi0, pi1, split.target)
        rows.append({"outer": int(outer), "split": split_name, **score})
    return rows


def run_calibration_selection(
    context: ExperimentContext,
    raw_by_outer: Mapping[int, RawOuterInputs],
    *,
    device: str,
) -> SelectionRun:
    grid = candidate_grid_from_config(context.config, device=device)
    outer_results: dict[int, CandidateGenerationResult] = {}
    groupings: dict[int, dict[str, Any]] = {}
    grouping_hashes: dict[int, str] = {}
    atom_rows: list[dict[str, Any]] = []
    reference_dates: np.ndarray | None = None
    for outer in sorted(raw_by_outer):
        raw = raw_by_outer[outer]
        bundle = context.bundles[outer]
        dates = np.unique(raw.day.astype("datetime64[D]"))
        if dates.shape != (50,):
            raise RuntimeError(
                f"outer{outer} calibration must contain exactly 50 unique dates"
            )
        if reference_dates is None:
            reference_dates = dates
        elif not np.array_equal(reference_dates, dates):
            raise RuntimeError("outer calibration archives do not share the frozen 50 dates")
        raw_stack = np.stack(raw.scenarios_by_seed)
        grouping = GroupingProtocol.fit(raw_stack, raw.zone)
        grouping_payload = grouping.to_dict()
        assignments = grouping.assign(raw_stack, raw.zone)
        groupings[outer] = grouping_payload
        grouping_hashes[outer] = outer_runner.canonical_json_sha256(grouping_payload)
        a1 = crossfit_legacy_linear(
            raw.scenarios_by_seed,
            raw.observations,
            raw.zone,
            raw.day,
            folds=int(grid.folds),
            fold_seed=int(grid.fold_seed),
            regularization=float(A1_PROTOCOL["regularization"]),
            strength=float(A1_PROTOCOL["strength"]),
            epochs=int(A1_PROTOCOL["epochs"]),
            device=device,
        )
        result = generate_oof_candidates(
            raw.scenarios_by_seed,
            raw.observations,
            raw.zone,
            raw.day,
            train_condition=bundle.train.condition,
            train_target=bundle.train.target,
            calibration_condition=bundle.calibration.condition,
            assignments=assignments,
            grid=grid,
            train_day=bundle.train.day,
            a1_scenarios_by_seed=a1.scenarios_by_seed,
            a1_metadata={
                "config": dict(A1_PROTOCOL),
                "fold_assignments": a1.fold_assignments,
                "fit_records": a1.fit_records,
            },
        )
        outer_results[outer] = result
        atom_rows.extend(_atom_score_rows(outer, result, bundle))
    assert reference_dates is not None
    aggregated = _aggregate_catalog(outer_results)
    baseline_records = aggregated["A0"]
    if len(baseline_records) != 1 or baseline_records[0].name != "A0":
        raise RuntimeError("aggregated candidate catalog lacks one exact A0 record")
    baseline = baseline_records[0]
    selector_kwargs = _selector_kwargs(context.config)
    selector_audits: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for lock_key, family, decision_key in (
        ("main_A4", "A4", "constrained_A4"),
        ("A1", "A1", "constrained_A4"),
        ("A2", "A2", "constrained_A4"),
        ("A3", "A3", "constrained_A4"),
        ("A5", "A5", "unconstrained_A5"),
        ("A6", "A6", "constrained_A4"),
    ):
        audit = select_candidates(
            [baseline, *aggregated[family]],
            baseline_name="A0",
            day_ids=reference_dates,
            **selector_kwargs,
        )
        selector_audits[lock_key] = audit
        selected[lock_key] = _selection_entry(
            family=family,
            selector_audit=audit,
            decision_key=decision_key,
            outer_results=outer_results,
            config=context.config,
        )
    return SelectionRun(
        groupings,
        grouping_hashes,
        outer_results,
        aggregated,
        selector_audits,
        selected,
        atom_rows,
    )


def _selection_summary_rows(run: SelectionRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, audit in run.selector_audits.items():
        decision_key = run.selected[path]["decision_key"]
        decision = audit[decision_key]
        for candidate in audit["candidates"]:
            rows.append(
                {
                    "selection_path": path,
                    "candidate": candidate["name"],
                    "selected": candidate["name"] == decision["selected"],
                    "fallback_path": bool(decision["fallback"]),
                    "selection_policy": candidate["selection_policy"],
                    "constraints_passed": candidate["all_constraints_passed"],
                    "conditional_ace90": candidate["conditional_ace90"]["mean"],
                    **{
                        f"point_{metric}": candidate["point_metrics"][metric]
                        for metric in METRIC_KEYS
                    },
                    "metadata": candidate["metadata"],
                }
            )
    return rows


def _selection_audit_payload(
    context: ExperimentContext,
    raw_by_outer: Mapping[int, RawOuterInputs],
    run: SelectionRun,
    grid: CandidateGrid,
) -> dict[str, Any]:
    return {
        "schema": "caa_rahc_calibration_selection_audit_v1",
        "protocol": {
            "test_access": "none; only calibration archives were loaded",
            "candidate_generator": "caa_rahc.candidates_nested.generate_oof_candidates",
            "outer_aggregation": "retain common date axis; concatenate 3 seeds x 3 outers",
            "aggregated_model_replicates": 9,
            "width_cap_reference": "atom_only",
            "grid": grid.to_dict(),
            "a1": dict(A1_PROTOCOL),
        },
        "calibration_archives": {
            f"outer{outer}": {
                f"seed{seed}": {
                    "archive_sha256": raw.archive_sha256_by_seed[seed],
                    "audit": raw.audit_by_seed[seed],
                }
                for seed in sorted(raw.archive_sha256_by_seed)
            }
            for outer, raw in sorted(raw_by_outer.items())
        },
        "grouping_protocols": {
            f"outer{outer}": {
                "sha256": run.grouping_sha256[outer],
                "protocol": run.groupings[outer],
            }
            for outer in sorted(run.groupings)
        },
        "outer_candidate_audits": {
            f"outer{outer}": result.audit
            for outer, result in sorted(run.outer_results.items())
        },
        "selector_audits": run.selector_audits,
        "selected": run.selected,
    }


def _final_fit_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    caa = config["caa"]
    baseline = config["baseline"]
    return {
        "device": str(config.get("device", "cpu")),
        "c0": {
            "fit": "separate per seed on full outer calibration",
            "strength": float(baseline["calibration_strength"]),
        },
        "structural_atom": {
            "fit_split": "base train only",
            "regularization_c": float(caa["atom_regularization_c"]),
            "maximum_iterations": int(caa.get("zero_model_maximum_iterations", 1000)),
        },
        "gates": {
            "fit": "pooled three-seed full-calibration A0/features",
            "regularized": float(caa["gate_regularization"]),
            "no_shrink": float(caa["gate_no_shrink_regularization"]),
            "anchor": float(caa["anchor"]),
            "maximum_lambda": float(caa["maximum_lambda"]),
            "steps": int(caa["gate_steps"]),
            "batch_cells": int(caa["gate_batch_cells"]),
            "learning_rate": float(caa["gate_learning_rate"]),
            "random_seed_rule": "baseline.fold_seed + outer (+1000 for no-shrink)",
        },
        "a1": {
            **dict(A1_PROTOCOL),
            "fit": "pooled three-seed full calibration",
            "random_seed_rule": "baseline.fold_seed + outer",
        },
    }


def select_and_lock(
    config: str | Path = DEFAULT_CONFIG, *, device: str | None = None
) -> dict[str, Any]:
    context = load_context(config)
    lock_path = context.root / outer_runner.SELECTION_LOCK_NAME
    if lock_path.exists():
        return validate_caa_lock(context)
    configured_device = str(context.config.get("device", "cpu"))
    if device is not None and str(device) != configured_device:
        raise RuntimeError(
            f"selection device {device!r} differs from frozen config device "
            f"{configured_device!r}"
        )
    # This is the only raw-input request before the lock, and the literal split
    # is calibration.  No test path is constructed in this phase.
    raw_by_outer = {
        outer: load_raw_inputs(context, outer=outer, split="calibration")
        for outer in sorted(context.bundles)
    }
    selected_device = configured_device
    run = run_calibration_selection(context, raw_by_outer, device=selected_device)
    grid = candidate_grid_from_config(context.config, device=selected_device)
    selection_dir = context.root / SELECTION_DIR_NAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    for outer, result in sorted(run.outer_results.items()):
        outer_dir = selection_dir / f"outer{outer}"
        _write_json_atomic(outer_dir / "candidate_generation.audit.json", result.audit)
        _write_json_atomic(
            outer_dir / "candidate_records.json",
            {
                family: [_record_payload(record) for record in result.records_for_family(family)]
                for family in FAMILIES
            },
        )
        _write_json_atomic(
            outer_dir / "grouping_protocol.json",
            {
                "sha256": run.grouping_sha256[outer],
                "protocol": run.groupings[outer],
            },
        )
    _write_json_atomic(
        selection_dir / "aggregate_records.json",
        {
            family: [_record_payload(record) for record in records]
            for family, records in run.aggregated_records.items()
        },
    )
    analysis = analysis_code_fingerprint()
    selected_configs = _selected_configs_payload(context.config, run.selected)
    audit_payload = _selection_audit_payload(context, raw_by_outer, run, grid)
    audit_payload.update(
        {
            "authoritative_module": AUTHORITATIVE_MODULE,
            "full_code": analysis,
            "full_code_sha256": analysis["combined_sha256"],
            "selected_configs": selected_configs,
        }
    )
    audit_path = selection_dir / "selection.audit.json"
    _write_json_atomic(audit_path, audit_payload)
    _write_csv_atomic(selection_dir / "selection_summary.csv", _selection_summary_rows(run))
    _write_csv_atomic(selection_dir / "analytic_atom_scores.csv", run.atom_score_rows)
    selection_analysis_sha256 = outer_runner.sha256_file(audit_path)
    lock = {
        "schema": "caa_rahc_selection_lock_v1",
        **outer_runner.selection_lock_identity(context.root, context.manifest),
        "analysis_code_sha256": analysis["combined_sha256"],
        "analysis_code": analysis,
        "full_code_sha256": analysis["combined_sha256"],
        "full_code": analysis,
        "authoritative_module": AUTHORITATIVE_MODULE,
        "selection_audit": str(audit_path.relative_to(context.root).as_posix()),
        "selection_audit_sha256": selection_analysis_sha256,
        "analysis_sha256": selection_analysis_sha256,
        "analysis": {
            "path": str(audit_path.relative_to(context.root).as_posix()),
            "sha256": selection_analysis_sha256,
        },
        "frozen_manifest_sha256": outer_runner.sha256_file(
            context.root / outer_runner.FROZEN_MANIFEST_NAME
        ),
        "locked_at_utc": _utc_now(),
        "calibration_archive_sha256": {
            f"outer{outer}": {
                f"seed{seed}": digest
                for seed, digest in sorted(raw.archive_sha256_by_seed.items())
            }
            for outer, raw in sorted(raw_by_outer.items())
        },
        "grouping_protocols": {
            f"outer{outer}": {
                "sha256": run.grouping_sha256[outer],
                "protocol": run.groupings[outer],
            }
            for outer in sorted(run.groupings)
        },
        "candidate_grid": grid.to_dict(),
        "a1_protocol": dict(A1_PROTOCOL),
        "final_fit_protocol": _final_fit_protocol(context.config),
        "selected": run.selected,
        "selected_configs": selected_configs,
        "fallback_contract": "exact A0 array; no transformation call",
        "test_archives_accessed": False,
    }
    _write_json_atomic(lock_path, lock)
    return lock


def validate_caa_lock(context: ExperimentContext) -> dict[str, Any]:
    lock = outer_runner.validate_selection_lock(context.root, context.manifest)
    if lock.get("schema") != "caa_rahc_selection_lock_v1":
        raise RuntimeError("selection lock has the wrong CAA schema")
    analysis = analysis_code_fingerprint()
    if lock.get("analysis_code_sha256") != analysis["combined_sha256"]:
        raise RuntimeError("selection lock analysis-code hash drift")
    if lock.get("analysis_code") != analysis:
        raise RuntimeError("selection lock analysis-code registry drift")
    if lock.get("full_code_sha256") != analysis["combined_sha256"]:
        raise RuntimeError("selection lock full-code hash drift")
    if lock.get("full_code") != analysis:
        raise RuntimeError("selection lock full-code registry drift")
    if lock.get("authoritative_module") != AUTHORITATIVE_MODULE:
        raise RuntimeError("selection lock authoritative module drift")
    _utc_datetime(lock.get("locked_at_utc"), field="locked_at_utc")
    if lock.get("frozen_manifest_sha256") != lock.get("manifest_sha256"):
        raise RuntimeError("selection lock frozen-manifest hash alias drift")
    if tuple(sorted(lock.get("selected", {}))) != tuple(sorted(LOCK_SELECTION_KEYS)):
        raise RuntimeError("selection lock does not contain the six registered decisions")
    audit_relative = lock.get("selection_audit")
    if not isinstance(audit_relative, str):
        raise RuntimeError("selection lock lacks its audit path")
    audit_path = context.root / audit_relative
    if not audit_path.is_file() or outer_runner.sha256_file(audit_path) != lock.get(
        "selection_audit_sha256"
    ):
        raise RuntimeError("selection audit is missing or hash-drifted")
    if lock.get("analysis_sha256") != lock.get("selection_audit_sha256"):
        raise RuntimeError("selection lock analysis hash alias drift")
    if lock.get("analysis") != {
        "path": audit_relative,
        "sha256": lock["selection_audit_sha256"],
    }:
        raise RuntimeError("selection lock analysis identity drift")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_selected_configs = _selected_configs_payload(
        context.config, lock["selected"]
    )
    if audit_payload.get("authoritative_module") != AUTHORITATIVE_MODULE:
        raise RuntimeError("selection audit authoritative module drift")
    if audit_payload.get("full_code") != analysis:
        raise RuntimeError("selection audit full-code registry drift")
    if audit_payload.get("full_code_sha256") != analysis["combined_sha256"]:
        raise RuntimeError("selection audit full-code hash drift")
    if audit_payload.get("selected_configs") != expected_selected_configs:
        raise RuntimeError("selection audit selected-config catalog drift")
    if lock.get("selected_configs") != expected_selected_configs:
        raise RuntimeError("selection lock selected-config catalog drift")
    if lock.get("final_fit_protocol") != _final_fit_protocol(context.config):
        raise RuntimeError("selection lock final-fit protocol drift")
    for key, entry in lock["selected"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
            raise RuntimeError(f"selection lock entry {key} is malformed")
        if bool(entry.get("fallback")) != (entry.get("selected") == "A0"):
            raise RuntimeError(f"selection lock entry {key} has inconsistent fallback state")
        if (
            not entry.get("fallback")
            and entry["config"].get("width_cap_reference", "atom_only") != "atom_only"
            and key != "A1"
        ):
            raise RuntimeError(f"selection lock entry {key} has wrong width-cap reference")
    return lock


def _c0_path(directory: Path, seed: int) -> Path:
    return directory / f"c0_seed{seed}.npz"


def _save_c0(path: Path, model: CopulaPITCalibrator, metadata: Mapping[str, Any]) -> None:
    _write_npz_atomic(
        path,
        sorted_pits=np.asarray(model.sorted_pits),
        strength=np.asarray(float(model.strength)),
        grid_size=np.asarray(int(model.grid_size)),
        metadata=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
    )


def _load_c0(path: Path) -> tuple[CopulaPITCalibrator, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"sorted_pits", "strength", "grid_size", "metadata"}
        if set(stored.files) != required:
            raise RuntimeError(f"C0 archive schema mismatch: {path}")
        sorted_pits = np.asarray(stored["sorted_pits"]).copy()
        strength = float(np.asarray(stored["strength"]).item())
        grid_size = int(np.asarray(stored["grid_size"]).item())
        metadata = _load_metadata_scalar(np.asarray(stored["metadata"]))
    if sorted_pits.ndim != 2 or sorted_pits.shape[1] != 24:
        raise RuntimeError("C0 sorted PIT array must be [calibration_cells,24]")
    if not np.isfinite(sorted_pits).all():
        raise RuntimeError("C0 sorted PIT array contains non-finite values")
    return CopulaPITCalibrator(sorted_pits, strength, grid_size), metadata


def _artifact_metadata(
    context: ExperimentContext,
    lock: Mapping[str, Any],
    raw: RawOuterInputs,
    *,
    outer: int,
    artifact: str,
) -> dict[str, Any]:
    lock_path = context.root / outer_runner.SELECTION_LOCK_NAME
    generated_at_utc = _utc_now()
    if _utc_datetime(generated_at_utc, field="generated_at_utc") < _utc_datetime(
        lock["locked_at_utc"], field="locked_at_utc"
    ):
        raise RuntimeError("final calibrator timestamp predates selection lock")
    return {
        "schema": "caa_rahc_final_calibrator_metadata_v1",
        "artifact": artifact,
        "outer": int(outer),
        "config_sha256": lock["config_sha256"],
        "protocol_sha256": lock["protocol_sha256"][f"outer{outer}"],
        "selection_lock_sha256": outer_runner.sha256_file(lock_path),
        "analysis_code_sha256": lock["analysis_code_sha256"],
        "selection_analysis_sha256": lock["selection_audit_sha256"],
        "full_code_sha256": lock["full_code_sha256"],
        "generated_after_selection_lock": True,
        "generated_at_utc": generated_at_utc,
        "grouping_protocol_sha256": lock["grouping_protocols"][f"outer{outer}"]["sha256"],
        "calibration_archive_sha256": {
            f"seed{seed}": digest
            for seed, digest in sorted(raw.archive_sha256_by_seed.items())
        },
        "test_data_used": False,
    }


def _final_directory(context: ExperimentContext, outer: int) -> Path:
    return context.root / f"outer{outer}" / FINAL_DIR_NAME


def _registry(directory: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": outer_runner.sha256_file(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]


def validate_registered_files(directory: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on missing, path-escaping, size-drifted, or hash-drifted files."""

    root = directory.resolve()
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("artifact registry contains an invalid path")
        path = (directory / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("artifact registry path escapes final directory") from error
        if not path.is_file():
            raise RuntimeError(f"registered final calibrator is missing: {path}")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise RuntimeError(f"registered final calibrator size drift: {path}")
        if outer_runner.sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"registered final calibrator hash drift: {path}")


def fit_final(
    config: str | Path = DEFAULT_CONFIG, *, device: str | None = None
) -> dict[int, dict[str, Any]]:
    context = load_context(config)
    lock = validate_caa_lock(context)
    # Still calibration-only: the final fitting phase never asks for test paths.
    raw_by_outer = {
        outer: load_raw_inputs(context, outer=outer, split="calibration")
        for outer in sorted(context.bundles)
    }
    for outer, raw in raw_by_outer.items():
        expected_hashes = lock["calibration_archive_sha256"][f"outer{outer}"]
        actual_hashes = {
            f"seed{seed}": digest
            for seed, digest in sorted(raw.archive_sha256_by_seed.items())
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(f"outer{outer} calibration archive hash drift after selection")
    selected_device = str(lock["final_fit_protocol"]["device"])
    if device is not None and str(device) != selected_device:
        raise RuntimeError(
            f"final-fit device {device!r} differs from locked device "
            f"{selected_device!r}"
        )
    outputs: dict[int, dict[str, Any]] = {}
    for outer, raw in sorted(raw_by_outer.items()):
        directory = _final_directory(context, outer)
        manifest_path = directory / "final_manifest.json"
        if manifest_path.exists():
            loaded = load_final_outer(context, lock, outer=outer)
            outputs[outer] = loaded.manifest
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"partial final-calibrator directory lacks a manifest: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        grouping_payload = lock["grouping_protocols"][f"outer{outer}"]["protocol"]
        grouping = GroupingProtocol.fit(np.stack(raw.scenarios_by_seed), raw.zone)
        if grouping.to_dict() != grouping_payload:
            raise RuntimeError(f"outer{outer} refitted grouping differs from selection lock")
        common = _artifact_metadata(
            context, lock, raw, outer=outer, artifact="common"
        )
        grouping_path = directory / "grouping_protocol.json"
        _write_json_atomic(
            grouping_path,
            {
                "metadata": {**common, "artifact": "grouping_protocol"},
                "protocol": grouping_payload,
            },
        )
        c0_models = fit_c0(
            raw.scenarios_by_seed,
            raw.observations,
            strength=float(context.config["baseline"]["calibration_strength"]),
        )
        c0_paths: list[Path] = []
        for seed, model in zip(context.config["model_seeds"], c0_models):
            path = _c0_path(directory, int(seed))
            _save_c0(
                path,
                model,
                {**common, "artifact": "c0", "model_seed": int(seed)},
            )
            c0_paths.append(path)
        a0 = transform_c0(c0_models, raw.scenarios_by_seed)
        caa = context.config["caa"]
        structural = StructuralZeroModel.fit(
            context.bundles[outer].train.condition,
            context.bundles[outer].train.target,
            regularization_c=float(caa["atom_regularization_c"]),
            maximum_iterations=int(caa.get("zero_model_maximum_iterations", 1000)),
        )
        structural_path = directory / "structural_atom.pt"
        structural.save(
            structural_path,
            {**common, "artifact": "structural_atom", "fit_split": "base train only"},
        )
        features = [
            build_features(raw_seed, a0_seed)
            for raw_seed, a0_seed in zip(raw.scenarios_by_seed, a0)
        ]
        gate_paths: list[Path] = []
        for label, regularization, offset in (
            ("regularized", float(caa["gate_regularization"]), 0),
            ("no_shrink", float(caa["gate_no_shrink_regularization"]), 1000),
        ):
            gate = fit_tail_gate(
                a0,
                raw.observations,
                features,
                raw.zone,
                anchor=float(caa["anchor"]),
                maximum_lambda=float(caa["maximum_lambda"]),
                regularization=regularization,
                steps=int(caa["gate_steps"]),
                batch_cells=int(caa["gate_batch_cells"]),
                learning_rate=float(caa["gate_learning_rate"]),
                random_seed=int(context.config["baseline"]["fold_seed"]) + outer + offset,
                device=selected_device,
            )
            path = directory / f"gate_{label}.pt"
            gate.save(
                path,
                {
                    **common,
                    "artifact": f"gate_{label}",
                    "regularization": regularization,
                    "fit_split": "full calibration pooled across three model seeds",
                },
            )
            gate_paths.append(path)
        legacy = RAHCalibratorV2.fit(
            np.concatenate(raw.scenarios_by_seed),
            np.concatenate([raw.observations for _ in raw.scenarios_by_seed]),
            np.concatenate([raw.zone for _ in raw.scenarios_by_seed]),
            variant=str(A1_PROTOCOL["variant"]),
            regularization=float(A1_PROTOCOL["regularization"]),
            epochs=int(A1_PROTOCOL["epochs"]),
            learning_rate=float(A1_PROTOCOL["learning_rate"]),
            seed=int(context.config["baseline"]["fold_seed"]) + outer,
            device=selected_device,
        )
        legacy_path = directory / "legacy_a1.pt"
        legacy.save(
            legacy_path,
            {
                **common,
                "artifact": "legacy_a1",
                "protocol": dict(A1_PROTOCOL),
                "fit_split": "full calibration pooled across three model seeds",
            },
        )
        artifacts = [
            grouping_path,
            *c0_paths,
            structural_path,
            *gate_paths,
            legacy_path,
        ]
        manifest = {
            "schema": "caa_rahc_final_calibrators_v1",
            "outer": int(outer),
            "config_sha256": lock["config_sha256"],
            "protocol_sha256": lock["protocol_sha256"][f"outer{outer}"],
            "selection_lock_sha256": outer_runner.sha256_file(
                context.root / outer_runner.SELECTION_LOCK_NAME
            ),
            "analysis_code_sha256": lock["analysis_code_sha256"],
            "selection_analysis_sha256": lock["selection_audit_sha256"],
            "full_code_sha256": lock["full_code_sha256"],
            "generated_after_selection_lock": True,
            "generated_at_utc": common["generated_at_utc"],
            "grouping_protocol_sha256": lock["grouping_protocols"][f"outer{outer}"]["sha256"],
            "calibration_archive_sha256": common["calibration_archive_sha256"],
            "fit_protocol": lock["final_fit_protocol"],
            "files": _registry(directory, artifacts),
            "test_data_used": False,
        }
        _write_json_atomic(manifest_path, manifest)
        outputs[outer] = manifest
    return outputs


def _validate_artifact_metadata(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any], *, outer: int, artifact: str
) -> None:
    expected = {
        "schema": "caa_rahc_final_calibrator_metadata_v1",
        "artifact": artifact,
        "outer": int(outer),
        "config_sha256": manifest["config_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "selection_lock_sha256": manifest["selection_lock_sha256"],
        "analysis_code_sha256": manifest["analysis_code_sha256"],
        "selection_analysis_sha256": manifest["selection_analysis_sha256"],
        "full_code_sha256": manifest["full_code_sha256"],
        "generated_after_selection_lock": True,
        "generated_at_utc": manifest["generated_at_utc"],
        "grouping_protocol_sha256": manifest["grouping_protocol_sha256"],
        "calibration_archive_sha256": manifest["calibration_archive_sha256"],
        "test_data_used": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"outer{outer} {artifact} metadata mismatch for {key}")


def load_final_outer(
    context: ExperimentContext, lock: Mapping[str, Any], *, outer: int
) -> LoadedFinalOuter:
    directory = _final_directory(context, outer)
    manifest_path = directory / "final_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"outer{outer} final calibrator manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "caa_rahc_final_calibrators_v1",
        "outer": int(outer),
        "config_sha256": lock["config_sha256"],
        "protocol_sha256": lock["protocol_sha256"][f"outer{outer}"],
        "selection_lock_sha256": outer_runner.sha256_file(
            context.root / outer_runner.SELECTION_LOCK_NAME
        ),
        "analysis_code_sha256": lock["analysis_code_sha256"],
        "selection_analysis_sha256": lock["selection_audit_sha256"],
        "full_code_sha256": lock["full_code_sha256"],
        "generated_after_selection_lock": True,
        "grouping_protocol_sha256": lock["grouping_protocols"][f"outer{outer}"]["sha256"],
        "calibration_archive_sha256": lock["calibration_archive_sha256"][f"outer{outer}"],
        "fit_protocol": lock["final_fit_protocol"],
        "test_data_used": False,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"outer{outer} final manifest mismatch for {key}")
    generated_at = _utc_datetime(
        manifest.get("generated_at_utc"), field="generated_at_utc"
    )
    if generated_at < _utc_datetime(lock["locked_at_utc"], field="locked_at_utc"):
        raise RuntimeError(f"outer{outer} final manifest predates selection lock")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("final manifest lacks a file registry")
    validate_registered_files(directory, records)
    grouping_stored = json.loads((directory / "grouping_protocol.json").read_text(encoding="utf-8"))
    _validate_artifact_metadata(
        grouping_stored["metadata"], manifest, outer=outer, artifact="grouping_protocol"
    )
    if grouping_stored.get("protocol") != lock["grouping_protocols"][f"outer{outer}"]["protocol"]:
        raise RuntimeError(f"outer{outer} grouping protocol differs from selection lock")
    grouping = GroupingProtocol.from_dict(grouping_stored["protocol"])
    c0_by_seed: dict[int, CopulaPITCalibrator] = {}
    for seed in context.config["model_seeds"]:
        model, metadata = _load_c0(_c0_path(directory, int(seed)))
        _validate_artifact_metadata(metadata, manifest, outer=outer, artifact="c0")
        if int(metadata.get("model_seed", -1)) != int(seed):
            raise RuntimeError(f"outer{outer} C0 model-seed metadata mismatch")
        c0_by_seed[int(seed)] = model
    structural, structural_metadata = StructuralZeroModel.load(
        directory / "structural_atom.pt"
    )
    _validate_artifact_metadata(
        structural_metadata, manifest, outer=outer, artifact="structural_atom"
    )
    regularized, regularized_metadata = FittedTailGate.load(
        directory / "gate_regularized.pt"
    )
    _validate_artifact_metadata(
        regularized_metadata, manifest, outer=outer, artifact="gate_regularized"
    )
    no_shrink, no_shrink_metadata = FittedTailGate.load(
        directory / "gate_no_shrink.pt"
    )
    _validate_artifact_metadata(
        no_shrink_metadata, manifest, outer=outer, artifact="gate_no_shrink"
    )
    legacy, legacy_metadata = RAHCalibratorV2.load(
        directory / "legacy_a1.pt", device="cpu"
    )
    _validate_artifact_metadata(
        legacy_metadata, manifest, outer=outer, artifact="legacy_a1"
    )
    return LoadedFinalOuter(
        int(outer),
        c0_by_seed,
        structural,
        regularized,
        no_shrink,
        legacy,
        grouping,
        manifest,
    )


def _lock_entry(lock: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    key = "main_A4" if family == "A4" else family
    if key not in lock["selected"]:
        raise RuntimeError(f"selection lock lacks {family} decision")
    return lock["selected"][key]


def apply_locked_family(
    family: str,
    entry: Mapping[str, Any],
    baseline_a0: np.ndarray,
    *,
    raw: np.ndarray | None = None,
    zone: np.ndarray | None = None,
    legacy_a1: RAHCalibratorV2 | None = None,
    structural_zero_probability: np.ndarray | None = None,
    structural_upper_conditional_probability: float | None = None,
    gate: FittedTailGate | None = None,
    features: np.ndarray | None = None,
    anchor: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one locked family, preserving exact A0 on every fallback path."""

    baseline = np.asarray(baseline_a0)
    if bool(entry.get("fallback")) or entry.get("selected") == "A0":
        if not (bool(entry.get("fallback")) and entry.get("selected") == "A0"):
            raise RuntimeError(f"{family} lock has inconsistent fallback state")
        output = baseline.copy()
        if not np.array_equal(output, baseline):
            raise AssertionError("exact A0 fallback was not preserved")
        return output, {
            "family": family,
            "selected": "A0",
            "fallback": True,
            "exact_A0": True,
            "transform_called": False,
        }
    config = entry.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{family} selected lock entry lacks a config")
    if family == "A1":
        if raw is None or zone is None or legacy_a1 is None:
            raise ValueError("A1 application needs raw scenarios, zone, and final A1 model")
        output = legacy_a1.transform(
            np.asarray(raw),
            np.asarray(zone),
            strength=float(config["strength"]),
            tail_rule=str(config["tail_rule"]),
        )
        return output, {
            "family": family,
            "selected": entry["selected"],
            "fallback": False,
            "config": _jsonable(config),
        }
    if family not in {"A2", "A3", "A4", "A5", "A6"}:
        raise ValueError(f"unknown locked family {family!r}")
    if config.get("width_cap_reference") != "atom_only":
        raise RuntimeError(f"{family} width-cap reference is not atom_only")
    atom_enabled = bool(config["atom_enabled"])
    tail_enabled = bool(config["tail_enabled"])
    if atom_enabled:
        if structural_zero_probability is None or structural_upper_conditional_probability is None:
            raise ValueError(f"{family} atom application lacks structural probabilities")
        pi0: np.ndarray | float = blend_zero_probability(
            baseline,
            np.asarray(structural_zero_probability),
            float(config["atom_strength"]),
        )
        rho1 = float(structural_upper_conditional_probability)
    else:
        pi0 = 0.0
        rho1 = 0.0
    if tail_enabled:
        if gate is None or features is None or zone is None:
            raise ValueError(f"{family} tail application lacks gate/features/zone")
        predicted = gate.predict(
            np.asarray(features), np.asarray(zone), strength=float(config["tail_strength"])
        )
        lower_gate: np.ndarray | float = predicted[..., 0]
        upper_gate: np.ndarray | float = predicted[..., 1]
    else:
        lower_gate = 0.0
        upper_gate = 0.0
    transformed = atom_aware_logit_hinge_tail(
        baseline,
        pi0,
        rho1,
        lower_gate,
        upper_gate,
        dmax=float(config["dmax"]),
        width_delta_cap=float(config["width_delta_cap"]),
        width_cap_reference="atom_only",
        atom_enabled=atom_enabled,
        tail_enabled=tail_enabled,
        anchor=float(anchor),
    )
    output = transformed.atom_only if family == "A2" else transformed.calibrated
    return output, {
        "family": family,
        "selected": entry["selected"],
        "fallback": False,
        "config": _jsonable(config),
        "transformation": transformed.diagnostics.summary(),
    }


def _save_test_scenario(
    path: Path,
    scenarios: np.ndarray,
    raw: RawOuterInputs,
    metadata: Mapping[str, Any],
) -> None:
    if path.exists():
        raise RuntimeError(f"test application output already exists: {path}")
    audit_path = outer_runner.archive_audit_path(path)
    if audit_path.exists():
        raise RuntimeError(f"orphan test output audit already exists: {audit_path}")
    _write_npz_atomic(
        path,
        scenarios=np.asarray(scenarios),
        observations=raw.observations,
        zone=raw.zone,
        day=raw.day,
        metadata=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
    )
    _write_json_atomic(
        audit_path,
        {
            "schema": "caa_rahc_locked_test_scenario_audit_v1",
            "archive": str(path.resolve()),
            "archive_sha256": outer_runner.sha256_file(path),
            "shape": list(np.asarray(scenarios).shape),
            "dtype": str(np.asarray(scenarios).dtype),
            "metadata_sha256": outer_runner.canonical_json_sha256(metadata),
            "selection_lock_sha256": metadata["selection_lock_sha256"],
            "selection_analysis_sha256": metadata["selection_analysis_sha256"],
            "full_code_sha256": metadata["full_code_sha256"],
            "generated_after_selection_lock": metadata[
                "generated_after_selection_lock"
            ],
            "generated_at_utc": metadata["generated_at_utc"],
            "final_manifest_sha256": metadata["final_manifest_sha256"],
            "raw_test_archive_sha256": metadata["raw_test_archive_sha256"],
            "source_test_raw_sha256": metadata["source_test_raw_sha256"],
            "selected_config_sha256": metadata["selected_config_sha256"],
        },
    )


def apply_test(config: str | Path = DEFAULT_CONFIG) -> dict[int, dict[str, Any]]:
    context = load_context(config)
    lock = validate_caa_lock(context)
    # Validate all three final manifests, their file registries, and embedded
    # metadata first.  A failure here occurs before any test path is requested.
    final_by_outer = {
        outer: load_final_outer(context, lock, outer=outer)
        for outer in sorted(context.bundles)
    }
    # Only after every final calibrator passed do we construct/open test paths.
    raw_by_outer = {
        outer: load_raw_inputs(context, outer=outer, split="test")
        for outer in sorted(context.bundles)
    }
    lock_sha256 = outer_runner.sha256_file(
        context.root / outer_runner.SELECTION_LOCK_NAME
    )
    outputs: dict[int, dict[str, Any]] = {}
    for outer in sorted(raw_by_outer):
        raw = raw_by_outer[outer]
        final = final_by_outer[outer]
        directory = context.root / f"outer{outer}" / TEST_OUTPUT_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        owned_paths: list[Path] = []
        generated_at_utc = _utc_now()
        if _utc_datetime(
            generated_at_utc, field="generated_at_utc"
        ) < _utc_datetime(lock["locked_at_utc"], field="locked_at_utc"):
            raise RuntimeError("test output timestamp predates selection lock")
        raw_stack = np.stack(raw.scenarios_by_seed)
        assignments = final.grouping.assign(raw_stack, raw.zone)
        grouping_metadata = {
            "outer": outer,
            "fit_split": "calibration raw only",
            "assignment_split": "test raw",
            "grouping_protocol_sha256": final.manifest["grouping_protocol_sha256"],
            "selection_lock_sha256": lock_sha256,
            "selection_analysis_sha256": lock["selection_audit_sha256"],
            "full_code_sha256": lock["full_code_sha256"],
            "generated_after_selection_lock": True,
            "generated_at_utc": generated_at_utc,
        }
        grouping_path = directory / "group_assignments_test_final.npz"
        if grouping_path.exists():
            raise RuntimeError(f"test grouping output already exists: {grouping_path}")
        _write_npz_atomic(
            grouping_path,
            **assignments,
            metadata=np.asarray(json.dumps(grouping_metadata, sort_keys=True)),
        )
        owned_paths.append(grouping_path)
        diagnostics: list[dict[str, Any]] = []
        c0_models = [final.c0_by_seed[int(seed)] for seed in context.config["model_seeds"]]
        a0_by_seed = transform_c0(c0_models, raw.scenarios_by_seed)
        structural_probability = final.structural_atom.predict_zero(
            context.bundles[outer].test.condition
        )
        for seed, raw_seed, a0 in zip(
            context.config["model_seeds"], raw.scenarios_by_seed, a0_by_seed
        ):
            seed = int(seed)
            features = build_features(raw_seed, a0)
            common_metadata = {
                "schema": "caa_rahc_locked_test_scenarios_v1",
                "outer": outer,
                "model_seed": seed,
                "selection_lock_sha256": lock_sha256,
                "selection_analysis_sha256": lock["selection_audit_sha256"],
                "full_code_sha256": lock["full_code_sha256"],
                "generated_after_selection_lock": True,
                "generated_at_utc": generated_at_utc,
                "final_manifest_sha256": outer_runner.sha256_file(
                    _final_directory(context, outer) / "final_manifest.json"
                ),
                "raw_test_archive_sha256": raw.archive_sha256_by_seed[seed],
                "source_test_raw_sha256": raw.archive_sha256_by_seed[seed],
                "protocol_sha256": lock["protocol_sha256"][f"outer{outer}"],
                "grouping_protocol_sha256": final.manifest["grouping_protocol_sha256"],
            }
            a0_config = _baseline_locked_config(context.config)
            a0_config_sha256 = outer_runner.canonical_json_sha256(a0_config)
            a0_rank_audit = rank_audit(raw_seed, a0)
            a0_transform_diagnostics = {
                "method": "C0",
                "rank_audit_raw_to_A0": a0_rank_audit,
            }
            a0_path = directory / f"A0_seed{seed}_test_final.npz"
            _save_test_scenario(
                a0_path,
                a0,
                raw,
                {
                    **common_metadata,
                    "family": "A0",
                    "selected": "A0",
                    "fallback": False,
                    "config": a0_config,
                    "selected_config": a0_config,
                    "selected_config_sha256": a0_config_sha256,
                    "transform_diagnostics": a0_transform_diagnostics,
                },
            )
            owned_paths.extend((a0_path, outer_runner.archive_audit_path(a0_path)))
            diagnostics.append(
                {
                    "outer": outer,
                    "model_seed": seed,
                    "family": "A0",
                    "selected": "A0",
                    "fallback": False,
                    "exact_A0": True,
                    "rank_audit_raw_to_A0": a0_rank_audit,
                }
            )
            structural_pi1 = (
                1.0 - structural_probability
            ) * float(final.structural_atom.upper_conditional_probability)
            analytic_arrays: dict[str, np.ndarray] = {
                "structural_pi0": structural_probability,
                "structural_pi1": structural_pi1,
            }
            analytic_sources: dict[str, str] = {}
            for family in APPLIED_FAMILIES[1:]:
                entry = _lock_entry(lock, family)
                gate = final.no_shrink_gate if family == "A6" else final.regularized_gate
                calibrated, diagnostic = apply_locked_family(
                    family,
                    entry,
                    a0,
                    raw=raw_seed,
                    zone=raw.zone,
                    legacy_a1=final.legacy_a1,
                    structural_zero_probability=structural_probability,
                    structural_upper_conditional_probability=(
                        final.structural_atom.upper_conditional_probability
                    ),
                    gate=gate,
                    features=features,
                    anchor=float(context.config["caa"]["anchor"]),
                )
                if bool(entry["fallback"]) and not np.array_equal(calibrated, a0):
                    raise AssertionError(f"{family} fallback is not exact A0")
                diagnostic["rank_audit_A0_to_family"] = rank_audit(a0, calibrated)
                if family == "A1":
                    diagnostic["rank_audit_raw_to_A1"] = rank_audit(
                        raw_seed, calibrated
                    )
                if family in {"A2", "A4", "A5", "A6"}:
                    if bool(entry["fallback"]):
                        members = a0.shape[1]
                        atom_pi0 = (np.sum(a0 == 0.0, axis=1) + 0.5) / (
                            members + 1.0
                        )
                        atom_pi1 = (np.sum(a0 == 1.0, axis=1) + 0.5) / (
                            members + 1.0
                        )
                        analytic_sources[family] = "exact_A0 finite-ensemble smoothed atoms"
                    else:
                        family_config = entry["config"]
                        if not bool(family_config.get("atom_enabled")):
                            raise RuntimeError(
                                f"{family} non-fallback selection unexpectedly disables atoms"
                            )
                        atom_pi0 = blend_zero_probability(
                            a0,
                            structural_probability,
                            float(family_config["atom_strength"]),
                        )
                        atom_pi1 = (
                            1.0 - atom_pi0
                        ) * float(
                            final.structural_atom.upper_conditional_probability
                        )
                        analytic_sources[family] = (
                            "locked structural/base blend before finite-M quantization"
                        )
                    if (
                        not np.isfinite(atom_pi0).all()
                        or not np.isfinite(atom_pi1).all()
                        or np.any(atom_pi0 < 0.0)
                        or np.any(atom_pi1 < 0.0)
                        or np.any(atom_pi0 + atom_pi1 > 1.0 + 1e-12)
                    ):
                        raise RuntimeError(f"{family} analytic atom probabilities are invalid")
                    analytic_arrays[f"{family}_pi0"] = atom_pi0
                    analytic_arrays[f"{family}_pi1"] = atom_pi1
                family_path = directory / f"{family}_seed{seed}_test_final.npz"
                _save_test_scenario(
                    family_path,
                    calibrated,
                    raw,
                    {
                        **common_metadata,
                        "family": family,
                        "selected": entry["selected"],
                        "fallback": bool(entry["fallback"]),
                        "config": entry["config"],
                        "selected_config": entry["config"],
                        "selected_config_sha256": outer_runner.canonical_json_sha256(
                            entry["config"]
                        ),
                        "transform_diagnostics": diagnostic,
                    },
                )
                owned_paths.extend(
                    (family_path, outer_runner.archive_audit_path(family_path))
                )
                diagnostics.append({"outer": outer, "model_seed": seed, **diagnostic})
            expected_atom_keys = {
                "structural_pi0",
                "structural_pi1",
                *{
                    f"{family}_{suffix}"
                    for family in ("A2", "A4", "A5", "A6")
                    for suffix in ("pi0", "pi1")
                },
            }
            if set(analytic_arrays) != expected_atom_keys:
                raise RuntimeError("analytic atom output catalog is incomplete")
            atom_metadata = {
                **common_metadata,
                "schema": "caa_rahc_test_analytic_atoms_v1",
                "families": ["A2", "A4", "A5", "A6"],
                "selected_configs": {
                    family: lock["selected_configs"][family]
                    for family in ("A2", "A4", "A5", "A6")
                },
                "sources": analytic_sources,
                "probability_definition": (
                    "pre-quantization locked analytic probabilities; exact-A0 "
                    "fallback uses finite-ensemble Jeffreys smoothing"
                ),
            }
            atom_path = directory / f"analytic_atoms_seed{seed}_test_final.npz"
            if atom_path.exists() or outer_runner.archive_audit_path(atom_path).exists():
                raise RuntimeError(f"analytic atom output already exists: {atom_path}")
            _write_npz_atomic(
                atom_path,
                **analytic_arrays,
                observations=raw.observations,
                day=raw.day,
                zone=raw.zone,
                metadata=np.asarray(json.dumps(atom_metadata, sort_keys=True)),
            )
            atom_audit_path = outer_runner.archive_audit_path(atom_path)
            _write_json_atomic(
                atom_audit_path,
                {
                    "schema": "caa_rahc_test_analytic_atoms_audit_v1",
                    "archive": str(atom_path.resolve()),
                    "archive_sha256": outer_runner.sha256_file(atom_path),
                    "arrays": sorted(analytic_arrays),
                    "shape": list(raw.observations.shape),
                    "selection_lock_sha256": lock_sha256,
                    "selection_analysis_sha256": lock["selection_audit_sha256"],
                    "full_code_sha256": lock["full_code_sha256"],
                    "generated_after_selection_lock": True,
                    "generated_at_utc": generated_at_utc,
                    "final_manifest_sha256": common_metadata["final_manifest_sha256"],
                    "raw_test_archive_sha256": common_metadata["raw_test_archive_sha256"],
                    "source_test_raw_sha256": common_metadata["source_test_raw_sha256"],
                    "metadata_sha256": outer_runner.canonical_json_sha256(atom_metadata),
                },
            )
            owned_paths.extend((atom_path, atom_audit_path))
        diagnostics_path = directory / "caa_test_final_diagnostics.json"
        _write_json_atomic(
            diagnostics_path,
            {
                "schema": "caa_rahc_locked_test_application_audit_v1",
                "outer": outer,
                "selection_lock_sha256": lock_sha256,
                "selection_analysis_sha256": lock["selection_audit_sha256"],
                "full_code_sha256": lock["full_code_sha256"],
                "generated_after_selection_lock": True,
                "generated_at_utc": generated_at_utc,
                "grouping": grouping_metadata,
                "records": diagnostics,
            },
        )
        owned_paths.append(diagnostics_path)
        output_manifest = {
            "schema": "caa_rahc_locked_test_outputs_v1",
            "outer": outer,
            "selection_lock_sha256": lock_sha256,
            "selection_analysis_sha256": lock["selection_audit_sha256"],
            "full_code_sha256": lock["full_code_sha256"],
            "generated_after_selection_lock": True,
            "generated_at_utc": generated_at_utc,
            "selected_configs": lock["selected_configs"],
            "final_manifest_sha256": outer_runner.sha256_file(
                _final_directory(context, outer) / "final_manifest.json"
            ),
            "families": list(APPLIED_FAMILIES),
            "files": _registry(directory, owned_paths),
        }
        _write_json_atomic(directory / "caa_test_final_manifest.json", output_manifest)
        outputs[outer] = output_manifest
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-closed CAA selection/final-fit/test-application phases"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--phase",
        required=True,
        choices=("select-and-lock", "fit-final", "apply-test"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.phase == "select-and-lock":
        result = select_and_lock(args.config, device=args.device)
    elif args.phase == "fit-final":
        result = fit_final(args.config, device=args.device)
    else:
        result = apply_test(args.config)
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "A1_PROTOCOL",
    "ANALYSIS_CODE_PATHS",
    "ExperimentContext",
    "LoadedFinalOuter",
    "RawOuterInputs",
    "SelectionRun",
    "analysis_code_fingerprint",
    "apply_locked_family",
    "apply_test",
    "candidate_grid_from_config",
    "fit_final",
    "load_context",
    "load_final_outer",
    "load_raw_inputs",
    "run_calibration_selection",
    "select_and_lock",
    "validate_caa_lock",
    "validate_registered_files",
]








