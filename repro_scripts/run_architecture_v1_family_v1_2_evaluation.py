#!/usr/bin/env python3
"""Generate and adjudicate frozen family-v1.2 validation scenarios.

The runner evaluates all 48 retained adapter checkpoints against the reused
D0-v baseline with common atom allocations and common initial Gaussian noise.
It also applies the two inference-only permutations to every chronological
checkpoint.  Selection, calibration, R-SEEN, and external-final targets are
never materialized.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_2_probe.json"
AMENDMENT = (
    ROOT
    / "repro_configs"
    / "architecture_v1_family_v1_2_evaluation_amendment.json"
)
DATA_AMENDMENT = (
    ROOT
    / "repro_configs"
    / "architecture_v1_family_v1_2_data_provenance_amendment.json"
)
TRAINING_ROOT = (
    ROOT / "outputs" / "architecture_v1_family_v1_2_probe" / "formal_training"
)
OUTPUT_ROOT = (
    ROOT / "outputs" / "architecture_v1_family_v1_2_probe" / "formal_evaluation"
)
ARCHIVE_SCHEMA = "architecture_v1_family_v1_2_validation_scenarios_v1"
RESULT_SCHEMA = "architecture_v1_family_v1_2_probe_result_v1"

from architecture_v1.atom import AtomAllocation
from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.family_v1_2_evaluation import (
    PRIMARY_METRICS,
    REGIME_LABELS,
    SAFETY_METRICS,
    adjudicate,
    block_by_nwp_interaction_omnibus,
    daily_contribution,
    daily_harm,
    max_t_simultaneous_bands,
    nwp_regime_homogeneity_omnibus,
    stage_homogeneity_omnibus,
)
from architecture_v1.family_v1_2_probe import assign_nwp_dynamicity
from architecture_v1.formal_evaluation import (
    aggregate_sampling_replicates,
    validation_per_day_metrics,
)
from architecture_v1.mechanism_evaluation import average_training_seeds
from architecture_v1.training import canonical_sha256, file_sha256
import repro_scripts.run_architecture_v1_family_v1 as family_v1
import repro_scripts.run_architecture_v1_family_v1_2_probe as probe_runner


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verified(path: Path, expected: str) -> str:
    return probe_runner._verified(path, expected)


def _verified_sidecar(path: Path) -> str:
    return probe_runner._verified_sidecar(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return probe_runner._atomic_json(path, value)


def _atomic_npz(
    path: Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **arrays,
            metadata=np.asarray(
                json.dumps(
                    probe_runner._jsonable(metadata),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _load_amendment() -> dict[str, Any]:
    _verified_sidecar(AMENDMENT)
    amendment = _read(AMENDMENT)
    if amendment.get("schema") != "architecture_v1_family_v1_2_evaluation_amendment_v1":
        raise ValueError("unexpected family-v1.2 evaluation-amendment schema")
    if amendment.get("status") != (
        "frozen_after_48_run_training_freeze_before_family_v1_2_validation_materialization"
    ):
        raise RuntimeError("family-v1.2 evaluation amendment is not frozen")
    if amendment["authorization"][
        "formal_family_v1_2_validation_scenario_generation_authorized_after_all_lineage_checks_pass"
    ] is not True:
        raise RuntimeError("formal family-v1.2 validation is not authorized")
    if amendment["authorization"]["allowed_target_roles"] != ["train", "validation"]:
        raise RuntimeError("evaluation target-role allowlist drifted")
    for key in (
        "selection_access_authorized",
        "calibration_access_authorized",
        "r_seen_target_access_authorized",
        "external_final_access_authorized",
    ):
        if amendment["authorization"][key] is not False:
            raise RuntimeError(f"evaluation amendment opened forbidden role: {key}")
    for path_key, hash_key in (
        ("base_probe_config", "base_probe_config_sha256"),
        ("data_provenance_amendment", "data_provenance_amendment_sha256"),
        ("P0_result", "P0_result_sha256"),
        ("NWP_dynamicity_registry", "NWP_dynamicity_registry_file_sha256"),
        ("formal_training_result", "formal_training_result_sha256"),
        ("formal_training_freeze", "formal_training_freeze_sha256"),
        ("formal_training_identity", "formal_training_identity_sha256"),
        ("family_v1_1_comparison", "family_v1_1_comparison_sha256"),
        ("family_v1_1_evaluation_identity", "family_v1_1_evaluation_identity_sha256"),
    ):
        _verified(ROOT / amendment["lineage"][path_key], amendment["lineage"][hash_key])
    return amendment


def _validate_training(
    amendment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    lineage = amendment["lineage"]
    result_path = ROOT / lineage["formal_training_result"]
    freeze_path = ROOT / lineage["formal_training_freeze"]
    identity_path = ROOT / lineage["formal_training_identity"]
    result = _read(result_path)
    freeze = _read(freeze_path)
    identity = _read(identity_path)
    required_status = lineage["required_training_status"]
    if result.get("status") != required_status or freeze.get("status") != required_status:
        raise RuntimeError("family-v1.2 retained-training status drifted")
    if len(result.get("runs", {})) != int(lineage["required_completed_runs"]):
        raise RuntimeError("family-v1.2 retained-training run count drifted")
    if result.get("all_training_gates_passed") is not True:
        raise RuntimeError("a family-v1.2 retained training gate failed")
    if freeze.get("validation_scenario_generation_authorized") is not True:
        raise RuntimeError("training freeze did not authorize validation generation")
    if freeze.get("training_result_sha256") != lineage["formal_training_result_sha256"]:
        raise RuntimeError("training freeze/result identity mismatch")
    for role in ("validation", "selection", "calibration", "r_seen", "final"):
        if result.get(f"{role}_target_accessed") is not False:
            raise RuntimeError(f"retained training accessed forbidden role: {role}")
    for record in identity["code_manifest"]["files"]:
        _verified(ROOT / record["path"], record["sha256"])

    completions: dict[tuple[int, int, str], dict[str, Any]] = {}
    for seed in (3, 4, 5):
        for block in range(8):
            for order_mode in ("chronological", "shuffle"):
                key = f"seed{seed}_block{block}_{order_mode}"
                record = result["runs"][key]
                completion_path = Path(record["completion"])
                completion_sha = _verified_sidecar(completion_path)
                completion = _read(completion_path)
                if completion.get("status") != "complete" or completion.get(
                    "training_gate_passed"
                ) is not True:
                    raise RuntimeError(f"ineligible adapter completion: {key}")
                if (
                    int(completion["backbone_seed"]) != seed
                    or int(completion["stage_block_index"]) != block
                    or completion["order_mode"] != order_mode
                ):
                    raise RuntimeError(f"adapter completion identity drifted: {key}")
                checkpoint = Path(record["final_adapter_checkpoint"])
                checkpoint_sha = _verified(
                    checkpoint, record["final_adapter_checkpoint_sha256"]
                )
                if (
                    checkpoint_sha != completion["final_adapter_checkpoint_sha256"]
                    or completion["optimizer_updates"] != 2040
                    or completion["EMA_updates"] != 2040
                ):
                    raise RuntimeError(f"adapter checkpoint/completion mismatch: {key}")
                completions[(seed, block, order_mode)] = {
                    **completion,
                    "completion": str(completion_path.resolve()),
                    "completion_sha256": completion_sha,
                }
    return {
        "result": result,
        "freeze": freeze,
        "identity": identity,
        "result_sha256": lineage["formal_training_result_sha256"],
        "freeze_sha256": lineage["formal_training_freeze_sha256"],
        "identity_sha256": lineage["formal_training_identity_sha256"],
    }, completions


def _load_nwp_registry(amendment: Mapping[str, Any]) -> dict[str, Any]:
    lineage = amendment["lineage"]
    path = ROOT / lineage["NWP_dynamicity_registry"]
    _verified(path, lineage["NWP_dynamicity_registry_file_sha256"])
    registry = _read(path)
    if registry.get("registry_sha256") != lineage[
        "NWP_dynamicity_registry_payload_sha256"
    ]:
        raise RuntimeError("NWP dynamicity registry payload drifted")
    return registry


def _code_manifest(amendment: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        CONFIG.resolve(),
        AMENDMENT.resolve(),
        DATA_AMENDMENT.resolve(),
        ROOT / "architecture_v1" / "family_v1_2_evaluation.py",
        ROOT / "architecture_v1" / "family_v1_2_probe.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "evaluation.py",
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_2_probe.py",
    ]
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    core = {
        "schema": "architecture_v1_family_v1_2_evaluation_code_v1",
        "files": records,
        "evaluation_amendment_sha256": file_sha256(AMENDMENT),
        "training_identity_sha256": amendment["lineage"][
            "formal_training_identity_sha256"
        ],
    }
    return {**core, "code_sha256": canonical_sha256(core)}


def _validate_fit_data_bridge(
    bundle: Any,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    data_amendment = _read(DATA_AMENDMENT)
    bridge = amendment["validation_data_bridge"]
    drift = data_amendment["audit_source_drift"]
    protocol = bundle.protocol.manifest
    manifest = bundle.manifest
    checks = {
        "roles": bundle.materialized_roles == ("train", "validation"),
        "current_protocol": protocol["protocol_sha256"]
        == bridge["current_repository_protocol_sha256"],
        "current_fit_bundle": manifest["fit_data_bundle_sha256"]
        == bridge["current_repository_fit_data_bundle_sha256"],
        "train_array": manifest["data_audit"]["split_array_sha256"]["train"]
        == bridge["train_split_array_sha256"],
        "validation_array": manifest["data_audit"]["split_array_sha256"][
            "validation"
        ]
        == bridge["validation_split_array_sha256"],
        "forbidden_arrays_absent": manifest["formal_fit_target_access"][
            "forbidden_target_arrays_materialized"
        ]
        is False,
    }
    projected_protocol = copy.deepcopy(protocol)
    matches = [
        value
        for value in projected_protocol["r_seen_sources"]
        if value["path"] == drift["path"]
    ]
    if len(matches) != 1:
        raise RuntimeError("validation bridge R-SEEN audit source is ambiguous")
    matches[0]["file_sha256"] = drift["legacy_file_sha256"]
    projected_protocol.pop("protocol_sha256")
    protocol_sha = canonical_sha256(projected_protocol)
    checks["projected_protocol"] = protocol_sha == bridge[
        "legacy_D0_v_protocol_sha256"
    ]
    projected_manifest = copy.deepcopy(manifest)
    matches = [
        value
        for value in projected_manifest["r_seen_sources"]
        if value["path"] == drift["path"]
    ]
    if len(matches) != 1:
        raise RuntimeError("fit-bundle R-SEEN audit source is ambiguous")
    matches[0]["file_sha256"] = drift["legacy_file_sha256"]
    projected_manifest["protocol_sha256"] = protocol_sha
    projected_manifest.pop("fit_data_bundle_sha256")
    projected_fit = canonical_sha256(projected_manifest)
    checks["projected_fit_bundle"] = projected_fit == bridge[
        "projected_legacy_fit_data_bundle_sha256"
    ]
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"family-v1.2 validation data bridge failed: {failed}")
    return {
        "schema": "architecture_v1_family_v1_2_validation_data_bridge_v1",
        "passed": True,
        "checks": checks,
        "current_protocol_sha256": protocol["protocol_sha256"],
        "current_fit_data_bundle_sha256": manifest["fit_data_bundle_sha256"],
        "projected_legacy_protocol_sha256": protocol_sha,
        "projected_legacy_fit_data_bundle_sha256": projected_fit,
        "train_split_array_sha256": manifest["data_audit"]["split_array_sha256"][
            "train"
        ],
        "validation_split_array_sha256": manifest["data_audit"][
            "split_array_sha256"
        ]["validation"],
        "changed_model_facing_arrays": 0,
        "materialized_target_roles": list(bundle.materialized_roles),
        "forbidden_target_arrays_materialized": False,
    }


def _validate_scenario_arrays(
    values: Mapping[str, Any],
    *,
    expected_shape: Sequence[int] = (50, 100, 10, 24),
) -> None:
    scenarios = np.asarray(values["scenarios"])
    states = np.asarray(values["states"])
    truth = np.asarray(values["observations"])
    observed = np.asarray(values["observed_mask"])
    missing = np.asarray(values["raw_missing_mask"])
    shape = tuple(int(value) for value in expected_shape)
    if scenarios.shape != shape or states.shape != shape:
        raise ValueError("scenario/state archive shape drifted")
    if truth.shape != (shape[0], shape[2], shape[3]):
        raise ValueError("scenario truth shape drifted")
    if observed.shape != truth.shape or missing.shape != truth.shape:
        raise ValueError("scenario masks do not align with truth")
    if not np.array_equal(observed, ~missing):
        raise ValueError("observed/raw-missing masks disagree")
    if not np.isfinite(scenarios).all() or scenarios.min() < 0.0 or scenarios.max() > 1.0:
        raise FloatingPointError("scenario values are non-finite or outside [0,1]")
    if not np.isin(states, (0, 1, 2)).all():
        raise ValueError("scenario atom states are invalid")
    if not np.all(scenarios[states == 0] == 0.0):
        raise ValueError("zero atoms are not exact")
    if not np.all(scenarios[states == 2] == 1.0):
        raise ValueError("one atoms are not exact")
    interior = scenarios[states == 1]
    if np.any(interior <= 0.0) or np.any(interior >= 1.0):
        raise ValueError("interior scenario reached an exact boundary")
    zero = np.asarray(values["zero_probability"])
    one = np.asarray(values["one_probability"])
    if zero.shape != truth.shape or one.shape != truth.shape:
        raise ValueError("atom probabilities do not align with truth")
    if np.any(zero < 0.0) or np.any(one < 0.0) or np.any(zero + one > 1.0 + 1e-6):
        raise ValueError("atom probability law is invalid")
    wall = np.asarray(values["wall_seconds_per_day"], dtype=np.float64)
    memory = np.asarray(values["peak_memory_bytes_per_day"])
    if wall.shape != (shape[0],) or not np.isfinite(wall).all() or np.any(wall <= 0):
        raise ValueError("per-day scenario runtime is invalid")
    if memory.shape != (shape[0],) or np.any(memory < 0):
        raise ValueError("per-day scenario peak memory is invalid")


def _load_npz(
    path: Path,
    *,
    expected_sha256: str | None,
    schema: str,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    digest = _verified(path, expected_sha256) if expected_sha256 else _verified_sidecar(path)
    if expected_sha256 is not None:
        sidecar_digest = _verified_sidecar(path)
        if sidecar_digest != digest:
            raise RuntimeError("scenario sidecar and frozen hash disagree")
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "scenarios",
            "states",
            "zero_probability",
            "one_probability",
            "observations",
            "observed_mask",
            "raw_missing_mask",
            "day",
            "zones",
            "wall_seconds_per_day",
            "peak_memory_bytes_per_day",
            "metadata",
        }
        missing = required.difference(stored.files)
        if missing:
            raise ValueError(f"scenario archive is incomplete: {sorted(missing)}")
        result = {name: stored[name].copy() for name in required if name != "metadata"}
        for optional in (
            "shuffle_bank_index_per_day",
            "context_residual_abs_max_per_day",
        ):
            if optional in stored.files:
                result[optional] = stored[optional].copy()
        metadata = json.loads(str(stored["metadata"].item()))
    if metadata.get("schema") != schema:
        raise ValueError(f"unexpected scenario archive schema: {path}")
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise ValueError(f"scenario archive metadata mismatch for {name}: {path}")
    result["metadata"] = metadata
    result["archive_sha256"] = digest
    _validate_scenario_arrays(result)
    return result


def _baseline_checkpoint_sha(config: Mapping[str, Any], seed: int) -> str:
    return str(config["frozen_D0_v_backbones"][f"seed{seed}"]["sha256"])


def _load_baselines(
    *,
    amendment: Mapping[str, Any],
    config: Mapping[str, Any],
    bundle: Any,
) -> tuple[
    dict[tuple[int, int], dict[str, np.ndarray]],
    dict[int, Mapping[str, np.ndarray]],
    dict[str, Any],
]:
    contract = amendment["reused_D0_v_baseline"]
    random_contract: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    per_seed: dict[int, Mapping[str, np.ndarray]] = {}
    records: dict[str, Any] = {}
    sampling_seeds = tuple(
        int(value) for value in amendment["scenario_contract"]["sampling_seeds"]
    )
    reference_by_sampling: dict[int, dict[str, np.ndarray]] = {}
    for seed in (3, 4, 5):
        replicates: list[Mapping[str, np.ndarray]] = []
        for sampling_seed in sampling_seeds:
            key = f"seed{seed}_sampling{sampling_seed}"
            record = contract["archives"][key]
            archive = _load_npz(
                ROOT / record["path"],
                expected_sha256=record["sha256"],
                schema=contract["archive_schema"],
                expected_metadata={
                    "family": "D0",
                    "training_seed": seed,
                    "sampling_seed": sampling_seed,
                    "checkpoint_sha256": _baseline_checkpoint_sha(config, seed),
                    "code_sha256": contract["code_sha256_recorded_in_archives"],
                },
            )
            if not all(
                (
                    np.array_equal(archive["day"], bundle.validation.day),
                    np.array_equal(archive["zones"], bundle.validation.zones),
                    np.array_equal(archive["observations"], bundle.validation.target),
                    np.array_equal(
                        archive["observed_mask"], bundle.validation.observed_mask
                    ),
                    np.array_equal(
                        archive["raw_missing_mask"],
                        bundle.validation.raw_missing_mask,
                    ),
                )
            ):
                raise RuntimeError(f"reused D0-v validation arrays drifted: {key}")
            current = {
                "states": np.asarray(archive["states"], dtype=np.int8),
                "zero_probability": np.asarray(
                    archive["zero_probability"], dtype=np.float32
                ),
                "one_probability": np.asarray(
                    archive["one_probability"], dtype=np.float32
                ),
                "archive_sha256": str(archive["archive_sha256"]),
            }
            if sampling_seed not in reference_by_sampling:
                reference_by_sampling[sampling_seed] = current
            else:
                reference = reference_by_sampling[sampling_seed]
                if not all(
                    np.array_equal(current[name], reference[name])
                    for name in ("states", "zero_probability", "one_probability")
                ):
                    raise RuntimeError("reused D0-v atom law differs across seeds")
            random_contract[(seed, sampling_seed)] = current
            replicates.append(
                validation_per_day_metrics(
                    archive["scenarios"],
                    archive["observations"],
                    archive["observed_mask"],
                    zero_probability=archive["zero_probability"],
                    one_probability=archive["one_probability"],
                )
            )
            records[f"D0_v_{key}"] = {
                "path": str((ROOT / record["path"]).resolve()),
                "sha256": record["sha256"],
                "reused": True,
            }
            del archive
        per_seed[seed] = aggregate_sampling_replicates(replicates)["per_day"]
    return random_contract, per_seed, records


def _shuffle_bank_indices(
    *,
    sampling_seed: int,
    days: int,
    amendment: Mapping[str, Any],
) -> np.ndarray:
    contract = amendment["scenario_contract"]
    position = int(contract["sampling_seed_positions"][str(int(sampling_seed))])
    eligible = np.asarray(
        contract["shuffle_trained_adapter_inference_rule"]["eligible_bank_indices"],
        dtype=np.int64,
    )
    if len(eligible) != 6:
        raise RuntimeError("shuffle evaluation bank must contain six entries")
    result = eligible[(3 * np.arange(int(days)) + position) % len(eligible)]
    return np.ascontiguousarray(result, dtype=np.int64)


def _variant_orders(
    variant: str,
    *,
    sampling_seed: int,
    days: int,
    amendment: Mapping[str, Any],
) -> np.ndarray:
    if variant == "chronological":
        return np.full(days, -1, dtype=np.int64)
    if variant == "shuffle":
        return _shuffle_bank_indices(
            sampling_seed=sampling_seed, days=days, amendment=amendment
        )
    if variant.startswith("chronological_heldout"):
        index = int(variant.removeprefix("chronological_heldout"))
        allowed = amendment["scenario_contract"]["same_checkpoint_falsification"][
            "held_out_bank_indices"
        ]
        if index not in allowed:
            raise ValueError("unknown inference-only permutation")
        return np.full(days, index, dtype=np.int64)
    raise ValueError(f"unknown evaluation variant: {variant}")


@torch.no_grad()
def _allocation_and_noise(
    probe: Any,
    condition: torch.Tensor,
    *,
    baseline_states: np.ndarray,
    baseline_zero: np.ndarray,
    baseline_one: np.ndarray,
    common_seeds: Sequence[int],
    members: int,
) -> tuple[AtomAllocation, torch.Tensor, float]:
    encoded = probe.backbone.encode_condition(condition)
    statistics = probe.backbone.atom(encoded)
    probabilities = statistics.probabilities
    zero = probabilities[..., 0].detach().cpu().numpy()
    one = probabilities[..., 2].detach().cpu().numpy()
    maximum = max(
        float(np.max(np.abs(zero - baseline_zero))),
        float(np.max(np.abs(one - baseline_one))),
    )
    if maximum > 1e-6:
        raise RuntimeError(
            f"frozen atom probability differs from D0-v baseline by {maximum:.3g}"
        )
    states = torch.from_numpy(
        np.ascontiguousarray(baseline_states, dtype=np.int64)
    ).to(condition.device)
    active = states == 1
    realized = F.one_hot(states, num_classes=3).float().mean(dim=1)
    allocation = AtomAllocation(
        states=states,
        active_mask=active,
        analytic_probabilities=probabilities,
        realized_probabilities=realized,
    )
    pieces: list[torch.Tensor] = []
    shape = (1, int(members), probe.backbone.zones, probe.backbone.hours)
    for common_seed in common_seeds:
        generator = torch.Generator(device=condition.device)
        generator.manual_seed(int(common_seed) + 2)
        pieces.append(
            torch.randn(
                shape,
                dtype=condition.dtype,
                device=condition.device,
                generator=generator,
            )
        )
    return allocation, torch.cat(pieces, dim=0), maximum


@torch.no_grad()
def _warmup(probe: Any, condition: np.ndarray, device: torch.device) -> None:
    sample_condition = torch.from_numpy(
        np.ascontiguousarray(condition[:1], dtype=np.float32)
    ).to(device)
    probe.sample_ddim(
        sample_condition,
        members=100,
        steps=31,
        eta=0.0,
        seed=990001,
        member_chunk=10,
        intervention=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _sample_archive(
    *,
    probe: Any,
    variant: str,
    trained_order_mode: str,
    seed: int,
    block: int,
    sampling_seed: int,
    checkpoint_sha256: str,
    completion_sha256: str,
    baseline: Mapping[str, Any],
    bundle: Any,
    bank: Sequence[Sequence[int]],
    amendment: Mapping[str, Any],
    code_sha256: str,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    expected_metadata = {
        "variant": variant,
        "trained_order_mode": trained_order_mode,
        "backbone_seed": int(seed),
        "stage_block_index": int(block),
        "sampling_seed": int(sampling_seed),
        "checkpoint_sha256": checkpoint_sha256,
        "completion_sha256": completion_sha256,
        "evaluation_code_sha256": code_sha256,
        "evaluation_amendment_sha256": file_sha256(AMENDMENT),
        "baseline_archive_sha256": baseline["archive_sha256"],
    }
    if output_path.exists():
        return _load_npz(
            output_path,
            expected_sha256=None,
            schema=ARCHIVE_SCHEMA,
            expected_metadata=expected_metadata,
        )

    contract = amendment["scenario_contract"]
    members = int(contract["members"])
    member_chunk = int(contract["member_chunk"])
    day_batch = int(contract["validation_day_batch_size"])
    days = len(bundle.validation)
    orders = _variant_orders(
        variant,
        sampling_seed=sampling_seed,
        days=days,
        amendment=amendment,
    )
    scenarios = np.empty((days, members, 10, 24), dtype=np.float32)
    states = np.empty((days, members, 10, 24), dtype=np.int8)
    zero = np.empty((days, 10, 24), dtype=np.float32)
    one = np.empty((days, 10, 24), dtype=np.float32)
    wall = np.empty(days, dtype=np.float64)
    memory = np.empty(days, dtype=np.int64)
    residual = np.empty(days, dtype=np.float64)
    probability_max = 0.0
    inactive_max = 0.0
    calls: set[int] = set()
    intervention_calls: set[int] = set()

    for order_index in sorted(set(int(value) for value in orders.tolist())):
        selected = np.flatnonzero(orders == order_index)
        hour_order = None if order_index < 0 else tuple(int(value) for value in bank[order_index])
        for start in range(0, len(selected), day_batch):
            indices = selected[start : start + day_batch]
            condition = torch.from_numpy(
                np.ascontiguousarray(
                    bundle.validation.condition[indices], dtype=np.float32
                )
            ).to(device)
            common_seeds = [
                int(sampling_seed + int(day_index) * 1009) for day_index in indices
            ]
            allocation, initial_noise, difference = _allocation_and_noise(
                probe,
                condition,
                baseline_states=np.asarray(baseline["states"])[indices],
                baseline_zero=np.asarray(baseline["zero_probability"])[indices],
                baseline_one=np.asarray(baseline["one_probability"])[indices],
                common_seeds=common_seeds,
                members=members,
            )
            probability_max = max(probability_max, difference)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            scenario = probe.sample_ddim(
                condition,
                members=members,
                steps=int(contract["DDIM_steps"]),
                eta=float(contract["DDIM_eta"]),
                seed=common_seeds[0],
                member_chunk=member_chunk,
                allocation=allocation,
                initial_noise=initial_noise,
                hour_order=hour_order,
                intervention=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = float(time.perf_counter() - started)
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            produced_states = scenario.states.detach().cpu().numpy().astype(np.int8)
            if not np.array_equal(produced_states, np.asarray(baseline["states"])[indices]):
                raise RuntimeError("probe sampler changed the common D0-v atom allocation")
            scenarios[indices] = scenario.values.detach().cpu().numpy().astype(np.float32)
            states[indices] = produced_states
            # The D0-v probabilities are the frozen common atom-law record.
            zero[indices] = np.asarray(baseline["zero_probability"])[indices]
            one[indices] = np.asarray(baseline["one_probability"])[indices]
            wall[indices] = elapsed / float(len(indices))
            memory[indices] = peak
            residual[indices] = float(scenario.context_residual_abs_max)
            inactive = scenario.interior_latent[~scenario.active_mask]
            if inactive.numel():
                inactive_max = max(inactive_max, float(inactive.abs().max().cpu()))
            calls.add(int(scenario.batched_forward_calls))
            intervention_calls.add(int(scenario.context_intervention_calls))

    arrays = {
        "scenarios": scenarios,
        "states": states,
        "zero_probability": zero,
        "one_probability": one,
        "observations": bundle.validation.target.astype(np.float32),
        "observed_mask": bundle.validation.observed_mask.astype(bool),
        "raw_missing_mask": bundle.validation.raw_missing_mask.astype(bool),
        "day": bundle.validation.day.astype("datetime64[D]"),
        "zones": bundle.validation.zones.astype(np.int64),
        "wall_seconds_per_day": wall,
        "peak_memory_bytes_per_day": memory,
        "shuffle_bank_index_per_day": orders,
        "context_residual_abs_max_per_day": residual,
    }
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        **expected_metadata,
        "split_role": "validation",
        "members": members,
        "member_chunk": member_chunk,
        "validation_day_batch_size": day_batch,
        "sampler": "DDIM_31_step_eta0",
        "per_path_NFE": 31,
        "batched_forward_calls_per_group": sorted(calls),
        "context_intervention_calls_per_group": sorted(intervention_calls),
        "inactive_latent_max_abs": inactive_max,
        "runtime_atom_probability_max_abs_difference_to_D0_v": probability_max,
        "archived_atom_probabilities_and_allocations_exactly_reused_from_D0_v": True,
        "common_seed_formula": "sampling_seed + zero_based_validation_day_index*1009",
        "atom_allocation_seed_offset": 1,
        "initial_noise_seed_offset": 2,
        "EMA_adapter_weights_only": True,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }
    _validate_scenario_arrays(arrays)
    digest = _atomic_npz(output_path, arrays, metadata)
    return {**arrays, "metadata": metadata, "archive_sha256": digest}


def _metrics(archive: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return validation_per_day_metrics(
        np.asarray(archive["scenarios"], dtype=np.float32),
        np.asarray(archive["observations"], dtype=np.float32),
        np.asarray(archive["observed_mask"], dtype=bool),
        zero_probability=np.asarray(archive["zero_probability"], dtype=np.float32),
        one_probability=np.asarray(archive["one_probability"], dtype=np.float32),
    )


def _mean_by_block(
    per_day: Mapping[str, Mapping[int, Mapping[int, Mapping[str, np.ndarray]]]],
    variant: str,
) -> dict[str, np.ndarray]:
    blocks: list[Mapping[str, np.ndarray]] = []
    for block in range(8):
        blocks.append(
            average_training_seeds(
                {seed: per_day[variant][seed][block] for seed in (3, 4, 5)}
            )
        )
    names = tuple(sorted(blocks[0]))
    return {
        name: np.stack([np.asarray(value[name]) for value in blocks], axis=1)
        for name in names
    }


def _descriptive_contrast(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    relative_names = {
        "normalized_joint_ES",
        "lagged_increment_variogram_score",
        "width90",
    }
    for name in (*PRIMARY_METRICS, *SAFETY_METRICS):
        contribution = daily_contribution(
            reference[name], candidate[name], relative=name in relative_names
        )
        result[name] = {
            "scale": "relative" if name in relative_names else "absolute",
            "positive_means_candidate_better": name != "coverage90",
            "estimate_by_block": contribution.mean(axis=0),
        }
    return result


def _write_block_csv(
    path: Path,
    *,
    config: Mapping[str, Any],
    primary: Mapping[str, Any],
    same: Mapping[str, Any],
    safety: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "block",
        "label",
        "ramp_benefit",
        "ramp_simultaneous_low",
        "lagged_relative_benefit",
        "lagged_simultaneous_low",
        "same_checkpoint_ramp_low",
        "same_checkpoint_lagged_low",
        "level_harm_upper",
        "joint_relative_harm_upper",
        "coverage_max_abs_band",
        "width_relative_harm_upper",
        "primary_support",
        "same_checkpoint_support",
        "safety_support",
        "full_support",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in range(8):
            writer.writerow(
                {
                    "block": block,
                    "label": config["DDIM_stage_blocks"][block]["label"],
                    "ramp_benefit": primary["estimate"][0, block],
                    "ramp_simultaneous_low": primary["simultaneous_low"][0, block],
                    "lagged_relative_benefit": primary["estimate"][1, block],
                    "lagged_simultaneous_low": primary["simultaneous_low"][1, block],
                    "same_checkpoint_ramp_low": same["simultaneous_low"][0, block],
                    "same_checkpoint_lagged_low": same["simultaneous_low"][1, block],
                    "level_harm_upper": safety["simultaneous_high"][0, block],
                    "joint_relative_harm_upper": safety["simultaneous_high"][1, block],
                    "coverage_max_abs_band": max(
                        abs(safety["simultaneous_low"][2, block]),
                        abs(safety["simultaneous_high"][2, block]),
                    ),
                    "width_relative_harm_upper": safety["simultaneous_high"][3, block],
                    "primary_support": bool(decision["primary_block_support"][block]),
                    "same_checkpoint_support": bool(
                        decision["same_checkpoint_block_support"][block]
                    ),
                    "safety_support": bool(
                        decision["distribution_safety_block_support"][block]
                    ),
                    "full_support": bool(decision["full_block_support"][block]),
                }
            )
    temporary.replace(path)


def _dry_run() -> dict[str, Any]:
    amendment = _load_amendment()
    config, _, _, stages, bank = probe_runner._load_config(CONFIG)
    training, completions = _validate_training(amendment)
    registry = _load_nwp_registry(amendment)
    baseline_records = amendment["reused_D0_v_baseline"]["archives"]
    for record in baseline_records.values():
        _verified(ROOT / record["path"], record["sha256"])
        _verified_sidecar(ROOT / record["path"])
    assignments = [
        int(value)
        for sampling_seed in amendment["scenario_contract"]["sampling_seeds"]
        for value in _shuffle_bank_indices(
            sampling_seed=int(sampling_seed), days=50, amendment=amendment
        )
    ]
    counts = {str(index): assignments.count(index) for index in range(6)}
    return {
        "schema": "architecture_v1_family_v1_2_evaluation_dry_run_v1",
        "evaluation_amendment_sha256": file_sha256(AMENDMENT),
        "training_result_sha256": training["result_sha256"],
        "training_status": training["result"]["status"],
        "verified_completions": len(completions),
        "DDIM_stage_blocks": len(stages),
        "permutation_bank_entries": len(bank),
        "NWP_registry_sha256": registry["registry_sha256"],
        "verified_reused_D0_v_archives": len(baseline_records),
        "balanced_shuffle_bank_counts": counts,
        "new_primary_archives": amendment["scenario_contract"][
            "new_primary_archives"
        ],
        "new_same_checkpoint_archives": amendment["scenario_contract"][
            "new_same_checkpoint_archives"
        ],
        "validation_days": 50,
        "members": amendment["scenario_contract"]["members"],
        "target_roles_when_executed": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_flag": "--execute",
    }


def execute(*, resume: bool) -> dict[str, Any]:
    amendment = _load_amendment()
    config, base_config, model_config, stages, bank = probe_runner._load_config(CONFIG)
    training, completions = _validate_training(amendment)
    registry = _load_nwp_registry(amendment)
    code = _code_manifest(amendment)
    result_path = OUTPUT_ROOT / "FAMILY_V1_2_PROBE_RESULT.json"
    freeze_path = OUTPUT_ROOT / "evaluation.freeze.json"
    if result_path.exists() and freeze_path.exists():
        _verified_sidecar(result_path)
        _verified_sidecar(freeze_path)
        return _read(result_path)
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()) and not resume:
        raise FileExistsError("family-v1.2 evaluation output is non-empty; use --resume")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    device, runtime = family_v1._configure_cuda()
    data_config = ROOT / config["lineage"]["data_protocol_config"]
    bundle = build_architecture_v1_fit_data(config_path=data_config)
    data_bridge = _validate_fit_data_bridge(bundle, amendment)
    nwp_score, nwp_label = assign_nwp_dynamicity(
        bundle.validation.raw_condition, registry
    )
    nwp_counts = {
        label: int(np.count_nonzero(nwp_label == label)) for label in REGIME_LABELS
    }
    if sum(nwp_counts.values()) != len(bundle.validation) or min(nwp_counts.values()) < 2:
        raise RuntimeError("validation NWP-regime assignment is invalid")

    identity = {
        "schema": "architecture_v1_family_v1_2_evaluation_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG.resolve()),
        "config_sha256": file_sha256(CONFIG),
        "evaluation_amendment": str(AMENDMENT.resolve()),
        "evaluation_amendment_sha256": file_sha256(AMENDMENT),
        "training_result_sha256": training["result_sha256"],
        "training_freeze_sha256": training["freeze_sha256"],
        "code_manifest": code,
        "runtime": runtime,
        "data_provenance_bridge": data_bridge,
        "NWP_dynamicity_registry_file_sha256": amendment["lineage"][
            "NWP_dynamicity_registry_file_sha256"
        ],
        "NWP_dynamicity_registry_payload_sha256": registry["registry_sha256"],
        "materialized_target_roles": ["train", "validation"],
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    identity_path = OUTPUT_ROOT / "identity.json"
    if identity_path.exists():
        previous = _read(identity_path)
        previous.pop("created_utc", None)
        current = dict(identity)
        current.pop("created_utc", None)
        if previous != current:
            raise RuntimeError("family-v1.2 evaluation resume identity drifted")
    else:
        _atomic_json(identity_path, identity)
    _atomic_json(
        OUTPUT_ROOT / "NWP_VALIDATION_ASSIGNMENT.json",
        {
            "schema": "architecture_v1_family_v1_2_validation_NWP_assignment_v1",
            "registry_sha256": registry["registry_sha256"],
            "target_power_used": False,
            "day": bundle.validation.day.astype(str).tolist(),
            "score": nwp_score.tolist(),
            "label": nwp_label.tolist(),
            "counts": nwp_counts,
        },
    )
    print(
        f"[family-v1.2 evaluation] GPU={runtime['device_name']}; "
        "validation=50 days; 288 new resumable archives",
        flush=True,
    )

    baseline_random, baseline_per_seed, archive_records = _load_baselines(
        amendment=amendment, config=config, bundle=bundle
    )
    baseline = average_training_seeds(baseline_per_seed)
    per_day: dict[str, dict[int, dict[int, Mapping[str, np.ndarray]]]] = {
        name: {seed: {} for seed in (3, 4, 5)}
        for name in (
            "chronological",
            "shuffle",
            "chronological_heldout6",
            "chronological_heldout7",
        )
    }
    aggregate_by_checkpoint: dict[str, Any] = {}
    runtime_atom_difference_max = 0.0
    kwargs = probe_runner._model_kwargs(base_config, model_config)
    sampling_seeds = tuple(
        int(value) for value in amendment["scenario_contract"]["sampling_seeds"]
    )

    try:
        for seed in (3, 4, 5):
            for block in range(8):
                checkpoint_payload: dict[str, Any] = {}
                for trained_order_mode, variants in (
                    (
                        "chronological",
                        (
                            "chronological",
                            "chronological_heldout6",
                            "chronological_heldout7",
                        ),
                    ),
                    ("shuffle", ("shuffle",)),
                ):
                    completion = completions[(seed, block, trained_order_mode)]
                    probe, backbone = probe_runner._new_probe(
                        seed=seed,
                        stage=stages[block],
                        config=config,
                        kwargs=kwargs,
                        device=device,
                    )
                    trainer = probe_runner._make_trainer(
                        probe, config=config, backbone_identity=backbone
                    )
                    trainer.load_checkpoint(
                        Path(completion["final_adapter_checkpoint"]),
                        expected_identity=completion["identity"],
                        restore_rng=False,
                    )
                    if trainer.optimizer_updates != 2040:
                        raise RuntimeError("evaluation loaded a non-final adapter checkpoint")
                    with trainer.ema_weights() as ema_probe:
                        ema_probe.eval()
                        _warmup(ema_probe, bundle.validation.condition, device)
                        for variant in variants:
                            replicate_metrics: list[Mapping[str, np.ndarray]] = []
                            paths: list[dict[str, Any]] = []
                            for sampling_seed in sampling_seeds:
                                filename = (
                                    f"seed{seed}_block{block}_{variant}_"
                                    f"sampling{sampling_seed}.npz"
                                )
                                path = OUTPUT_ROOT / "archives" / filename
                                archive = _sample_archive(
                                    probe=ema_probe,
                                    variant=variant,
                                    trained_order_mode=trained_order_mode,
                                    seed=seed,
                                    block=block,
                                    sampling_seed=sampling_seed,
                                    checkpoint_sha256=completion[
                                        "final_adapter_checkpoint_sha256"
                                    ],
                                    completion_sha256=completion["completion_sha256"],
                                    baseline=baseline_random[(seed, sampling_seed)],
                                    bundle=bundle,
                                    bank=bank,
                                    amendment=amendment,
                                    code_sha256=code["code_sha256"],
                                    output_path=path,
                                    device=device,
                                )
                                if not all(
                                    (
                                        np.array_equal(
                                            archive["states"],
                                            baseline_random[(seed, sampling_seed)]["states"],
                                        ),
                                        np.array_equal(
                                            archive["zero_probability"],
                                            baseline_random[(seed, sampling_seed)][
                                                "zero_probability"
                                            ],
                                        ),
                                        np.array_equal(
                                            archive["one_probability"],
                                            baseline_random[(seed, sampling_seed)][
                                                "one_probability"
                                            ],
                                        ),
                                    )
                                ):
                                    raise RuntimeError("new archive violated common atom law")
                                runtime_atom_difference_max = max(
                                    runtime_atom_difference_max,
                                    float(
                                        archive["metadata"][
                                            "runtime_atom_probability_max_abs_difference_to_D0_v"
                                        ]
                                    ),
                                )
                                replicate_metrics.append(_metrics(archive))
                                digest = _verified_sidecar(path)
                                key = filename.removesuffix(".npz")
                                archive_records[key] = {
                                    "path": str(path.resolve()),
                                    "sha256": digest,
                                    "reused": False,
                                }
                                paths.append(archive_records[key])
                                print(
                                    f"[family-v1.2 evaluation] seed{seed}/block{block}/"
                                    f"{variant}/sampling{sampling_seed}: complete",
                                    flush=True,
                                )
                                del archive
                            aggregated = aggregate_sampling_replicates(replicate_metrics)
                            per_day[variant][seed][block] = aggregated["per_day"]
                            checkpoint_payload[variant] = {
                                "aggregate": aggregated["aggregate"],
                                "per_day": {
                                    name: value.tolist()
                                    for name, value in aggregated["per_day"].items()
                                },
                                "archives": paths,
                            }
                    del trainer, probe
                    torch.cuda.empty_cache()
                checkpoint_key = f"seed{seed}_block{block}"
                aggregate_by_checkpoint[checkpoint_key] = checkpoint_payload
                _atomic_json(
                    OUTPUT_ROOT / "per_checkpoint" / f"{checkpoint_key}.json",
                    {
                        "schema": "architecture_v1_family_v1_2_checkpoint_validation_v1",
                        "backbone_seed": seed,
                        "stage_block_index": block,
                        "variants": checkpoint_payload,
                        "selection_target_accessed": False,
                        "calibration_target_accessed": False,
                    },
                )

        averaged = {
            variant: _mean_by_block(per_day, variant) for variant in per_day
        }
        baseline_block = {
            name: np.broadcast_to(value[:, None], (len(value), 8)).copy()
            for name, value in baseline.items()
        }
        heldout_mean = {
            name: 0.5
            * (
                averaged["chronological_heldout6"][name]
                + averaged["chronological_heldout7"][name]
            )
            for name in averaged["chronological"]
        }
        ordered_contributions = np.stack(
            (
                daily_contribution(
                    averaged["shuffle"][PRIMARY_METRICS[0]],
                    averaged["chronological"][PRIMARY_METRICS[0]],
                    relative=False,
                ),
                daily_contribution(
                    averaged["shuffle"][PRIMARY_METRICS[1]],
                    averaged["chronological"][PRIMARY_METRICS[1]],
                    relative=True,
                ),
            ),
            axis=1,
        )
        same_checkpoint_contributions = np.stack(
            (
                daily_contribution(
                    heldout_mean[PRIMARY_METRICS[0]],
                    averaged["chronological"][PRIMARY_METRICS[0]],
                    relative=False,
                ),
                daily_contribution(
                    heldout_mean[PRIMARY_METRICS[1]],
                    averaged["chronological"][PRIMARY_METRICS[1]],
                    relative=True,
                ),
            ),
            axis=1,
        )
        safety_contributions = np.stack(
            (
                daily_harm(
                    averaged["chronological"]["level_CRPS"],
                    baseline_block["level_CRPS"],
                    relative=False,
                ),
                daily_harm(
                    averaged["chronological"]["normalized_joint_ES"],
                    baseline_block["normalized_joint_ES"],
                    relative=True,
                ),
                daily_harm(
                    averaged["chronological"]["coverage90"],
                    baseline_block["coverage90"],
                    relative=False,
                ),
                daily_harm(
                    averaged["chronological"]["width90"],
                    baseline_block["width90"],
                    relative=True,
                ),
            ),
            axis=1,
        )
        resampling = amendment["resampling_and_multiplicity"]
        repetitions = int(resampling["repetitions"])
        confidence = float(resampling["confidence_level"])
        primary_bands = max_t_simultaneous_bands(
            ordered_contributions,
            repetitions=repetitions,
            seed=int(resampling["primary_max_T"]["seed"]),
            confidence=confidence,
        )
        same_bands = max_t_simultaneous_bands(
            same_checkpoint_contributions,
            repetitions=repetitions,
            seed=int(resampling["same_checkpoint_max_T"]["seed"]),
            confidence=confidence,
        )
        safety_bands = max_t_simultaneous_bands(
            safety_contributions,
            repetitions=repetitions,
            seed=int(resampling["safety_max_T"]["seed"]),
            confidence=confidence,
        )
        stage_test = stage_homogeneity_omnibus(
            ordered_contributions,
            repetitions=repetitions,
            seed=int(resampling["stage_omnibus"]["seed"]),
        )
        nwp_test = nwp_regime_homogeneity_omnibus(
            ordered_contributions,
            nwp_label,
            repetitions=repetitions,
            seed=int(resampling["NWP_regime_omnibus"]["seed"]),
        )
        interaction_test = block_by_nwp_interaction_omnibus(
            ordered_contributions,
            nwp_label,
            repetitions=repetitions,
            seed=int(resampling["block_by_NWP_interaction_omnibus"]["seed"]),
        )
        margins = config["practical_gates"]
        decision = adjudicate(
            primary_bands=primary_bands,
            same_checkpoint_bands=same_bands,
            safety_bands=safety_bands,
            stage_omnibus=stage_test,
            nwp_omnibus=nwp_test,
            interaction_omnibus=interaction_test,
            margins=margins,
            alpha=0.05,
        )

        atom_names = ("zero_Brier", "one_Brier", "atom_state_Brier")
        atom_metric_checks: dict[str, bool] = {}
        for variant in averaged:
            for name in atom_names:
                atom_metric_checks[f"{variant}:{name}"] = np.array_equal(
                    averaged[variant][name], baseline_block[name]
                )
        atom_audit = {
            "scenario_archives_states_and_probabilities_exactly_common": True,
            "atom_metric_day_vectors_exactly_common": atom_metric_checks,
            "all_atom_metric_checks_passed": all(atom_metric_checks.values()),
            "runtime_atom_probability_max_abs_difference_to_reused_baseline": runtime_atom_difference_max,
            "runtime_probability_tolerance": 1e-6,
            "passed": all(atom_metric_checks.values())
            and runtime_atom_difference_max <= 1e-6,
        }
        if atom_audit["passed"] is not True:
            raise RuntimeError("family-v1.2 exact common atom-law audit failed")

        per_day_path = OUTPUT_ROOT / "seed_averaged_per_day_metrics.npz"
        per_day_sha = _atomic_npz(
            per_day_path,
            {
                **{f"D0_v__{name}": value for name, value in baseline.items()},
                **{
                    f"{variant}__{name}": value
                    for variant, metrics in averaged.items()
                    for name, value in metrics.items()
                },
                "ordered_contributions": ordered_contributions,
                "same_checkpoint_contributions": same_checkpoint_contributions,
                "safety_contributions": safety_contributions,
                "NWP_dynamicity_score": nwp_score,
                "NWP_dynamicity_label": nwp_label,
                "day": bundle.validation.day.astype("datetime64[D]"),
            },
            {
                "schema": "architecture_v1_family_v1_2_seed_averaged_per_day_v1",
                "unit_of_inference": "calendar_day",
                "training_seeds_averaged_within_day": [3, 4, 5],
                "sampling_seeds_averaged_within_day": list(sampling_seeds),
                "primary_metric_order": list(PRIMARY_METRICS),
                "safety_metric_order": list(SAFETY_METRICS),
            },
        )
        _write_block_csv(
            OUTPUT_ROOT / "block_summary.csv",
            config=config,
            primary=primary_bands,
            same=same_bands,
            safety=safety_bands,
            decision=decision,
        )

        aggregate_metrics = {
            "D0_v": {name: float(np.mean(value)) for name, value in baseline.items()},
            **{
                variant: {
                    name: np.mean(value, axis=0) for name, value in metrics.items()
                }
                for variant, metrics in averaged.items()
            },
        }
        net_contrast = _descriptive_contrast(
            baseline_block, averaged["chronological"]
        )
        capacity_contrast = _descriptive_contrast(
            baseline_block, averaged["shuffle"]
        )
        heldout6_contrast = _descriptive_contrast(
            averaged["chronological_heldout6"], averaged["chronological"]
        )
        heldout7_contrast = _descriptive_contrast(
            averaged["chronological_heldout7"], averaged["chronological"]
        )
        result = {
            "schema": RESULT_SCHEMA,
            "status": decision["status"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_question": config["scientific_estimand"]["primary_question"],
            "decision": decision,
            "primary_metric_order": list(PRIMARY_METRICS),
            "safety_metric_order": list(SAFETY_METRICS),
            "ordered_mechanism_max_T": primary_bands,
            "same_checkpoint_falsification_max_T": same_bands,
            "distribution_safety_vs_D0_v_max_T": safety_bands,
            "stage_homogeneity_omnibus": stage_test,
            "NWP_regime_homogeneity_omnibus": nwp_test,
            "block_by_NWP_interaction_omnibus": interaction_test,
            "descriptive_contrasts": {
                "D0_v_minus_chronological": net_contrast,
                "D0_v_minus_shuffle": capacity_contrast,
                "heldout6_minus_same_checkpoint_chronological": heldout6_contrast,
                "heldout7_minus_same_checkpoint_chronological": heldout7_contrast,
            },
            "aggregate_metrics": aggregate_metrics,
            "NWP_dynamicity": {
                "registry_sha256": registry["registry_sha256"],
                "validation_counts": nwp_counts,
                "target_power_used_for_assignment": False,
            },
            "atom_invariance_audit": atom_audit,
            "archive_records": archive_records,
            "archive_count": len(archive_records),
            "new_archive_count": sum(
                not value["reused"] for value in archive_records.values()
            ),
            "reused_archive_count": sum(
                value["reused"] for value in archive_records.values()
            ),
            "per_day_metrics": {
                "path": str(per_day_path.resolve()),
                "sha256": per_day_sha,
            },
            "evaluation_identity": str(identity_path.resolve()),
            "evaluation_identity_sha256": _verified_sidecar(identity_path),
            "evaluation_code_sha256": code["code_sha256"],
            "evaluation_amendment_sha256": file_sha256(AMENDMENT),
            "training_result_sha256": training["result_sha256"],
            "unit_of_inference": "calendar_day",
            "bootstrap_or_permutation_repetitions": repetitions,
            "materialized_target_roles": ["train", "validation"],
            "validation_only_mechanism_discovery": True,
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
            "r_seen_target_accessed": False,
            "final_target_accessed": False,
            "external_confirmation_required": True,
            "next_action": decision["status"],
        }
        result_sha = _atomic_json(result_path, result)
        freeze = {
            "schema": "architecture_v1_family_v1_2_evaluation_freeze_v1",
            "status": decision["status"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "result": str(result_path.resolve()),
            "result_sha256": result_sha,
            "identity": str(identity_path.resolve()),
            "identity_sha256": _verified_sidecar(identity_path),
            "evaluation_amendment_sha256": file_sha256(AMENDMENT),
            "evaluation_code_sha256": code["code_sha256"],
            "training_result_sha256": training["result_sha256"],
            "archive_manifest_sha256": canonical_sha256(archive_records),
            "new_archives": result["new_archive_count"],
            "reused_D0_v_archives": result["reused_archive_count"],
            "atom_invariance_passed": True,
            "validation_evaluation_closed": True,
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "external_confirmation_required": True,
        }
        _atomic_json(freeze_path, freeze)
        return result
    except Exception as error:
        _atomic_json(
            OUTPUT_ROOT / "EVALUATION_FAILURE.json",
            {
                "schema": "architecture_v1_family_v1_2_evaluation_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "resume_boundary": "verified_complete_scenario_archive",
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
                "r_seen_target_accessed": False,
                "final_target_accessed": False,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(_dry_run(), ensure_ascii=False, indent=2))
        return 0
    result = execute(resume=args.resume)
    print(json.dumps(probe_runner._jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
