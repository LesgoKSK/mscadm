#!/usr/bin/env python3
"""Generate family-v1 validation scenarios and adjudicate Flow versus DDPM."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
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
FORMAL_ROOT = ROOT / "outputs" / "architecture_v1_family_v1" / "formal_training"
OUTPUT_ROOT = ROOT / "outputs" / "architecture_v1_family_v1" / "formal_evaluation"
ARCHIVE_SCHEMA = "architecture_v1_family_v1_validation_scenarios_v1"

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.formal_evaluation import (
    aggregate_sampling_replicates,
    validation_per_day_metrics,
)
from architecture_v1.mechanism_evaluation import (
    average_training_seeds,
    paired_mean_bootstrap,
)
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import (
    canonical_sha256,
    file_sha256,
    shared_ea_state_sha256,
)
from repro_scripts.run_architecture_v1_family_v1 import (
    _atomic_json,
    _configure_cuda,
    _load_config,
    _load_shared_ea,
    _model_kwargs,
    _verified,
)
from repro_scripts.run_architecture_v1_family_v1_formal import (
    _make_trainer,
    _verified_sidecar,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _atomic_npz(path: Path, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **arrays,
            metadata=np.asarray(
                json.dumps(
                    metadata,
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
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_tmp.replace(sidecar)
    return digest


def _load_archive(path: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    _verified_sidecar(path)
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
            raise ValueError(f"scenario archive is incomplete: {missing}")
        result = {name: stored[name].copy() for name in required if name != "metadata"}
        metadata = json.loads(str(stored["metadata"].item()))
    if metadata.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError("unexpected family scenario archive schema")
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"scenario archive metadata mismatch: {name}")
    result["metadata"] = metadata
    _validate_scenario_arrays(result)
    return result


def _validate_scenario_arrays(values: Mapping[str, Any]) -> None:
    scenarios = np.asarray(values["scenarios"])
    states = np.asarray(values["states"])
    observations = np.asarray(values["observations"])
    observed = np.asarray(values["observed_mask"])
    missing = np.asarray(values["raw_missing_mask"])
    if scenarios.shape != (50, 100, 10, 24):
        raise ValueError("family scenarios must have shape [50,100,10,24]")
    if states.shape != scenarios.shape or observations.shape != (50, 10, 24):
        raise ValueError("family scenario state/truth shapes differ")
    if observed.shape != observations.shape or missing.shape != observations.shape:
        raise ValueError("family scenario masks differ from truth")
    if not np.array_equal(observed, ~missing):
        raise ValueError("observed and raw-missing masks are inconsistent")
    if not np.isfinite(scenarios).all() or scenarios.min() < 0.0 or scenarios.max() > 1.0:
        raise FloatingPointError("family scenarios are non-finite or out of [0,1]")
    if not np.isin(states, (0, 1, 2)).all():
        raise ValueError("family scenario states are invalid")
    if not np.all(scenarios[states == 0] == 0.0):
        raise ValueError("zero atoms are not exact")
    if not np.all(scenarios[states == 2] == 1.0):
        raise ValueError("one atoms are not exact")
    interior = scenarios[states == 1]
    if np.any(interior <= 0.0) or np.any(interior >= 1.0):
        raise ValueError("interior scenario values reached an exact boundary")
    zero = np.asarray(values["zero_probability"])
    one = np.asarray(values["one_probability"])
    if zero.shape != observations.shape or one.shape != observations.shape:
        raise ValueError("atom probability shapes differ from truth")
    if np.any(zero < 0.0) or np.any(one < 0.0) or np.any(zero + one > 1.0 + 1e-6):
        raise ValueError("atom probability law is invalid")
    wall = np.asarray(values["wall_seconds_per_day"], dtype=np.float64)
    memory = np.asarray(values["peak_memory_bytes_per_day"])
    if wall.shape != (50,) or not np.isfinite(wall).all() or np.any(wall <= 0.0):
        raise FloatingPointError("per-day sampling wall times are invalid")
    if memory.shape != (50,) or np.any(memory < 0):
        raise ValueError("per-day peak memory values are invalid")


def _code_manifest(config_path: Path) -> dict[str, Any]:
    training_identity = _read(FORMAL_ROOT / "identity.json")
    for record in training_identity["code_manifest"]["files"]:
        _verified(ROOT / record["path"], record["sha256"])
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        DATA_CONFIG,
        ROOT / "architecture_v1" / "evaluation.py",
        ROOT / "architecture_v1" / "formal_evaluation.py",
        ROOT / "architecture_v1" / "mechanism_evaluation.py",
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1_formal.py",
        ROOT / "tests" / "test_architecture_v1_family_v1_evaluation.py",
    ]
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    core = {
        "schema": "architecture_v1_family_v1_evaluation_code_v1",
        "files": records,
        "formal_training_code_sha256": training_identity["code_manifest"][
            "code_sha256"
        ],
    }
    return {**core, "code_sha256": canonical_sha256(core)}


def _validate_training() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    result_path = FORMAL_ROOT / "TRAINING_RESULT.json"
    result_sha = _verified_sidecar(result_path)
    result = _read(result_path)
    if result.get("status") != "SIX_OF_SIX_TRAINING_COMPLETE":
        raise RuntimeError("six retained family runs are not complete")
    if result.get("all_training_gates_passed") is not True:
        raise RuntimeError("a formal family training gate failed")
    if result.get("selection_target_accessed") is not False:
        raise RuntimeError("training result accessed selection")
    completions: dict[tuple[str, int], dict[str, Any]] = {}
    for family in ("F0", "D0"):
        for seed in (3, 4, 5):
            record = result["runs"][f"{family}_seed{seed}"]
            completion_path = Path(record["completion"])
            _verified_sidecar(completion_path)
            completion = _read(completion_path)
            if completion.get("status") != "complete" or completion.get(
                "training_gate_passed"
            ) is not True:
                raise RuntimeError(f"training completion is not eligible: {family}/{seed}")
            _verified(Path(record["best_checkpoint"]), record["best_checkpoint_sha256"])
            if record["best_checkpoint_sha256"] != completion["best_checkpoint_sha256"]:
                raise RuntimeError("training result/completion checkpoint mismatch")
            completions[(family, seed)] = completion
    return {"path": str(result_path.resolve()), "sha256": result_sha, "payload": result}, completions


@torch.no_grad()
def _sample_archive(
    *,
    family: str,
    seed: int,
    sampling_seed: int,
    trainer: Any,
    bundle: Any,
    config: Mapping[str, Any],
    checkpoint_sha256: str,
    code_sha256: str,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    expected_metadata = {
        "family": family,
        "training_seed": int(seed),
        "sampling_seed": int(sampling_seed),
        "checkpoint_sha256": checkpoint_sha256,
        "code_sha256": code_sha256,
    }
    if output_path.exists():
        return _load_archive(output_path, expected=expected_metadata)
    model = trainer.model
    diffusion = trainer.diffusion
    members = int(config["evaluation"]["members"])
    member_chunk = 10
    scenario_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    zero_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    wall: list[float] = []
    peak_memory: list[int] = []
    inactive_max = 0.0
    calls: list[int] = []
    with trainer.ema_weights() as ema_model:
        ema_model.eval()
        # One untimed full-shape warmup removes lazy-kernel effects from timing.
        warm_condition = torch.from_numpy(
            np.ascontiguousarray(bundle.validation.condition[:1], dtype=np.float32)
        ).to(device)
        if family == "F0":
            ema_model.sample(
                warm_condition,
                members=members,
                steps=16,
                seed=990001,
                method="heun",
                member_chunk=member_chunk,
            )
        else:
            assert diffusion is not None
            diffusion.sample_ddim(
                ema_model,
                warm_condition,
                members=members,
                steps=31,
                eta=0.0,
                seed=990001,
                member_chunk=member_chunk,
            )
        torch.cuda.synchronize(device)
        for day_index in range(len(bundle.validation)):
            condition = torch.from_numpy(
                np.ascontiguousarray(
                    bundle.validation.condition[day_index : day_index + 1],
                    dtype=np.float32,
                )
            ).to(device)
            common_seed = int(sampling_seed + day_index * 1009)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if family == "F0":
                scenario = ema_model.sample(
                    condition,
                    members=members,
                    steps=16,
                    seed=common_seed,
                    method="heun",
                    member_chunk=member_chunk,
                )
            else:
                assert diffusion is not None
                scenario = diffusion.sample_ddim(
                    ema_model,
                    condition,
                    members=members,
                    steps=31,
                    eta=0.0,
                    seed=common_seed,
                    member_chunk=member_chunk,
                )
            torch.cuda.synchronize(device)
            wall.append(float(time.perf_counter() - started))
            peak_memory.append(int(torch.cuda.max_memory_allocated(device)))
            scenario_parts.append(scenario.values.cpu().numpy().astype(np.float32))
            state_parts.append(scenario.states.cpu().numpy().astype(np.int8))
            zero_parts.append(
                scenario.atom_statistics.zero_probability.cpu().numpy().astype(np.float32)
            )
            one_parts.append(
                scenario.atom_statistics.one_probability.cpu().numpy().astype(np.float32)
            )
            calls.append(int(scenario.batched_forward_calls))
            if bool((~scenario.active_mask).any()):
                inactive_max = max(
                    inactive_max,
                    float(scenario.interior_latent[~scenario.active_mask].abs().max().cpu()),
                )
    arrays = {
        "scenarios": np.concatenate(scenario_parts, axis=0),
        "states": np.concatenate(state_parts, axis=0),
        "zero_probability": np.concatenate(zero_parts, axis=0),
        "one_probability": np.concatenate(one_parts, axis=0),
        "observations": bundle.validation.target.astype(np.float32),
        "observed_mask": bundle.validation.observed_mask.astype(bool),
        "raw_missing_mask": bundle.validation.raw_missing_mask.astype(bool),
        "day": bundle.validation.day.astype("datetime64[D]"),
        "zones": bundle.validation.zones.astype(np.int64),
        "wall_seconds_per_day": np.asarray(wall, dtype=np.float64),
        "peak_memory_bytes_per_day": np.asarray(peak_memory, dtype=np.int64),
    }
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        **expected_metadata,
        "split_role": "validation",
        "members": members,
        "member_chunk": member_chunk,
        "sampler": "Heun_16_step" if family == "F0" else "DDIM_31_step_eta0",
        "per_path_NFE": 31,
        "batched_forward_calls_per_day": sorted(set(calls)),
        "inactive_latent_max_abs": inactive_max,
        "common_seed_formula": "sampling_seed + zero_based_validation_day_index*1009",
        "EMA_weights_only": True,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
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


def _per_seed_gate(
    *,
    aggregate: Mapping[str, float],
    completion: Mapping[str, Any],
    archives: Sequence[Mapping[str, Any]],
    shared_unchanged: bool,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    finite = all(math.isfinite(float(value)) for value in aggregate.values())
    inactive = max(float(item["metadata"]["inactive_latent_max_abs"]) for item in archives)
    exact_atoms = all(
        np.all(np.asarray(item["scenarios"])[np.asarray(item["states"]) == 0] == 0.0)
        and np.all(np.asarray(item["scenarios"])[np.asarray(item["states"]) == 2] == 1.0)
        for item in archives
    )
    checks = {
        "training_gate": completion["training_gate_passed"] is True,
        "finite": finite,
        "coverage90_min": aggregate["coverage90"] >= float(rules["coverage90_min"]),
        "coverage90_max": aggregate["coverage90"] <= float(rules["coverage90_max"]),
        "width90_min": aggregate["width90"] >= float(rules["width90_min"]),
        "width90_max": aggregate["width90"] <= float(rules["width90_max"]),
        "level_CRPS": aggregate["level_CRPS"] <= float(rules["level_CRPS_max"]),
        "ramp_CRPS": aggregate["ramp_CRPS"] <= float(rules["ramp_CRPS_max"]),
        "normalized_joint_ES": aggregate["normalized_joint_ES"]
        <= float(rules["normalized_joint_ES_max"]),
        "inactive_latent": inactive <= float(rules["inactive_latent_max_abs"]),
        "atom_boundary_values_exact": exact_atoms
        is bool(rules["atom_boundary_values_exact"]),
        "shared_EA_unchanged": shared_unchanged
        is bool(rules["shared_EA_unchanged"]),
    }
    return {
        "validation_aggregate": dict(aggregate),
        "inactive_latent_max_abs": inactive,
        "atom_boundary_values_exact": exact_atoms,
        "shared_EA_unchanged": shared_unchanged,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _seed_stability(
    per_seed_aggregate: Mapping[int, Mapping[str, float]], rules: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = {
        name: np.asarray(
            [per_seed_aggregate[seed][name] for seed in sorted(per_seed_aggregate)],
            dtype=np.float64,
        )
        for name in ("level_CRPS", "ramp_CRPS", "normalized_joint_ES", "coverage90")
    }
    spread = {name: float(values.max() - values.min()) for name, values in metrics.items()}
    joint_relative = spread["normalized_joint_ES"] / float(
        np.median(metrics["normalized_joint_ES"])
    )
    checks = {
        "level_CRPS": spread["level_CRPS"]
        <= float(rules["level_CRPS_max_minus_min_max"]),
        "ramp_CRPS": spread["ramp_CRPS"]
        <= float(rules["ramp_CRPS_max_minus_min_max"]),
        "normalized_joint_ES": joint_relative
        <= float(rules["normalized_joint_ES_relative_max_minus_min_max"]),
        "coverage90": spread["coverage90"]
        <= float(rules["coverage90_max_minus_min_max"]),
    }
    return {
        "values_by_seed": {name: values.tolist() for name, values in metrics.items()},
        "spread": spread,
        "normalized_joint_ES_relative_spread": joint_relative,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _bootstrap_endpoint(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    relative: bool,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    # Positive superiority means reference score minus candidate score > 0.
    superiority = paired_mean_bootstrap(
        reference,
        candidate,
        repetitions=repetitions,
        seed=seed,
        transform="relative_difference" if relative else "difference",
    )
    # Non-inferiority uses candidate minus reference and an upper margin.
    noninferiority = paired_mean_bootstrap(
        candidate,
        reference,
        repetitions=repetitions,
        seed=seed + 100,
        transform="relative_difference" if relative else "difference",
    )
    return {"superiority": superiority, "noninferiority": noninferiority}


def _directional_family_gate(
    candidate_name: str,
    reference_name: str,
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    practical = config["family_comparison"]["practical_margins"]
    ni = config["family_comparison"]["noninferiority_margins"]
    definitions = {
        "level_CRPS": (False, float(practical["level_CRPS_absolute"]), float(ni["level_CRPS_absolute"])),
        "ramp_CRPS": (False, float(practical["ramp_CRPS_absolute"]), float(ni["ramp_CRPS_absolute"])),
        "normalized_joint_ES": (True, float(practical["normalized_joint_ES_relative"]), float(ni["normalized_joint_ES_relative"])),
        "lagged_increment_variogram_score": (True, float(practical["lagged_increment_variogram_score_relative"]), float(ni["lagged_increment_variogram_score_relative"])),
    }
    endpoint: dict[str, Any] = {}
    superior: dict[str, bool] = {}
    noninferior: dict[str, bool] = {}
    for offset, (name, (relative, practical_margin, ni_margin)) in enumerate(definitions.items()):
        result = _bootstrap_endpoint(
            np.asarray(candidate[name]),
            np.asarray(reference[name]),
            relative=relative,
            repetitions=repetitions,
            seed=seed + offset,
        )
        endpoint[name] = result
        superior[name] = (
            result["superiority"]["estimate"] >= practical_margin
            and result["superiority"]["ci_low"] > 0.0
        )
        noninferior[name] = result["noninferiority"]["ci_high"] <= ni_margin
    coverage = paired_mean_bootstrap(
        np.asarray(candidate["coverage90"]),
        np.asarray(reference["coverage90"]),
        repetitions=repetitions,
        seed=seed + 10,
    )
    width = paired_mean_bootstrap(
        np.asarray(candidate["width90"]),
        np.asarray(reference["width90"]),
        repetitions=repetitions,
        seed=seed + 11,
        transform="relative_difference",
    )
    noninferior["coverage90"] = coverage["absolute_draw_q975"] <= float(
        ni["coverage90_absolute"]
    )
    noninferior["width90"] = width["ci_high"] <= float(
        ni["width90_relative_increase"]
    )
    endpoint["coverage90"] = coverage
    endpoint["width90"] = width
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "endpoint": endpoint,
        "superiority_checks": superior,
        "noninferiority_checks": noninferior,
        "noninferior_on_all": all(noninferior.values()),
        "superior_on_at_least_one_proper_score": any(superior.values()),
        "dominance_endpoint_gate": all(noninferior.values()) and any(superior.values()),
    }


def _equivalence_gate(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    margins = config["family_comparison"]["noninferiority_margins"]
    definitions = {
        "level_CRPS": (False, float(margins["level_CRPS_absolute"])),
        "ramp_CRPS": (False, float(margins["ramp_CRPS_absolute"])),
        "normalized_joint_ES": (True, float(margins["normalized_joint_ES_relative"])),
        "lagged_increment_variogram_score": (True, float(margins["lagged_increment_variogram_score_relative"])),
        "coverage90": (False, float(margins["coverage90_absolute"])),
        "width90": (True, float(margins["width90_relative_increase"])),
    }
    records: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for offset, (name, (relative, margin)) in enumerate(definitions.items()):
        result = paired_mean_bootstrap(
            np.asarray(left[name]),
            np.asarray(right[name]),
            repetitions=repetitions,
            seed=53000 + offset,
            transform="relative_difference" if relative else "difference",
        )
        records[name] = result
        checks[name] = result["absolute_draw_q975"] <= margin
    return {"endpoint": records, "checks": checks, "passed": all(checks.values())}


def _efficiency_gate(
    wall: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, Any]:
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    f0 = np.asarray(wall["F0"], dtype=np.float64)
    d0 = np.asarray(wall["D0"], dtype=np.float64)
    d0_faster = paired_mean_bootstrap(
        f0, d0, repetitions=repetitions, seed=54001, transform="relative_difference"
    )
    f0_faster = paired_mean_bootstrap(
        d0, f0, repetitions=repetitions, seed=54002, transform="relative_difference"
    )
    margin = float(
        config["family_comparison"]["efficiency_tiebreak"][
            "sampling_wall_time_relative_improvement_min"
        ]
    )
    checks = {
        "D0": d0_faster["estimate"] >= margin and d0_faster["ci_low"] > 0.0,
        "F0": f0_faster["estimate"] >= margin and f0_faster["ci_low"] > 0.0,
    }
    return {
        "D0_relative_faster_than_F0": d0_faster,
        "F0_relative_faster_than_D0": f0_faster,
        "minimum_relative_improvement": margin,
        "checks": checks,
        "winner": "D0" if checks["D0"] else "F0" if checks["F0"] else None,
    }


def _write_summary_csv(
    path: Path, per_seed: Mapping[str, Mapping[int, Mapping[str, float]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(next(iter(next(iter(per_seed.values())).values())).keys())
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "training_seed", *names])
        writer.writeheader()
        for family in ("F0", "D0"):
            for seed in (3, 4, 5):
                writer.writerow(
                    {"family": family, "training_seed": seed, **per_seed[family][seed]}
                )
    temporary.replace(path)


def execute(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, model_config = _load_config(config_path)
    training, completions = _validate_training()
    if (FORMAL_ROOT / "training.freeze.json").exists():
        raise RuntimeError("family-v1 training/evaluation is already frozen")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()) and not resume:
        raise FileExistsError("formal evaluation output is non-empty; use --resume")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device, runtime = _configure_cuda()
    code = _code_manifest(config_path)
    bundle = build_architecture_v1_fit_data(config_path=DATA_CONFIG)
    if bundle.materialized_roles != ("train", "validation"):
        raise RuntimeError("family evaluation materialized forbidden target roles")
    if bundle.manifest["fit_data_bundle_sha256"] != config["lineage"]["fit_data_bundle_sha256"]:
        raise RuntimeError("family evaluation fit-data identity mismatch")
    kwargs = _model_kwargs(config, model_config)
    shared = _load_shared_ea(config, kwargs, torch.device("cpu"))
    identity = {
        "schema": "architecture_v1_family_v1_evaluation_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config_path),
        "training_result_sha256": training["sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "fit_data_bundle_sha256": bundle.manifest["fit_data_bundle_sha256"],
        "code_manifest": code,
        "runtime": runtime,
        "target_role": "validation",
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
            raise RuntimeError("family evaluation resume identity drifted")
    else:
        _atomic_json(identity_path, identity)
    print(
        f"[evaluation] runtime={runtime['device_name']}; validation=50 days; "
        "members=100; sampling replicates=3",
        flush=True,
    )

    per_day: dict[str, dict[int, Mapping[str, np.ndarray]]] = {"F0": {}, "D0": {}}
    per_seed_aggregate: dict[str, dict[int, Mapping[str, float]]] = {"F0": {}, "D0": {}}
    per_seed_gate: dict[str, dict[int, Mapping[str, Any]]] = {"F0": {}, "D0": {}}
    wall_by_seed: dict[str, dict[int, np.ndarray]] = {"F0": {}, "D0": {}}
    archive_records: dict[str, Any] = {}
    try:
        for seed in (3, 4, 5):
            for family in ("F0", "D0"):
                completion = completions[(family, seed)]
                torch.manual_seed(12000 + seed)
                torch.cuda.manual_seed_all(12000 + seed)
                model = T0StableSourceRectifiedFlow(**kwargs).to(device)
                model.load_shared_from(shared, freeze=True)
                trainer = _make_trainer(model, family=family, config=config)
                checkpoint = Path(completion["best_checkpoint"])
                trainer.load_checkpoint(
                    checkpoint,
                    expected_identity=completion["identity"],
                    restore_rng=False,
                )
                if shared_ea_state_sha256(model) != config["lineage"]["shared_EA_state_sha256"]:
                    raise RuntimeError(f"shared E/A changed: {family}/seed{seed}")
                replicate_metrics: list[Mapping[str, np.ndarray]] = []
                archives: list[Mapping[str, Any]] = []
                wall_replicates: list[np.ndarray] = []
                for sampling_seed in config["evaluation"]["sampling_seeds"]:
                    sampling_seed = int(sampling_seed)
                    path = (
                        OUTPUT_ROOT
                        / "archives"
                        / f"{family}_seed{seed}_sampling{sampling_seed}.npz"
                    )
                    archive = _sample_archive(
                        family=family,
                        seed=seed,
                        sampling_seed=sampling_seed,
                        trainer=trainer,
                        bundle=bundle,
                        config=config,
                        checkpoint_sha256=completion["best_checkpoint_sha256"],
                        code_sha256=code["code_sha256"],
                        output_path=path,
                        device=device,
                    )
                    metrics = _metrics(archive)
                    replicate_metrics.append(metrics)
                    archives.append(archive)
                    wall_replicates.append(
                        np.asarray(archive["wall_seconds_per_day"], dtype=np.float64)
                    )
                    archive_records[f"{family}_seed{seed}_sampling{sampling_seed}"] = {
                        "path": str(path.resolve()),
                        "sha256": _verified_sidecar(path),
                    }
                    print(
                        f"[evaluation] {family}/seed{seed}/sampling{sampling_seed}: "
                        "archive+metrics complete",
                        flush=True,
                    )
                aggregate = aggregate_sampling_replicates(replicate_metrics)
                gate = _per_seed_gate(
                    aggregate=aggregate["aggregate"],
                    completion=completion,
                    archives=archives,
                    shared_unchanged=shared_ea_state_sha256(model)
                    == config["lineage"]["shared_EA_state_sha256"],
                    rules=config["per_seed_hard_gates"],
                )
                per_day[family][seed] = aggregate["per_day"]
                per_seed_aggregate[family][seed] = aggregate["aggregate"]
                per_seed_gate[family][seed] = gate
                wall_by_seed[family][seed] = np.mean(
                    np.stack(wall_replicates, axis=0), axis=0
                )
                seed_path = OUTPUT_ROOT / "per_seed" / f"{family}_seed{seed}.json"
                _atomic_json(
                    seed_path,
                    {
                        "schema": "architecture_v1_family_v1_validation_seed_v1",
                        "family": family,
                        "training_seed": seed,
                        "aggregate": aggregate["aggregate"],
                        "per_day": {
                            name: value.tolist() for name, value in aggregate["per_day"].items()
                        },
                        "per_day_sampling_wall_seconds": wall_by_seed[family][seed].tolist(),
                        "gate": gate,
                        "selection_target_accessed": False,
                        "calibration_target_accessed": False,
                    },
                )
                print(
                    f"[evaluation] {family}/seed{seed}: hard gate "
                    f"{'PASS' if gate['passed'] else 'FAIL'}",
                    flush=True,
                )
                del trainer, model
                torch.cuda.empty_cache()

        averaged = {
            family: average_training_seeds(per_day[family]) for family in ("F0", "D0")
        }
        averaged_wall = {
            family: np.mean(
                np.stack([wall_by_seed[family][seed] for seed in (3, 4, 5)], axis=0),
                axis=0,
            )
            for family in ("F0", "D0")
        }
        candidate: dict[str, Any] = {}
        for family in ("F0", "D0"):
            stability = _seed_stability(
                per_seed_aggregate[family], config["three_seed_stability"]
            )
            all_hard = all(per_seed_gate[family][seed]["passed"] for seed in (3, 4, 5))
            candidate[family] = {
                "all_per_seed_hard_gates_passed": all_hard,
                "three_seed_stability": stability,
                "eligible": all_hard and stability["passed"],
                "aggregate_across_seed_and_sampling_replicates": {
                    name: float(np.mean(value)) for name, value in averaged[family].items()
                },
                "per_seed": {str(seed): per_seed_gate[family][seed] for seed in (3, 4, 5)},
            }
        d0_direction = _directional_family_gate(
            "D0", "F0", averaged["D0"], averaged["F0"], config=config, seed=51000
        )
        f0_direction = _directional_family_gate(
            "F0", "D0", averaged["F0"], averaged["D0"], config=config, seed=52000
        )
        equivalence = _equivalence_gate(averaged["F0"], averaged["D0"], config=config)
        efficiency = _efficiency_gate(averaged_wall, config)
        d0_dominates = candidate["D0"]["eligible"] and d0_direction[
            "dominance_endpoint_gate"
        ]
        f0_dominates = candidate["F0"]["eligible"] and f0_direction[
            "dominance_endpoint_gate"
        ]
        if not candidate["F0"]["eligible"] and not candidate["D0"]["eligible"]:
            status = "FAMILY_V1_UNRESOLVED_BOTH_INELIGIBLE"
            selected = None
        elif candidate["D0"]["eligible"] and not candidate["F0"]["eligible"]:
            selected = "D0" if d0_direction["noninferior_on_all"] else None
            status = "FAMILY_V1_SELECT_D0" if selected else "FAMILY_V1_UNRESOLVED_ONE_ELIGIBLE_NOT_NONINFERIOR"
        elif candidate["F0"]["eligible"] and not candidate["D0"]["eligible"]:
            selected = "F0" if f0_direction["noninferior_on_all"] else None
            status = "FAMILY_V1_SELECT_F0" if selected else "FAMILY_V1_UNRESOLVED_ONE_ELIGIBLE_NOT_NONINFERIOR"
        elif d0_dominates and not f0_dominates:
            status, selected = "FAMILY_V1_SELECT_D0_DOMINANCE", "D0"
        elif f0_dominates and not d0_dominates:
            status, selected = "FAMILY_V1_SELECT_F0_DOMINANCE", "F0"
        elif equivalence["passed"] and efficiency["winner"] is not None:
            selected = efficiency["winner"]
            status = f"FAMILY_V1_SELECT_{selected}_EFFICIENCY_AFTER_EQUIVALENCE"
        else:
            status, selected = "FAMILY_V1_UNRESOLVED_TRADEOFF_OR_NO_DOMINANCE", None

        per_day_path = OUTPUT_ROOT / "family_seed_averaged_per_day.npz"
        _atomic_npz(
            per_day_path,
            {
                **{
                    f"{family}__{name}": values
                    for family in ("F0", "D0")
                    for name, values in averaged[family].items()
                },
                "day": bundle.validation.day.astype("datetime64[D]"),
                "F0__sampling_wall_seconds": averaged_wall["F0"],
                "D0__sampling_wall_seconds": averaged_wall["D0"],
            },
            {
                "schema": "architecture_v1_family_v1_seed_averaged_per_day_v1",
                "unit_of_inference": "calendar_day",
                "training_seeds_averaged_within_day": [3, 4, 5],
                "sampling_seeds_averaged_within_day": config["evaluation"][
                    "sampling_seeds"
                ],
            },
        )
        _write_summary_csv(OUTPUT_ROOT / "per_seed_summary.csv", per_seed_aggregate)
        result = {
            "schema": "architecture_v1_family_v1_formal_comparison_v1",
            "status": status,
            "selected_family": selected,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate,
            "directional_gate": {"D0_vs_F0": d0_direction, "F0_vs_D0": f0_direction},
            "equivalence_gate": equivalence,
            "efficiency_gate": efficiency,
            "archive_records": archive_records,
            "unit_of_inference": "calendar_day",
            "bootstrap_repetitions": int(config["evaluation"]["bootstrap_repetitions"]),
            "selection_state": "sealed",
            "calibration_state": "sealed",
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
            "next_action": (
                "freeze_selected_family_then_create_separate_selection_access_state_machine"
                if selected is not None
                else "family_v1_unresolved_follow_frozen_decision_tree"
            ),
        }
        result_path = OUTPUT_ROOT / "FAMILY_COMPARISON.json"
        result_sha = _atomic_json(result_path, result)
        freeze = {
            "schema": "architecture_v1_family_v1_training_evaluation_freeze_v1",
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_result": str((FORMAL_ROOT / "TRAINING_RESULT.json").resolve()),
            "training_result_sha256": training["sha256"],
            "family_comparison": str(result_path.resolve()),
            "family_comparison_sha256": result_sha,
            "six_best_checkpoint_sha256": {
                f"{family}_seed{seed}": completions[(family, seed)][
                    "best_checkpoint_sha256"
                ]
                for family in ("F0", "D0")
                for seed in (3, 4, 5)
            },
            "training_closed": True,
            "selection_state": "sealed",
            "calibration_state": "sealed",
        }
        _atomic_json(FORMAL_ROOT / "training.freeze.json", freeze)
        print(f"[evaluation] COMPLETE: {status}; selected={selected}", flush=True)
        return result
    except Exception as error:
        _atomic_json(
            OUTPUT_ROOT / "EVALUATION_FAILURE.json",
            {
                "schema": "architecture_v1_family_v1_evaluation_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "resume": "rerun_with_--resume; completed archives are hash-verified",
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not args.execute:
        config, _ = _load_config(config_path)
        training, _ = _validate_training()
        code = _code_manifest(config_path)
        print(
            json.dumps(
                {
                    "schema": "architecture_v1_family_v1_evaluation_dry_run_v1",
                    "training_status": training["payload"]["status"],
                    "training_result_sha256": training["sha256"],
                    "evaluation_code_sha256": code["code_sha256"],
                    "runs": 6,
                    "sampling_replicates_per_run": 3,
                    "validation_days": 50,
                    "members": int(config["evaluation"]["members"]),
                    "matched_NFE": int(config["evaluation"]["primary_matched_NFE"]),
                    "target_role": "validation",
                    "selection_state": "sealed",
                    "calibration_state": "sealed",
                    "next_flag": "--execute",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = execute(config_path, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
