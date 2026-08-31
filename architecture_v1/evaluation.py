"""Mask-aware scoring and auditable scenario archives for architecture-v1.

Only raw-observed target cells contribute to a score.  A ramp contributes
only when both endpoints were observed in the source data.  Ensemble members
describe a finite predictive set and are never used as inferential replicates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .model import JointRectifiedFlowBase
from .training import shared_ea_state_sha256


SCENARIO_ARCHIVE_SCHEMA = "architecture_v1_scenario_archive_v1"
SCENARIO_MANIFEST_SCHEMA = "architecture_v1_scenario_manifest_v1"
ALLOWED_EVALUATION_ROLES = frozenset({"validation", "calibration", "selection"})
REQUIRED_METADATA_FIELDS = (
    "candidate_id",
    "run_id",
    "training_seed",
    "sampling_seed",
    "split_role",
    "config_sha256",
    "protocol_sha256",
    "data_bundle_sha256",
    "checkpoint_sha256",
    "shared_EA_state_sha256",
    "integrator",
    "integration_steps",
    "flow_nfe",
    "batched_flow_forward_calls",
    "member_chunk",
    "allocation_semantics",
    "estimand",
    "score_semantics",
    "common_random_numbers_group",
    "plan_id",
    "scientific_status",
    "sampling_wall_seconds",
    "peak_memory_allocated_bytes",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"metadata {name} must be a lowercase SHA256 hex digest")
    return text


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
    raise TypeError(f"metadata value is not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def validate_archive_metadata(
    metadata: Mapping[str, Any],
    *,
    members: int,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(_jsonable(metadata))
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in result]
    if missing:
        raise ValueError(f"scenario metadata is missing required fields: {missing}")
    schema = result.get("schema", SCENARIO_ARCHIVE_SCHEMA)
    if schema != SCENARIO_ARCHIVE_SCHEMA:
        raise ValueError("unexpected architecture-v1 scenario archive schema")
    result["schema"] = SCENARIO_ARCHIVE_SCHEMA
    if result["split_role"] not in ALLOWED_EVALUATION_ROLES:
        raise ValueError(
            "scenario scoring is allowed only on validation/calibration/selection; "
            "r_seen and final fail closed"
        )
    for field in (
        "config_sha256",
        "protocol_sha256",
        "data_bundle_sha256",
        "checkpoint_sha256",
        "shared_EA_state_sha256",
    ):
        result[field] = _validate_sha256(result[field], name=field)
    for field in ("training_seed", "sampling_seed", "integration_steps", "flow_nfe"):
        if not isinstance(result[field], int) or isinstance(result[field], bool):
            raise TypeError(f"metadata {field} must be an integer")
    if int(result["integration_steps"]) < 1 or int(result["flow_nfe"]) < 1:
        raise ValueError("integration steps and flow NFE must be positive")
    integrator = str(result["integrator"])
    if integrator not in {"euler", "heun"}:
        raise ValueError("integrator must be euler or heun")
    expected_nfe = (
        int(result["integration_steps"])
        if integrator == "euler"
        else 2 * int(result["integration_steps"]) - 1
    )
    if int(result["flow_nfe"]) != expected_nfe:
        raise ValueError("metadata flow_nfe does not match the registered integrator")
    for field in (
        "batched_flow_forward_calls",
        "member_chunk",
        "peak_memory_allocated_bytes",
    ):
        if not isinstance(result[field], int) or isinstance(result[field], bool):
            raise TypeError(f"metadata {field} must be an integer")
    if int(result["member_chunk"]) < 1 or int(result["member_chunk"]) > members:
        raise ValueError("metadata member_chunk is incompatible with members")
    expected_calls = expected_nfe * (
        (int(members) + int(result["member_chunk"]) - 1)
        // int(result["member_chunk"])
    )
    if int(result["batched_flow_forward_calls"]) != expected_calls:
        raise ValueError(
            "metadata batched flow forward calls do not match NFE/member chunking"
        )
    if int(result["peak_memory_allocated_bytes"]) < 0:
        raise ValueError("peak memory must be non-negative")
    wall = float(result["sampling_wall_seconds"])
    if not np.isfinite(wall) or wall < 0.0:
        raise ValueError("sampling wall time must be finite and non-negative")
    if not str(result["candidate_id"]) or not str(result["run_id"]):
        raise ValueError("candidate_id and run_id must be non-empty")
    if not str(result["allocation_semantics"]):
        raise ValueError("allocation_semantics must be explicit")
    if result["estimand"] != "finite_dependent_scenario_set":
        raise ValueError("unsupported scenario estimand")
    if result["score_semantics"] != "empirical_v_stat":
        raise ValueError("unsupported CRPS score semantics")
    for field in (
        "common_random_numbers_group",
        "plan_id",
        "scientific_status",
    ):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"metadata {field} must be a non-empty string")
    declared_members = result.get("members", members)
    if int(declared_members) != members:
        raise ValueError("metadata member count disagrees with scenario array")
    result["members"] = int(members)
    result["zones"] = 10
    result["hours"] = 24
    if expected is not None:
        for field, value in expected.items():
            if result.get(field) != value:
                raise ValueError(f"scenario metadata mismatch for {field}")
    return result


def _as_finite_float(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must be floating point")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_arrays(
    *,
    scenarios: Any,
    observations: Any,
    observed_mask: Any,
    raw_missing_mask: Any,
    states: Any,
    zero_probability: Any,
    one_probability: Any,
    day: Any,
    zones: Any,
) -> dict[str, np.ndarray]:
    sample = _as_finite_float(scenarios, name="scenarios")
    truth = _as_finite_float(observations, name="observations")
    if sample.ndim != 4 or sample.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [D,M,10,24]")
    if sample.shape[1] < 2:
        raise ValueError("scenario archives require at least two members")
    expected = (sample.shape[0], 10, 24)
    if truth.shape != expected:
        raise ValueError("observations must have shape [D,10,24]")
    if sample.min() < 0.0 or sample.max() > 1.0:
        raise ValueError("scenarios must lie in [0,1]")
    if truth.min() < 0.0 or truth.max() > 1.0:
        raise ValueError("observations must lie in [0,1]")
    observed = np.asarray(observed_mask)
    missing = np.asarray(raw_missing_mask)
    if observed.dtype != np.bool_ or missing.dtype != np.bool_:
        raise TypeError("archive observation masks must be boolean")
    if observed.shape != expected or missing.shape != expected:
        raise ValueError("archive observation masks do not align with observations")
    if not np.array_equal(observed, ~missing):
        raise ValueError("observed_mask must complement raw_missing_mask")
    if np.any(observed.reshape(len(observed), -1).sum(axis=1) == 0):
        raise ValueError("every archived day must contain observed target cells")
    state = np.asarray(states)
    if state.shape != sample.shape or not np.issubdtype(state.dtype, np.integer):
        raise ValueError("states must be an integer [D,M,10,24] array")
    if not np.isin(state, (0, 1, 2)).all():
        raise ValueError("states must contain only exact codes 0/1/2")
    if not np.all(sample[state == 0] == 0.0):
        raise ValueError("state-zero scenario coordinates must be exact zero")
    if not np.all(sample[state == 2] == 1.0):
        raise ValueError("state-one scenario coordinates must be exact one")
    interior_values = sample[state == 1]
    if np.any(interior_values <= 0.0) or np.any(interior_values >= 1.0):
        raise ValueError("interior-state scenario coordinates must lie strictly in (0,1)")
    zero = _as_finite_float(zero_probability, name="zero_probability")
    one = _as_finite_float(one_probability, name="one_probability")
    if zero.shape != expected or one.shape != expected:
        raise ValueError("analytic atom probabilities must have shape [D,10,24]")
    if np.any(zero < 0.0) or np.any(one < 0.0) or np.any(zero + one > 1.0 + 1e-6):
        raise ValueError("atom probabilities do not define a categorical law")
    dates = np.asarray(day).astype("datetime64[D]")
    if dates.shape != (sample.shape[0],) or len(np.unique(dates)) != len(dates):
        raise ValueError("day must be a unique [D] datetime vector")
    zone_ids = np.asarray(zones)
    if not np.array_equal(zone_ids, np.arange(1, 11, dtype=zone_ids.dtype)):
        raise ValueError("zones must be ordered identifiers 1..10")
    return {
        "scenarios": sample.astype(np.float32),
        "observations": truth.astype(np.float32),
        "observed_mask": observed,
        "raw_missing_mask": missing,
        "states": state.astype(np.int8),
        "zero_probability": zero.astype(np.float32),
        "one_probability": one.astype(np.float32),
        "day": dates,
        "zones": zone_ids.astype(np.int64),
    }


def _empirical_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Exact empirical V-statistic CRPS with member axis one."""

    values = np.asarray(samples, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    if values.ndim < 2 or observed.shape != (values.shape[0], *values.shape[2:]):
        raise ValueError("CRPS sample/truth shapes do not align")
    members = values.shape[1]
    if members < 2:
        raise ValueError("CRPS requires at least two members")
    first = np.mean(np.abs(values - observed[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    weights = 2.0 * np.arange(1, members + 1, dtype=np.float64) - members - 1.0
    weight_shape = (1, members, *([1] * (values.ndim - 2)))
    half_pairwise = np.sum(
        ordered * weights.reshape(weight_shape), axis=1
    ) / float(members * members)
    score = first - half_pairwise
    if np.any(score < -1e-10):
        raise RuntimeError("empirical CRPS became negative")
    return np.maximum(score, 0.0)


def _daily_masked_mean(values: np.ndarray, mask: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if score.shape != valid.shape:
        raise ValueError(f"{name} score and mask do not align")
    flat_score = score.reshape(len(score), -1)
    flat_mask = valid.reshape(len(valid), -1)
    count = flat_mask.sum(axis=1)
    if np.any(count == 0):
        raise ValueError(f"a day has no valid {name} cells")
    total = np.where(flat_mask, flat_score, 0.0).sum(axis=1)
    return total / count, count.astype(np.int64)


def masked_per_day_metrics(
    scenarios: Any,
    observations: Any,
    observed_mask: Any,
    *,
    zero_probability: Any | None = None,
    one_probability: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return daily observed-only level/ramp CRPS and atom Brier scores."""

    values = _as_finite_float(scenarios, name="scenarios")
    truth = _as_finite_float(observations, name="observations")
    mask = np.asarray(observed_mask)
    if values.ndim != 4 or values.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [D,M,10,24]")
    if truth.shape != (values.shape[0], 10, 24) or mask.shape != truth.shape:
        raise ValueError("observations/mask do not align with scenarios")
    if mask.dtype != np.bool_:
        raise TypeError("observed_mask must be boolean")
    if values.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("scenarios must lie in [0,1]")
    if truth.min() < 0.0 or truth.max() > 1.0:
        raise ValueError("observations must lie in [0,1]")

    level_cell = _empirical_crps(values, truth)
    level, level_count = _daily_masked_mean(level_cell, mask, name="level")
    ramp_mask = mask[..., :-1] & mask[..., 1:]
    ramp_cell = _empirical_crps(np.diff(values, axis=-1), np.diff(truth, axis=-1))
    ramp, ramp_count = _daily_masked_mean(ramp_cell, ramp_mask, name="ramp")

    if zero_probability is None:
        zero = np.mean(values == 0.0, axis=1)
    else:
        zero = _as_finite_float(zero_probability, name="zero_probability")
    if one_probability is None:
        one = np.mean(values == 1.0, axis=1)
    else:
        one = _as_finite_float(one_probability, name="one_probability")
    if zero.shape != truth.shape or one.shape != truth.shape:
        raise ValueError("atom probabilities do not align with observations")
    if np.any(zero < 0.0) or np.any(one < 0.0) or np.any(zero + one > 1.0 + 1e-6):
        raise ValueError("atom probabilities do not define a categorical law")
    observed_zero = truth == 0.0
    observed_one = truth == 1.0
    zero_brier, _ = _daily_masked_mean(
        (zero - observed_zero) ** 2, mask, name="zero atom"
    )
    one_brier, _ = _daily_masked_mean(
        (one - observed_one) ** 2, mask, name="one atom"
    )
    interior = 1.0 - zero - one
    observed_interior = ~(observed_zero | observed_one)
    state_brier_cell = (
        (zero - observed_zero) ** 2
        + (one - observed_one) ** 2
        + (interior - observed_interior) ** 2
    )
    state_brier, _ = _daily_masked_mean(state_brier_cell, mask, name="atom state")
    return {
        "level_CRPS": level,
        "ramp_CRPS": ramp,
        "zero_Brier": zero_brier,
        "one_Brier": one_brier,
        "atom_state_Brier": state_brier,
        "valid_level_cells": level_count,
        "valid_ramp_cells": ramp_count,
    }


# Explicit alias used by some runners and notebooks.
mask_aware_per_day_metrics = masked_per_day_metrics


@dataclass(frozen=True)
class ScenarioArchive:
    scenarios: np.ndarray
    observations: np.ndarray
    observed_mask: np.ndarray
    raw_missing_mask: np.ndarray
    states: np.ndarray
    zero_probability: np.ndarray
    one_probability: np.ndarray
    day: np.ndarray
    zones: np.ndarray
    metadata: Mapping[str, Any]
    path: Path
    sha256: str


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def write_scenario_archive(
    path: str | Path,
    *,
    scenarios: Any,
    observations: Any,
    observed_mask: Any,
    raw_missing_mask: Any,
    states: Any,
    zero_probability: Any,
    one_probability: Any,
    day: Any,
    zones: Any,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write one architecture-v1 scenario archive."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("scenario archive path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(destination)
    if not overwrite and (destination.exists() or manifest_path.exists()):
        raise FileExistsError("refusing to overwrite an existing scenario artifact")
    arrays = _validate_arrays(
        scenarios=scenarios,
        observations=observations,
        observed_mask=observed_mask,
        raw_missing_mask=raw_missing_mask,
        states=states,
        zero_probability=zero_probability,
        one_probability=one_probability,
        day=day,
        zones=zones,
    )
    meta = validate_archive_metadata(
        metadata, members=arrays["scenarios"].shape[1]
    )
    temporary = destination.with_name(destination.stem + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **arrays,
            metadata=np.asarray(
                json.dumps(
                    meta,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = file_sha256(destination)
    _atomic_json(
        manifest_path,
        {
            "schema": SCENARIO_MANIFEST_SCHEMA,
            "archive": destination.name,
            "bytes": int(destination.stat().st_size),
            "sha256": digest,
            "scenario_schema": SCENARIO_ARCHIVE_SCHEMA,
            "candidate_id": meta["candidate_id"],
            "run_id": meta["run_id"],
            "split_role": meta["split_role"],
            "day_count": int(len(arrays["day"])),
            "member_count": int(arrays["scenarios"].shape[1]),
        },
    )
    return destination


def load_scenario_archive(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    verify_manifest: bool = True,
) -> ScenarioArchive:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = file_sha256(source)
    if verify_manifest:
        manifest_path = _manifest_path(source)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"scenario manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCENARIO_MANIFEST_SCHEMA:
            raise ValueError("unexpected scenario manifest schema")
        if manifest.get("archive") != source.name or manifest.get("sha256") != digest:
            raise RuntimeError("scenario manifest/archive SHA256 mismatch")
        if int(manifest.get("bytes", -1)) != source.stat().st_size:
            raise RuntimeError("scenario manifest/archive size mismatch")
    with np.load(source, allow_pickle=False) as stored:
        required = {
            "scenarios",
            "observations",
            "observed_mask",
            "raw_missing_mask",
            "states",
            "zero_probability",
            "one_probability",
            "day",
            "zones",
            "metadata",
        }
        missing = required.difference(stored.files)
        if missing:
            raise ValueError(f"scenario archive is missing arrays: {sorted(missing)}")
        raw_metadata = stored["metadata"]
        if raw_metadata.shape != ():
            raise ValueError("scenario metadata must be a scalar JSON string")
        metadata = json.loads(str(raw_metadata.item()))
        arrays = _validate_arrays(
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
    metadata = validate_archive_metadata(
        metadata,
        members=arrays["scenarios"].shape[1],
        expected=expected_metadata,
    )
    return ScenarioArchive(
        **arrays,
        metadata=metadata,
        path=source.resolve(),
        sha256=digest,
    )


def evaluate_archive(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    archive = load_scenario_archive(
        path,
        expected_metadata=expected_metadata,
        verify_manifest=verify_manifest,
    )
    per_day = masked_per_day_metrics(
        archive.scenarios,
        archive.observations,
        archive.observed_mask,
        zero_probability=archive.zero_probability,
        one_probability=archive.one_probability,
    )
    score_names = (
        "level_CRPS",
        "ramp_CRPS",
        "zero_Brier",
        "one_Brier",
        "atom_state_Brier",
    )
    summary = {name: float(np.mean(per_day[name])) for name in score_names}
    summary.update(
        {
            "days": int(len(archive.day)),
            "members": int(archive.scenarios.shape[1]),
            "valid_level_cells": int(per_day["valid_level_cells"].sum()),
            "valid_ramp_cells": int(per_day["valid_ramp_cells"].sum()),
        }
    )
    return {
        "schema": "architecture_v1_evaluation_v1",
        "archive": str(archive.path),
        "archive_sha256": archive.sha256,
        "metadata": dict(archive.metadata),
        "day": archive.day,
        "per_day": per_day,
        "summary": summary,
    }


def _cpu_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate_and_save_scenarios(
    model: JointRectifiedFlowBase,
    condition: torch.Tensor,
    observations: Any,
    observed_mask: Any,
    raw_missing_mask: Any,
    day: Any,
    zones: Any,
    archive_path: str | Path,
    *,
    members: int,
    steps: int,
    sampling_seed: int,
    split_role: str,
    member_chunk: int,
    config_sha256: str,
    protocol_sha256: str,
    data_bundle_sha256: str,
    checkpoint_sha256: str,
    shared_EA_state_sha256: str,
    training_seed: int,
    run_id: str,
    common_random_numbers_group: str,
    sampling_plan_id: str,
    scientific_status: str,
    integrator: str = "heun",
    allocation: Any | None = None,
    initial_noise: torch.Tensor | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Sample once, atomically archive, then score under the strict contract.

    File-checkpoint identity is an explicit input.  It is intentionally not
    inferred from model tensors: a model-state hash is not a checkpoint-file
    provenance hash.  The common E/A hash *is* checked against the live model
    before sampling, which makes paired R0/T0 comparisons fail closed.
    """

    if not torch.is_tensor(condition) or condition.dtype != torch.float32:
        raise TypeError("condition must be a float32 torch tensor")
    if condition.ndim != 4 or tuple(condition.shape[1:]) != (10, 24, 20):
        raise ValueError("condition must have shape [D,10,24,20]")
    declared_shared_hash = _validate_sha256(
        shared_EA_state_sha256, name="shared_EA_state_sha256"
    )
    if shared_ea_state_sha256(model) != declared_shared_hash:
        raise ValueError("declared shared E/A hash does not match the live model")
    for name, value in (
        ("config_sha256", config_sha256),
        ("protocol_sha256", protocol_sha256),
        ("data_bundle_sha256", data_bundle_sha256),
        ("checkpoint_sha256", checkpoint_sha256),
    ):
        _validate_sha256(value, name=name)

    device = condition.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sampled = model.sample(
        condition,
        members=int(members),
        steps=int(steps),
        seed=int(sampling_seed),
        method=integrator,
        member_chunk=int(member_chunk),
        allocation=allocation,
        initial_noise=initial_noise,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    wall_seconds = float(time.perf_counter() - started)
    effective_chunk = min(int(member_chunk), int(members))
    metadata = {
        "candidate_id": model.variant,
        "run_id": str(run_id),
        "training_seed": int(training_seed),
        "sampling_seed": int(sampling_seed),
        "split_role": str(split_role),
        "config_sha256": str(config_sha256),
        "protocol_sha256": str(protocol_sha256),
        "data_bundle_sha256": str(data_bundle_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "shared_EA_state_sha256": declared_shared_hash,
        "integrator": str(integrator),
        "integration_steps": int(steps),
        "flow_nfe": int(sampled.per_path_nfe),
        "batched_flow_forward_calls": int(sampled.batched_forward_calls),
        "member_chunk": effective_chunk,
        "allocation_semantics": model.atom.allocation_semantics,
        "estimand": "finite_dependent_scenario_set",
        "score_semantics": "empirical_v_stat",
        "common_random_numbers_group": str(common_random_numbers_group),
        "plan_id": str(sampling_plan_id),
        "sampling_plan_id": str(sampling_plan_id),
        "scientific_status": str(scientific_status),
        "sampling_wall_seconds": wall_seconds,
        "peak_memory_allocated_bytes": peak_memory,
    }
    write_scenario_archive(
        archive_path,
        scenarios=_cpu_numpy(sampled.values),
        observations=_cpu_numpy(observations),
        observed_mask=_cpu_numpy(observed_mask),
        raw_missing_mask=_cpu_numpy(raw_missing_mask),
        states=_cpu_numpy(sampled.states),
        zero_probability=_cpu_numpy(sampled.atom_statistics.zero_probability),
        one_probability=_cpu_numpy(sampled.atom_statistics.one_probability),
        day=_cpu_numpy(day),
        zones=_cpu_numpy(zones),
        metadata=metadata,
        overwrite=overwrite,
    )
    return evaluate_archive(
        archive_path,
        expected_metadata={
            "candidate_id": model.variant,
            "checkpoint_sha256": str(checkpoint_sha256),
            "shared_EA_state_sha256": declared_shared_hash,
            "common_random_numbers_group": str(common_random_numbers_group),
            "plan_id": str(sampling_plan_id),
        },
    )


__all__ = [
    "ALLOWED_EVALUATION_ROLES",
    "REQUIRED_METADATA_FIELDS",
    "SCENARIO_ARCHIVE_SCHEMA",
    "SCENARIO_MANIFEST_SCHEMA",
    "ScenarioArchive",
    "evaluate_and_save_scenarios",
    "evaluate_archive",
    "file_sha256",
    "load_scenario_archive",
    "mask_aware_per_day_metrics",
    "masked_per_day_metrics",
    "validate_archive_metadata",
    "write_scenario_archive",
]
