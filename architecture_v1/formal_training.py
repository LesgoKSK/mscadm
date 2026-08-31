"""Fail-closed primitives for formal architecture-v1 training.

The smoke trainer in :mod:`architecture_v1.training` deliberately implements
only short, fixed-step wiring checks.  This module adds the stateful pieces a
formal runner needs without weakening that smoke boundary:

* validation noise and flow times are fixed by ``(plan, day, replicate)``;
* caller-provided validation minibatches are canonicalised before evaluation,
  so changing their order or partition cannot change checkpoint selection;
* every training epoch uses a registered, deterministic day permutation;
* the last *complete* epoch is saved atomically with optimizer, early-stop and
  Python/NumPy/Torch RNG state, making interruption recovery auditable; and
* a training-freeze manifest can be written only after every registered seed
  has a verified, completed flow-stage artifact.

This file does not read calibration, selection, R-SEEN or final targets.  Role
access remains the responsibility of the formal runner/data facade; keeping
the trainer role-agnostic also makes that boundary straightforward to test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback
from typing import Any, Literal

import numpy as np
import torch

from .atom import INTERIOR_STATE
from .model import JointRectifiedFlowBase, ScenarioBatch
from .training import (
    ArchitectureBatch,
    canonical_sha256,
    configure_stage,
    file_sha256,
    parameter_manifest,
    shared_ea_state_sha256,
    tensor_state_sha256,
    train_step,
)


FORMAL_VALIDATION_BANK_SCHEMA = "architecture_v1_formal_validation_bank_v1"
FORMAL_RESUME_SCHEMA = "architecture_v1_formal_epoch_resume_v1"
FORMAL_COMPLETION_SCHEMA = "architecture_v1_formal_stage_completion_v1"
FORMAL_FREEZE_SCHEMA = "architecture_v1_formal_training_freeze_v1"
FORMAL_HISTORY_SCHEMA = "architecture_v1_formal_epoch_history_v1"
FORMAL_FAILURE_SCHEMA = "architecture_v1_formal_epoch_failure_v1"
FORMAL_GRADIENT_PREFLIGHT_SCHEMA = (
    "architecture_v1_formal_train_only_gradient_preflight_seed_v1"
)
FORMAL_SAMPLING_CHUNK_AUDIT_SCHEMA = (
    "architecture_v1_formal_sampling_chunk_audit_v1"
)

Stage = Literal["atom", "flow"]


def _validate_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
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
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[D]").astype(str).tolist()
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonically JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temporary.replace(sidecar)
    return digest


def _verified_file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecar = source.with_name(source.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"hash sidecar is missing: {sidecar}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if not pieces:
        raise ValueError(f"empty hash sidecar: {sidecar}")
    declared = _validate_sha256(pieces[0], name=f"{source.name} sidecar hash")
    actual = file_sha256(source)
    if declared != actual:
        raise RuntimeError(f"file SHA256 mismatch: {source}")
    return actual


def _stable_seed(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (
        2**63 - 1
    )


def deterministic_epoch_permutation(
    length: int,
    *,
    training_seed: int,
    stage: Stage,
    epoch: int,
    namespace: str = "architecture_v1_formal_day_shuffle_v1",
) -> np.ndarray:
    """Return the registered permutation for one zero-based epoch.

    The result depends on epoch identity rather than mutable RNG state.  An
    interrupted epoch can therefore be discarded and replayed exactly from the
    previous safe checkpoint.
    """

    if length < 1:
        raise ValueError("permutation length must be positive")
    if stage not in ("atom", "flow"):
        raise ValueError(f"unknown formal training stage: {stage}")
    if epoch < 0:
        raise ValueError("epoch must be zero based and non-negative")
    seed = _stable_seed(namespace, int(training_seed), stage, int(epoch))
    order = np.random.default_rng(seed).permutation(length).astype(np.int64)
    if not np.array_equal(np.sort(order), np.arange(length, dtype=np.int64)):
        raise RuntimeError("epoch permutation is not a complete bijection")
    return order


# Short alias for runner code.
epoch_permutation = deterministic_epoch_permutation


def epoch_permutation_sha256(order: Sequence[int] | np.ndarray) -> str:
    values = np.ascontiguousarray(order, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("epoch permutation must be one-dimensional")
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _concatenate_batches(
    value: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
) -> ArchitectureBatch:
    """Canonicalise a split/batch collection into day-sorted CPU storage."""

    if isinstance(value, ArchitectureBatch):
        batches = [value]
    elif hasattr(value, "condition") and hasattr(value, "day") and not isinstance(
        value, Sequence
    ):
        batches = [ArchitectureBatch.from_split(value, np.arange(len(value)))]
    else:
        batches = list(value)
    if not batches or not all(isinstance(batch, ArchitectureBatch) for batch in batches):
        raise TypeError("expected an ArchitectureBatch, split, or non-empty batch sequence")

    cpu = [batch.to("cpu") for batch in batches]
    combined = ArchitectureBatch(
        condition=torch.cat([batch.condition for batch in cpu], dim=0),
        target=torch.cat([batch.target for batch in cpu], dim=0),
        state=torch.cat([batch.state for batch in cpu], dim=0),
        observed_mask=torch.cat([batch.observed_mask for batch in cpu], dim=0),
        raw_missing_mask=torch.cat([batch.raw_missing_mask for batch in cpu], dim=0),
        day_index=torch.cat([batch.day_index for batch in cpu], dim=0),
    )
    if len(torch.unique(combined.day_index)) != len(combined.day_index):
        raise ValueError("formal day collection contains duplicate day indices")
    order = torch.argsort(combined.day_index, stable=True)
    return _take_batch(combined, order)


def _take_batch(batch: ArchitectureBatch, indices: Any) -> ArchitectureBatch:
    selected = torch.as_tensor(indices, dtype=torch.long, device=batch.condition.device)
    if selected.ndim != 1 or selected.numel() < 1:
        raise ValueError("batch selection must be a non-empty vector")
    return ArchitectureBatch(
        condition=batch.condition.index_select(0, selected),
        target=batch.target.index_select(0, selected),
        state=batch.state.index_select(0, selected),
        observed_mask=batch.observed_mask.index_select(0, selected),
        raw_missing_mask=batch.raw_missing_mask.index_select(0, selected),
        day_index=batch.day_index.index_select(0, selected),
    )


def _chunks(batch: ArchitectureBatch, batch_days: int) -> list[ArchitectureBatch]:
    if batch_days < 1:
        raise ValueError("batch_days must be positive")
    return [
        _take_batch(
            batch,
            torch.arange(
                start, min(start + batch_days, batch.condition.shape[0])
            ),
        )
        for start in range(0, batch.condition.shape[0], batch_days)
    ]


def _finite_distribution(values: Sequence[float], *, name: str) -> dict[str, float | int]:
    """Return deterministic descriptive statistics for a finite scalar trace."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty scalar sequence")
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def summarize_preclip_gradient_norms(
    update_records: Sequence[Mapping[str, Any]],
    *,
    audit_epoch_start: int,
    audit_epoch_end: int,
    gradient_clip: float,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise and optionally gate a one-based inclusive epoch window.

    ``preclip_gradient_norm`` is the total L2 norm returned by
    :func:`torch.nn.utils.clip_grad_norm_` *before* clipping.  Keeping the raw
    update trace avoids trying to infer a tail statistic from epoch means.
    Accepted gate keys are intentionally small and explicit:

    ``preclip_gradient_norm_max`` / ``preclip_gradient_norm_p99_max`` and
    ``clip_fraction_max``.  ``gradient_clip_fraction_max`` is accepted as a
    descriptive alias for the latter so a frozen registry can use either
    spelling without changing semantics.
    """

    start = int(audit_epoch_start)
    end = int(audit_epoch_end)
    clip = float(gradient_clip)
    if start < 1 or end < start:
        raise ValueError("gradient audit epochs must be one-based and inclusive")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("gradient clip must be finite and positive")
    all_records = [dict(record) for record in update_records]
    if not all_records:
        raise ValueError("gradient audit contains no update records")
    def epoch_number(record: Mapping[str, Any]) -> int:
        return int(record.get("epoch_number", record.get("epoch_1_based", -1)))

    def preclip_norm(record: Mapping[str, Any]) -> float:
        return float(
            record.get(
                "preclip_gradient_norm", record.get("preclip_total_l2_norm")
            )
        )

    selected = [
        dict(record)
        for record in all_records
        if start <= epoch_number(record) <= end
    ]
    if not selected:
        raise ValueError("gradient audit window contains no update records")
    norms = [preclip_norm(record) for record in selected]
    distribution = _finite_distribution(norms, name="preclip gradient norms")
    all_update_distribution = _finite_distribution(
        [preclip_norm(record) for record in all_records],
        name="all-update preclip gradient norms",
    )
    exceed_count = int(sum(value > clip for value in norms))
    clip_fraction = float(exceed_count / len(norms))
    thresholds = dict(gate or {})
    unknown = set(thresholds).difference(
        {
            "preclip_gradient_norm_max",
            "preclip_gradient_norm_median_max",
            "preclip_gradient_norm_p99_max",
            "preclip_gradient_norm_p99_to_median_max",
            "all_update_preclip_gradient_norm_max",
            "preclip_gradient_norm_max_over_all_680_flow_updates",
            "preclip_gradient_norm_max_over_all_flow_updates",
            "clip_fraction_max",
            "gradient_clip_fraction_max",
            "minimum_updates",
            "require_all_finite",
        }
    )
    if unknown:
        raise ValueError(f"unknown gradient gate keys: {sorted(unknown)}")
    if (
        "clip_fraction_max" in thresholds
        and "gradient_clip_fraction_max" in thresholds
        and float(thresholds["clip_fraction_max"])
        != float(thresholds["gradient_clip_fraction_max"])
    ):
        raise ValueError("gradient clip-fraction aliases disagree")
    clip_fraction_max = thresholds.get(
        "clip_fraction_max", thresholds.get("gradient_clip_fraction_max")
    )
    checks: dict[str, bool] = {
        "all_finite": True,
        "window_complete": min(epoch_number(record) for record in selected)
        == start
        and max(epoch_number(record) for record in selected) == end,
    }
    if "minimum_updates" in thresholds:
        checks["minimum_updates"] = len(selected) >= int(thresholds["minimum_updates"])
    if "preclip_gradient_norm_max" in thresholds:
        checks["preclip_gradient_norm_max"] = distribution["maximum"] <= float(
            thresholds["preclip_gradient_norm_max"]
        )
    if "preclip_gradient_norm_median_max" in thresholds:
        checks["preclip_gradient_norm_median_max"] = distribution[
            "median"
        ] <= float(thresholds["preclip_gradient_norm_median_max"])
    if "preclip_gradient_norm_p99_max" in thresholds:
        checks["preclip_gradient_norm_p99_max"] = distribution["p99"] <= float(
            thresholds["preclip_gradient_norm_p99_max"]
        )
    median = float(distribution["median"])
    p99_to_median = float(distribution["p99"]) / median if median > 0.0 else None
    if "preclip_gradient_norm_p99_to_median_max" in thresholds:
        checks["preclip_gradient_norm_p99_to_median_max"] = (
            p99_to_median is not None
            and p99_to_median
            <= float(thresholds["preclip_gradient_norm_p99_to_median_max"])
        )
    if "all_update_preclip_gradient_norm_max" in thresholds:
        checks["all_update_preclip_gradient_norm_max"] = all_update_distribution[
            "maximum"
        ] <= float(thresholds["all_update_preclip_gradient_norm_max"])
    if "preclip_gradient_norm_max_over_all_680_flow_updates" in thresholds:
        checks["all_680_flow_updates_present"] = len(all_records) == 680
        checks[
            "preclip_gradient_norm_max_over_all_680_flow_updates"
        ] = all_update_distribution["maximum"] <= float(
            thresholds["preclip_gradient_norm_max_over_all_680_flow_updates"]
        )
    if "preclip_gradient_norm_max_over_all_flow_updates" in thresholds:
        checks[
            "preclip_gradient_norm_max_over_all_flow_updates"
        ] = all_update_distribution["maximum"] <= float(
            thresholds["preclip_gradient_norm_max_over_all_flow_updates"]
        )
    if clip_fraction_max is not None:
        checks["clip_fraction"] = clip_fraction <= float(clip_fraction_max)
    return {
        "schema": "architecture_v1_preclip_gradient_summary_v1",
        "audit_epoch_start": start,
        "audit_epoch_end": end,
        "epoch_number_semantics": "one_based_inclusive",
        "quantile_method": "numpy.quantile linear",
        "gradient_clip": clip,
        "distribution": distribution,
        "p99_to_median": p99_to_median,
        "all_update_distribution": all_update_distribution,
        "updates_exceeding_clip": exceed_count,
        "clip_fraction": clip_fraction,
        "gate_thresholds": _jsonable(thresholds),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def run_discarded_train_only_stage(
    model: JointRectifiedFlowBase,
    train_data: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
    *,
    data_role: str,
    stage: Stage,
    epochs: int,
    batch_days: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    training_seed: int,
    shuffle_seed: int,
    path_seed: int,
    device: str | torch.device,
    audit_epoch_start: int = 1,
    audit_epoch_end: int | None = None,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a fixed, train-only stage whose weights are never checkpointed.

    The API deliberately has no validation argument and requires the caller to
    declare the sole materialised target role as ``train``.  It writes no
    checkpoints; the formal runner persists only the returned scalar trace
    after deleting the in-memory model.  This makes the preflight unsuitable
    for checkpoint selection by construction.
    """

    if str(data_role) != "train":
        raise RuntimeError("discarded gradient preflight accepts train targets only")
    if stage not in ("atom", "flow"):
        raise ValueError(f"unknown formal training stage: {stage}")
    epochs = int(epochs)
    batch_days = int(batch_days)
    end = epochs if audit_epoch_end is None else int(audit_epoch_end)
    if epochs < 1 or batch_days < 1:
        raise ValueError("gradient preflight epochs and batch days must be positive")
    if not 1 <= int(audit_epoch_start) <= end <= epochs:
        raise ValueError("gradient audit window must lie inside fixed epochs")
    if learning_rate <= 0.0 or weight_decay < 0.0 or gradient_clip <= 0.0:
        raise ValueError("invalid discarded-stage optimizer configuration")

    train = _concatenate_batches(train_data)
    target_device = torch.device(device)
    model.to(target_device)
    parameters = configure_stage(model, stage)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    random.seed(int(path_seed))
    np.random.seed(int(path_seed) % (2**32 - 1))
    torch.manual_seed(int(path_seed))
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(int(path_seed))

    updates_per_epoch = int(math.ceil(len(train.condition) / batch_days))
    update_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(epochs):
        permutation = deterministic_epoch_permutation(
            len(train.condition),
            training_seed=int(shuffle_seed),
            stage=stage,
            epoch=epoch,
            namespace="architecture_v1_formal_v2_2_discarded_gradient_shuffle_v1",
        )
        epoch_norms: list[float] = []
        for update_in_epoch, begin in enumerate(range(0, len(permutation), batch_days), 1):
            indices = permutation[begin : begin + batch_days]
            batch = _take_batch(train, indices).to(target_device)
            global_step = epoch * updates_per_epoch + update_in_epoch
            generator = torch.Generator(device=target_device)
            generator.manual_seed(
                _stable_seed(
                    "architecture_v1_formal_v2_2_discarded_gradient_loss_v1",
                    int(path_seed),
                    stage,
                    global_step,
                )
            )
            metrics = train_step(
                model,
                batch,
                optimizer,
                stage=stage,
                generator=generator,
                gradient_clip=float(gradient_clip),
            )
            norm = float(metrics["gradient_norm"])
            if not np.isfinite(norm):
                raise FloatingPointError("non-finite preclip gradient norm")
            epoch_norms.append(norm)
            update_records.append(
                {
                    "epoch_1_based": int(epoch + 1),
                    "update_in_epoch_1_based": int(update_in_epoch),
                    "global_update_1_based": int(global_step),
                    "preclip_total_l2_norm": norm,
                    "gradient_clip": float(gradient_clip),
                    "clip_triggered": bool(norm > float(gradient_clip)),
                }
            )
        epoch_records.append(
            {
                "epoch": int(epoch),
                "epoch_number": int(epoch + 1),
                "permutation_sha256": epoch_permutation_sha256(permutation),
                "updates": len(epoch_norms),
                "preclip_gradient_norm": _finite_distribution(
                    epoch_norms, name=f"epoch {epoch + 1} preclip gradient norms"
                ),
            }
        )
    summary = summarize_preclip_gradient_norms(
        update_records,
        audit_epoch_start=int(audit_epoch_start),
        audit_epoch_end=end,
        gradient_clip=float(gradient_clip),
        gate=gate,
    )
    return {
        "schema": FORMAL_GRADIENT_PREFLIGHT_SCHEMA,
        "status": "complete_weights_in_memory_must_be_discarded",
        "candidate_id": model.variant,
        "training_seed": int(training_seed),
        "stage": stage,
        "data_role": "train",
        "materialized_target_roles": ["train"],
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
        "epochs": epochs,
        "batch_days": batch_days,
        "updates_per_epoch": updates_per_epoch,
        "updates": len(update_records),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": float(gradient_clip),
        "shuffle_seed": int(shuffle_seed),
        "path_seed": int(path_seed),
        "train_days_sha256": tensor_state_sha256({"day_index": train.day_index}),
        "epoch_records": epoch_records,
        "update_records": update_records,
        "gradient_audit": summary,
        "checkpoint_selection": "none",
        "checkpoint_written": False,
        "weights_retained": False,
    }


@dataclass(frozen=True)
class CommonRandomChunkSamples:
    """Two chunkings and a replay generated from one allocation/noise pair."""

    primary: ScenarioBatch
    alternate: ScenarioBatch
    replay: ScenarioBatch
    audit: Mapping[str, Any]


def _absolute_difference_summary(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    if first.shape != second.shape:
        raise ValueError("sampling-audit tensors do not align")
    difference = (first.detach().cpu().to(torch.float64) - second.detach().cpu().to(torch.float64)).abs()
    if mask is not None:
        if mask.shape != first.shape or mask.dtype != torch.bool:
            raise ValueError("sampling-audit mask does not align")
        difference = difference[mask.detach().cpu()]
    values = difference.reshape(-1).numpy()
    return _finite_distribution(values, name="sampling absolute differences")


@torch.no_grad()
def common_random_chunk_samples(
    model: JointRectifiedFlowBase,
    condition: torch.Tensor,
    *,
    members: int,
    steps: int,
    seed: int,
    method: Literal["euler", "heun"],
    primary_member_chunk: int,
    alternate_member_chunk: int,
    value_allclose_atol: float = 5e-7,
    value_allclose_rtol: float = 1e-6,
) -> CommonRandomChunkSamples:
    """Audit two inference chunk sizes under identical stochastic inputs.

    Allocation and initial Gaussian noise are created once, hashed, and passed
    explicitly to both chunkings and to a primary replay.  Thus any continuous
    discrepancy is attributable to numerical batching, not RNG consumption.
    """

    if condition.ndim != 4 or tuple(condition.shape[1:3]) != (model.zones, model.hours):
        raise ValueError("sampling-audit condition has an invalid joint-day shape")
    if members < 1 or steps < 1:
        raise ValueError("sampling-audit members and steps must be positive")
    chunks = (int(primary_member_chunk), int(alternate_member_chunk))
    if any(value < 1 for value in chunks) or chunks[0] == chunks[1]:
        raise ValueError("sampling-audit chunk sizes must be distinct and positive")
    if value_allclose_atol < 0.0 or value_allclose_rtol < 0.0:
        raise ValueError("sampling-audit allclose tolerances must be non-negative")
    was_training = model.training
    model.eval()
    try:
        encoded = model.encode_condition(condition)
        statistics = model.atom(encoded)
        allocation = model.atom.allocate(
            statistics, members=int(members), seed=int(seed) + 1
        )
        generator = torch.Generator(device=condition.device)
        generator.manual_seed(int(seed) + 2)
        initial_noise = torch.randn(
            (len(condition), int(members), model.zones, model.hours),
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        kwargs = {
            "members": int(members),
            "steps": int(steps),
            "seed": int(seed),
            "method": method,
            "allocation": allocation,
            "initial_noise": initial_noise,
        }
        primary = model.sample(
            condition, member_chunk=chunks[0], **kwargs
        )
        alternate = model.sample(
            condition, member_chunk=chunks[1], **kwargs
        )
        replay = model.sample(
            condition, member_chunk=chunks[0], **kwargs
        )
    finally:
        model.train(was_training)

    state_exact = torch.equal(primary.states, alternate.states)
    analytic_probability_exact = torch.equal(
        primary.atom_allocation.analytic_probabilities,
        alternate.atom_allocation.analytic_probabilities,
    )
    realized_probability_exact = torch.equal(
        primary.atom_allocation.realized_probabilities,
        alternate.atom_allocation.realized_probabilities,
    )
    replay_exact = (
        torch.equal(primary.states, replay.states)
        and torch.equal(primary.values, replay.values)
        and torch.equal(primary.interior_latent, replay.interior_latent)
        and torch.equal(
            primary.atom_statistics.probabilities,
            replay.atom_statistics.probabilities,
        )
    )
    boundary = ~primary.active_mask
    boundary_values_cross_chunk_exact = torch.equal(
        primary.values[boundary], alternate.values[boundary]
    )
    primary_boundary_reconstruction_exact = bool(
        torch.all(primary.values[primary.states == 0] == 0.0)
        and torch.all(primary.values[primary.states == 2] == 1.0)
    )
    alternate_boundary_reconstruction_exact = bool(
        torch.all(alternate.values[alternate.states == 0] == 0.0)
        and torch.all(alternate.values[alternate.states == 2] == 1.0)
    )
    audit = {
        "schema": FORMAL_SAMPLING_CHUNK_AUDIT_SCHEMA,
        "common_random_numbers": True,
        "members": int(members),
        "steps": int(steps),
        "seed": int(seed),
        "method": method,
        "primary_member_chunk": chunks[0],
        "alternate_member_chunk": chunks[1],
        "allocation_sha256": tensor_state_sha256(
            {
                "states": allocation.states,
                "active_mask": allocation.active_mask,
                "analytic_probabilities": allocation.analytic_probabilities,
                "realized_probabilities": allocation.realized_probabilities,
            }
        ),
        "initial_noise_sha256": tensor_state_sha256(
            {"initial_noise": initial_noise}
        ),
        "state_exact": bool(state_exact),
        "active_mask_exact": bool(
            torch.equal(primary.active_mask, alternate.active_mask)
        ),
        "analytic_probability_exact": bool(analytic_probability_exact),
        "realized_probability_exact": bool(realized_probability_exact),
        "atom_probability_exact": bool(
            torch.equal(
                primary.atom_statistics.probabilities,
                alternate.atom_statistics.probabilities,
            )
        ),
        "zero_probability_exact": bool(
            torch.equal(
                primary.atom_statistics.zero_probability,
                alternate.atom_statistics.zero_probability,
            )
        ),
        "one_probability_exact": bool(
            torch.equal(
                primary.atom_statistics.one_probability,
                alternate.atom_statistics.one_probability,
            )
        ),
        "same_seed_primary_replay_bitwise_exact": bool(replay_exact),
        "atom_latent_strictly_zero": bool(
            torch.all(primary.interior_latent[~primary.active_mask] == 0.0)
            and torch.all(alternate.interior_latent[~alternate.active_mask] == 0.0)
        ),
        "atom_boundary_values_cross_chunk_exact": bool(
            boundary_values_cross_chunk_exact
        ),
        "primary_atom_boundary_reconstruction_exact": (
            primary_boundary_reconstruction_exact
        ),
        "alternate_atom_boundary_reconstruction_exact": (
            alternate_boundary_reconstruction_exact
        ),
        "atom_boundary_values_exact": bool(
            boundary_values_cross_chunk_exact
            and primary_boundary_reconstruction_exact
            and alternate_boundary_reconstruction_exact
        ),
        "value_bitwise_exact": bool(torch.equal(primary.values, alternate.values)),
        "value_allclose_atol": float(value_allclose_atol),
        "value_allclose_rtol": float(value_allclose_rtol),
        "value_allclose": bool(
            torch.allclose(
                primary.values,
                alternate.values,
                atol=float(value_allclose_atol),
                rtol=float(value_allclose_rtol),
            )
        ),
        "value_absolute_difference": _absolute_difference_summary(
            primary.values, alternate.values
        ),
        "interior_value_absolute_difference": _absolute_difference_summary(
            primary.values,
            alternate.values,
            mask=primary.active_mask,
        ),
    }
    return CommonRandomChunkSamples(
        primary=primary,
        alternate=alternate,
        replay=replay,
        audit=audit,
    )


def evaluate_sampling_chunk_gate(
    audit: Mapping[str, Any],
    *,
    score_deltas: Mapping[str, float],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the v2.2 exact-discrete/tolerant-continuous sampling gate."""

    thresholds = dict(gate)
    allowed = {
        "value_abs_max",
        "value_abs_mean",
        "value_abs_p99",
        "value_allclose_atol",
        "value_allclose_rtol",
        "score_abs_delta_max",
    }
    unknown = set(thresholds).difference(allowed)
    if unknown:
        raise ValueError(f"unknown sampling chunk gate keys: {sorted(unknown)}")
    required = {
        "value_abs_max",
        "value_allclose_atol",
        "value_allclose_rtol",
        "score_abs_delta_max",
    }
    missing = required.difference(thresholds)
    if missing:
        raise ValueError(f"sampling chunk gate is missing: {sorted(missing)}")
    score_thresholds = thresholds["score_abs_delta_max"]
    if isinstance(score_thresholds, Mapping):
        absent = set(score_deltas).difference(score_thresholds)
        if absent:
            raise ValueError(f"score-delta thresholds are missing: {sorted(absent)}")
    elif not isinstance(score_thresholds, (int, float, np.generic)):
        raise TypeError("score_abs_delta_max must be a scalar or metric mapping")
    finite_scores = {
        str(name): float(value) for name, value in score_deltas.items()
    }
    if not finite_scores or not all(np.isfinite(value) and value >= 0.0 for value in finite_scores.values()):
        raise ValueError("score deltas must be non-empty finite absolute values")
    value = audit.get("value_absolute_difference", {})
    exact_checks = {
        "state_exact": audit.get("state_exact") is True,
        "active_mask_exact": audit.get("active_mask_exact") is True,
        "analytic_probability_exact": audit.get("analytic_probability_exact") is True,
        "realized_probability_exact": audit.get("realized_probability_exact") is True,
        "atom_probability_exact": audit.get("atom_probability_exact") is True,
        "zero_probability_exact": audit.get("zero_probability_exact") is True,
        "one_probability_exact": audit.get("one_probability_exact") is True,
        "same_seed_primary_replay_bitwise_exact": audit.get(
            "same_seed_primary_replay_bitwise_exact"
        )
        is True,
        "atom_latent_strictly_zero": audit.get("atom_latent_strictly_zero") is True,
        "atom_boundary_values_exact": audit.get("atom_boundary_values_exact") is True,
    }
    tolerance_checks = {
        "value_abs_max": float(value["maximum"])
        <= float(thresholds["value_abs_max"]),
        "value_allclose": audit.get("value_allclose") is True
        and float(audit.get("value_allclose_atol", -1.0))
        == float(thresholds["value_allclose_atol"])
        and float(audit.get("value_allclose_rtol", -1.0))
        == float(thresholds["value_allclose_rtol"]),
    }
    if "value_abs_mean" in thresholds:
        tolerance_checks["value_abs_mean"] = float(value["mean"]) <= float(
            thresholds["value_abs_mean"]
        )
    if "value_abs_p99" in thresholds:
        tolerance_checks["value_abs_p99"] = float(value["p99"]) <= float(
            thresholds["value_abs_p99"]
        )
    score_checks = {
        name: delta
        <= float(
            score_thresholds[name]
            if isinstance(score_thresholds, Mapping)
            else score_thresholds
        )
        for name, delta in finite_scores.items()
    }
    checks = {
        **exact_checks,
        **tolerance_checks,
        "score_deltas": bool(all(score_checks.values())),
    }
    return {
        "schema": "architecture_v1_formal_sampling_chunk_gate_v1",
        "thresholds": _jsonable(thresholds),
        "score_absolute_deltas": finite_scores,
        "score_checks": score_checks,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


@dataclass(frozen=True)
class FormalValidationBank:
    """Fixed flow noise/time draws for the complete validation calendar.

    ``noise`` has shape ``[K,D,10,24]`` and ``flow_time`` has shape ``[K,D]``.
    Both live on CPU and are moved only for a canonical validation chunk.  The
    model is always evaluated on day-sorted chunks of ``evaluation_batch_days``;
    the caller may consequently pass the same days in any order/partition.
    """

    day_index: torch.Tensor
    noise: torch.Tensor
    flow_time: torch.Tensor
    plan_id: str
    base_seed: int
    evaluation_batch_days: int = 16
    noise_seeds: tuple[int, ...] = ()
    time_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.day_index.device.type != "cpu" or self.day_index.dtype != torch.long:
            raise TypeError("validation day_index must be a CPU torch.long tensor")
        if self.day_index.ndim != 1 or self.day_index.numel() < 1:
            raise ValueError("validation bank requires at least one day")
        if len(torch.unique(self.day_index)) != len(self.day_index):
            raise ValueError("validation bank day indices must be unique")
        if not torch.equal(self.day_index, torch.sort(self.day_index).values):
            raise ValueError("validation bank day indices must be sorted")
        expected_noise = (self.replicates, len(self.day_index), 10, 24)
        if self.noise.device.type != "cpu" or self.noise.dtype != torch.float32:
            raise TypeError("validation noise must be a CPU float32 tensor")
        if tuple(self.noise.shape) != expected_noise:
            raise ValueError(f"validation noise must have shape {expected_noise}")
        if self.flow_time.device.type != "cpu" or self.flow_time.dtype != torch.float32:
            raise TypeError("validation flow times must be a CPU float32 tensor")
        if tuple(self.flow_time.shape) != (self.replicates, len(self.day_index)):
            raise ValueError("validation flow times do not align with noise")
        if not bool(torch.isfinite(self.noise).all()):
            raise ValueError("validation noise contains non-finite values")
        if not bool(torch.isfinite(self.flow_time).all()):
            raise ValueError("validation flow times contain non-finite values")
        if bool(((self.flow_time < 0.0) | (self.flow_time >= 1.0)).any()):
            raise ValueError("validation flow times must lie in [0,1)")
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("validation plan_id must be non-empty")
        if self.evaluation_batch_days < 1:
            raise ValueError("evaluation_batch_days must be positive")
        if bool(self.noise_seeds) != bool(self.time_seeds):
            raise ValueError("explicit validation noise/time seeds must be paired")
        if self.noise_seeds:
            if len(self.noise_seeds) != self.replicates or len(self.time_seeds) != self.replicates:
                raise ValueError("explicit validation seed counts must equal replications")
            if len(set(self.noise_seeds)) != self.replicates:
                raise ValueError("validation noise seeds must be unique")
            if len(set(self.time_seeds)) != self.replicates:
                raise ValueError("validation time seeds must be unique")

    @property
    def replicates(self) -> int:
        return int(self.noise.shape[0])

    @classmethod
    def from_day_indices(
        cls,
        day_indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        replicates: int = 4,
        base_seed: int,
        plan_id: str,
        evaluation_batch_days: int = 16,
    ) -> "FormalValidationBank":
        if replicates < 1:
            raise ValueError("validation replicates must be positive")
        days = torch.as_tensor(day_indices, dtype=torch.long, device="cpu")
        if days.ndim != 1 or days.numel() < 1:
            raise ValueError("validation day indices must be a non-empty vector")
        days = torch.sort(days).values
        if len(torch.unique(days)) != len(days):
            raise ValueError("validation day indices contain duplicates")
        noise = torch.empty((replicates, len(days), 10, 24), dtype=torch.float32)
        flow_time = torch.empty((replicates, len(days)), dtype=torch.float32)
        for replicate in range(replicates):
            for offset, day in enumerate(days.tolist()):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    _stable_seed(
                        FORMAL_VALIDATION_BANK_SCHEMA,
                        plan_id,
                        int(base_seed),
                        int(day),
                        replicate,
                    )
                )
                noise[replicate, offset] = torch.randn(
                    (10, 24), dtype=torch.float32, generator=generator
                )
                flow_time[replicate, offset] = torch.rand(
                    (), dtype=torch.float32, generator=generator
                )
        return cls(
            day_index=days.contiguous(),
            noise=noise.contiguous(),
            flow_time=flow_time.contiguous(),
            plan_id=str(plan_id),
            base_seed=int(base_seed),
            evaluation_batch_days=int(evaluation_batch_days),
        )

    @classmethod
    def from_explicit_seeds(
        cls,
        day_indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        noise_seeds: Sequence[int],
        time_seeds: Sequence[int],
        plan_id: str,
        evaluation_batch_days: int = 16,
    ) -> "FormalValidationBank":
        """Build the preregistered bank from distinct noise/time seed lists."""

        noise_seed_tuple = tuple(int(value) for value in noise_seeds)
        time_seed_tuple = tuple(int(value) for value in time_seeds)
        if not noise_seed_tuple or len(noise_seed_tuple) != len(time_seed_tuple):
            raise ValueError("explicit noise/time seed lists must be non-empty and equal")
        days = torch.as_tensor(day_indices, dtype=torch.long, device="cpu")
        if days.ndim != 1 or days.numel() < 1:
            raise ValueError("validation day indices must be a non-empty vector")
        days = torch.sort(days).values
        if len(torch.unique(days)) != len(days):
            raise ValueError("validation day indices contain duplicates")
        replicates = len(noise_seed_tuple)
        noise = torch.empty((replicates, len(days), 10, 24), dtype=torch.float32)
        flow_time = torch.empty((replicates, len(days)), dtype=torch.float32)
        for replicate, (noise_seed, time_seed) in enumerate(
            zip(noise_seed_tuple, time_seed_tuple)
        ):
            for offset, day in enumerate(days.tolist()):
                noise_generator = torch.Generator(device="cpu")
                noise_generator.manual_seed(
                    _stable_seed(
                        FORMAL_VALIDATION_BANK_SCHEMA,
                        plan_id,
                        "noise",
                        noise_seed,
                        int(day),
                    )
                )
                time_generator = torch.Generator(device="cpu")
                time_generator.manual_seed(
                    _stable_seed(
                        FORMAL_VALIDATION_BANK_SCHEMA,
                        plan_id,
                        "time",
                        time_seed,
                        int(day),
                    )
                )
                noise[replicate, offset] = torch.randn(
                    (10, 24), dtype=torch.float32, generator=noise_generator
                )
                flow_time[replicate, offset] = torch.rand(
                    (), dtype=torch.float32, generator=time_generator
                )
        return cls(
            day_index=days.contiguous(),
            noise=noise.contiguous(),
            flow_time=flow_time.contiguous(),
            plan_id=str(plan_id),
            base_seed=-1,
            evaluation_batch_days=int(evaluation_batch_days),
            noise_seeds=noise_seed_tuple,
            time_seeds=time_seed_tuple,
        )

    @classmethod
    def from_batches(
        cls,
        batches: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
        *,
        replicates: int = 4,
        base_seed: int,
        plan_id: str,
        evaluation_batch_days: int = 16,
    ) -> "FormalValidationBank":
        combined = _concatenate_batches(batches)
        return cls.from_day_indices(
            combined.day_index,
            replicates=replicates,
            base_seed=base_seed,
            plan_id=plan_id,
            evaluation_batch_days=evaluation_batch_days,
        )

    @property
    def tensor_sha256(self) -> str:
        return tensor_state_sha256(
            {
                "day_index": self.day_index,
                "noise": self.noise,
                "flow_time": self.flow_time,
            }
        )

    @property
    def manifest(self) -> dict[str, Any]:
        core = {
            "schema": FORMAL_VALIDATION_BANK_SCHEMA,
            "plan_id": self.plan_id,
            "base_seed": int(self.base_seed),
            "replicates": self.replicates,
            "day_count": len(self.day_index),
            "day_index_sha256": tensor_state_sha256({"day_index": self.day_index}),
            "tensor_sha256": self.tensor_sha256,
            "evaluation_batch_days": int(self.evaluation_batch_days),
            "noise_seeds": list(self.noise_seeds),
            "time_seeds": list(self.time_seeds),
            "seed_key": (
                "sha256(schema,plan_id,kind,registered_seed,day_index)"
                if self.noise_seeds
                else "sha256(schema,plan_id,base_seed,day_index,replicate)"
            ),
        }
        return {**core, "bank_sha256": canonical_sha256(core)}

    @property
    def sha256(self) -> str:
        return str(self.manifest["bank_sha256"])

    def _offsets(self, days: torch.Tensor) -> torch.Tensor:
        requested = days.detach().cpu().to(torch.long)
        offsets = torch.searchsorted(self.day_index, requested)
        valid = offsets < len(self.day_index)
        if bool(valid.any()):
            valid = valid & (self.day_index[offsets.clamp_max(len(self.day_index) - 1)] == requested)
        if not bool(valid.all()):
            missing = requested[~valid].tolist()
            raise ValueError(f"days are absent from validation bank: {missing}")
        return offsets

    @torch.no_grad()
    def evaluate(
        self,
        model: JointRectifiedFlowBase,
        batches: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
        *,
        stage: Stage,
        device: str | torch.device | None = None,
    ) -> dict[str, float]:
        """Evaluate one stage with batch/order-invariant validation draws."""

        if stage not in ("atom", "flow"):
            raise ValueError(f"unknown formal validation stage: {stage}")
        combined = _concatenate_batches(batches)
        if not torch.equal(combined.day_index, self.day_index):
            raise ValueError("validation data days do not exactly match the frozen bank")
        selected_device = torch.device(device) if device is not None else next(
            model.parameters()
        ).device
        canonical = _chunks(combined, self.evaluation_batch_days)
        was_training = model.training
        model.eval()
        try:
            if stage == "atom":
                observed_totals = {
                    "atom_nll": 0.0,
                    "accuracy": 0.0,
                    "zero_brier": 0.0,
                }
                location_total = 0.0
                interior_total = 0
                observed_total = 0
                cell_total = 0
                for cpu_batch in canonical:
                    batch = cpu_batch.to(selected_device)
                    values = model.atom_loss(
                        batch.condition,
                        states=batch.state,
                        observed_mask=batch.observed_mask,
                        location_target=(
                            batch.target
                            if model.atom_location_auxiliary_weight > 0.0
                            else None
                        ),
                    )
                    weight = int(batch.observed_mask.sum().item())
                    interior_weight = int(
                        (
                            batch.observed_mask
                            & (batch.state == INTERIOR_STATE)
                        ).sum().item()
                    )
                    observed_total += weight
                    interior_total += interior_weight
                    cell_total += int(batch.observed_mask.numel())
                    for name in observed_totals:
                        scalar = float(values[name].detach().cpu())
                        if not np.isfinite(scalar):
                            raise FloatingPointError(f"non-finite validation atom {name}")
                        observed_totals[name] += scalar * weight
                    location_scalar = float(
                        values["interior_location_smooth_l1"].detach().cpu()
                    )
                    if not np.isfinite(location_scalar):
                        raise FloatingPointError(
                            "non-finite validation interior location loss"
                        )
                    location_total += location_scalar * interior_weight
                if observed_total < 1:
                    raise ValueError("validation contains no observed atom targets")
                if model.atom_location_auxiliary_weight > 0.0 and interior_total < 1:
                    raise ValueError(
                        "validation contains no observed interior location targets"
                    )
                reduced = {
                    name: value / observed_total
                    for name, value in observed_totals.items()
                }
                location_loss = (
                    location_total / interior_total if interior_total else 0.0
                )
                reduced["interior_location_smooth_l1"] = location_loss
                reduced["loss"] = (
                    reduced["atom_nll"]
                    + model.atom_location_auxiliary_weight * location_loss
                )
                return {
                    **reduced,
                    "observed_fraction": observed_total / cell_total,
                    "interior_location_count": float(interior_total),
                    "replicates": 1.0,
                }

            squared_total = 0.0
            active_total = 0
            all_cells = 0
            inactive_velocity_max = 0.0
            for cpu_batch in canonical:
                offsets = self._offsets(cpu_batch.day_index)
                batch = cpu_batch.to(selected_device)
                active = model.atom.active_mask(batch.state, batch.observed_mask)
                active_count = int(active.sum().item())
                if active_count < 1:
                    raise ValueError("a validation chunk has no observed interior cells")
                safe_observation = torch.where(
                    active,
                    batch.target,
                    torch.full_like(batch.target, 0.5),
                )
                target = torch.logit(
                    safe_observation.clamp(
                        model.logit_epsilon, 1.0 - model.logit_epsilon
                    )
                )
                target = torch.where(active, target, torch.zeros_like(target))
                encoded = model.encode_condition(batch.condition)
                context = model.condition_context(encoded)
                for replicate in range(self.replicates):
                    noise = self.noise[replicate].index_select(0, offsets).to(
                        selected_device
                    )
                    noise = torch.where(active, noise, torch.zeros_like(noise))
                    noise = model.prepare_source_noise(noise, encoded, active)
                    flow_time = self.flow_time[replicate].index_select(
                        0, offsets
                    ).to(selected_device)
                    path = (
                        (1.0 - flow_time[:, None, None]) * noise
                        + flow_time[:, None, None] * target
                    )
                    prediction = model.flow(path, flow_time, context, active)
                    desired = target - noise
                    squared = (prediction - desired).square()
                    selected_squared = squared[active]
                    if not bool(torch.isfinite(selected_squared).all()):
                        raise FloatingPointError("non-finite fixed validation flow loss")
                    squared_total += float(
                        selected_squared.double().sum().detach().cpu()
                    )
                    if bool((~active).any()):
                        inactive_velocity_max = max(
                            inactive_velocity_max,
                            float(prediction[~active].abs().max().detach().cpu()),
                        )
                    active_total += active_count
                    all_cells += int(active.numel())
            if active_total < 1:
                raise ValueError("validation contains no observed interior targets")
            loss = squared_total / active_total
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite reduced validation flow loss")
            return {
                "loss": float(loss),
                "velocity_mse": float(loss),
                "active_fraction": float(active_total / all_cells),
                "inactive_velocity_max": float(inactive_velocity_max),
                "replicates": float(self.replicates),
            }
        finally:
            model.train(was_training)

    # An explicit verb reads naturally in runner code and tests.
    validate = evaluate


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("formal CUDA device requested but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": properties.name,
                "cuda_compute_capability": [properties.major, properties.minor],
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
    return result


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


class FormalEpochTrainer:
    """Epoch-based formal trainer with exact epoch-boundary recovery."""

    def __init__(
        self,
        model: JointRectifiedFlowBase,
        output_dir: str | Path,
        *,
        resolved_config: Mapping[str, Any],
        protocol_sha256: str,
        data_bundle_sha256: str,
        code_sha256: str,
        validation_bank: FormalValidationBank,
        training_seed: int,
        shuffle_seed: int | None = None,
        path_seed: int | None = None,
        run_id: str | None = None,
        device: str | torch.device = "cuda",
        extra_identity_hashes: Mapping[str, str] | None = None,
        training_freeze_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.resolved_config = _jsonable(resolved_config)
        self.config_sha256 = canonical_sha256(self.resolved_config)
        self.validation_bank = validation_bank
        self.training_seed = int(training_seed)
        self.shuffle_seed = (
            self.training_seed if shuffle_seed is None else int(shuffle_seed)
        )
        self.path_seed = self.training_seed if path_seed is None else int(path_seed)
        self.run_id = run_id or f"architecture_v1_{model.variant}_seed{training_seed}"
        self.device = torch.device(device)
        self.runtime_identity = _runtime_identity(self.device)
        hashes = {
            "config_sha256": self.config_sha256,
            "protocol_sha256": _validate_sha256(
                protocol_sha256, name="protocol_sha256"
            ),
            "data_bundle_sha256": _validate_sha256(
                data_bundle_sha256, name="data_bundle_sha256"
            ),
            "code_sha256": _validate_sha256(code_sha256, name="code_sha256"),
            "validation_bank_sha256": validation_bank.sha256,
            "runtime_identity_sha256": canonical_sha256(self.runtime_identity),
        }
        for name, value in dict(extra_identity_hashes or {}).items():
            if name in hashes:
                raise ValueError(f"duplicate formal identity hash: {name}")
            hashes[str(name)] = _validate_sha256(value, name=str(name))
        self.identity_hashes = hashes
        self.training_freeze_path = (
            None if training_freeze_path is None else Path(training_freeze_path)
        )
        self.model.to(self.device)

    def _assert_training_open(self) -> None:
        candidates = [self.output_dir / "training.freeze.json"]
        if self.training_freeze_path is not None:
            candidates.append(self.training_freeze_path)
        existing = [path for path in candidates if path.exists()]
        if existing:
            raise RuntimeError(f"formal training is frozen; refusing mutation: {existing[0]}")

    def _stage_paths(self, stage: Stage) -> dict[str, Path]:
        root = self.output_dir / stage
        return {
            "root": root,
            "resume": root / "latest_safe.pt",
            "best": root / "best.pt",
            "history": root / "history.json",
            "completion": root / "completion.json",
            "failure": root / "failure.json",
        }

    def _stage_identity(self, stage: Stage, specification: Mapping[str, Any]) -> dict[str, Any]:
        identity = {
            "schema": "architecture_v1_formal_stage_identity_v1",
            "run_id": self.run_id,
            "candidate_id": self.model.variant,
            "training_seed": self.training_seed,
            "stage": stage,
            "shuffle_seed": self.shuffle_seed,
            "path_seed": self.path_seed,
            "stage_specification": _jsonable(specification),
            "identity_hashes": dict(self.identity_hashes),
            "parameter_manifest_sha256": parameter_manifest(self.model)[
                "structural_sha256"
            ],
        }
        return {**identity, "identity_sha256": canonical_sha256(identity)}

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda_all": (
                [value.cpu() for value in torch.cuda.get_rng_state_all()]
                if self.device.type == "cuda"
                else None
            ),
        }

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda_all")
        if self.device.type == "cuda" and cuda_state is not None:
            if len(cuda_state) != torch.cuda.device_count():
                raise RuntimeError("resume CUDA RNG device count changed")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])

    def _checkpoint_payload(
        self,
        *,
        stage_identity: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        next_epoch: int,
        best_score: float,
        best_epoch: int,
        stale_validations: int,
        history: Sequence[Mapping[str, Any]],
        status: str,
        stop_reason: str | None,
        stage_start_state_sha256: str,
    ) -> dict[str, Any]:
        model_state = _cpu_state_dict(self.model)
        return {
            "schema": FORMAL_RESUME_SCHEMA,
            "status": status,
            "stop_reason": stop_reason,
            "stage_identity": dict(stage_identity),
            "next_epoch": int(next_epoch),
            "best_validation_loss": float(best_score),
            "best_epoch": int(best_epoch),
            "stale_validations": int(stale_validations),
            "history": [_jsonable(record) for record in history],
            "model_state_dict": model_state,
            "model_state_sha256": tensor_state_sha256(model_state),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": self._rng_state(),
            "parameter_manifest": parameter_manifest(self.model),
            "shared_EA_state_sha256": shared_ea_state_sha256(self.model),
            "stage_start_model_state_sha256": stage_start_state_sha256,
            "runtime_identity": dict(self.runtime_identity),
        }

    def _load_resume(
        self,
        path: Path,
        *,
        stage_identity: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        _verified_file_sha256(path)  # Verify bytes before torch.load/unpickling.
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != FORMAL_RESUME_SCHEMA:
            raise ValueError("unexpected formal resume schema")
        if payload.get("stage_identity") != dict(stage_identity):
            raise RuntimeError("formal resume identity changed")
        if payload.get("parameter_manifest") != parameter_manifest(self.model):
            raise RuntimeError("formal resume model topology changed")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        loaded_state = _cpu_state_dict(self.model)
        if tensor_state_sha256(loaded_state) != payload.get("model_state_sha256"):
            raise RuntimeError("formal resume model-state hash mismatch")
        if shared_ea_state_sha256(self.model) != payload.get(
            "shared_EA_state_sha256"
        ):
            raise RuntimeError("formal resume shared E/A hash mismatch")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to(optimizer, self.device)
        if not _optimizer_finite(optimizer):
            raise FloatingPointError("formal resume optimizer is non-finite")
        if any(
            not bool(torch.isfinite(parameter).all())
            for parameter in self.model.parameters()
        ):
            raise FloatingPointError("formal resume model is non-finite")
        self._restore_rng_state(payload["rng_state"])
        return payload

    def fit_stage(
        self,
        stage: Stage,
        train_data: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
        validation_data: ArchitectureBatch | Sequence[ArchitectureBatch] | Any,
        *,
        max_epochs: int,
        batch_days: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        validate_every_epochs: int = 1,
        early_stopping_patience: int = 8,
        early_stopping_minimum_delta: float = 1e-5,
        early_stopping_relative_delta: float = 0.0,
        minimum_epochs_before_early_stop: int = 1,
        resume: bool = False,
        epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
        restore_best: bool = True,
        record_update_preclip_gradient_norms: bool = False,
        gradient_audit_epoch_start: int | None = None,
        gradient_audit_epoch_end: int | None = None,
        gradient_audit_gate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit one registered stage and checkpoint only complete epochs.

        ``epoch_callback`` runs *after* ``latest_safe.pt`` is committed.  A
        callback exception is therefore a convenient way to exercise exact
        interruption/resume behaviour without adding a non-scientific CLI flag.
        """

        self._assert_training_open()
        if stage not in ("atom", "flow"):
            raise ValueError(f"unknown formal stage: {stage}")
        if max_epochs < 1 or batch_days < 1 or validate_every_epochs < 1:
            raise ValueError("epochs, batch days and validation interval must be positive")
        if learning_rate <= 0.0 or weight_decay < 0.0 or gradient_clip <= 0.0:
            raise ValueError("invalid formal optimizer configuration")
        if (
            early_stopping_patience < 1
            or early_stopping_minimum_delta < 0.0
            or early_stopping_relative_delta < 0.0
            or minimum_epochs_before_early_stop < 1
            or minimum_epochs_before_early_stop > max_epochs
        ):
            raise ValueError("invalid formal early-stopping configuration")
        if not record_update_preclip_gradient_norms and (
            gradient_audit_epoch_start is not None
            or gradient_audit_epoch_end is not None
            or gradient_audit_gate is not None
        ):
            raise ValueError("gradient audit requires per-update preclip recording")
        if record_update_preclip_gradient_norms:
            audit_start = int(gradient_audit_epoch_start or 1)
            audit_end = (
                None
                if gradient_audit_epoch_end is None
                else int(gradient_audit_epoch_end)
            )
            if not 1 <= audit_start <= max_epochs or (
                audit_end is not None
                and not audit_start <= audit_end <= max_epochs
            ):
                raise ValueError("formal gradient audit window lies outside max epochs")
        else:
            audit_start = 1
            audit_end = None

        train = _concatenate_batches(train_data)
        validation = _concatenate_batches(validation_data)
        if not torch.equal(validation.day_index, self.validation_bank.day_index):
            raise ValueError("validation data do not match the frozen validation bank")
        train_days_sha256 = tensor_state_sha256({"day_index": train.day_index})
        validation_days_sha256 = tensor_state_sha256(
            {"day_index": validation.day_index}
        )
        stage_specification = {
            "max_epochs": int(max_epochs),
            "batch_days": int(batch_days),
            "updates_per_epoch": int(
                math.ceil(train.condition.shape[0] / batch_days)
            ),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip": float(gradient_clip),
            "validate_every_epochs": int(validate_every_epochs),
            "early_stopping_patience_validations": int(early_stopping_patience),
            "early_stopping_minimum_delta": float(early_stopping_minimum_delta),
            "early_stopping_relative_delta": float(early_stopping_relative_delta),
            "minimum_epochs_before_early_stop": int(
                minimum_epochs_before_early_stop
            ),
            "shuffle_algorithm": "numpy.default_rng(PCG64).permutation keyed by stage/epoch",
            "train_days_sha256": train_days_sha256,
            "validation_days_sha256": validation_days_sha256,
        }
        if record_update_preclip_gradient_norms:
            stage_specification["update_gradient_audit"] = {
                "persist_preclip_gradient_norm_per_update": True,
                "audit_epoch_start": audit_start,
                "audit_epoch_end": (
                    "stage_completion" if audit_end is None else audit_end
                ),
                "epoch_number_semantics": "one_based_inclusive",
                "gate": _jsonable(dict(gradient_audit_gate or {})),
            }
        stage_identity = self._stage_identity(stage, stage_specification)
        paths = self._stage_paths(stage)
        stage_artifacts = [
            paths["resume"],
            paths["best"],
            paths["history"],
            paths["completion"],
        ]
        if not resume and any(path.exists() for path in stage_artifacts):
            raise FileExistsError(
                "formal stage artifacts already exist; use resume only for the exact identity"
            )
        if resume and not paths["resume"].is_file():
            raise FileNotFoundError("--resume requested but latest_safe.pt is absent")
        if paths["completion"].exists():
            raise RuntimeError("formal stage is already complete")

        parameters = configure_stage(self.model, stage)
        optimizer = torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        stage_start_state = _cpu_state_dict(self.model)
        stage_start_state_sha256 = tensor_state_sha256(stage_start_state)
        start_epoch = 0
        best_score = float("inf")
        best_epoch = -1
        stale_validations = 0
        history: list[dict[str, Any]] = []
        if resume:
            payload = self._load_resume(
                paths["resume"],
                stage_identity=stage_identity,
                optimizer=optimizer,
            )
            if payload.get("status") == "complete":
                raise RuntimeError("formal stage resume is already marked complete")
            start_epoch = int(payload["next_epoch"])
            best_score = float(payload["best_validation_loss"])
            best_epoch = int(payload["best_epoch"])
            stale_validations = int(payload["stale_validations"])
            history = [dict(record) for record in payload["history"]]
            stage_start_state_sha256 = str(payload["stage_start_model_state_sha256"])
            if start_epoch < 0 or start_epoch > max_epochs:
                raise RuntimeError("formal resume next_epoch is outside the registered schedule")
        else:
            random.seed(self.path_seed)
            np.random.seed(self.path_seed % (2**32 - 1))
            torch.manual_seed(self.path_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.path_seed)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        stopped_early = bool(
            resume and stale_validations >= early_stopping_patience
        )
        stop_reason = (
            "early_stopping_patience" if stopped_early else "max_epochs"
        )
        updates_per_epoch = int(stage_specification["updates_per_epoch"])
        paths["root"].mkdir(parents=True, exist_ok=True)
        epoch_range = range(start_epoch, max_epochs) if not stopped_early else range(0)
        for epoch in epoch_range:
            permutation = deterministic_epoch_permutation(
                train.condition.shape[0],
                training_seed=self.shuffle_seed,
                stage=stage,
                epoch=epoch,
            )
            train_sums: dict[str, float] = {}
            train_weight_total = 0
            gradient_clipped_updates = 0
            epoch_updates = 0
            update_gradient_audit: list[dict[str, Any]] = []
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            epoch_started = datetime.now(timezone.utc)
            wall_started = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            wall_finished = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if wall_started is not None:
                wall_started.record()
            cpu_started = time.perf_counter()
            try:
                for batch_offset, begin in enumerate(
                    range(0, len(permutation), batch_days)
                ):
                    indices = permutation[begin : begin + batch_days]
                    batch = _take_batch(train, indices).to(self.device)
                    global_step = epoch * updates_per_epoch + batch_offset + 1
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(
                        _stable_seed(
                            "architecture_v1_formal_training_loss_v1",
                            self.path_seed,
                            stage,
                            global_step,
                        )
                    )
                    metrics = train_step(
                        self.model,
                        batch,
                        optimizer,
                        stage=stage,
                        generator=generator,
                        gradient_clip=gradient_clip,
                    )
                    epoch_updates += 1
                    if float(metrics["gradient_norm"]) > gradient_clip:
                        gradient_clipped_updates += 1
                    if record_update_preclip_gradient_norms:
                        update_gradient_audit.append(
                            {
                                "update_in_epoch": int(batch_offset + 1),
                                "global_step": int(global_step),
                                "preclip_gradient_norm": float(
                                    metrics["gradient_norm"]
                                ),
                                "gradient_clip": float(gradient_clip),
                                "clipped": bool(
                                    float(metrics["gradient_norm"])
                                    > float(gradient_clip)
                                ),
                                "batch_day_index_sha256": tensor_state_sha256(
                                    {"day_index": batch.day_index.detach().cpu()}
                                ),
                            }
                        )
                    if stage == "atom":
                        weight = int(batch.observed_mask.sum().item())
                    else:
                        weight = int(
                            (
                                batch.observed_mask
                                & (batch.state == INTERIOR_STATE)
                            ).sum().item()
                        )
                    if weight < 1:
                        raise ValueError("formal train batch contains no supervised cells")
                    train_weight_total += weight
                    for name, value in metrics.items():
                        if name == "wall_seconds":
                            continue
                        scalar = float(value)
                        if not np.isfinite(scalar):
                            raise FloatingPointError(f"non-finite train metric {name}")
                        train_sums[name] = train_sums.get(name, 0.0) + scalar * weight
                if self.device.type == "cuda":
                    assert wall_finished is not None and wall_started is not None
                    wall_finished.record()
                    torch.cuda.synchronize(self.device)
                    epoch_wall = wall_started.elapsed_time(wall_finished) / 1000.0
                else:
                    epoch_wall = time.perf_counter() - cpu_started
                train_metrics = {
                    name: total / train_weight_total
                    for name, total in train_sums.items()
                }
                validation_metrics: dict[str, float] | None = None
                should_validate = (
                    (epoch + 1) % validate_every_epochs == 0
                    or epoch + 1 == max_epochs
                )
                improved = False
                if should_validate:
                    validation_metrics = self.validation_bank.evaluate(
                        self.model,
                        validation,
                        stage=stage,
                        device=self.device,
                    )
                    score = float(validation_metrics["loss"])
                    if not np.isfinite(score):
                        raise FloatingPointError("non-finite formal validation loss")
                    required_delta = early_stopping_minimum_delta
                    if np.isfinite(best_score):
                        required_delta = max(
                            required_delta,
                            early_stopping_relative_delta * best_score,
                        )
                    improved = score < best_score - required_delta
                    if improved:
                        best_score = score
                        best_epoch = epoch
                        stale_validations = 0
                    else:
                        stale_validations += 1
                record = {
                    "epoch": int(epoch),
                    "epoch_number": int(epoch + 1),
                    "global_step_end": int((epoch + 1) * updates_per_epoch),
                    "permutation_sha256": epoch_permutation_sha256(permutation),
                    "train": train_metrics,
                    "updates": int(epoch_updates),
                    "gradient_clipped_updates": int(gradient_clipped_updates),
                    "gradient_clip_fraction": float(
                        gradient_clipped_updates / max(epoch_updates, 1)
                    ),
                    "validation": validation_metrics,
                    "validation_improved": bool(improved),
                    "stale_validations": int(stale_validations),
                    "epoch_wall_seconds_synchronized": float(epoch_wall),
                    "started_utc": epoch_started.isoformat(),
                }
                if record_update_preclip_gradient_norms:
                    record["update_gradient_audit"] = update_gradient_audit
                history.append(record)
                checkpoint = self._checkpoint_payload(
                    stage_identity=stage_identity,
                    optimizer=optimizer,
                    next_epoch=epoch + 1,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    stale_validations=stale_validations,
                    history=history,
                    status="running",
                    stop_reason=None,
                    stage_start_state_sha256=stage_start_state_sha256,
                )
                if improved:
                    _atomic_torch_save(paths["best"], checkpoint)
                _atomic_torch_save(paths["resume"], checkpoint)
                _atomic_json(
                    paths["history"],
                    {
                        "schema": FORMAL_HISTORY_SCHEMA,
                        "stage_identity": stage_identity,
                        "records": history,
                    },
                )
                if epoch_callback is not None:
                    epoch_callback(record)
                if (
                    should_validate
                    and epoch + 1 >= minimum_epochs_before_early_stop
                    and stale_validations >= early_stopping_patience
                ):
                    stopped_early = True
                    stop_reason = "early_stopping_patience"
                    break
            except Exception as error:
                _atomic_json(
                    paths["failure"],
                    {
                        "schema": FORMAL_FAILURE_SCHEMA,
                        "stage_identity": stage_identity,
                        "failed_epoch": int(epoch),
                        "latest_safe": str(paths["resume"]),
                        "latest_safe_exists": paths["resume"].is_file(),
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "traceback": traceback.format_exc(),
                        "scientific_status": "failed; resume only from the previous complete epoch",
                    },
                )
                raise

        next_epoch = len(history)
        if not paths["best"].is_file() or best_epoch < 0 or not np.isfinite(best_score):
            raise RuntimeError("formal stage ended without a validation-selected checkpoint")
        completed_payload = self._checkpoint_payload(
            stage_identity=stage_identity,
            optimizer=optimizer,
            next_epoch=next_epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            stale_validations=stale_validations,
            history=history,
            status="complete",
            stop_reason=stop_reason,
            stage_start_state_sha256=stage_start_state_sha256,
        )
        latest_sha256 = _atomic_torch_save(paths["resume"], completed_payload)
        best_sha256 = _verified_file_sha256(paths["best"])
        if restore_best:
            best_payload = torch.load(
                paths["best"], map_location=self.device, weights_only=False
            )
            if best_payload.get("stage_identity") != stage_identity:
                raise RuntimeError("best checkpoint identity mismatch")
            self.model.load_state_dict(best_payload["model_state_dict"], strict=True)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        completion = {
            "schema": FORMAL_COMPLETION_SCHEMA,
            "status": "complete",
            "scientific_status": "formal_training_complete_selection_not_opened",
            "run_id": self.run_id,
            "candidate_id": self.model.variant,
            "training_seed": self.training_seed,
            "stage": stage,
            "stage_identity": stage_identity,
            "epochs_completed": int(next_epoch),
            "max_epochs": int(max_epochs),
            "stopped_early": bool(stopped_early),
            "stop_reason": stop_reason,
            "best_epoch": int(best_epoch),
            "best_epoch_number": int(best_epoch + 1),
            "best_validation_loss": float(best_score),
            "best_checkpoint": str(paths["best"].resolve()),
            "best_checkpoint_sha256": best_sha256,
            "latest_safe_checkpoint": str(paths["resume"].resolve()),
            "latest_safe_checkpoint_sha256": latest_sha256,
            "history": str(paths["history"].resolve()),
            "history_sha256": _verified_file_sha256(paths["history"]),
            "validation_bank": self.validation_bank.manifest,
            "model_spec": self.model.model_spec(),
            "shared_EA_state_sha256": shared_ea_state_sha256(self.model),
            "runtime_identity": self.runtime_identity,
            "peak_memory_allocated_bytes": peak_memory,
            "selection_target_accessed": False,
            "calibration_target_accessed": False,
            "r_seen_target_accessed": False,
        }
        if record_update_preclip_gradient_norms:
            updates = [
                {
                    "epoch": int(record["epoch"]),
                    "epoch_number": int(record["epoch_number"]),
                    **dict(update),
                }
                for record in history
                for update in record.get("update_gradient_audit", [])
            ]
            # Early stopping may end before a preregistered audit endpoint.  In
            # that case fail closed instead of silently shortening the window.
            if (
                audit_end is not None
                and (not history or int(history[-1]["epoch_number"]) < audit_end)
            ):
                raise RuntimeError(
                    "formal stage ended before the registered gradient audit window"
                )
            actual_audit_end = (
                int(history[-1]["epoch_number"])
                if audit_end is None
                else audit_end
            )
            completion["preclip_gradient_audit"] = summarize_preclip_gradient_norms(
                updates,
                audit_epoch_start=audit_start,
                audit_epoch_end=actual_audit_end,
                gradient_clip=float(gradient_clip),
                gate=gradient_audit_gate,
            )
        completion_sha256 = _atomic_json(paths["completion"], completion)
        return {
            **completion,
            "completion": str(paths["completion"].resolve()),
            "completion_sha256": completion_sha256,
        }


def write_training_freeze(
    destination: str | Path,
    *,
    completion_files: Mapping[int, str | Path],
    expected_seeds: Sequence[int] = (0, 1, 2),
    candidate_id: str = "R0",
    required_stage: Stage = "flow",
    gate_config_sha256: str,
    selection_plan_sha256: str,
    protocol_sha256: str,
    data_bundle_sha256: str,
    code_sha256: str,
) -> dict[str, Any]:
    """Freeze a complete formal training family before selection is opened.

    The helper is intentionally one-way: an existing freeze file or sidecar is
    never overwritten.  It certifies training closure but explicitly does not
    authorize selection access; a separate release/consumption state machine
    must bind itself to the returned freeze-file hash.
    """

    path = Path(destination)
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"training freeze already exists: {path}")
    registered = tuple(int(seed) for seed in expected_seeds)
    if len(set(registered)) != len(registered) or not registered:
        raise ValueError("expected seeds must be unique and non-empty")
    supplied = {int(seed): Path(value) for seed, value in completion_files.items()}
    if set(supplied) != set(registered):
        raise ValueError("completion files must contain exactly every registered seed")
    common_hashes = {
        "protocol_sha256": _validate_sha256(
            protocol_sha256, name="protocol_sha256"
        ),
        "data_bundle_sha256": _validate_sha256(
            data_bundle_sha256, name="data_bundle_sha256"
        ),
        "code_sha256": _validate_sha256(code_sha256, name="code_sha256"),
    }
    gate_hash = _validate_sha256(gate_config_sha256, name="gate_config_sha256")
    plan_hash = _validate_sha256(
        selection_plan_sha256, name="selection_plan_sha256"
    )
    records: list[dict[str, Any]] = []
    for seed in registered:
        completion_path = supplied[seed]
        completion_file_sha = _verified_file_sha256(completion_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("schema") != FORMAL_COMPLETION_SCHEMA:
            raise ValueError(f"seed {seed} has an unexpected completion schema")
        if completion.get("status") != "complete":
            raise RuntimeError(f"seed {seed} formal training is incomplete")
        if int(completion.get("training_seed", -1)) != seed:
            raise ValueError(f"completion seed mismatch for seed {seed}")
        if completion.get("candidate_id") != candidate_id:
            raise ValueError(f"completion candidate mismatch for seed {seed}")
        if completion.get("stage") != required_stage:
            raise ValueError(f"completion stage mismatch for seed {seed}")
        if any(
            completion.get(field) is not False
            for field in (
                "selection_target_accessed",
                "calibration_target_accessed",
                "r_seen_target_accessed",
            )
        ):
            raise RuntimeError(f"forbidden target access declared by seed {seed}")
        identity_hashes = completion.get("stage_identity", {}).get(
            "identity_hashes", {}
        )
        for field, expected in common_hashes.items():
            if identity_hashes.get(field) != expected:
                raise RuntimeError(f"seed {seed} completion {field} mismatch")
        checkpoint = Path(completion["best_checkpoint"])
        checkpoint_sha = _verified_file_sha256(checkpoint)
        if checkpoint_sha != completion.get("best_checkpoint_sha256"):
            raise RuntimeError(f"seed {seed} best checkpoint hash mismatch")
        records.append(
            {
                "training_seed": seed,
                "completion": str(completion_path.resolve()),
                "completion_sha256": completion_file_sha,
                "run_id": completion["run_id"],
                "stage_identity_sha256": completion["stage_identity"][
                    "identity_sha256"
                ],
                "best_checkpoint": str(checkpoint.resolve()),
                "best_checkpoint_sha256": checkpoint_sha,
                "best_epoch": int(completion["best_epoch"]),
                "best_validation_loss": float(completion["best_validation_loss"]),
                "shared_EA_state_sha256": completion[
                    "shared_EA_state_sha256"
                ],
            }
        )
    payload = {
        "schema": FORMAL_FREEZE_SCHEMA,
        "status": "training_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "required_stage": required_stage,
        "registered_training_seeds": list(registered),
        "common_identity_hashes": common_hashes,
        "gate_config_sha256": gate_hash,
        "selection_plan_sha256": plan_hash,
        "seed_completions": records,
        "training_closed": True,
        "selection_authorized": False,
        "selection_policy": (
            "a separate release must bind this freeze file hash, the gate hash "
            "and the selection-plan hash before any selection target access"
        ),
    }
    digest = _atomic_json(path, payload)
    return {
        **payload,
        "training_freeze": str(path.resolve()),
        "training_freeze_sha256": digest,
    }


# Explicit longer alias for discoverability.
freeze_formal_training = write_training_freeze


__all__ = [
    "FORMAL_COMPLETION_SCHEMA",
    "FORMAL_FAILURE_SCHEMA",
    "FORMAL_FREEZE_SCHEMA",
    "FORMAL_GRADIENT_PREFLIGHT_SCHEMA",
    "FORMAL_HISTORY_SCHEMA",
    "FORMAL_RESUME_SCHEMA",
    "FORMAL_SAMPLING_CHUNK_AUDIT_SCHEMA",
    "FORMAL_VALIDATION_BANK_SCHEMA",
    "FormalEpochTrainer",
    "FormalValidationBank",
    "CommonRandomChunkSamples",
    "common_random_chunk_samples",
    "deterministic_epoch_permutation",
    "epoch_permutation",
    "epoch_permutation_sha256",
    "evaluate_sampling_chunk_gate",
    "freeze_formal_training",
    "run_discarded_train_only_stage",
    "summarize_preclip_gradient_norms",
    "write_training_freeze",
]
