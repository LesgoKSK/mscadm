"""Restartable stochastic-day training with audited midpoint reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import ps_dfsc.pipeline_runtime_publication as runtime
from .manifest import file_sha256
from .stable_training_metrics import (
    SMOOTHING_EPSILON,
    SURROGATE_SCHEMA,
)
from .training_v7 import train_calibrator


DECISION_BATCHES_PER_EPOCH = 2


def _legacy_metadata_matches(old: dict, new: dict) -> bool:
    keys = (
        "proper_score_surrogate",
        "proper_score_smoothing_epsilon",
        "input_sha256",
        "mapping_sha256",
        "beta",
        "epochs",
        "warmup_epochs",
        "batch_size",
        "learning_rate",
        "strong_convexity",
        "seed",
        "clusters",
    )
    schema = old.get("schema")
    if schema not in (
        "ps_dfsc_epoch_resume_metadata_v2",
        "ps_dfsc_epoch_resume_metadata_v3",
    ):
        return False
    if not all(old.get(key) == new.get(key) for key in keys):
        return False
    if schema == "ps_dfsc_epoch_resume_metadata_v3":
        return (
            old.get("decision_batches_per_epoch")
            == new.get("decision_batches_per_epoch")
            and old.get("decision_sampling_unit")
            == new.get("decision_sampling_unit")
        )
    return True


def _refresh_records(output: Path) -> list[dict]:
    cache = output.with_suffix(".refresh_cache")
    records = []
    for day in range(100):
        path = cache / f"day_{day:03d}.npz"
        if not path.is_file():
            raise RuntimeError(
                f"midpoint refresh cache is incomplete: missing {path}"
            )
        with np.load(path, allow_pickle=False) as value:
            records.append(
                {
                    "day_index": day,
                    "mip_gap": float(value["mip_gap"]),
                    "solver_success": bool(value["solver_success"]),
                    "solver_status": str(value["solver_status"]),
                    "dual_bound": float(value["dual_bound"]),
                    "node_count": int(value["node_count"]),
                    "solve_time_seconds": float(value["solve_time_seconds"]),
                    "cache_origin": str(value["cache_origin"]),
                }
            )
    return records


def _postprocess(output: Path, *, midpoint_time_limit: float) -> None:
    records = _refresh_records(output)
    threshold = 0.001 * (1.0 + 1e-6) + 1e-12
    summary = {
        "cases": len(records),
        "solver_success_rate": float(
            np.mean([row["solver_success"] for row in records])
        ),
        "target_gap_pass_rate": float(
            np.mean([row["mip_gap"] <= threshold for row in records])
        ),
        "maximum_mip_gap": float(max(row["mip_gap"] for row in records)),
    }
    sampling = {
        "schema": "ps_dfsc_stochastic_day_training_v1",
        "proper_score_warmup_days_per_epoch": 100,
        "decision_batches_per_epoch": DECISION_BATCHES_PER_EPOCH,
        "batch_size": 4,
        "decision_days_per_epoch": 8,
        "sampling_unit": "complete day",
        "validation_uses_all_days": True,
        "midpoint_refresh_days": 100,
        "midpoint_time_limit_seconds": float(midpoint_time_limit),
    }
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["commitment_refresh_calls"] = 1
    payload["commitment_refresh_summary"] = summary
    payload["training_sampling"] = sampling
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    history_path = output.with_suffix(".history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["commitment_refresh_calls"] = 1
    history["commitment_refresh_summary"] = summary
    history["commitment_refresh_records"] = records
    history["training_sampling"] = sampling
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_command(args) -> None:
    output = Path(args.output)
    resume_path = output.with_suffix(".epoch_resume.pt")
    temporary_path = output.with_suffix(".epoch_resume.tmp")
    metadata = {
        "schema": "ps_dfsc_epoch_resume_metadata_v3",
        "proper_score_surrogate": SURROGATE_SCHEMA,
        "proper_score_smoothing_epsilon": SMOOTHING_EPSILON,
        "input_sha256": file_sha256(args.input),
        "mapping_sha256": (
            file_sha256(args.mapping)
            if getattr(args, "mapping", None) is not None
            else "fit_from_registered_training_archive"
        ),
        "beta": float(args.beta),
        "epochs": int(args.epochs),
        "warmup_epochs": int(args.warmup_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "strong_convexity": float(args.strong_convexity),
        "seed": int(args.seed),
        "clusters": int(args.clusters),
        "decision_batches_per_epoch": DECISION_BATCHES_PER_EPOCH,
        "decision_sampling_unit": "complete_day",
        "midpoint_time_limit_seconds": float(args.time_limit),
    }
    original = runtime.train_calibrator

    def resumable(
        model,
        *values,
        commitment_refresh=None,
        **kwargs,
    ):
        resume_state = None
        refresh_completed = False
        transition = None
        if resume_path.exists():
            payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            old_metadata = payload.get("metadata", {})
            if old_metadata == metadata:
                pass
            elif _legacy_metadata_matches(old_metadata, metadata):
                legacy_v2 = (
                    old_metadata.get("schema")
                    == "ps_dfsc_epoch_resume_metadata_v2"
                )
                transition = {
                    "from": (
                        "full_day_decision_epochs"
                        if legacy_v2
                        else "stochastic_complete_day_decision_epochs"
                    ),
                    "to": "stochastic_complete_day_decision_epochs",
                    "effective_next_epoch": int(
                        payload["state"]["next_epoch"]
                    ),
                    "reason": (
                        "formal runtime feasibility on registered hardware"
                        if legacy_v2
                        else "restore registered 600-second exact midpoint limit"
                    ),
                }
            else:
                raise ValueError(
                    "epoch resume metadata does not match this training run"
                )
            resume_state = payload["state"]
            refresh_completed = bool(
                payload.get("midpoint_refresh_completed", False)
            )
        refresh_flag = {"completed": refresh_completed}

        def wrapped_refresh(current_model):
            result = commitment_refresh(current_model)
            refresh_flag["completed"] = True
            return result

        def save_epoch(state):
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": metadata,
                    "protocol_transition": transition,
                    "midpoint_refresh_completed": (
                        refresh_flag["completed"]
                    ),
                    "state": state,
                },
                temporary_path,
            )
            temporary_path.replace(resume_path)

        return train_calibrator(
            model,
            *values,
            commitment_refresh=(
                wrapped_refresh
                if commitment_refresh is not None
                else None
            ),
            resume_state=resume_state,
            epoch_callback=save_epoch,
            decision_batches_per_epoch=DECISION_BATCHES_PER_EPOCH,
            **kwargs,
        )

    runtime.train_calibrator = resumable
    try:
        runtime.train_command(args)
        _postprocess(output, midpoint_time_limit=float(args.time_limit))
    finally:
        runtime.train_calibrator = original


__all__ = ["DECISION_BATCHES_PER_EPOCH", "train_command"]
