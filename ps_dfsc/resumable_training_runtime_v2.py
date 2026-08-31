"""Atomic epoch checkpoints for finite-gradient publication training."""

from __future__ import annotations

from pathlib import Path

import torch

import ps_dfsc.pipeline_runtime_publication as runtime
from .manifest import file_sha256
from .stable_training_metrics import (
    SMOOTHING_EPSILON,
    SURROGATE_SCHEMA,
)
from .training_v5 import train_calibrator


def train_command(args) -> None:
    output = Path(args.output)
    resume_path = output.with_suffix(".epoch_resume.pt")
    temporary_path = output.with_suffix(".epoch_resume.tmp")
    metadata = {
        "schema": "ps_dfsc_epoch_resume_metadata_v2",
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
        if resume_path.exists():
            payload = torch.load(
                resume_path,
                map_location=kwargs.get("device", "cpu"),
                weights_only=False,
            )
            if payload.get("metadata") != metadata:
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

        if (
            resume_state is not None
            and refresh_completed
            and commitment_refresh is not None
        ):
            model.load_state_dict(resume_state["model_state"])
            commitment_refresh(model)

        def save_epoch(state):
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": metadata,
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
            **kwargs,
        )

    runtime.train_calibrator = resumable
    try:
        runtime.train_command(args)
    finally:
        runtime.train_calibrator = original


__all__ = ["train_command"]
