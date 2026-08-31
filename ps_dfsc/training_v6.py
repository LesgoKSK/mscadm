"""Finite-gradient trainer with fail-closed checkpoint validation."""

from __future__ import annotations

import math

import numpy as np
import torch

from .training_v5 import train_calibrator as _finite_gradient_train


def _validate_state(state: dict) -> None:
    if not all(
        bool(torch.isfinite(value).all())
        for value in state["model_state"].values()
    ):
        raise FloatingPointError(
            "candidate model contains a non-finite parameter"
        )
    if not all(
        math.isfinite(float(value))
        for record in state["history_epochs"]
        for value in record.values()
    ):
        raise FloatingPointError(
            "candidate training history contains a non-finite value"
        )
    if not all(
        math.isfinite(float(value))
        for value in state["augmented_dual"].values()
    ):
        raise FloatingPointError(
            "candidate augmented-Lagrangian dual is non-finite"
        )
    if not np.isfinite(state["cached_commitment"]).all():
        raise FloatingPointError(
            "candidate commitment cache is non-finite"
        )


def train_calibrator(*args, epoch_callback=None, **kwargs):
    def validated_callback(state):
        _validate_state(state)
        if epoch_callback is not None:
            epoch_callback(state)

    return _finite_gradient_train(
        *args,
        epoch_callback=validated_callback,
        **kwargs,
    )


__all__ = ["train_calibrator"]
