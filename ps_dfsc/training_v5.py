"""Publication trainer with finite-gradient proper-score surrogates."""

from __future__ import annotations

import ps_dfsc.training_v4 as _base

from .stable_training_metrics import (
    smooth_energy_score,
    smooth_variogram_score,
)


def train_calibrator(*args, **kwargs):
    original_energy = _base._energy_score
    original_variogram = _base._variogram_score
    _base._energy_score = smooth_energy_score
    _base._variogram_score = smooth_variogram_score
    try:
        return _base.train_calibrator(*args, **kwargs)
    finally:
        _base._energy_score = original_energy
        _base._variogram_score = original_variogram


__all__ = ["train_calibrator"]
