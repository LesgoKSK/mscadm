"""Leakage-safe empirical and legacy-linear calibration baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cr_mscadm.calibration import CopulaPITCalibrator
from rahc.v2_calibration import RAHCalibratorV2
from rahc.validation import grouped_day_folds


@dataclass
class CrossfitResult:
    scenarios_by_seed: list[np.ndarray]
    fold_assignments: np.ndarray
    fit_records: list[dict[str, Any]]


def _validate_seed_arrays(
    scenarios_by_seed: Sequence[np.ndarray], observations: np.ndarray, day: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    values = [np.asarray(item) for item in scenarios_by_seed]
    truth = np.asarray(observations)
    dates = np.asarray(day).astype("datetime64[D]")
    if not values:
        raise ValueError("at least one model-seed array is required")
    shape = values[0].shape
    if any(item.shape != shape for item in values):
        raise ValueError("all model-seed scenario arrays must align")
    if len(shape) != 3 or truth.shape != (shape[0], shape[2]) or dates.shape != (shape[0],):
        raise ValueError("scenario, observation, or day shape mismatch")
    return values, truth, dates


def crossfit_c0(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    day: np.ndarray,
    *,
    folds: int = 5,
    fold_seed: int = 20260719,
    strength: float = 1.0,
) -> CrossfitResult:
    """Date-grouped OOF C0, fitted separately for each model seed."""

    values, truth, dates = _validate_seed_arrays(scenarios_by_seed, observations, day)
    assignments = grouped_day_folds(dates, folds=int(folds), seed=int(fold_seed))
    outputs = [np.empty_like(item) for item in values]
    records: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        train = assignments != fold
        held = ~train
        if set(np.unique(dates[train])) & set(np.unique(dates[held])):
            raise AssertionError("date leakage in C0 cross-fitting")
        for seed, scenarios in enumerate(values):
            calibrator = CopulaPITCalibrator.fit(
                scenarios[train], truth[train], strength=float(strength)
            )
            outputs[seed][held] = calibrator.transform(scenarios[held])
            records.append(
                {
                    "fold": fold,
                    "model_seed_index": seed,
                    "training_dates": int(len(np.unique(dates[train]))),
                    "held_dates": int(len(np.unique(dates[held]))),
                    "strength": float(strength),
                    "tail_rule": "linear",
                }
            )
    return CrossfitResult(outputs, assignments, records)


def fit_c0(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    *,
    strength: float = 1.0,
) -> list[CopulaPITCalibrator]:
    values = [np.asarray(item) for item in scenarios_by_seed]
    truth = np.asarray(observations)
    return [
        CopulaPITCalibrator.fit(item, truth, strength=float(strength)) for item in values
    ]


def transform_c0(
    calibrators: Sequence[CopulaPITCalibrator], scenarios_by_seed: Sequence[np.ndarray]
) -> list[np.ndarray]:
    if len(calibrators) != len(scenarios_by_seed):
        raise ValueError("one C0 calibrator is required per model seed")
    return [calibrator.transform(np.asarray(values)) for calibrator, values in zip(calibrators, scenarios_by_seed)]


def crossfit_legacy_linear(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    zone: np.ndarray,
    day: np.ndarray,
    *,
    folds: int = 5,
    fold_seed: int = 20260719,
    regularization: float = 0.1,
    strength: float = 1.0,
    epochs: int = 160,
    device: str = "cpu",
) -> CrossfitResult:
    """Prior full RAHC with linear tails, retained only as ablation A1."""

    values, truth, dates = _validate_seed_arrays(scenarios_by_seed, observations, day)
    zones = np.asarray(zone)
    if zones.shape != (len(truth),):
        raise ValueError("zone shape mismatch")
    assignments = grouped_day_folds(dates, folds=int(folds), seed=int(fold_seed))
    outputs = [np.empty_like(item) for item in values]
    records: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        train = assignments != fold
        held = ~train
        pooled_scenarios = np.concatenate([item[train] for item in values])
        pooled_truth = np.concatenate([truth[train] for _ in values])
        pooled_zone = np.concatenate([zones[train] for _ in values])
        model = RAHCalibratorV2.fit(
            pooled_scenarios,
            pooled_truth,
            pooled_zone,
            variant="full",
            regularization=float(regularization),
            epochs=int(epochs),
            learning_rate=0.03,
            seed=int(fold_seed) + fold,
            device=device,
        )
        for seed, scenarios in enumerate(values):
            transformed = model.transform(
                scenarios[held], zones[held], strength=float(strength), tail_rule="linear"
            )
            if not np.isfinite(transformed).all():
                raise FloatingPointError("legacy A1 produced non-finite scenarios")
            outputs[seed][held] = transformed
        records.append({"fold": fold, **model.fit_summary.to_dict(), "tail_rule": "linear"})
    return CrossfitResult(outputs, assignments, records)


__all__ = [
    "CrossfitResult",
    "crossfit_c0",
    "crossfit_legacy_linear",
    "fit_c0",
    "transform_c0",
]
