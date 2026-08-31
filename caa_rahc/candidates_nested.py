"""Authoritative strict-nested CAA OOF candidate generation.

The early :mod:`caa_rahc.candidates` implementation correctly withheld the
current calendar dates from ``fit_tail_gate`` itself, but its gate-training
curves came from one global C0 crossfit.  For a gate-training date in fold J,
that global C0 calibrator could have used the current gate-held fold K.  This
module closes that upstream dependency:

* the main five-fold C0 still supplies the held-fold A0 prediction;
* inside every gate fold, an additional C0 crossfit is run using only that
  gate fold's training dates;
* the inner OOF A0 curves and their forecast-only features are the only arrays
  passed to ``fit_tail_gate``; and
* every inner train/held date list is exposed in the returned audit.

The mature streaming/scoring implementation remains in the prototype module
and is invoked through a guarded dependency adapter.  The adapter is restored
in ``finally`` and serialized by a process-local lock, so no patched global
state escapes this call.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

import numpy as np

from rahc.validation import grouped_day_folds

from . import candidates as _prototype
from .baselines import crossfit_c0
from .candidates import (
    FAMILIES,
    CandidateGenerationResult,
    CandidateGrid,
)
from .features import build_features
from .gate import fit_tail_gate
from .selection import CandidateRecord


_ADAPTER_LOCK = threading.RLock()


def _date_fold_audit(
    dates: np.ndarray,
    assignments: np.ndarray,
    outer_held_dates: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(assignments)):
        held = assignments == fold
        training = ~held
        inner_training_dates = np.unique(dates[training])
        inner_held_dates = np.unique(dates[held])
        if np.intersect1d(inner_training_dates, inner_held_dates).size:
            raise AssertionError("inner C0 date leaked between train and held")
        if np.intersect1d(inner_training_dates, outer_held_dates).size:
            raise AssertionError("outer gate-held date entered inner C0 training")
        if np.intersect1d(inner_held_dates, outer_held_dates).size:
            raise AssertionError("outer gate-held date entered inner C0 held data")
        records.append(
            {
                "inner_fold": fold,
                "training_dates": [str(value) for value in inner_training_dates],
                "held_dates": [str(value) for value in inner_held_dates],
                "outer_gate_held_overlap_count": 0,
                "training_cases": int(training.sum()),
                "held_cases": int(held.sum()),
            }
        )
    return records


def _with_nested_metadata(record: CandidateRecord) -> CandidateRecord:
    if record.metadata.get("family") not in {"A3", "A4", "A5", "A6"}:
        return record
    return CandidateRecord(
        name=record.name,
        metrics=record.metrics,
        conditional_ace90=record.conditional_ace90,
        gate_maximum=record.gate_maximum,
        width_cap=record.width_cap,
        shrinkage=record.shrinkage,
        selection_policy=record.selection_policy,
        metadata={
            **dict(record.metadata),
            "gate_training_protocol": (
                "strict outer-date fold plus C0 inner crossfit using outer-training dates only"
            ),
            "outer_gate_held_dates_used_by_inner_C0": 0,
        },
    )


def generate_oof_candidates(
    raw_scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    zone: np.ndarray,
    day: np.ndarray,
    *,
    train_condition: np.ndarray,
    train_target: np.ndarray,
    calibration_condition: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    grid: CandidateGrid | None = None,
    train_day: np.ndarray | None = None,
    a1_scenarios_by_seed: Sequence[np.ndarray] | None = None,
    a1_metadata: Mapping[str, Any] | None = None,
) -> CandidateGenerationResult:
    """Generate A0--A6 records with strict nested C0 gate training.

    The public signature matches the prototype so formal calibration scripts
    can switch imports without changing experiment inputs.
    """

    protocol = grid or CandidateGrid()
    raw = [np.asarray(value) for value in raw_scenarios_by_seed]
    truth = np.asarray(observations)
    zones = np.asarray(zone)
    dates = np.asarray(day).astype("datetime64[D]")
    if len(raw) != 3:
        raise ValueError("the registered nested protocol requires exactly three model seeds")
    if truth.ndim != 2 or truth.shape[1] != 24:
        raise ValueError("observations must have shape [case,24]")
    if dates.shape != (len(truth),) or zones.shape != (len(truth),):
        raise ValueError("day and zone must align with observations")

    outer_assignments = grouped_day_folds(
        dates, folds=int(protocol.folds), seed=int(protocol.fold_seed)
    )
    nested_cache: dict[int, dict[str, Any]] = {}
    nested_audit: list[dict[str, Any]] = []
    adapter_calls: list[dict[str, Any]] = []
    call_index = 0

    def nested_gate_fit(
        scenarios_by_seed: Sequence[np.ndarray],
        fold_observations: np.ndarray,
        features_by_seed: Sequence[np.ndarray],
        fold_zone: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        nonlocal call_index
        # The prototype makes exactly two calls per outer fold: regularized,
        # then no-shrink.  Validate that contract rather than silently guessing.
        outer_fold = call_index // 2
        call_in_fold = call_index % 2
        call_index += 1
        if outer_fold >= int(protocol.folds):
            raise RuntimeError("prototype made more gate-fit calls than the frozen protocol")
        outer_training = outer_assignments != outer_fold
        outer_held = ~outer_training
        expected_cases = int(outer_training.sum())
        if len(fold_observations) != expected_cases or len(fold_zone) != expected_cases:
            raise AssertionError("prototype gate fold does not match strict outer assignment")
        if len(scenarios_by_seed) != 3 or len(features_by_seed) != 3:
            raise AssertionError("gate fit did not pool exactly three model seeds")

        if outer_fold not in nested_cache:
            outer_training_dates = np.unique(dates[outer_training])
            outer_held_dates = np.unique(dates[outer_held])
            if np.intersect1d(outer_training_dates, outer_held_dates).size:
                raise AssertionError("outer gate dates overlap")
            inner_folds = min(int(protocol.folds), int(len(outer_training_dates)))
            if inner_folds < 2:
                raise ValueError("outer gate training needs at least two dates for inner C0")
            inner_seed = int(protocol.fold_seed) + 10_000 + outer_fold
            inner = crossfit_c0(
                [value[outer_training] for value in raw],
                truth[outer_training],
                dates[outer_training],
                folds=inner_folds,
                fold_seed=inner_seed,
                strength=float(protocol.c0_strength),
            )
            inner_features = [
                build_features(raw_seed[outer_training], inner_seed_scenarios)
                for raw_seed, inner_seed_scenarios in zip(raw, inner.scenarios_by_seed)
            ]
            inner_dates = dates[outer_training]
            inner_fold_records = _date_fold_audit(
                inner_dates,
                inner.fold_assignments,
                outer_held_dates,
            )
            audit = {
                "outer_fold": outer_fold,
                "outer_training_dates": [str(value) for value in outer_training_dates],
                "outer_held_dates": [str(value) for value in outer_held_dates],
                "outer_overlap_count": 0,
                "inner_folds": inner_folds,
                "inner_fold_seed": inner_seed,
                "inner_C0_folds": inner_fold_records,
                "outer_held_dates_used_anywhere_in_inner_C0": 0,
                "inner_C0_fit_records": _prototype._json(inner.fit_records),
            }
            nested_cache[outer_fold] = {
                "scenarios": inner.scenarios_by_seed,
                "features": inner_features,
                "observations": truth[outer_training],
                "zone": zones[outer_training],
                "audit": audit,
            }
            nested_audit.append(audit)

        cached = nested_cache[outer_fold]
        regularization = float(kwargs.get("regularization", np.nan))
        expected_regularization = (
            float(protocol.gate_regularization)
            if call_in_fold == 0
            else float(protocol.gate_no_shrink_regularization)
        )
        if not np.isclose(regularization, expected_regularization, rtol=0.0, atol=0.0):
            raise AssertionError("gate fit call order/regularization differs from frozen protocol")
        adapter_calls.append(
            {
                "outer_fold": outer_fold,
                "call_in_fold": call_in_fold,
                "regularization": regularization,
                "training_cases": expected_cases,
                "outer_held_dates_used": 0,
            }
        )
        return fit_tail_gate(
            cached["scenarios"],
            cached["observations"],
            cached["features"],
            cached["zone"],
            **kwargs,
        )

    # The prototype keeps the well-tested streaming score path.  Patch only
    # its gate-fit dependency, under a lock, and restore it even on failure.
    with _ADAPTER_LOCK:
        original_gate_fit = _prototype.fit_tail_gate
        _prototype.fit_tail_gate = nested_gate_fit
        try:
            result = _prototype.generate_oof_candidates(
                raw,
                truth,
                zones,
                dates,
                train_condition=train_condition,
                train_target=train_target,
                calibration_condition=calibration_condition,
                assignments=assignments,
                grid=protocol,
                train_day=train_day,
                a1_scenarios_by_seed=a1_scenarios_by_seed,
                a1_metadata=a1_metadata,
            )
        finally:
            _prototype.fit_tail_gate = original_gate_fit

    expected_calls = int(protocol.folds) * 2
    if call_index != expected_calls:
        raise RuntimeError(
            f"prototype made {call_index} gate fits; frozen protocol requires {expected_calls}"
        )
    if len(nested_audit) != int(protocol.folds):
        raise RuntimeError("not every outer gate fold received an inner C0 crossfit")

    result.records_by_family = {
        family: [_with_nested_metadata(record) for record in records]
        for family, records in result.records_by_family.items()
    }
    result.audit["protocol"]["name"] = (
        "CAA strict nested calibration-only five-date-fold OOF candidate generation v2"
    )
    result.audit["protocol"]["gate_fit"] = (
        "for every outer-held fold, gate-training A0 is inner-cross-fitted using only "
        "outer-training dates; main A0 OOF is used only on the outer-held fold"
    )
    result.audit["protocol"]["authoritative_module"] = "caa_rahc.candidates_nested"
    result.audit["protocol"]["prototype_role"] = (
        "streaming A0-A6 transformation/scoring only; gate-fit dependency replaced under lock"
    )
    result.audit["strict_nested_C0_gate_training"] = nested_audit
    result.audit["strict_gate_adapter_calls"] = adapter_calls
    result.audit["strict_nested_invariant"] = {
        "outer_gate_held_dates_used_anywhere_in_inner_C0": 0,
        "gate_fit_calls": call_index,
        "adapter_restored_after_call": _prototype.fit_tail_gate is original_gate_fit,
    }
    return result


def build_oof_candidates(*args: Any, **kwargs: Any) -> CandidateGenerationResult:
    """Alias for :func:`generate_oof_candidates`."""

    return generate_oof_candidates(*args, **kwargs)


__all__ = [
    "FAMILIES",
    "CandidateGenerationResult",
    "CandidateGrid",
    "build_oof_candidates",
    "generate_oof_candidates",
]
