"""Leakage-safe calibration OOF candidate generation for one outer split.

This module is an orchestration layer.  It fits the train-only structural-zero
model, creates date-cross-fitted A0 scenarios, fits regularized and no-shrink
tail gates on the non-held dates, and streams each ablation/configuration into
daily score sufficient statistics.  Only A0 and the small gate/feature arrays
are retained while the grid is evaluated; full scenario grids are never
stored in the returned result.

Conditional group assignments are supplied by the caller.  Candidate output
therefore cannot choose groups from its own calibrated scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from rahc.evaluation import family_equal_conditional_ace

from .baselines import crossfit_c0
from .features import build_features
from .gate import fit_tail_gate
from .hinge_tail import WidthCapInfeasibleError, atom_aware_logit_hinge_tail
from .metrics import per_date_metrics
from .selection import CandidateRecord, METRIC_KEYS
from .structural_atom import (
    StructuralZeroModel,
    blend_zero_probability,
)


FAMILIES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")


@dataclass(frozen=True)
class CandidateGrid:
    """Frozen candidate and fitting grid, with cheap overrides for tests."""

    atom_strengths: tuple[float, ...] = (0.5, 1.0)
    tail_strengths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    maximum_logit_shifts: tuple[float, ...] = (0.25, 0.5)
    maximum_width_increases: tuple[float, ...] = (0.02, 0.05)
    atom_regularization_c: float = 0.1
    zero_model_maximum_iterations: int = 1000
    anchor: float = 0.10
    maximum_lambda: float = 1.0
    gate_regularization: float = 0.03
    gate_no_shrink_regularization: float = 0.0
    gate_steps: int = 600
    gate_batch_cells: int = 4096
    gate_learning_rate: float = 0.03
    gate_device: str = "cpu"
    folds: int = 5
    fold_seed: int = 20260719
    c0_strength: float = 1.0

    def __post_init__(self) -> None:
        for name, values in (
            ("atom_strengths", self.atom_strengths),
            ("tail_strengths", self.tail_strengths),
            ("maximum_logit_shifts", self.maximum_logit_shifts),
            ("maximum_width_increases", self.maximum_width_increases),
        ):
            if not values:
                raise ValueError(f"{name} must be non-empty")
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any((np.asarray(self.atom_strengths) < 0.0) | (np.asarray(self.atom_strengths) > 1.0)):
            raise ValueError("atom strengths must be in [0,1]")
        if np.any((np.asarray(self.tail_strengths) < 0.0) | (np.asarray(self.tail_strengths) > 1.0)):
            raise ValueError("tail strengths must be in [0,1]")
        if np.any(np.asarray(self.maximum_logit_shifts) < 0.0):
            raise ValueError("maximum logit shifts must be non-negative")
        if np.any(np.asarray(self.maximum_width_increases) < 0.0):
            raise ValueError("maximum width increases must be non-negative")
        if not 0.0 < float(self.anchor) < 0.5:
            raise ValueError("anchor must be in (0,0.5)")
        if self.maximum_lambda < 0.0:
            raise ValueError("maximum_lambda must be non-negative")
        if self.atom_regularization_c <= 0.0:
            raise ValueError("atom_regularization_c must be positive")
        if self.gate_regularization < 0.0 or self.gate_no_shrink_regularization < 0.0:
            raise ValueError("gate regularization values must be non-negative")
        if self.zero_model_maximum_iterations < 1 or self.gate_steps < 1:
            raise ValueError("optimization iteration counts must be positive")
        if self.gate_batch_cells < 1 or self.gate_learning_rate <= 0.0:
            raise ValueError("gate batch size and learning rate must be positive")
        if self.folds != 5:
            raise ValueError("the registered calibration protocol requires exactly five folds")
        if not 0.0 <= self.c0_strength <= 1.0:
            raise ValueError("c0_strength must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_strengths": list(self.atom_strengths),
            "tail_strengths": list(self.tail_strengths),
            "maximum_logit_shifts": list(self.maximum_logit_shifts),
            "maximum_width_increases": list(self.maximum_width_increases),
            "atom_regularization_c": self.atom_regularization_c,
            "zero_model_maximum_iterations": self.zero_model_maximum_iterations,
            "anchor": self.anchor,
            "maximum_lambda": self.maximum_lambda,
            "gate_regularization": self.gate_regularization,
            "gate_no_shrink_regularization": self.gate_no_shrink_regularization,
            "gate_steps": self.gate_steps,
            "gate_batch_cells": self.gate_batch_cells,
            "gate_learning_rate": self.gate_learning_rate,
            "gate_device": self.gate_device,
            "folds": self.folds,
            "fold_seed": self.fold_seed,
            "c0_strength": self.c0_strength,
        }


@dataclass
class CandidateGenerationResult:
    """Candidate records plus fitted train-only model and compact OOF audit."""

    records_by_family: dict[str, list[CandidateRecord]]
    zero_model: StructuralZeroModel
    audit: dict[str, Any]

    def records_for_family(self, family: str) -> list[CandidateRecord]:
        if family not in FAMILIES:
            raise ValueError(f"unknown candidate family {family!r}")
        return list(self.records_by_family.get(family, ()))

    @property
    def all_records(self) -> list[CandidateRecord]:
        return [
            record
            for family in FAMILIES
            for record in self.records_by_family.get(family, ())
        ]

    @property
    def primary_constrained_records(self) -> list[CandidateRecord]:
        """The only records permitted in the primary constrained selector."""

        return self.records_for_family("A0") + self.records_for_family("A4")

    @property
    def primary_unconstrained_records(self) -> list[CandidateRecord]:
        """The only records permitted in the A5 unconstrained selector."""

        return self.records_for_family("A0") + self.records_for_family("A5")


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _token(value: float) -> str:
    return np.format_float_positional(float(value), trim="-").replace("-", "m").replace(".", "p")


def _assignment_fingerprint(assignments: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(assignments):
        array = np.ascontiguousarray(np.asarray(assignments[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_inputs(
    raw_scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    zone: np.ndarray,
    day: np.ndarray,
    train_condition: np.ndarray,
    train_target: np.ndarray,
    calibration_condition: np.ndarray,
    assignments: Mapping[str, np.ndarray],
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    raw = [np.asarray(value) for value in raw_scenarios_by_seed]
    if len(raw) != 3:
        raise ValueError("the registered experiment requires exactly three model seeds")
    shape = raw[0].shape
    if len(shape) != 3 or shape[2] != 24 or shape[1] < 4:
        raise ValueError("raw scenarios must have shape [case,member,24] with >=4 members")
    if any(value.shape != shape for value in raw):
        raise ValueError("all three raw scenario archives must align")
    if any(not np.isfinite(value).all() for value in raw):
        raise ValueError("raw scenarios contain non-finite values")
    truth = np.asarray(observations, dtype=np.float64)
    zones = np.asarray(zone, dtype=np.int64)
    dates = np.asarray(day).astype("datetime64[D]")
    condition = np.asarray(calibration_condition)
    if truth.shape != (shape[0], 24):
        raise ValueError("observations must align as [case,24]")
    if zones.shape != (shape[0],) or np.any((zones < 1) | (zones > 10)):
        raise ValueError("zone must align and contain values 1..10")
    if dates.shape != (shape[0],):
        raise ValueError("day must align with cases")
    if len(np.unique(dates)) < 5:
        raise ValueError("calibration needs at least five unique dates")
    if condition.shape != (shape[0], 24, 20):
        raise ValueError("calibration_condition must have shape [case,24,20]")
    train_values = np.asarray(train_condition)
    train_truth = np.asarray(train_target)
    if train_values.ndim != 3 or train_values.shape[1:] != (24, 20):
        raise ValueError("train_condition must have shape [case,24,20]")
    if train_truth.shape != train_values.shape[:2]:
        raise ValueError("train_target must align with train_condition")
    if not assignments:
        raise ValueError("caller must supply at least one frozen conditional family")
    checked_assignments: dict[str, np.ndarray] = {}
    for name, values in assignments.items():
        array = np.asarray(values)
        if array.shape != truth.shape:
            raise ValueError(f"assignment family {name!r} must have shape {truth.shape}")
        checked_assignments[str(name)] = array.copy()
    return raw, truth, zones, dates, condition, checked_assignments


def _record_from_scenarios(
    name: str,
    family: str,
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    day: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    *,
    selection_policy: str,
    gate_maximum: float,
    width_cap: float,
    shrinkage: float,
    metadata: Mapping[str, Any],
) -> CandidateRecord:
    """Score caller-provided A0/A1 arrays; no scenario is copied into output."""

    metric_parts = {metric: [] for metric in METRIC_KEYS}
    ace: list[float] = []
    expected_dates = np.unique(np.asarray(day).astype("datetime64[D]"))
    for values in scenarios_by_seed:
        scored = per_date_metrics(values, observations, day)
        if not np.array_equal(scored["dates"], expected_dates):
            raise AssertionError("daily metric ordering changed")
        for metric in METRIC_KEYS:
            metric_parts[metric].append(scored[metric])
        ace.append(
            family_equal_conditional_ace(
                values, observations, dict(assignments), nominal=0.90
            )
        )
    return CandidateRecord(
        name=name,
        metrics={metric: np.stack(parts) for metric, parts in metric_parts.items()},
        conditional_ace90=np.asarray(ace, dtype=np.float64),
        gate_maximum=float(gate_maximum),
        width_cap=float(width_cap),
        shrinkage=float(shrinkage),
        selection_policy=selection_policy,
        metadata={"family": family, **dict(metadata)},
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
    """Generate calibration-only OOF records for one sealed outer split.

    The structural-zero model is fitted once using only ``train_condition``
    and ``train_target``.  C0 and both tail gates are cross-fitted by calendar
    date.  The held dates are never passed to their fold's C0/gate fit.
    ``assignments`` are only read during scoring and are never derived from a
    candidate's scenarios.
    """

    protocol = grid or CandidateGrid()
    raw, truth, zones, dates, condition, frozen_assignments = _validate_inputs(
        raw_scenarios_by_seed,
        observations,
        zone,
        day,
        train_condition,
        train_target,
        calibration_condition,
        assignments,
    )
    train_dates_audit: dict[str, Any]
    if train_day is not None:
        train_dates = np.asarray(train_day).astype("datetime64[D]")
        if train_dates.shape != (len(np.asarray(train_condition)),):
            raise ValueError("train_day must align with train_condition cases")
        overlap = np.intersect1d(np.unique(train_dates), np.unique(dates))
        if overlap.size:
            raise ValueError("structural-zero train dates overlap calibration dates")
        train_dates_audit = {
            "provided": True,
            "unique_dates": int(len(np.unique(train_dates))),
            "calibration_overlap_count": 0,
        }
    else:
        train_dates_audit = {
            "provided": False,
            "calibration_overlap_count": "not independently auditable without train_day",
        }

    zero_model = StructuralZeroModel.fit(
        np.asarray(train_condition),
        np.asarray(train_target),
        regularization_c=float(protocol.atom_regularization_c),
        maximum_iterations=int(protocol.zero_model_maximum_iterations),
    )
    structural_zero = zero_model.predict_zero(condition)

    c0 = crossfit_c0(
        raw,
        truth,
        dates,
        folds=int(protocol.folds),
        fold_seed=int(protocol.fold_seed),
        strength=float(protocol.c0_strength),
    )
    a0 = c0.scenarios_by_seed
    fold_assignments = c0.fold_assignments
    features = [build_features(raw_seed, a0_seed) for raw_seed, a0_seed in zip(raw, a0)]

    gate_arrays = {
        "regularized": [np.empty((len(truth), 24, 2), dtype=np.float64) for _ in raw],
        "no_shrink": [np.empty((len(truth), 24, 2), dtype=np.float64) for _ in raw],
    }
    gate_summaries: dict[str, list[dict[str, Any]]] = {
        "regularized": [],
        "no_shrink": [],
    }
    fold_audit: list[dict[str, Any]] = []
    for fold in range(int(protocol.folds)):
        training = fold_assignments != fold
        held = ~training
        training_dates = np.unique(dates[training])
        held_dates = np.unique(dates[held])
        overlap = np.intersect1d(training_dates, held_dates)
        if overlap.size:
            raise AssertionError("calendar date leaked between gate train and held fold")
        fold_record: dict[str, Any] = {
            "fold": fold,
            "training_dates": [str(value) for value in training_dates],
            "held_dates": [str(value) for value in held_dates],
            "overlap_count": 0,
            "training_cases": int(training.sum()),
            "held_cases": int(held.sum()),
            "gate_models": {},
        }
        for label, regularization in (
            ("regularized", protocol.gate_regularization),
            ("no_shrink", protocol.gate_no_shrink_regularization),
        ):
            fitted = fit_tail_gate(
                [values[training] for values in a0],
                truth[training],
                [value[training] for value in features],
                zones[training],
                anchor=float(protocol.anchor),
                maximum_lambda=float(protocol.maximum_lambda),
                regularization=float(regularization),
                steps=int(protocol.gate_steps),
                batch_cells=int(protocol.gate_batch_cells),
                learning_rate=float(protocol.gate_learning_rate),
                random_seed=int(protocol.fold_seed) + 1000 * (label == "no_shrink") + fold,
                device=protocol.gate_device,
            )
            for seed_index in range(3):
                gate_arrays[label][seed_index][held] = fitted.predict(
                    features[seed_index][held], zones[held], strength=1.0
                )
            summary = {
                "fold": fold,
                "regularization": float(regularization),
                **_json(fitted.fit_summary),
            }
            gate_summaries[label].append(summary)
            fold_record["gate_models"][label] = summary
        fold_audit.append(fold_record)
    for label, values_by_seed in gate_arrays.items():
        if any(not np.isfinite(value).all() for value in values_by_seed):
            raise FloatingPointError(f"{label} OOF gate contains unfilled/non-finite cells")

    records: dict[str, list[CandidateRecord]] = {family: [] for family in FAMILIES}
    catalog: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    a0_record = _record_from_scenarios(
        "A0",
        "A0",
        a0,
        truth,
        dates,
        frozen_assignments,
        selection_policy="baseline",
        gate_maximum=0.0,
        width_cap=0.0,
        shrinkage=0.0,
        metadata={
            "definition": "five-date-fold cross-fitted C0 linear empirical calibration",
            "fold_seed": int(protocol.fold_seed),
        },
    )
    records["A0"].append(a0_record)
    catalog.append({"name": "A0", "family": "A0", "status": "valid"})

    if a1_scenarios_by_seed is not None:
        a1_values = [np.asarray(value) for value in a1_scenarios_by_seed]
        if len(a1_values) != 3 or any(value.shape != raw[0].shape for value in a1_values):
            raise ValueError("caller-provided A1 needs three aligned OOF scenario arrays")
        a1_record = _record_from_scenarios(
            "A1",
            "A1",
            a1_values,
            truth,
            dates,
            frozen_assignments,
            selection_policy="constrained",
            gate_maximum=0.0,
            width_cap=0.0,
            shrinkage=0.0,
            metadata={
                "definition": "caller-provided legacy linear RAHC OOF ablation",
                "caller_responsible_for_date_crossfit": True,
                **dict(a1_metadata or {}),
            },
        )
        records["A1"].append(a1_record)
        catalog.append({"name": "A1", "family": "A1", "status": "valid"})

    unique_dates = np.unique(dates)
    blended_zero: dict[float, list[np.ndarray]] = {
        float(strength): [
            blend_zero_probability(a0_seed, structural_zero, float(strength))
            for a0_seed in a0
        ]
        for strength in protocol.atom_strengths
    }
    rho1 = float(zero_model.upper_conditional_probability)

    def stream_candidate(
        *,
        name: str,
        family: str,
        atom_strength: float,
        tail_strength: float,
        dmax: float,
        width_cap: float,
        gate_kind: str,
        atom_enabled: bool,
        tail_enabled: bool,
        selection_policy: str,
        shrinkage: float,
    ) -> CandidateRecord | None:
        metric_parts = {metric: [] for metric in METRIC_KEYS}
        ace_values: list[float] = []
        diagnostic_summaries: list[dict[str, Any]] = []
        config = {
            "atom_strength": float(atom_strength),
            "tail_strength": float(tail_strength),
            "dmax": float(dmax),
            "width_delta_cap": float(width_cap),
            "width_cap_reference": "atom_only",
            "gate_kind": gate_kind,
            "atom_enabled": bool(atom_enabled),
            "tail_enabled": bool(tail_enabled),
        }
        for seed_index, baseline in enumerate(a0):
            if atom_enabled:
                pi0 = blended_zero[float(atom_strength)][seed_index]
                upper = rho1
            else:
                pi0 = 0.0
                upper = 0.0
            if tail_enabled:
                gate = gate_arrays[gate_kind][seed_index] * float(tail_strength)
                lower_gate = gate[..., 0]
                upper_gate = gate[..., 1]
            else:
                lower_gate = 0.0
                upper_gate = 0.0
            try:
                transformed = atom_aware_logit_hinge_tail(
                    baseline,
                    pi0,
                    upper,
                    lower_gate,
                    upper_gate,
                    dmax=float(dmax),
                    width_delta_cap=float(width_cap),
                    width_cap_reference="atom_only",
                    atom_enabled=atom_enabled,
                    tail_enabled=tail_enabled,
                    anchor=float(protocol.anchor),
                )
            except WidthCapInfeasibleError as error:
                failure = {
                    "name": name,
                    "family": family,
                    "status": "invalid",
                    "reason": "WidthCapInfeasibleError",
                    "message": str(error),
                    "model_seed_index": seed_index,
                    "config": config,
                    "silently_replaced": False,
                }
                invalid.append(failure)
                catalog.append(failure)
                return None
            scenarios = transformed.atom_only if family == "A2" else transformed.calibrated
            scored = per_date_metrics(scenarios, truth, dates)
            if not np.array_equal(scored["dates"], unique_dates):
                raise AssertionError("candidate daily date ordering changed")
            for metric in METRIC_KEYS:
                metric_parts[metric].append(scored[metric])
            ace_values.append(
                family_equal_conditional_ace(
                    scenarios, truth, dict(frozen_assignments), nominal=0.90
                )
            )
            diagnostic_summaries.append(
                {"model_seed_index": seed_index, **transformed.diagnostics.summary()}
            )
            # Do not retain this grid member's full scenarios after scoring.
            del scenarios, transformed
        candidate = CandidateRecord(
            name=name,
            metrics={metric: np.stack(parts) for metric, parts in metric_parts.items()},
            conditional_ace90=np.asarray(ace_values, dtype=np.float64),
            gate_maximum=(
                float(protocol.maximum_lambda) * float(tail_strength)
                if tail_enabled
                else 0.0
            ),
            width_cap=float(width_cap),
            shrinkage=float(shrinkage),
            selection_policy=selection_policy,
            metadata={
                "family": family,
                "config": config,
                "transformation_diagnostics_by_seed": diagnostic_summaries,
                "scenario_storage": "streamed per seed; not retained after daily scoring",
                "conditional_assignments": "caller supplied and method independent",
            },
        )
        records[family].append(candidate)
        catalog.append({"name": name, "family": family, "status": "valid", "config": config})
        return candidate

    # A2: structural atom mixture only.  The local cap is relative to that
    # atom-only curve and is zero because no tail shift is requested.
    for atom_strength in protocol.atom_strengths:
        name = f"A2_atom{_token(atom_strength)}"
        stream_candidate(
            name=name,
            family="A2",
            atom_strength=float(atom_strength),
            tail_strength=0.0,
            dmax=0.0,
            width_cap=0.0,
            gate_kind="regularized",
            atom_enabled=True,
            tail_enabled=False,
            selection_policy="constrained",
            shrinkage=float(protocol.atom_regularization_c),
        )

    # A3: regularized local tail only, without atom changes.
    for tail_strength in protocol.tail_strengths:
        for dmax in protocol.maximum_logit_shifts:
            for width_cap in protocol.maximum_width_increases:
                name = (
                    f"A3_tail{_token(tail_strength)}_d{_token(dmax)}_w{_token(width_cap)}"
                )
                stream_candidate(
                    name=name,
                    family="A3",
                    atom_strength=0.0,
                    tail_strength=float(tail_strength),
                    dmax=float(dmax),
                    width_cap=float(width_cap),
                    gate_kind="regularized",
                    atom_enabled=False,
                    tail_enabled=True,
                    selection_policy="constrained",
                    shrinkage=float(protocol.gate_regularization),
                )

    # A4 is the primary full constrained family.  A5 mirrors each valid A4
    # score/config exactly but is labelled unconstrained for a separate path.
    for atom_strength in protocol.atom_strengths:
        for tail_strength in protocol.tail_strengths:
            for dmax in protocol.maximum_logit_shifts:
                for width_cap in protocol.maximum_width_increases:
                    suffix = (
                        f"atom{_token(atom_strength)}_tail{_token(tail_strength)}_"
                        f"d{_token(dmax)}_w{_token(width_cap)}"
                    )
                    a4 = stream_candidate(
                        name=f"A4_{suffix}",
                        family="A4",
                        atom_strength=float(atom_strength),
                        tail_strength=float(tail_strength),
                        dmax=float(dmax),
                        width_cap=float(width_cap),
                        gate_kind="regularized",
                        atom_enabled=True,
                        tail_enabled=True,
                        selection_policy="constrained",
                        shrinkage=float(protocol.gate_regularization),
                    )
                    if a4 is not None:
                        mirrored = CandidateRecord(
                            name=f"A5_{suffix}",
                            metrics={
                                metric: np.asarray(a4.metrics[metric]).copy()
                                for metric in METRIC_KEYS
                            },
                            conditional_ace90=np.asarray(a4.conditional_ace90).copy(),
                            gate_maximum=a4.gate_maximum,
                            width_cap=a4.width_cap,
                            shrinkage=a4.shrinkage,
                            selection_policy="unconstrained",
                            metadata={
                                "family": "A5",
                                "mirrors_A4_record": a4.name,
                                "config": dict(a4.metadata["config"]),
                                "definition": "same OOF candidate as A4; constraints ignored only in selection",
                            },
                        )
                        records["A5"].append(mirrored)
                        catalog.append(
                            {
                                "name": mirrored.name,
                                "family": "A5",
                                "status": "valid",
                                "mirrors_A4_record": a4.name,
                            }
                        )

    # A6: identical full grid with the no-shrink gate fit.
    for atom_strength in protocol.atom_strengths:
        for tail_strength in protocol.tail_strengths:
            for dmax in protocol.maximum_logit_shifts:
                for width_cap in protocol.maximum_width_increases:
                    name = (
                        f"A6_atom{_token(atom_strength)}_tail{_token(tail_strength)}_"
                        f"d{_token(dmax)}_w{_token(width_cap)}"
                    )
                    stream_candidate(
                        name=name,
                        family="A6",
                        atom_strength=float(atom_strength),
                        tail_strength=float(tail_strength),
                        dmax=float(dmax),
                        width_cap=float(width_cap),
                        gate_kind="no_shrink",
                        atom_enabled=True,
                        tail_enabled=True,
                        selection_policy="constrained",
                        shrinkage=float(protocol.gate_no_shrink_regularization),
                    )

    audit = {
        "protocol": {
            "name": "CAA calibration-only five-date-fold OOF candidate generation v1",
            "grid": protocol.to_dict(),
            "model_seeds": 3,
            "unique_calibration_dates": int(len(np.unique(dates))),
            "cases": int(len(truth)),
            "hours": 24,
            "members": int(raw[0].shape[1]),
            "A0_definition": "crossfit_c0 linear empirical, fitted separately per seed",
            "gate_fit": "pooled three-seed A0 OOF curves/features on non-held dates",
            "width_cap_reference": "atom_only for local transformation",
            "global_atom_width_guard": "selector point and U95 width-vs-A0 constraints",
            "scenario_grid_storage": "stream one candidate seed at a time; no returned grid scenarios",
            "conditional_assignments_source": "caller supplied",
            "assignment_fingerprint_sha256": _assignment_fingerprint(frozen_assignments),
            "assignment_families": sorted(frozen_assignments),
            "primary_constrained_filter": ["A0", "A4"],
            "primary_unconstrained_filter": ["A0", "A5"],
        },
        "zero_model": {
            **zero_model.to_dict(),
            "train_date_audit": train_dates_audit,
            "calibration_prediction_summary": {
                "mean": float(structural_zero.mean()),
                "minimum": float(structural_zero.min()),
                "maximum": float(structural_zero.max()),
            },
            "calibration_analytic_scores_descriptive_only": zero_model.analytic_scores(
                condition, truth
            ),
        },
        "folds": fold_audit,
        "c0_fit_records": _json(c0.fit_records),
        "gate_summaries": _json(gate_summaries),
        "candidate_counts": {
            family: int(len(records[family])) for family in FAMILIES
        },
        "catalog": _json(catalog),
        "invalid_candidates": _json(invalid),
        "invalid_handling": (
            "WidthCapInfeasibleError is recorded as invalid and never replaced by A0 or another curve"
        ),
    }
    return CandidateGenerationResult(records, zero_model, audit)


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
