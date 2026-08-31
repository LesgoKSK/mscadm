"""Atom-aware asymmetric logit-hinge tail transformation.

This module is deliberately independent from model fitting.  It consumes a
baseline ensemble and analytic state probabilities and returns both the
atom-only ensemble and the locally tail-adjusted ensemble.  All arithmetic is
performed in float64 and all invalid numerical states are rejected explicitly.

``pi0`` is the unconditional zero-atom probability.  ``rho1`` is the upper
atom probability conditional on a non-zero realization, so the unconditional
upper mass is ``(1 - pi0) * rho1``.  The remaining mass is represented by the
empirical distribution of strictly interior baseline members.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import expit


HINGE_ANCHOR = 0.10
DEFAULT_EPSILON = 1e-6
WidthCapReference = Literal["baseline", "atom_only"]


class WidthCapInfeasibleError(ValueError):
    """Raised when the requested atoms alone violate the per-cell W90 cap."""


@dataclass(frozen=True)
class HingeTailDiagnostics:
    """Auditable transformation diagnostics.

    Array-valued fields have shape ``[case, hour]``.  ``summary`` returns a
    compact JSON-serializable view without discarding the arrays stored here.
    """

    requested_pi0: np.ndarray
    requested_rho1: np.ndarray
    requested_pi1: np.ndarray
    realized_pi0: np.ndarray
    realized_pi1: np.ndarray
    baseline_width_90: np.ndarray
    atom_only_width_90: np.ndarray
    calibrated_width_90: np.ndarray
    tail_scale: np.ndarray
    atom_enabled: bool
    tail_enabled: bool
    members: int
    anchor: float
    dmax: float
    width_delta_cap: float
    width_cap_reference: WidthCapReference
    width_cap_reference_90: np.ndarray
    member_probability_resolution: float
    midpoint_atom_quantization_bound: float
    atom_quantization_applicable: bool
    is_hundred_member_ensemble: bool
    zero_atom_quantization_mae: float
    zero_atom_quantization_max_abs: float
    one_atom_quantization_mae: float
    one_atom_quantization_max_abs: float
    atom_preprojection_adjacent_violations: int
    atom_preprojection_max_violation: float
    tail_preprojection_adjacent_violations: int
    tail_preprojection_max_violation: float
    requested_max_abs_logit_shift: float
    applied_max_abs_logit_shift: float
    logit_cap_hit_values: int
    boundary_shift_skipped_values: int
    width_cap_activated_cells: int
    width_cap_infeasible_cells: int
    maximum_final_width_cap_excess: float
    central_values_changed_by_tail: int
    strict_reversals: int
    nonfinite_input_values: int
    nonfinite_intermediate_values: int

    def summary(self) -> dict[str, Any]:
        """Return aggregate diagnostics suitable for JSON metadata."""

        return {
            "atom_enabled": self.atom_enabled,
            "tail_enabled": self.tail_enabled,
            "members": self.members,
            "anchor": self.anchor,
            "dmax": self.dmax,
            "width_delta_cap": self.width_delta_cap,
            "width_cap_reference": self.width_cap_reference,
            "member_probability_resolution": self.member_probability_resolution,
            "midpoint_atom_quantization_bound": self.midpoint_atom_quantization_bound,
            "atom_quantization_applicable": self.atom_quantization_applicable,
            "is_hundred_member_ensemble": self.is_hundred_member_ensemble,
            "requested_pi0_mean": float(self.requested_pi0.mean()),
            "requested_pi1_mean": float(self.requested_pi1.mean()),
            "realized_pi0_mean": float(self.realized_pi0.mean()),
            "realized_pi1_mean": float(self.realized_pi1.mean()),
            "zero_atom_quantization_mae": self.zero_atom_quantization_mae,
            "zero_atom_quantization_max_abs": self.zero_atom_quantization_max_abs,
            "one_atom_quantization_mae": self.one_atom_quantization_mae,
            "one_atom_quantization_max_abs": self.one_atom_quantization_max_abs,
            "atom_preprojection_adjacent_violations": (
                self.atom_preprojection_adjacent_violations
            ),
            "atom_preprojection_max_violation": self.atom_preprojection_max_violation,
            "tail_preprojection_adjacent_violations": (
                self.tail_preprojection_adjacent_violations
            ),
            "tail_preprojection_max_violation": self.tail_preprojection_max_violation,
            "requested_max_abs_logit_shift": self.requested_max_abs_logit_shift,
            "applied_max_abs_logit_shift": self.applied_max_abs_logit_shift,
            "logit_cap_hit_values": self.logit_cap_hit_values,
            "boundary_shift_skipped_values": self.boundary_shift_skipped_values,
            "width_cap_activated_cells": self.width_cap_activated_cells,
            "width_cap_infeasible_cells": self.width_cap_infeasible_cells,
            "maximum_final_width_cap_excess": self.maximum_final_width_cap_excess,
            "maximum_atom_only_width_change": float(
                np.max(self.atom_only_width_90 - self.baseline_width_90)
            ),
            "maximum_calibrated_width_change": float(
                np.max(self.calibrated_width_90 - self.baseline_width_90)
            ),
            "maximum_atom_only_width_change_from_baseline": float(
                np.max(self.atom_only_width_90 - self.baseline_width_90)
            ),
            "maximum_final_width_change_from_baseline": float(
                np.max(self.calibrated_width_90 - self.baseline_width_90)
            ),
            "maximum_final_width_change_from_atom_only": float(
                np.max(self.calibrated_width_90 - self.atom_only_width_90)
            ),
            "central_values_changed_by_tail": self.central_values_changed_by_tail,
            "strict_reversals": self.strict_reversals,
            "nonfinite_input_values": self.nonfinite_input_values,
            "nonfinite_intermediate_values": self.nonfinite_intermediate_values,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`summary` used by experiment metadata writers."""

        return self.summary()


@dataclass(frozen=True)
class HingeTailResult:
    """Atom-only and atom-plus-tail scenarios with their diagnostics."""

    atom_only: np.ndarray
    calibrated: np.ndarray
    diagnostics: HingeTailDiagnostics


def _as_cell_array(value: np.ndarray | float, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=np.float64)
    elif array.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    else:
        array = array.copy()
    if not np.isfinite(array).all():
        count = int(array.size - np.isfinite(array).sum())
        raise FloatingPointError(f"{name} contains {count} non-finite values")
    return array


def _validate_probability(array: np.ndarray, name: str) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        minimum = float(array.min())
        maximum = float(array.max())
        raise ValueError(f"{name} must lie in [0,1], found range [{minimum},{maximum}]")


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    case = np.arange(values.shape[0])[:, None, None]
    hour = np.arange(values.shape[2])[None, None, :]
    ranks[case, order, hour] = np.arange(values.shape[1])[None, :, None]
    return ranks


def _width_90(sorted_values: np.ndarray) -> float:
    quantiles = np.quantile(sorted_values, (0.05, 0.95))
    return float(quantiles[1] - quantiles[0])


def _shifted_value(
    atom_curve: np.ndarray,
    shift: np.ndarray,
    index: int,
    scale: float,
) -> float:
    value = float(atom_curve[index])
    displacement = float(scale) * float(shift[index])
    if displacement == 0.0 or value == 0.0 or value == 1.0:
        return value
    z = np.log(value) - np.log1p(-value) + displacement
    result = float(expit(z))
    if not np.isfinite(result):
        raise FloatingPointError("non-finite shifted quantile value")
    return result


def _shifted_width_90(
    atom_curve: np.ndarray, shift: np.ndarray, scale: float
) -> float:
    """Evaluate W90 from four order statistics without materializing a curve."""

    values: list[float] = []
    last = len(atom_curve) - 1
    for probability in (0.05, 0.95):
        position = last * probability
        left = int(np.floor(position))
        right = int(np.ceil(position))
        weight = position - left
        left_value = _shifted_value(atom_curve, shift, left, scale)
        right_value = _shifted_value(atom_curve, shift, right, scale)
        values.append((1.0 - weight) * left_value + weight * right_value)
    return float(values[1] - values[0])


def _batch_width_cap_scales(
    atom_curves: np.ndarray,
    shifts: np.ndarray,
    limits: np.ndarray,
    *,
    iterations: int = 56,
) -> np.ndarray:
    """Solve independent W90 caps using four fixed order statistics in batch."""

    if atom_curves.ndim != 2 or shifts.shape != atom_curves.shape:
        raise ValueError("batched atom curves and shifts must share [cell,member] shape")
    if limits.shape != (len(atom_curves),):
        raise ValueError("batched width limits must have one value per cell")
    members = atom_curves.shape[1]
    last = members - 1
    position = last * np.asarray((0.05, 0.95), dtype=np.float64)
    left = np.floor(position).astype(np.int64)
    right = np.ceil(position).astype(np.int64)
    weight = position - left
    indices = np.asarray((left[0], right[0], left[1], right[1]), dtype=np.int64)
    base = np.take(atom_curves, indices, axis=1)
    displacement = np.take(shifts, indices, axis=1)
    active = (displacement != 0.0) & (base > 0.0) & (base < 1.0)
    z0 = np.zeros_like(base)
    z0[active] = np.log(base[active]) - np.log1p(-base[active])
    if not np.isfinite(z0).all():
        raise FloatingPointError("non-finite batched width-cap logits")

    low = np.zeros(len(atom_curves), dtype=np.float64)
    high = np.ones(len(atom_curves), dtype=np.float64)
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        values = base.copy()
        shifted = z0 + middle[:, None] * displacement
        values[active] = expit(shifted[active])
        lower = (1.0 - weight[0]) * values[:, 0] + weight[0] * values[:, 1]
        upper = (1.0 - weight[1]) * values[:, 2] + weight[1] * values[:, 3]
        feasible = (upper - lower) <= limits
        low[feasible] = middle[feasible]
        high[~feasible] = middle[~feasible]
    return low

def _monotonicity_diagnostics(values: np.ndarray) -> tuple[int, float]:
    difference = np.diff(values)
    selected = difference < 0.0
    if not np.any(selected):
        return 0, 0.0
    return int(selected.sum()), float(np.max(-difference[selected]))


def _safe_logit(values: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    result = np.log(clipped) - np.log1p(-clipped)
    if not np.isfinite(result).all():
        count = int(result.size - np.isfinite(result).sum())
        raise FloatingPointError(f"logit transformation produced {count} non-finite values")
    return result


def _cell_atom_curve(
    sorted_baseline: np.ndarray,
    quantile_grid: np.ndarray,
    pi0: float,
    pi1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an atom mixture curve and its continuous probability positions.

    When the requested hurdle needs continuous positions but the baseline cell
    has no strictly interior member, its empirical conditional distribution is
    undefined. That cell falls back to the exact baseline curve with no hinge.
    """

    lower_atom = quantile_grid <= pi0
    # At an exact discontinuity q == pi0 == 1-pi1, the generalized inverse is
    # the lower atom.  The strict upper comparison avoids overlapping masks.
    upper_atom = (quantile_grid > 1.0 - pi1) & ~lower_atom
    continuous = ~(lower_atom | upper_atom)
    curve = np.empty_like(quantile_grid)
    curve[lower_atom] = 0.0
    curve[upper_atom] = 1.0
    continuous_probability = np.full_like(quantile_grid, np.nan)
    if np.any(continuous):
        continuous_mass = 1.0 - pi0 - pi1
        if continuous_mass <= 0.0:
            raise ValueError("positive continuous ensemble positions require positive continuous mass")
        interior = sorted_baseline[(sorted_baseline > 0.0) & (sorted_baseline < 1.0)]
        if len(interior) == 0:
            curve = sorted_baseline.copy()
            lower_atom = curve == 0.0
            upper_atom = curve == 1.0
            return curve, continuous_probability, lower_atom, upper_atom
        source = (np.arange(len(interior), dtype=np.float64) + 0.5) / len(interior)
        qc = (quantile_grid[continuous] - pi0) / continuous_mass
        if np.any((qc < 0.0) | (qc > 1.0)) or not np.isfinite(qc).all():
            raise FloatingPointError("invalid conditional continuous probabilities")
        continuous_probability[continuous] = qc
        curve[continuous] = np.interp(qc, source, interior)
    return curve, continuous_probability, lower_atom, upper_atom


def _tail_shift(
    atom_curve: np.ndarray,
    continuous_probability: np.ndarray,
    continuous: np.ndarray,
    lower_gate: float,
    upper_gate: float,
    *,
    anchor: float,
    dmax: float,
    epsilon: float,
) -> tuple[np.ndarray, float, int]:
    """Return frozen logit-hinge dilation, its raw maximum, and cap hits."""

    shift = np.zeros_like(continuous_probability)
    if not np.any(continuous):
        return shift, 0.0, 0
    probability = continuous_probability[continuous]
    z0 = _safe_logit(atom_curve[continuous], epsilon)
    if np.any(np.diff(probability) < 0.0) or np.any(np.diff(z0) < 0.0):
        raise RuntimeError("continuous atom curve is not monotone before hinge dilation")
    lower_anchor = float(np.interp(anchor, probability, z0))
    upper_anchor = float(np.interp(1.0 - anchor, probability, z0))
    lower = continuous & (continuous_probability < anchor)
    upper = continuous & (continuous_probability > 1.0 - anchor)
    if np.any(lower):
        lower_z = _safe_logit(atom_curve[lower], epsilon)
        shift[lower] = -float(lower_gate) * np.maximum(lower_anchor - lower_z, 0.0)
    if np.any(upper):
        upper_z = _safe_logit(atom_curve[upper], epsilon)
        shift[upper] = float(upper_gate) * np.maximum(upper_z - upper_anchor, 0.0)
    raw_maximum = float(np.max(np.abs(shift))) if shift.size else 0.0
    cap_hits = int(np.sum(np.abs(shift) > dmax)) if dmax > 0.0 else int(np.sum(shift != 0.0))
    return np.clip(shift, -dmax, dmax), raw_maximum, cap_hits


def _apply_shift(
    atom_curve: np.ndarray,
    shift: np.ndarray,
    scale: float,
    *,
    epsilon: float,
) -> tuple[np.ndarray, int, float]:
    result = atom_curve.copy()
    selected = shift != 0.0
    interior_value = selected & (atom_curve > 0.0) & (atom_curve < 1.0)
    skipped = int(np.sum(selected & ~interior_value))
    applied_max = 0.0
    if np.any(interior_value):
        applied = float(scale) * shift[interior_value]
        result[interior_value] = expit(
            _safe_logit(atom_curve[interior_value], epsilon) + applied
        )
        applied_max = float(np.max(np.abs(applied)))
    if not np.isfinite(result).all():
        count = int(result.size - np.isfinite(result).sum())
        raise FloatingPointError(f"tail mapping produced {count} non-finite values")
    return result, skipped, applied_max


def atom_aware_logit_hinge_tail(
    baseline_scenarios: np.ndarray,
    pi0: np.ndarray | float,
    rho1: np.ndarray | float,
    lower_gate: np.ndarray | float,
    upper_gate: np.ndarray | float,
    *,
    dmax: float,
    width_delta_cap: float,
    width_cap_reference: WidthCapReference = "baseline",
    atom_enabled: bool = True,
    tail_enabled: bool = True,
    anchor: float = HINGE_ANCHOR,
    epsilon: float = DEFAULT_EPSILON,
) -> HingeTailResult:
    """Create atom-only and asymmetric logit-hinge calibrated scenarios.

    The hinge is exactly zero for conditional continuous probabilities in
    ``[anchor, 1-anchor]``.  If ``z0`` denotes the atom-only continuous curve
    in logit space, its frozen dilation is

    ``-lower_gate * [z0(anchor) - z0(q)]_+`` below the lower anchor and
    ``+upper_gate * [z0(q) - z0(1-anchor)]_+`` above the upper anchor.

    Each shift is then clipped to ``[-dmax, dmax]``.  A per-cell bisection
    scales only these tail shifts until

    ``W90(calibrated) <= W90(reference) + width_delta_cap``.

    ``width_cap_reference="baseline"`` retains the strict original contract.
    If the requested atoms already violate that baseline cap, no tail scale can
    repair the conflict without changing the atom mixture, so
    :class:`WidthCapInfeasibleError` is raised.  With reference ``"atom_only"``
    the local cap isolates widening caused by the hinge; atom-induced widening
    remains visible in diagnostics for the outer selector to constrain against
    A0.  When atoms are disabled and both gates are zero, both returned
    ensembles are an exact float64 copy of the baseline.
    """

    baseline_input = np.asarray(baseline_scenarios)
    if baseline_input.ndim != 3:
        raise ValueError("baseline_scenarios must have shape [case,member,hour]")
    cases, members, hours = baseline_input.shape
    if cases < 1 or members < 2 or hours < 1:
        raise ValueError("baseline dimensions must be non-empty with at least two members")
    nonfinite_inputs = int(baseline_input.size - np.isfinite(baseline_input).sum())
    if nonfinite_inputs:
        raise FloatingPointError(
            f"baseline_scenarios contains {nonfinite_inputs} non-finite values"
        )
    baseline = baseline_input.astype(np.float64, copy=True)
    if np.any((baseline < 0.0) | (baseline > 1.0)):
        raise ValueError("baseline_scenarios must lie inside the physical interval [0,1]")
    if not np.isfinite(dmax) or float(dmax) < 0.0:
        raise ValueError("dmax must be finite and non-negative")
    if not np.isfinite(width_delta_cap) or float(width_delta_cap) < 0.0:
        raise ValueError("width_delta_cap must be finite and non-negative")
    if width_cap_reference not in ("baseline", "atom_only"):
        raise ValueError("width_cap_reference must be 'baseline' or 'atom_only'")
    if not np.isfinite(anchor) or not 0.0 < float(anchor) < 0.5:
        raise ValueError("anchor must be finite and strictly between zero and 0.5")
    if not np.isfinite(epsilon) or not 0.0 < float(epsilon) < 0.01:
        raise ValueError("epsilon must be finite and lie in (0,0.01)")

    cell_shape = (cases, hours)
    pi0_array = _as_cell_array(pi0, cell_shape, "pi0")
    rho1_array = _as_cell_array(rho1, cell_shape, "rho1")
    lower_gate_array = _as_cell_array(lower_gate, cell_shape, "lower_gate")
    upper_gate_array = _as_cell_array(upper_gate, cell_shape, "upper_gate")
    for name, array in (
        ("pi0", pi0_array),
        ("rho1", rho1_array),
        ("lower_gate", lower_gate_array),
        ("upper_gate", upper_gate_array),
    ):
        _validate_probability(array, name)

    requested_pi1 = (1.0 - pi0_array) * rho1_array
    sorted_baseline = np.sort(baseline, axis=1, kind="stable")
    quantile_grid = (np.arange(members, dtype=np.float64) + 0.5) / members
    atom_sorted = np.empty_like(sorted_baseline)
    calibrated_sorted = np.empty_like(sorted_baseline)
    baseline_width = np.empty(cell_shape, dtype=np.float64)
    atom_width = np.empty(cell_shape, dtype=np.float64)
    final_width = np.empty(cell_shape, dtype=np.float64)
    width_reference = np.empty(cell_shape, dtype=np.float64)
    tail_scale = np.ones(cell_shape, dtype=np.float64)
    shift_sorted = np.zeros_like(sorted_baseline)
    central_sorted = np.zeros_like(sorted_baseline, dtype=bool)
    width_cap_mask = np.zeros(cell_shape, dtype=bool)
    projection_fallback_mask = np.zeros(cell_shape, dtype=bool)
    realized_pi0 = np.empty(cell_shape, dtype=np.float64)
    realized_pi1 = np.empty(cell_shape, dtype=np.float64)

    atom_violation_count = 0
    atom_max_violation = 0.0
    tail_violation_count = 0
    tail_max_violation = 0.0
    requested_max_shift = 0.0
    applied_max_shift = 0.0
    cap_hit_values = 0
    skipped_boundary_values = 0
    width_cap_cells = 0
    infeasible_cells: list[tuple[int, int, float, float]] = []
    central_changed = 0
    tolerance = 2e-12

    no_tail_requested = (
        (not tail_enabled)
        or float(dmax) == 0.0
        or (not np.any(lower_gate_array) and not np.any(upper_gate_array))
    )

    for case in range(cases):
        for hour in range(hours):
            cell = sorted_baseline[case, :, hour]
            baseline_width[case, hour] = _width_90(cell)
            if atom_enabled:
                atom_curve, qc, lower_atom, upper_atom = _cell_atom_curve(
                    cell,
                    quantile_grid,
                    float(pi0_array[case, hour]),
                    float(requested_pi1[case, hour]),
                )
            else:
                atom_curve = cell.copy()
                qc = quantile_grid.copy()
                lower_atom = np.zeros(members, dtype=bool)
                upper_atom = np.zeros(members, dtype=bool)

            atom_count, atom_violation = _monotonicity_diagnostics(atom_curve)
            atom_violation_count += atom_count
            atom_max_violation = max(atom_max_violation, atom_violation)
            atom_curve = np.maximum.accumulate(atom_curve)
            atom_sorted[case, :, hour] = atom_curve
            atom_width[case, hour] = _width_90(atom_curve)
            realized_pi0[case, hour] = float(np.mean(atom_curve == 0.0))
            realized_pi1[case, hour] = float(np.mean(atom_curve == 1.0))

            if width_cap_reference == "baseline":
                width_reference[case, hour] = baseline_width[case, hour]
            else:
                width_reference[case, hour] = atom_width[case, hour]
            limit = width_reference[case, hour] + float(width_delta_cap)
            if atom_width[case, hour] > limit + tolerance:
                infeasible_cells.append(
                    (case, hour, float(atom_width[case, hour]), float(limit))
                )
                continue

            continuous = ~(lower_atom | upper_atom)
            shift, raw_shift_maximum, cell_cap_hits = _tail_shift(
                atom_curve,
                qc,
                continuous,
                float(lower_gate_array[case, hour]),
                float(upper_gate_array[case, hour]),
                anchor=float(anchor),
                dmax=float(dmax),
                epsilon=float(epsilon),
            )
            if no_tail_requested:
                shift.fill(0.0)
                raw_shift_maximum = 0.0
                cell_cap_hits = 0
            requested_max_shift = max(requested_max_shift, raw_shift_maximum)
            cap_hit_values += cell_cap_hits
            shift_sorted[case, :, hour] = shift
            central = continuous & (qc >= float(anchor)) & (qc <= 1.0 - float(anchor))
            central_sorted[case, :, hour] = central

            candidate, skipped, maximum_applied = _apply_shift(
                atom_curve, shift, 1.0, epsilon=float(epsilon)
            )
            skipped_boundary_values += skipped
            count, violation = _monotonicity_diagnostics(candidate)
            tail_violation_count += count
            tail_max_violation = max(tail_max_violation, violation)
            candidate = np.maximum.accumulate(candidate)
            candidate_width = _width_90(candidate)
            if candidate_width > limit + tolerance:
                width_cap_cells += 1
                width_cap_mask[case, hour] = True
                projection_fallback_mask[case, hour] = count > 0
                continue

            applied_max_shift = max(applied_max_shift, maximum_applied)
            final_width[case, hour] = candidate_width
            central_changed += int(np.sum(candidate[central] != atom_curve[central]))
            calibrated_sorted[case, :, hour] = candidate

    if infeasible_cells:
        first = infeasible_cells[0]
        raise WidthCapInfeasibleError(
            f"requested atoms violate the {width_cap_reference} W90 cap in "
            f"{len(infeasible_cells)} cells; first cell case={first[0]}, hour={first[1]} "
            f"has atom-only W90={first[2]:.12g} > limit={first[3]:.12g}"
        )
    batch_mask = width_cap_mask & ~projection_fallback_mask
    if np.any(batch_mask):
        batch_case, batch_hour = np.nonzero(batch_mask)
        batch_atoms = atom_sorted[batch_case, :, batch_hour]
        batch_shifts = shift_sorted[batch_case, :, batch_hour]
        batch_limits = (
            width_reference[batch_case, batch_hour] + float(width_delta_cap)
        )
        batch_scales = _batch_width_cap_scales(
            batch_atoms, batch_shifts, batch_limits, iterations=56
        )
        batch_candidates = batch_atoms.copy()
        batch_active = (
            (batch_shifts != 0.0)
            & (batch_atoms > 0.0)
            & (batch_atoms < 1.0)
        )
        if np.any(batch_active):
            batch_z0 = np.zeros_like(batch_atoms)
            clipped = np.clip(
                batch_atoms[batch_active], float(epsilon), 1.0 - float(epsilon)
            )
            batch_z0[batch_active] = np.log(clipped) - np.log1p(-clipped)
            displaced = batch_z0 + batch_scales[:, None] * batch_shifts
            batch_candidates[batch_active] = expit(displaced[batch_active])
            applied_max_shift = max(
                applied_max_shift,
                float(
                    np.max(
                        np.abs(batch_scales[:, None] * batch_shifts)[batch_active]
                    )
                ),
            )
        batch_candidates = np.maximum.accumulate(batch_candidates, axis=1)
        batch_quantiles = np.quantile(
            batch_candidates, (0.05, 0.95), axis=1
        )
        batch_widths = batch_quantiles[1] - batch_quantiles[0]
        calibrated_sorted[batch_case, :, batch_hour] = batch_candidates
        final_width[batch_case, batch_hour] = batch_widths
        tail_scale[batch_case, batch_hour] = batch_scales
        batch_central = central_sorted[batch_case, :, batch_hour]
        central_changed += int(
            np.sum(batch_candidates[batch_central] != batch_atoms[batch_central])
        )

    fallback_case, fallback_hour = np.nonzero(
        width_cap_mask & projection_fallback_mask
    )
    for case, hour in zip(fallback_case.tolist(), fallback_hour.tolist()):
        atom_curve = atom_sorted[case, :, hour]
        shift = shift_sorted[case, :, hour]
        limit = width_reference[case, hour] + float(width_delta_cap)
        low = 0.0
        high = 1.0
        for _ in range(56):
            middle = 0.5 * (low + high)
            trial, _, _ = _apply_shift(
                atom_curve, shift, middle, epsilon=float(epsilon)
            )
            trial_width = _width_90(np.maximum.accumulate(trial))
            if trial_width <= limit:
                low = middle
            else:
                high = middle
        candidate, _, maximum_applied = _apply_shift(
            atom_curve, shift, low, epsilon=float(epsilon)
        )
        candidate = np.maximum.accumulate(candidate)
        calibrated_sorted[case, :, hour] = candidate
        final_width[case, hour] = _width_90(candidate)
        tail_scale[case, hour] = low
        applied_max_shift = max(applied_max_shift, maximum_applied)
        central = central_sorted[case, :, hour]
        central_changed += int(np.sum(candidate[central] != atom_curve[central]))
    if central_changed:
        raise RuntimeError(
            f"tail transformation changed {central_changed} central continuous values"
        )
    if not np.isfinite(atom_sorted).all() or not np.isfinite(calibrated_sorted).all():
        count = int(
            atom_sorted.size
            + calibrated_sorted.size
            - np.isfinite(atom_sorted).sum()
            - np.isfinite(calibrated_sorted).sum()
        )
        raise FloatingPointError(f"transformation produced {count} non-finite values")

    maximum_cap_excess = float(
        np.max(final_width - (width_reference + float(width_delta_cap)))
    )
    if maximum_cap_excess > 5e-11:
        raise RuntimeError(
            f"width-cap enforcement failed by {maximum_cap_excess:.12g}"
        )

    ranks = _ordinal_ranks(baseline)
    atom_only = np.take_along_axis(atom_sorted, ranks, axis=1)
    calibrated = np.take_along_axis(calibrated_sorted, ranks, axis=1)
    # Re-sorting by the baseline's stable member order must recover a
    # nondecreasing calibrated curve.  This proves zero strict reversals;
    # newly created ties are deliberately allowed.
    order = np.argsort(baseline, axis=1, kind="stable")
    ordered_after = np.take_along_axis(calibrated, order, axis=1)
    if np.any(np.diff(ordered_after, axis=1) < 0.0):
        raise RuntimeError("stable-rank restoration introduced a strict reversal")

    effective_pi0 = pi0_array if atom_enabled else realized_pi0
    effective_pi1 = requested_pi1 if atom_enabled else realized_pi1
    zero_error = np.abs(realized_pi0 - effective_pi0)
    one_error = np.abs(realized_pi1 - effective_pi1)
    diagnostics = HingeTailDiagnostics(
        requested_pi0=pi0_array.copy(),
        requested_rho1=rho1_array.copy(),
        requested_pi1=requested_pi1.copy(),
        realized_pi0=realized_pi0,
        realized_pi1=realized_pi1,
        baseline_width_90=baseline_width,
        atom_only_width_90=atom_width,
        calibrated_width_90=final_width,
        tail_scale=tail_scale,
        atom_enabled=bool(atom_enabled),
        tail_enabled=bool(tail_enabled),
        members=int(members),
        anchor=float(anchor),
        dmax=float(dmax),
        width_delta_cap=float(width_delta_cap),
        width_cap_reference=width_cap_reference,
        width_cap_reference_90=width_reference,
        member_probability_resolution=1.0 / members,
        midpoint_atom_quantization_bound=0.5 / members,
        atom_quantization_applicable=bool(atom_enabled),
        is_hundred_member_ensemble=members == 100,
        zero_atom_quantization_mae=float(zero_error.mean()),
        zero_atom_quantization_max_abs=float(zero_error.max()),
        one_atom_quantization_mae=float(one_error.mean()),
        one_atom_quantization_max_abs=float(one_error.max()),
        atom_preprojection_adjacent_violations=atom_violation_count,
        atom_preprojection_max_violation=atom_max_violation,
        tail_preprojection_adjacent_violations=tail_violation_count,
        tail_preprojection_max_violation=tail_max_violation,
        requested_max_abs_logit_shift=requested_max_shift,
        applied_max_abs_logit_shift=min(applied_max_shift, float(dmax)),
        logit_cap_hit_values=cap_hit_values,
        boundary_shift_skipped_values=skipped_boundary_values,
        width_cap_activated_cells=width_cap_cells,
        width_cap_infeasible_cells=0,
        maximum_final_width_cap_excess=maximum_cap_excess,
        central_values_changed_by_tail=central_changed,
        strict_reversals=0,
        nonfinite_input_values=nonfinite_inputs,
        nonfinite_intermediate_values=0,
    )
    return HingeTailResult(
        atom_only=atom_only.astype(np.float64, copy=False),
        calibrated=calibrated.astype(np.float64, copy=False),
        diagnostics=diagnostics,
    )


def apply_hinge_tail(*args: Any, **kwargs: Any) -> HingeTailResult:
    """Concise alias for :func:`atom_aware_logit_hinge_tail`."""

    return atom_aware_logit_hinge_tail(*args, **kwargs)


__all__ = [
    "DEFAULT_EPSILON",
    "HINGE_ANCHOR",
    "HingeTailDiagnostics",
    "HingeTailResult",
    "WidthCapInfeasibleError",
    "WidthCapReference",
    "apply_hinge_tail",
    "atom_aware_logit_hinge_tail",
]
