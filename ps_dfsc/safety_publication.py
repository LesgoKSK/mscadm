"""Publication-grade paired safety gate and conditional ACE bootstrap."""

from __future__ import annotations

import numpy as np

from .safety import GateDecision, SafetyThresholds


def _assignment_cube(labels: np.ndarray, covered_shape: tuple[int, ...]) -> np.ndarray:
    days, zones, hours = covered_shape
    value = np.asarray(labels)
    if value.shape == covered_shape:
        return value
    if value.shape == (zones, hours):
        return np.broadcast_to(value[None], covered_shape)
    if value.shape == (days,):
        return np.broadcast_to(value[:, None, None], covered_shape)
    raise ValueError("conditional assignment has incompatible shape")


def bootstrap_conditional_ace(
    covered: np.ndarray,
    assignments: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    nominal: float = 0.90,
) -> np.ndarray:
    coverage = np.asarray(covered, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    if coverage.ndim != 3 or coverage.shape[1:] != (10, 24):
        raise ValueError("covered must have shape [day,10,24]")
    if selected.ndim != 2:
        raise ValueError("indices must have shape [replicate,draw]")
    family_values = []
    for labels in assignments.values():
        cube = _assignment_cube(np.asarray(labels), coverage.shape)
        groups = np.unique(cube)
        day_hits = np.zeros((len(coverage), len(groups)), dtype=np.float64)
        day_cells = np.zeros_like(day_hits)
        for group_index, group in enumerate(groups):
            mask = cube == group
            day_hits[:, group_index] = (coverage * mask).sum(axis=(1, 2))
            day_cells[:, group_index] = mask.sum(axis=(1, 2))
        hits = day_hits[selected].sum(axis=1)
        cells = day_cells[selected].sum(axis=1)
        rates = np.divide(
            hits,
            cells,
            out=np.full_like(hits, np.nan),
            where=cells > 0.0,
        )
        family_values.append(np.nanmean(np.abs(rates - nominal), axis=1))
    if not family_values:
        raise ValueError("at least one conditional assignment family is required")
    return np.mean(np.stack(family_values, axis=1), axis=1)


def evaluate_safety_gate(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    candidate_covered: np.ndarray,
    baseline_covered: np.ndarray,
    assignments: dict[str, np.ndarray],
    thresholds: SafetyThresholds = SafetyThresholds(),
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    structural_ok: bool = True,
) -> GateDecision:
    ratio_limits = {
        "CRPS": thresholds.crps_ratio,
        "ES": thresholds.es_ratio,
        "VS": thresholds.vs_ratio,
        "ramp_CRPS": thresholds.ramp_crps_ratio,
        "zero_Brier": thresholds.brier_ratio,
    }
    required = set(ratio_limits) | {"coverage_90"}
    if not required.issubset(candidate) or not required.issubset(baseline):
        raise ValueError("missing publication safety metrics")
    days = len(np.asarray(candidate["CRPS"]))
    if days < 2:
        raise ValueError("safety gate requires at least two dates")
    if np.asarray(candidate_covered).shape != (days, 10, 24):
        raise ValueError("candidate coverage mask is misaligned")
    if np.asarray(baseline_covered).shape != (days, 10, 24):
        raise ValueError("baseline coverage mask is misaligned")

    diagnostics: dict[str, float] = {}
    point_reasons: list[str] = []
    for metric, limit in ratio_limits.items():
        candidate_mean = float(np.mean(candidate[metric]))
        baseline_mean = float(np.mean(baseline[metric]))
        ratio = (
            1.0
            if baseline_mean == 0.0 and candidate_mean == 0.0
            else candidate_mean / baseline_mean
        )
        diagnostics[f"{metric}_ratio"] = ratio
        if not np.isfinite(ratio) or ratio > limit:
            point_reasons.append(f"{metric}_point_ratio")

    coverage = float(np.mean(candidate_covered))
    diagnostics["coverage_90"] = coverage
    if not thresholds.coverage_lower <= coverage <= thresholds.coverage_upper:
        point_reasons.append("coverage_point")
    identity_indices = np.arange(days, dtype=np.int64)[None]
    candidate_ace = float(
        bootstrap_conditional_ace(
            candidate_covered, assignments, identity_indices
        )[0]
    )
    baseline_ace = float(
        bootstrap_conditional_ace(
            baseline_covered, assignments, identity_indices
        )[0]
    )
    ace_difference = candidate_ace - baseline_ace
    diagnostics["conditional_ACE"] = candidate_ace
    diagnostics["baseline_conditional_ACE"] = baseline_ace
    diagnostics["conditional_ACE_difference"] = ace_difference
    if ace_difference > 0.0:
        point_reasons.append("conditional_ACE_point")
    if not structural_ok:
        point_reasons.append("structural_budget")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, days, size=(bootstrap_samples, days))
    confidence_reasons: list[str] = []
    for metric, limit in ratio_limits.items():
        candidate_values = np.asarray(candidate[metric], dtype=np.float64)
        baseline_values = np.asarray(baseline[metric], dtype=np.float64)
        candidate_mean = candidate_values[indices].mean(axis=1)
        baseline_mean = baseline_values[indices].mean(axis=1)
        ratios = np.divide(
            candidate_mean,
            baseline_mean,
            out=np.full_like(candidate_mean, np.inf),
            where=baseline_mean != 0.0,
        )
        ratios[(baseline_mean == 0.0) & (candidate_mean == 0.0)] = 1.0
        upper = float(np.quantile(ratios, thresholds.confidence))
        diagnostics[f"{metric}_ratio_upper"] = upper
        if not np.isfinite(upper) or upper > limit:
            confidence_reasons.append(f"{metric}_confidence_ratio")

    coverage_samples = np.asarray(candidate["coverage_90"])[indices].mean(axis=1)
    two_sided_tail = (1.0 - thresholds.confidence) / 2.0
    coverage_lower = float(np.quantile(coverage_samples, two_sided_tail))
    coverage_upper = float(np.quantile(coverage_samples, 1.0 - two_sided_tail))
    diagnostics["coverage_lower_bound"] = coverage_lower
    diagnostics["coverage_upper_bound"] = coverage_upper
    if (
        coverage_lower < thresholds.coverage_lower
        or coverage_upper > thresholds.coverage_upper
    ):
        confidence_reasons.append("coverage_confidence")

    ace_samples = bootstrap_conditional_ace(
        candidate_covered, assignments, indices
    ) - bootstrap_conditional_ace(baseline_covered, assignments, indices)
    ace_upper = float(np.quantile(ace_samples, thresholds.confidence))
    diagnostics["conditional_ACE_difference_upper"] = ace_upper
    if ace_upper > 0.0:
        confidence_reasons.append("conditional_ACE_confidence")

    point_passed = not point_reasons
    confidence_passed = not confidence_reasons
    return GateDecision(
        passed=point_passed and confidence_passed,
        point_passed=point_passed,
        confidence_passed=confidence_passed,
        reasons=tuple(point_reasons + confidence_reasons),
        diagnostics=diagnostics,
    )


__all__ = ["bootstrap_conditional_ace", "evaluate_safety_gate"]
