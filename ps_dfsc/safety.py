from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyThresholds:
    crps_ratio: float = 1.005
    es_ratio: float = 1.01
    vs_ratio: float = 1.01
    ramp_crps_ratio: float = 1.01
    brier_ratio: float = 1.01
    coverage_lower: float = 0.88
    coverage_upper: float = 0.92
    confidence: float = 0.95


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    point_passed: bool
    confidence_passed: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, float]


def evaluate_safety_gate(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    thresholds: SafetyThresholds = SafetyThresholds(),
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    structural_ok: bool = True,
) -> GateDecision:
    required = {
        "CRPS",
        "ES",
        "VS",
        "ramp_CRPS",
        "zero_Brier",
        "coverage_90",
        "conditional_ACE",
    }
    if not required.issubset(candidate) or not required.issubset(baseline):
        missing = sorted(required - candidate.keys() | required - baseline.keys())
        raise ValueError(f"missing safety metrics: {missing}")
    days = len(np.asarray(candidate["CRPS"]))
    if days < 2:
        raise ValueError("safety gate requires at least two dates")
    if any(len(np.asarray(candidate[key])) != days for key in required):
        raise ValueError("candidate metric lengths do not align")
    if any(len(np.asarray(baseline[key])) != days for key in required):
        raise ValueError("baseline metric lengths do not align")
    ratio_limits = {
        "CRPS": thresholds.crps_ratio,
        "ES": thresholds.es_ratio,
        "VS": thresholds.vs_ratio,
        "ramp_CRPS": thresholds.ramp_crps_ratio,
        "zero_Brier": thresholds.brier_ratio,
    }
    diagnostics: dict[str, float] = {}
    point_reasons: list[str] = []
    for metric, limit in ratio_limits.items():
        denominator = float(np.mean(baseline[metric]))
        ratio = (
            1.0
            if denominator == 0.0 and float(np.mean(candidate[metric])) == 0.0
            else float(np.mean(candidate[metric])) / denominator
        )
        diagnostics[f"{metric}_ratio"] = ratio
        if not np.isfinite(ratio) or ratio > limit:
            point_reasons.append(f"{metric}_point_ratio")
    coverage = float(np.mean(candidate["coverage_90"]))
    diagnostics["coverage_90"] = coverage
    if not thresholds.coverage_lower <= coverage <= thresholds.coverage_upper:
        point_reasons.append("coverage_point")
    ace_difference = float(
        np.mean(candidate["conditional_ACE"]) - np.mean(baseline["conditional_ACE"])
    )
    diagnostics["conditional_ACE_difference"] = ace_difference
    if ace_difference > 0.0:
        point_reasons.append("conditional_ACE_point")
    if not structural_ok:
        point_reasons.append("structural_budget")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, days, size=(bootstrap_samples, days))
    confidence_reasons: list[str] = []
    upper_quantile = thresholds.confidence
    lower_quantile = 1.0 - thresholds.confidence
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
        upper = float(np.quantile(ratios, upper_quantile))
        diagnostics[f"{metric}_ratio_upper"] = upper
        if not np.isfinite(upper) or upper > limit:
            confidence_reasons.append(f"{metric}_confidence_ratio")
    coverage_samples = np.asarray(candidate["coverage_90"])[indices].mean(axis=1)
    coverage_lower = float(np.quantile(coverage_samples, lower_quantile))
    coverage_upper = float(np.quantile(coverage_samples, upper_quantile))
    diagnostics["coverage_lower_bound"] = coverage_lower
    diagnostics["coverage_upper_bound"] = coverage_upper
    if (
        coverage_lower < thresholds.coverage_lower
        or coverage_upper > thresholds.coverage_upper
    ):
        confidence_reasons.append("coverage_confidence")
    ace_samples = (
        np.asarray(candidate["conditional_ACE"])[indices].mean(axis=1)
        - np.asarray(baseline["conditional_ACE"])[indices].mean(axis=1)
    )
    ace_upper = float(np.quantile(ace_samples, upper_quantile))
    diagnostics["conditional_ACE_difference_upper"] = ace_upper
    if ace_upper > 0.0:
        confidence_reasons.append("conditional_ACE_confidence")
    point_passed = not point_reasons
    confidence_passed = not confidence_reasons
    reasons = tuple(point_reasons + confidence_reasons)
    return GateDecision(
        passed=point_passed and confidence_passed,
        point_passed=point_passed,
        confidence_passed=confidence_passed,
        reasons=reasons,
        diagnostics=diagnostics,
    )


__all__ = ["GateDecision", "SafetyThresholds", "evaluate_safety_gate"]
