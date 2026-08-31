"""Publication runtime with complete safety metrics and diversity audit."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.metrics import weighted_per_date_metrics
from ps_dfsc.safety import SafetyThresholds
from ps_dfsc.safety_publication import evaluate_safety_gate
from repro_scripts.ps_dfsc_publication_runtime import (
    _conditional_assignments,
    _coverage_mask,
    _dates,
)
from repro_scripts.ps_dfsc_publication_runtime_v2 import calibrate_archive


def _diversity_per_date(scenarios, probabilities):
    values = np.asarray(scenarios, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    result = np.empty(len(values), dtype=np.float64)
    for day in range(len(values)):
        flat = values[day].reshape(len(values[day]), -1)
        distance = np.linalg.norm(
            flat[:, None, :] - flat[None, :, :], axis=2
        )
        result[day] = np.einsum(
            "i,j,ij->", weights[day], weights[day], distance, optimize=True
        )
    return result


def safety_gate(args) -> None:
    candidate = np.load(args.candidate, allow_pickle=False)
    baseline = np.load(args.baseline, allow_pickle=False)
    truth_archive = np.load(args.truth, allow_pickle=False)
    observations = truth_archive[args.observation_key]
    days = _dates(truth_archive, len(observations))
    assignments = _conditional_assignments(args, observations, days)
    candidate_metrics = weighted_per_date_metrics(
        candidate["full_scenarios"],
        candidate["probabilities"],
        observations,
    )
    baseline_metrics = weighted_per_date_metrics(
        baseline["full_scenarios"],
        baseline["probabilities"],
        observations,
    )
    candidate_covered = _coverage_mask(candidate, observations)
    baseline_covered = _coverage_mask(baseline, observations)
    probability_members = candidate["probabilities"].shape[1]
    finite_structural = all(
        np.isfinite(candidate[key]).all()
        for key in ("ess", "entropy", "transport_cost")
    )
    structural_ok = bool(
        finite_structural
        and np.all(candidate["ess"] >= args.ess_floor)
        and np.all(
            candidate["entropy"] / np.log(probability_members)
            >= args.entropy_floor
        )
        and np.all(candidate["transport_cost"] <= args.transport_budget)
    )
    decision = evaluate_safety_gate(
        candidate_metrics,
        baseline_metrics,
        candidate_covered=candidate_covered,
        baseline_covered=baseline_covered,
        assignments=assignments,
        thresholds=SafetyThresholds(),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        structural_ok=structural_ok,
    )
    candidate_diversity = _diversity_per_date(
        candidate["full_scenarios"], candidate["probabilities"]
    )
    baseline_diversity = _diversity_per_date(
        baseline["full_scenarios"], baseline["probabilities"]
    )
    metric_keys = sorted(candidate_metrics)
    result = asdict(decision)
    result.update(
        {
            "schema": "ps_dfsc_publication_safety_gate_v2",
            "candidate": str(Path(args.candidate).resolve()),
            "baseline": str(Path(args.baseline).resolve()),
            "truth": str(Path(args.truth).resolve()),
            "regime_reference": str(Path(args.regime_reference).resolve()),
            "bootstrap_unit": "paired_day",
            "bootstrap_samples": int(args.bootstrap_samples),
            "coverage_interval_quantiles": [0.025, 0.975],
            "conditional_families": [
                "season",
                "hour_block",
                "wind_level",
                "ramp_level",
            ],
            "conditional_group_weighting": (
                "equal within family; equal across families"
            ),
            "structural_ok_every_day": structural_ok,
            "candidate_metric_means": {
                key: float(np.mean(candidate_metrics[key]))
                for key in metric_keys
            },
            "baseline_metric_means": {
                key: float(np.mean(baseline_metrics[key]))
                for key in metric_keys
            },
            "candidate_distribution_diagnostics": {
                "ESS_mean": float(np.mean(candidate["ess"])),
                "ESS_min": float(np.min(candidate["ess"])),
                "normalized_entropy_mean": float(
                    np.mean(candidate["entropy"] / np.log(probability_members))
                ),
                "normalized_entropy_min": float(
                    np.min(candidate["entropy"] / np.log(probability_members))
                ),
                "transport_mean": float(
                    np.mean(candidate["transport_cost"])
                ),
                "transport_max": float(
                    np.max(candidate["transport_cost"])
                ),
                "weighted_pairwise_distance_mean": float(
                    np.mean(candidate_diversity)
                ),
                "fallback_days": int(np.sum(candidate["used_fallback"])),
            },
            "baseline_distribution_diagnostics": {
                "weighted_pairwise_distance_mean": float(
                    np.mean(baseline_diversity)
                )
            },
            "test_truth_accessed": False,
        }
    )
    output = Path(args.output)
    pipeline._write_json(output, result)
    metrics_path = output.with_suffix(".metrics.npz")
    np.savez_compressed(
        metrics_path,
        days=days,
        candidate_covered=candidate_covered,
        baseline_covered=baseline_covered,
        candidate_diversity=candidate_diversity,
        baseline_diversity=baseline_diversity,
        candidate_ESS=candidate["ess"],
        candidate_entropy=candidate["entropy"],
        candidate_transport=candidate["transport_cost"],
        candidate_used_fallback=candidate["used_fallback"],
        **{
            f"candidate_{key}": candidate_metrics[key]
            for key in metric_keys
        },
        **{
            f"baseline_{key}": baseline_metrics[key]
            for key in metric_keys
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "metrics": str(metrics_path.resolve()),
                "passed": decision.passed,
                "structural_ok_every_day": structural_ok,
            }
        )
    )


__all__ = ["calibrate_archive", "safety_gate"]
