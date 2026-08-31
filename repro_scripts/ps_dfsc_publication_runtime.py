"""Corrected calibration and safety commands used by the canonical CLI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.inference import calibrate
from ps_dfsc.metrics import (
    make_conditional_assignments,
    weighted_per_date_metrics,
    weighted_quantile,
)
from ps_dfsc.safety import SafetyThresholds
from ps_dfsc.safety_publication import evaluate_safety_gate


def _dates(archive, length: int) -> np.ndarray:
    if "days" in archive:
        values = archive["days"]
    elif "day" in archive:
        values = archive["day"]
    else:
        raise ValueError("archive lacks registered day labels")
    if len(values) != length:
        raise ValueError("day labels do not align with scenarios")
    return np.asarray(values).astype("datetime64[D]")


def calibrate_archive(args) -> None:
    source = np.load(args.input, allow_pickle=False)
    scenarios = source[args.scenario_key]
    days = _dates(source, len(scenarios))
    mapping = pipeline._mapping_from_args(args)
    model = pipeline._load_model(args.checkpoint, args.device)
    distributions = [
        calibrate(
            model,
            values,
            mapping,
            clusters=args.clusters,
            ess_floor=args.ess_floor,
            normalized_entropy_floor=args.entropy_floor,
            transport_budget=args.transport_budget,
            device=args.device,
        )
        for values in scenarios
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        full_scenarios=np.stack([item.full_scenarios for item in distributions]),
        probabilities=np.stack([item.probabilities for item in distributions]),
        suc_scenarios=np.stack([item.suc_scenarios for item in distributions]),
        suc_probabilities=np.stack(
            [item.suc_probabilities for item in distributions]
        ),
        ess=np.asarray([item.ess for item in distributions]),
        entropy=np.asarray([item.entropy for item in distributions]),
        transport_cost=np.asarray(
            [item.transport_cost for item in distributions]
        ),
        used_fallback=np.asarray(
            [item.used_fallback for item in distributions]
        ),
        fallback_reason=np.asarray(
            [item.fallback_reason for item in distributions]
        ),
        groups=np.asarray(
            [
                list(group) + [-1] * (2 - len(group))
                for group in mapping.groups
            ],
            dtype=np.int64,
        ),
        days=days,
    )
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "days": len(distributions),
                "fallback_days": sum(
                    item.used_fallback for item in distributions
                ),
                "date_field_preserved": True,
            }
        )
    )


def _conditional_assignments(args, observations, days):
    reference = np.load(args.regime_reference, allow_pickle=False)
    reference_truth = reference[args.observation_key]
    aggregate = reference_truth.mean(axis=1)
    wind_thresholds = tuple(
        np.quantile(aggregate.mean(axis=1), [1.0 / 3.0, 2.0 / 3.0])
    )
    ramp_thresholds = tuple(
        np.quantile(
            np.abs(np.diff(aggregate, axis=1)).mean(axis=1),
            [1.0 / 3.0, 2.0 / 3.0],
        )
    )
    return make_conditional_assignments(
        observations,
        days=days,
        wind_thresholds=wind_thresholds,
        ramp_thresholds=ramp_thresholds,
    )


def _coverage_mask(archive, observations):
    lower = weighted_quantile(
        archive["full_scenarios"], archive["probabilities"], 0.05
    )
    upper = weighted_quantile(
        archive["full_scenarios"], archive["probabilities"], 0.95
    )
    return (observations >= lower) & (observations <= upper)


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
    result = asdict(decision)
    result.update(
        {
            "schema": "ps_dfsc_publication_safety_gate_v1",
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
            "conditional_group_weighting": "equal within family; equal across families",
            "structural_ok_every_day": structural_ok,
            "test_truth_accessed": False,
        }
    )
    pipeline._write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "passed": decision.passed,
                "structural_ok_every_day": structural_ok,
            }
        )
    )


__all__ = ["calibrate_archive", "safety_gate"]
