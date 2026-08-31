"""Evaluation and success-gate aggregation for sealed CAA outer tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rahc.bootstrap import paired_calendar_day_bootstrap
from rahc.evaluation import family_equal_conditional_ace, overall_scores

from .metrics import stack_per_date_metrics
from .selection import CandidateRecord, METRIC_KEYS, select_candidates


def _validate(
    methods: Mapping[str, Mapping[int, Sequence[np.ndarray]]],
    observations: Mapping[int, np.ndarray],
    day: Mapping[int, np.ndarray],
    assignments: Mapping[int, Mapping[str, np.ndarray]],
    *,
    baseline_name: str,
) -> tuple[list[int], list[str]]:
    if baseline_name not in methods:
        raise ValueError("registered baseline method is missing")
    outers = sorted(observations)
    if not outers or set(day) != set(outers) or set(assignments) != set(outers):
        raise ValueError("observation/day/assignment outer catalogs differ")
    all_dates: list[np.ndarray] = []
    for outer in outers:
        truth = np.asarray(observations[outer])
        dates = np.asarray(day[outer]).astype("datetime64[D]")
        if truth.ndim != 2 or dates.shape != (len(truth),):
            raise ValueError(f"outer {outer} observation/day shape mismatch")
        all_dates.append(np.unique(dates))
        for family, codes in assignments[outer].items():
            if np.asarray(codes).shape != truth.shape:
                raise ValueError(f"outer {outer} assignment {family} shape mismatch")
    concatenated_dates = np.concatenate(all_dates)
    if len(np.unique(concatenated_dates)) != len(concatenated_dates):
        raise ValueError("sealed outer-test date blocks overlap")
    methods_order = list(methods)
    for method, by_outer in methods.items():
        if set(by_outer) != set(outers):
            raise ValueError(f"method {method} outer catalog mismatch")
        for outer in outers:
            values = list(by_outer[outer])
            if len(values) != 3:
                raise ValueError(f"method {method} outer {outer} needs three model seeds")
            expected = (len(observations[outer]), np.asarray(values[0]).shape[1], observations[outer].shape[1])
            if any(np.asarray(item).shape != expected for item in values):
                raise ValueError(f"method {method} outer {outer} seed arrays do not align")
    return outers, methods_order


def evaluate_outer_tests(
    methods: Mapping[str, Mapping[int, Sequence[np.ndarray]]],
    observations: Mapping[int, np.ndarray],
    day: Mapping[int, np.ndarray],
    assignments: Mapping[int, Mapping[str, np.ndarray]],
    *,
    baseline_name: str = "A0",
    primary_name: str = "A4",
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 20260721,
    success_config: Mapping[str, Any] | None = None,
    strict_reversals: int | None = None,
) -> dict[str, Any]:
    """Evaluate three-seed methods on disjoint outer-test date blocks."""

    outers, method_order = _validate(
        methods, observations, day, assignments, baseline_name=baseline_name
    )
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {}
    pooled_truth = np.concatenate([np.asarray(observations[outer]) for outer in outers])
    pooled_day = np.concatenate([np.asarray(day[outer]) for outer in outers])
    families = sorted(next(iter(assignments.values())))
    if any(sorted(assignments[outer]) != families for outer in outers):
        raise ValueError("conditional assignment families differ across outer tests")
    pooled_assignments = {
        family: np.concatenate([np.asarray(assignments[outer][family]) for outer in outers])
        for family in families
    }
    for method in method_order:
        pooled[method] = [
            np.concatenate([np.asarray(methods[method][outer][seed]) for outer in outers])
            for seed in range(3)
        ]
        for outer in outers:
            for seed, scenarios in enumerate(methods[method][outer]):
                rows.append(
                    {
                        "method": method,
                        "outer": int(outer),
                        "seed": seed,
                        **overall_scores(np.asarray(scenarios), np.asarray(observations[outer])),
                        "conditional_ACE90": family_equal_conditional_ace(
                            np.asarray(scenarios),
                            np.asarray(observations[outer]),
                            dict(assignments[outer]),
                            nominal=0.90,
                        ),
                    }
                )
        for seed, scenarios in enumerate(pooled[method]):
            rows.append(
                {
                    "method": method,
                    "outer": "pooled",
                    "seed": seed,
                    **overall_scores(scenarios, pooled_truth),
                    "conditional_ACE90": family_equal_conditional_ace(
                        scenarios, pooled_truth, pooled_assignments, nominal=0.90
                    ),
                }
            )

    records: list[CandidateRecord] = []
    for method in method_order:
        daily = stack_per_date_metrics(pooled[method], pooled_truth, pooled_day)
        records.append(
            CandidateRecord(
                name=method,
                metrics={metric: daily[metric] for metric in METRIC_KEYS},
                conditional_ace90=np.asarray(
                    [
                        family_equal_conditional_ace(
                            values, pooled_truth, pooled_assignments, nominal=0.90
                        )
                        for values in pooled[method]
                    ]
                ),
                selection_policy="baseline" if method == baseline_name else "constrained",
            )
        )
    noninferiority: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    baseline_values = pooled[baseline_name]
    for index, method in enumerate(method_order):
        if method == baseline_name:
            continue
        pair = [records[0] if records[0].name == baseline_name else next(r for r in records if r.name == baseline_name), next(r for r in records if r.name == method)]
        noninferiority[method] = select_candidates(
            pair,
            baseline_name=baseline_name,
            day_ids=np.unique(pooled_day.astype("datetime64[D]")),
            bootstrap_replicates=int(bootstrap_replicates),
            bootstrap_seed=int(bootstrap_seed) + index,
        )
        bootstrap[method] = paired_calendar_day_bootstrap(
            baseline_values,
            pooled[method],
            pooled_truth,
            pooled_day,
            pooled_assignments,
            replicates=int(bootstrap_replicates),
            random_seed=int(bootstrap_seed) + 100 + index,
            expected_unique_days=len(np.unique(pooled_day.astype("datetime64[D]"))),
        )

    gates: dict[str, Any] = {}
    if primary_name in methods:
        pooled_rows = [
            row for row in rows if row["method"] == primary_name and row["outer"] == "pooled"
        ]
        baseline_rows = [
            row for row in rows if row["method"] == baseline_name and row["outer"] == "pooled"
        ]
        primary_mean = _mean_numeric(pooled_rows)
        baseline_mean = _mean_numeric(baseline_rows)
        cfg = {
            "coverage_90_low": 0.88,
            "coverage_90_high": 0.92,
            "conditional_ace_relative_improvement": 0.30,
            "crps_relative_degradation": 0.005,
            "secondary_relative_degradation": 0.01,
            "width_absolute_increase": 0.05,
            "strict_rank_reversals": 0,
            **dict(success_config or {}),
        }
        conditional_reduction = (
            (baseline_mean["conditional_ACE90"] - primary_mean["conditional_ACE90"])
            / baseline_mean["conditional_ACE90"]
            if baseline_mean["conditional_ACE90"] > 0
            else 0.0
        )
        relative = {
            metric: (primary_mean[metric] - baseline_mean[metric]) / baseline_mean[metric]
            for metric in ("CRPS", "MAE", "VS", "ramp_CRPS", "winkler_90")
        }
        gate_values = {
            "coverage_90": cfg["coverage_90_low"] <= primary_mean["coverage_90"] <= cfg["coverage_90_high"],
            "conditional_ACE90_reduction": conditional_reduction >= cfg["conditional_ace_relative_improvement"],
            "CRPS_noninferiority": relative["CRPS"] <= cfg["crps_relative_degradation"],
            "MAE_noninferiority": relative["MAE"] <= cfg["secondary_relative_degradation"],
            "VS_noninferiority": relative["VS"] <= cfg["secondary_relative_degradation"],
            "ramp_CRPS_noninferiority": relative["ramp_CRPS"] <= cfg["secondary_relative_degradation"],
            "winkler_90_noninferiority": relative["winkler_90"] <= cfg["secondary_relative_degradation"],
            "width_90_increase": primary_mean["width_90"] - baseline_mean["width_90"] <= cfg["width_absolute_increase"],
            "strict_rank_reversals": strict_reversals is not None and strict_reversals <= cfg["strict_rank_reversals"],
        }
        gates = {
            "passed": int(sum(gate_values.values())),
            "total": len(gate_values),
            "all_passed": all(gate_values.values()),
            "gates": gate_values,
            "primary_mean": primary_mean,
            "baseline_mean": baseline_mean,
            "relative_degradation": relative,
            "conditional_ACE90_relative_reduction": conditional_reduction,
            "strict_reversals": strict_reversals,
        }
    return {
        "protocol": {
            "outers": outers,
            "model_seeds": 3,
            "unique_test_dates": int(len(np.unique(pooled_day.astype("datetime64[D]")))),
            "outer_dates_disjoint": True,
            "bootstrap_cluster": "calendar date",
        },
        "rows": rows,
        "noninferiority": noninferiority,
        "paired_bootstrap": bootstrap,
        "success_gates": gates,
    }


def _mean_numeric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty row set")
    keys = [
        key for key, value in rows[0].items() if key not in {"method", "outer", "seed"} and isinstance(value, (int, float, np.number))
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


__all__ = ["evaluate_outer_tests"]
