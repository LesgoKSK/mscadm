"""Aggregate the three locked outers with solver and failure sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ps_dfsc.evaluation import stratified_paired_bootstrap
from ps_dfsc.selection import empirical_cvar


def _method_summary(frame: pd.DataFrame) -> dict:
    available = frame.loc[frame["status"] == "solution_available"]
    result = {
        "cases": int(len(frame)),
        "solution_available": int(len(available)),
        "solution_available_rate": float(len(available) / max(len(frame), 1)),
    }
    if available.empty:
        return result
    cost = available["realized_total_cost"].to_numpy(np.float64)
    result.update(
        {
            "mean_cost": float(cost.mean()),
            "CVaR90": empirical_cvar(cost, 0.90),
            "CVaR95": empirical_cvar(cost, 0.95),
            "mean_cost_regret": float(available["cost_regret"].mean()),
            "mean_load_shedding": float(available["load_shedding"].mean()),
            "mean_wind_curtailment": float(
                available["wind_curtailment"].mean()
            ),
            "mean_reserve_shortage": float(
                available["reserve_shortage"].mean()
            ),
            "all_solver_success_rate": float(
                available["all_solver_success"].mean()
            ),
            "all_target_gaps_met_rate": float(
                available["all_target_gaps_met"].mean()
            ),
            "planned_gap_max": float(available["planned_mip_gap"].max()),
            "planned_time_mean": float(available["planned_solve_time"].mean()),
        }
    )
    return result


def _paired(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    joined = candidate.merge(
        baseline,
        on=["outer", "date"],
        suffixes=("_candidate", "_baseline"),
        how="outer",
        validate="one_to_one",
    )
    return joined.loc[
        (joined["status_candidate"] == "solution_available")
        & (joined["status_baseline"] == "solution_available")
    ].copy()


def _paired_effect(frame: pd.DataFrame) -> dict:
    candidate = frame["realized_total_cost_candidate"].to_numpy(np.float64)
    baseline = frame["realized_total_cost_baseline"].to_numpy(np.float64)
    if not len(frame):
        return {"paired_cases": 0}
    return {
        "paired_cases": int(len(frame)),
        "candidate_mean_cost": float(candidate.mean()),
        "baseline_mean_cost": float(baseline.mean()),
        "mean_cost_reduction_fraction": float(
            1.0 - candidate.mean() / baseline.mean()
        ),
        "candidate_CVaR90": empirical_cvar(candidate, 0.90),
        "baseline_CVaR90": empirical_cvar(baseline, 0.90),
        "CVaR90_reduction_fraction": float(
            1.0
            - empirical_cvar(candidate, 0.90)
            / empirical_cvar(baseline, 0.90)
        ),
        "candidate_CVaR95": empirical_cvar(candidate, 0.95),
        "baseline_CVaR95": empirical_cvar(baseline, 0.95),
        "CVaR95_reduction_fraction": float(
            1.0
            - empirical_cvar(candidate, 0.95)
            / empirical_cvar(baseline, 0.95)
        ),
    }


def _worst_case_penalty(
    candidate_frames: list[pd.DataFrame],
    baseline_frames: list[pd.DataFrame],
    multipliers=(1.25, 1.5, 2.0),
) -> dict[str, dict]:
    candidate = pd.concat(candidate_frames, ignore_index=True)
    baseline = pd.concat(baseline_frames, ignore_index=True)
    joined = candidate.merge(
        baseline,
        on=["outer", "date"],
        suffixes=("_candidate", "_baseline"),
        how="outer",
        validate="one_to_one",
    )
    available_values = np.concatenate(
        [
            candidate.loc[
                candidate["status"] == "solution_available",
                "realized_total_cost",
            ].to_numpy(np.float64),
            baseline.loc[
                baseline["status"] == "solution_available",
                "realized_total_cost",
            ].to_numpy(np.float64),
        ]
    )
    if not len(available_values):
        return {}
    reference = float(np.max(available_values))
    result = {}
    for multiplier in multipliers:
        penalty = multiplier * reference
        candidate_cost = joined["realized_total_cost_candidate"].to_numpy(
            np.float64
        )
        baseline_cost = joined["realized_total_cost_baseline"].to_numpy(
            np.float64
        )
        candidate_ok = (
            joined["status_candidate"].to_numpy() == "solution_available"
        ) & np.isfinite(candidate_cost)
        baseline_ok = (
            joined["status_baseline"].to_numpy() == "solution_available"
        ) & np.isfinite(baseline_cost)
        candidate_cost = np.where(candidate_ok, candidate_cost, penalty)
        baseline_cost = np.where(baseline_ok, baseline_cost, penalty)
        result[str(multiplier)] = {
            "penalty_cost": penalty,
            "candidate_mean_cost": float(candidate_cost.mean()),
            "baseline_mean_cost": float(baseline_cost.mean()),
            "mean_cost_reduction_fraction": float(
                1.0 - candidate_cost.mean() / baseline_cost.mean()
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication PS-DFSC confirmation report"
    )
    parser.add_argument(
        "--confirmation-root", default="outputs/ps_dfsc/confirmation"
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    root = Path(args.confirmation_root)
    candidate_frames = []
    baseline_frames = []
    paired_frames = []
    safety = {}
    by_outer = {}
    for outer in (1, 2, 3):
        directory = root / f"outer{outer}"
        candidate = pd.read_csv(directory / "exact_ps_dfsc.csv")
        baseline = pd.read_csv(directory / "exact_identity.csv")
        paired = _paired(candidate, baseline)
        candidate_frames.append(candidate)
        baseline_frames.append(baseline)
        paired_frames.append(paired)
        safety[str(outer)] = json.loads(
            (directory / "confirmation_safety.json").read_text(
                encoding="utf-8"
            )
        )
        by_outer[str(outer)] = {
            "candidate": _method_summary(candidate),
            "baseline": _method_summary(baseline),
            "paired_effect": _paired_effect(paired),
        }

    paired = pd.concat(paired_frames, ignore_index=True)
    candidate_cost = paired["realized_total_cost_candidate"].to_numpy(
        np.float64
    )
    baseline_cost = paired["realized_total_cost_baseline"].to_numpy(np.float64)
    outer_labels = paired["outer"].to_numpy()
    inference = (
        stratified_paired_bootstrap(
            candidate_cost,
            baseline_cost,
            outer_labels,
            samples=10_000,
            seed=20260801,
        )
        if len(paired)
        else {}
    )
    effect = _paired_effect(paired)
    all_safety_passed = all(bool(value["passed"]) for value in safety.values())
    complete_pairing = len(paired) == 150
    primary_passed = bool(
        complete_pairing
        and inference
        and inference["one_sided_95_upper"] < 0.0
    )
    summary = {
        "schema": "ps_dfsc_confirmation_report_v2",
        "paired_cases": int(len(paired)),
        "paired_success_rate": float(len(paired) / 150.0),
        **effect,
        "paired_cost_inference": inference,
        "primary_mean_cost_test_passed": primary_passed,
        "all_confirmation_safety_gates_passed": all_safety_passed,
        "safe_improvement_claim_allowed": bool(
            primary_passed and all_safety_passed
        ),
        "engineering_target_cost_1_to_3_percent": bool(
            effect
            and 0.01
            <= effect.get("mean_cost_reduction_fraction", -np.inf)
            <= 0.03
        ),
        "engineering_target_CVaR90_at_least_5_percent": bool(
            effect.get("CVaR90_reduction_fraction", -np.inf) >= 0.05
        ),
        "by_outer": by_outer,
        "safety_by_outer": safety,
        "failed_case_worst_case_penalty_sensitivity": _worst_case_penalty(
            candidate_frames, baseline_frames
        ),
        "test_results_used_for_model_selection": False,
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = f"""# PS-DFSC confirmation report

- Paired binary-MILP solutions: {len(paired)}/150
- Mean cost reduction: {100 * effect.get('mean_cost_reduction_fraction', float('nan')):.3f}%
- CVaR90 reduction: {100 * effect.get('CVaR90_reduction_fraction', float('nan')):.3f}%
- CVaR95 reduction: {100 * effect.get('CVaR95_reduction_fraction', float('nan')):.3f}%
- One-sided 95% upper bound for the paired cost difference: {inference.get('one_sided_95_upper', float('nan')):.3f}
- All confirmation safety audits passed: {all_safety_passed}
- Safe confirmatory improvement claim allowed: {summary['safe_improvement_claim_allowed']}

The 1–3% mean-cost reduction and at-least-5% CVaR90 reduction are engineering
targets, not guaranteed outcomes. Solver-success and 0.1% target-gap rates are
reported separately from binary-incumbent availability.
"""
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"report": str(json_path.resolve()), **summary}))


if __name__ == "__main__":
    main()
