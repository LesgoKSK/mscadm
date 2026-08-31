"""Build the paired exact validation objective for one candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ps_dfsc.selection import empirical_cvar


def summarize_pair(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    risk_weight: float = 0.25,
) -> dict:
    joined = candidate.merge(
        baseline,
        on=["outer", "date"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    paired = joined.loc[
        (joined["status_candidate"] == "solution_available")
        & (joined["status_baseline"] == "solution_available")
    ].copy()
    result = {
        "cases": int(len(joined)),
        "paired_cases": int(len(paired)),
        "paired_success_rate": float(len(paired) / max(len(joined), 1)),
        "risk_weight": float(risk_weight),
    }
    if paired.empty:
        return result
    candidate_cost = paired["realized_total_cost_candidate"].to_numpy(
        np.float64
    )
    baseline_cost = paired["realized_total_cost_baseline"].to_numpy(np.float64)
    candidate_cvar = empirical_cvar(candidate_cost, 0.90)
    baseline_cvar = empirical_cvar(baseline_cost, 0.90)
    objective = (
        candidate_cost.mean() / baseline_cost.mean()
        + risk_weight * candidate_cvar / baseline_cvar
    )
    result.update(
        {
            "exact_mean_cost": float(candidate_cost.mean()),
            "exact_cvar90": candidate_cvar,
            "baseline_mean_cost": float(baseline_cost.mean()),
            "baseline_cvar90": baseline_cvar,
            "exact_J_select": float(objective),
            "baseline_J_select": float(1.0 + risk_weight),
            "exact_not_worse_than_baseline": bool(
                objective <= 1.0 + risk_weight + 1e-12
            ),
            "candidate_target_gap_pass_rate": float(
                paired["all_target_gaps_met_candidate"].mean()
            ),
            "baseline_target_gap_pass_rate": float(
                paired["all_target_gaps_met_baseline"].mean()
            ),
            "candidate_solver_success_rate": float(
                paired["all_solver_success_candidate"].mean()
            ),
            "baseline_solver_success_rate": float(
                paired["all_solver_success_baseline"].mean()
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired candidate/baseline exact validation"
    )
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = {
        "schema": "ps_dfsc_paired_exact_validation_v1",
        **summarize_pair(
            pd.read_csv(args.candidate_csv),
            pd.read_csv(args.baseline_csv),
            risk_weight=args.risk_weight,
        ),
        "candidate_csv": str(Path(args.candidate_csv).resolve()),
        "baseline_csv": str(Path(args.baseline_csv).resolve()),
        "test_truth_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output.resolve()), **summary}))


if __name__ == "__main__":
    main()
