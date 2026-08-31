from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ps_dfsc.evaluation import stratified_paired_bootstrap
from ps_dfsc.selection import empirical_cvar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the three locked PS-DFSC confirmation outers"
    )
    parser.add_argument("--confirmation-root", default="outputs/ps_dfsc/confirmation")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    root = Path(args.confirmation_root)
    records = []
    safety = {}
    for outer in (1, 2, 3):
        directory = root / f"outer{outer}"
        candidate = pd.read_csv(directory / "exact_ps_dfsc.csv")
        baseline = pd.read_csv(directory / "exact_identity.csv")
        joined = candidate.merge(
            baseline,
            on=["outer", "date"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        joined = joined.loc[
            (joined["status_candidate"] == "ok")
            & (joined["status_baseline"] == "ok")
        ].copy()
        records.append(joined)
        safety[str(outer)] = json.loads(
            (directory / "confirmation_safety.json").read_text(encoding="utf-8")
        )
    paired = pd.concat(records, ignore_index=True)
    candidate_cost = paired["realized_total_cost_candidate"].to_numpy(np.float64)
    baseline_cost = paired["realized_total_cost_baseline"].to_numpy(np.float64)
    outer = paired["outer"].to_numpy()
    inference = stratified_paired_bootstrap(
        candidate_cost, baseline_cost, outer, samples=10_000, seed=20260801
    )
    mean_reduction = 1.0 - candidate_cost.mean() / baseline_cost.mean()
    cvar90_reduction = 1.0 - empirical_cvar(
        candidate_cost, 0.90
    ) / empirical_cvar(baseline_cost, 0.90)
    cvar95_reduction = 1.0 - empirical_cvar(
        candidate_cost, 0.95
    ) / empirical_cvar(baseline_cost, 0.95)
    all_safety_passed = all(
        bool(value["passed"]) for value in safety.values()
    )
    summary = {
        "schema": "ps_dfsc_confirmation_report_v1",
        "paired_cases": int(len(paired)),
        "paired_success_rate": float(len(paired) / 150.0),
        "candidate_mean_cost": float(candidate_cost.mean()),
        "baseline_mean_cost": float(baseline_cost.mean()),
        "mean_cost_reduction_fraction": float(mean_reduction),
        "candidate_CVaR90": empirical_cvar(candidate_cost, 0.90),
        "baseline_CVaR90": empirical_cvar(baseline_cost, 0.90),
        "CVaR90_reduction_fraction": float(cvar90_reduction),
        "candidate_CVaR95": empirical_cvar(candidate_cost, 0.95),
        "baseline_CVaR95": empirical_cvar(baseline_cost, 0.95),
        "CVaR95_reduction_fraction": float(cvar95_reduction),
        "paired_cost_inference": inference,
        "all_confirmation_safety_gates_passed": all_safety_passed,
        "engineering_target_cost_1_to_3_percent": bool(
            0.01 <= mean_reduction <= 0.03
        ),
        "engineering_target_CVaR90_at_least_5_percent": bool(
            cvar90_reduction >= 0.05
        ),
        "claim_allowed": bool(
            len(paired) == 150
            and inference["one_sided_95_upper"] < 0.0
            and all_safety_passed
        ),
        "safety_by_outer": safety,
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = f"""# PS-DFSC confirmation report

- Paired exact-MILP cases: {len(paired)}/150
- Mean cost reduction: {100 * mean_reduction:.3f}%
- CVaR90 reduction: {100 * cvar90_reduction:.3f}%
- CVaR95 reduction: {100 * cvar95_reduction:.3f}%
- One-sided 95% upper bound for paired cost difference: {inference['one_sided_95_upper']:.3f}
- All proper-score safety gates passed: {all_safety_passed}
- Confirmatory improvement claim allowed: {summary['claim_allowed']}

The 1–3% mean-cost and at-least-5% CVaR reductions are engineering targets,
not guaranteed outcomes.
"""
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"report": str(json_path.resolve()), **summary}))


if __name__ == "__main__":
    main()
