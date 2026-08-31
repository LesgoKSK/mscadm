from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ps_dfsc.safety import GateDecision
from ps_dfsc.selection import (
    CandidateRecord,
    choose_exact_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the locked PS-DFSC safety/exact selection rule"
    )
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--baseline-mean-cost", type=float, required=True)
    parser.add_argument("--baseline-cvar90", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    source_records = []
    for path in args.candidates:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        gate_value = value["gate"]
        gate = GateDecision(
            passed=bool(gate_value["passed"]),
            point_passed=bool(gate_value["point_passed"]),
            confidence_passed=bool(gate_value["confidence_passed"]),
            reasons=tuple(gate_value["reasons"]),
            diagnostics=dict(gate_value["diagnostics"]),
        )
        record = CandidateRecord(
            candidate_id=value["candidate_id"],
            beta=float(value["beta"]),
            proxy_objective=float(value["proxy_objective"]),
            gate=gate,
            mean_ess=float(value["mean_ess"]),
            mean_transport=float(value["mean_transport"]),
            checkpoint=value["checkpoint"],
            exact_mean_cost=(
                None
                if value.get("exact_mean_cost") is None
                else float(value["exact_mean_cost"])
            ),
            exact_cvar90=(
                None
                if value.get("exact_cvar90") is None
                else float(value["exact_cvar90"])
            ),
        )
        candidates.append(record)
        source_records.append(str(Path(path).resolve()))
    decision = choose_exact_candidate(
        candidates,
        baseline_mean_cost=args.baseline_mean_cost,
        baseline_cvar90=args.baseline_cvar90,
    )
    payload = {
        "schema": "ps_dfsc_candidate_selection_v1",
        **asdict(decision),
        "candidate": (
            None if decision.candidate is None else asdict(decision.candidate)
        ),
        "source_records": source_records,
        "baseline_mean_cost": args.baseline_mean_cost,
        "baseline_cvar90": args.baseline_cvar90,
        "test_truth_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "used_identity_fallback": decision.used_identity_fallback,
                "candidate_id": (
                    None
                    if decision.candidate is None
                    else decision.candidate.candidate_id
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
