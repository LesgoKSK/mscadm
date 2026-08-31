"""Fail-closed publication candidate selection from hashed records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ps_dfsc.manifest import file_sha256
from ps_dfsc.safety import GateDecision
from ps_dfsc.selection import (
    CandidateRecord,
    choose_exact_candidate,
    proxy_shortlist,
)


def _verify_record_artifacts(value: dict) -> None:
    checks = [
        (value["checkpoint"], value["checkpoint_sha256"]),
        (value["proxy_summary"], value["proxy_summary_sha256"]),
    ]
    if value.get("exact_pair_summary") is not None:
        checks.append(
            (
                value["exact_pair_summary"],
                value["exact_pair_summary_sha256"],
            )
        )
    for path, digest in checks:
        if file_sha256(path) != digest:
            raise ValueError(f"candidate record artifact changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the locked PS-DFSC validation candidate"
    )
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    payloads = {}
    for path in args.candidates:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        _verify_record_artifacts(value)
        gate_value = value["gate"]
        gate = GateDecision(
            passed=bool(gate_value["passed"]),
            point_passed=bool(gate_value["point_passed"]),
            confidence_passed=bool(gate_value["confidence_passed"]),
            reasons=tuple(gate_value["reasons"]),
            diagnostics=dict(gate_value["diagnostics"]),
        )
        complete_exact = (
            value.get("exact_pair_summary") is not None
            and int(value.get("exact_paired_cases") or 0) == 50
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
                float(value["exact_mean_cost"]) if complete_exact else None
            ),
            exact_cvar90=(
                float(value["exact_cvar90"]) if complete_exact else None
            ),
        )
        candidates.append(record)
        payloads[record.candidate_id] = value
    shortlist = proxy_shortlist(candidates)
    for item in shortlist:
        if payloads[item.candidate_id].get("exact_pair_summary") is None:
            raise RuntimeError(
                f"shortlisted candidate lacks exact evaluation: "
                f"{item.candidate_id}"
            )
    exact_summaries = [
        json.loads(
            Path(payloads[item.candidate_id]["exact_pair_summary"]).read_text(
                encoding="utf-8"
            )
        )
        for item in shortlist
        if int(payloads[item.candidate_id].get("exact_paired_cases") or 0) == 50
    ]
    if exact_summaries:
        baseline_mean = float(exact_summaries[0]["baseline_mean_cost"])
        baseline_cvar = float(exact_summaries[0]["baseline_cvar90"])
        for summary in exact_summaries[1:]:
            if not (
                abs(float(summary["baseline_mean_cost"]) - baseline_mean)
                <= 1e-8 * max(abs(baseline_mean), 1.0)
                and abs(float(summary["baseline_cvar90"]) - baseline_cvar)
                <= 1e-8 * max(abs(baseline_cvar), 1.0)
            ):
                raise ValueError("paired exact summaries disagree on baseline")
    else:
        baseline_mean = 1.0
        baseline_cvar = 1.0
    decision = choose_exact_candidate(
        candidates,
        baseline_mean_cost=baseline_mean,
        baseline_cvar90=baseline_cvar,
    )
    payload = {
        "schema": "ps_dfsc_candidate_selection_v2",
        **asdict(decision),
        "candidate": (
            None if decision.candidate is None else asdict(decision.candidate)
        ),
        "source_records": {
            str(Path(path).resolve()): file_sha256(path)
            for path in args.candidates
        },
        "baseline_mean_cost": baseline_mean,
        "baseline_cvar90": baseline_cvar,
        "exact_pairing_required": 50,
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
                "shortlist": [item.candidate_id for item in shortlist],
            }
        )
    )


if __name__ == "__main__":
    main()
