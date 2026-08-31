"""Assemble a hashed validation candidate record from audited artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ps_dfsc.manifest import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a publication PS-DFSC candidate record"
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-summary", required=True)
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--exact-pair-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibrated = np.load(args.calibrated, allow_pickle=False)
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    proxy = json.loads(
        Path(args.proxy_summary).read_text(encoding="utf-8")
    )
    exact = (
        {}
        if args.exact_pair_summary is None
        else json.loads(
            Path(args.exact_pair_summary).read_text(encoding="utf-8")
        )
    )
    if int(proxy["paired_cases"]) != len(calibrated["ess"]):
        raise ValueError("proxy summary does not cover every validation date")
    checkpoint = Path(args.checkpoint).resolve()
    value = {
        "schema": "ps_dfsc_candidate_record_v2",
        "candidate_id": args.candidate_id,
        "beta": float(args.beta),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "proxy_objective": float(proxy["proxy_objective"]),
        "proxy_summary": str(Path(args.proxy_summary).resolve()),
        "proxy_summary_sha256": file_sha256(args.proxy_summary),
        "mean_ess": float(np.mean(calibrated["ess"])),
        "minimum_ess": float(np.min(calibrated["ess"])),
        "mean_transport": float(np.mean(calibrated["transport_cost"])),
        "maximum_transport": float(np.max(calibrated["transport_cost"])),
        "fallback_days": int(np.sum(calibrated["used_fallback"])),
        "gate": gate,
        "gate_sha256": file_sha256(args.gate),
        "calibrated_sha256": file_sha256(args.calibrated),
        "exact_mean_cost": exact.get("exact_mean_cost"),
        "exact_cvar90": exact.get("exact_cvar90"),
        "exact_J_select": exact.get("exact_J_select"),
        "exact_not_worse_than_baseline": exact.get(
            "exact_not_worse_than_baseline"
        ),
        "exact_paired_cases": exact.get("paired_cases"),
        "exact_pair_summary": (
            None
            if args.exact_pair_summary is None
            else str(Path(args.exact_pair_summary).resolve())
        ),
        "exact_pair_summary_sha256": (
            None
            if args.exact_pair_summary is None
            else file_sha256(args.exact_pair_summary)
        ),
        "test_truth_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "candidate_id": args.candidate_id,
                "gate_passed": bool(gate["passed"]),
                "exact_paired_cases": value["exact_paired_cases"],
            }
        )
    )


if __name__ == "__main__":
    main()
