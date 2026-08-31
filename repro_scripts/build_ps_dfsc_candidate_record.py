from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble one auditable PS-DFSC validation candidate record"
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-objective", type=float, required=True)
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--exact-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibrated = np.load(args.calibrated)
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    exact = (
        {}
        if args.exact_summary is None
        else json.loads(Path(args.exact_summary).read_text(encoding="utf-8"))
    )
    value = {
        "schema": "ps_dfsc_candidate_record_v1",
        "candidate_id": args.candidate_id,
        "beta": args.beta,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "proxy_objective": args.proxy_objective,
        "mean_ess": float(np.mean(calibrated["ess"])),
        "minimum_ess": float(np.min(calibrated["ess"])),
        "mean_transport": float(np.mean(calibrated["transport_cost"])),
        "maximum_transport": float(np.max(calibrated["transport_cost"])),
        "fallback_days": int(np.sum(calibrated["used_fallback"])),
        "gate": gate,
        "exact_mean_cost": exact.get("realized_total_cost_mean"),
        "exact_cvar90": exact.get("CVaR90"),
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
            }
        )
    )


if __name__ == "__main__":
    main()
