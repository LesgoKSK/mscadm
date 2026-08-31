from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro.suc import reduce_scenarios, solve_suc


DEFAULT_DAYS = [37, 20, 134, 253, 153, 315, 420]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair RTS-24 SUC comparison for CR-MS-CADM")
    parser.add_argument("--root", default="outputs/cr_mscadm")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    sources = {
        "MS-CADM raw": root / "scenarios" / "baseline_mscadm_test_raw.npz",
        "MS-CADM calibrated": root / "scenarios" / "baseline_mscadm_test_calibrated.npz",
        "CR-MS-CADM raw": root / "scenarios" / "full_seed0_test_raw.npz",
        "CR-MS-CADM calibrated": root / "scenarios" / "full_seed0_test_calibrated.npz",
    }
    records: list[dict[str, float | int | str]] = []
    archive_map: dict[str, str] = {}
    for model, path in sources.items():
        archive_map[model] = str(path.resolve())
        archive = np.load(path, allow_pickle=False)
        for day_index in DEFAULT_DAYS:
            reduced, probability = reduce_scenarios(
                archive["scenarios"][day_index], args.clusters, seed=args.seed + day_index
            )
            planned = solve_suc(reduced, probability, time_limit=args.time_limit)
            realized = solve_suc(
                archive["observations"][day_index][None] * 1200.0,
                np.ones(1),
                fixed_commitment=planned["commitment"],
                time_limit=args.time_limit,
            )
            record = {
                "model": model,
                "day_index": day_index,
                "total_cost": float(realized["total_cost"]),
                "penalty_cost": float(realized["penalty_cost"]),
                "load_shedding": float(realized["load_shedding"]),
                "wind_curtailment": float(realized["wind_curtailment"]),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    output = root / "suc"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "daily_results.csv", index=False)
    summary = frame.groupby("model", as_index=False)[
        ["total_cost", "penalty_cost", "load_shedding", "wind_curtailment"]
    ].mean()
    summary.to_csv(output / "summary.csv", index=False)
    protocol = {
        "seed": args.seed,
        "common_day_indices": DEFAULT_DAYS,
        "clusters": args.clusters,
        "time_limit_seconds": args.time_limit,
        "wind_capacity_mw": 1200,
        "archives": archive_map,
        "fairness": "identical days, cluster count, capacity, solver and realized observations",
        "scope_warning": "seven-day sensitivity study inherited from the reconstruction, not a population estimate",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
