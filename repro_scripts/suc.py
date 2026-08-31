from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro.suc import reduce_scenarios, solve_suc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reconstructed RTS-24 two-stage SUC experiment")
    parser.add_argument("--root", default="outputs/full_reproduction")
    parser.add_argument("--models", nargs="*", default=["qrgbm", "wgan", "vae", "nf", "ddpm", "mscadm"])
    parser.add_argument("--days", type=int, default=7, help="Paper text says seven days but Table 5 says 100-day average")
    parser.add_argument("--clusters", type=int, default=10, help="The paper denotes k but does not disclose its value")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=300)
    args = parser.parse_args()
    root = Path(args.root); scenario_root = root / "scenarios"
    generator = np.random.default_rng(args.seed)
    records = []
    for model in args.models:
        matches = sorted(scenario_root.glob(f"{model}_test*.npz"))
        if not matches:
            print(f"skip {model}: no scenario archive")
            continue
        archive = np.load(matches[0], allow_pickle=False)
        chosen = generator.choice(len(archive["observations"]), size=min(args.days, len(archive["observations"])), replace=False)
        daily = []
        for day_index in chosen:
            reduced, probability = reduce_scenarios(archive["scenarios"][day_index], args.clusters, seed=args.seed + int(day_index))
            planned = solve_suc(reduced, probability, time_limit=args.time_limit)
            actual_wind = archive["observations"][day_index][None] * 1200
            realized = solve_suc(
                actual_wind,
                np.ones(1),
                fixed_commitment=planned["commitment"],
                time_limit=args.time_limit,
            )
            daily.append({
                "model": model,
                "day_index": int(day_index),
                "total_cost": realized["total_cost"],
                "penalty_cost": realized["penalty_cost"],
                "load_shedding": realized["load_shedding"],
                "wind_curtailment": realized["wind_curtailment"],
            })
            print(json.dumps(daily[-1]), flush=True)
        records.extend(daily)
    frame = pd.DataFrame(records)
    output = root / "suc"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_results.csv", index=False)
    if len(frame):
        summary = frame.groupby("model", as_index=False)[["total_cost","penalty_cost","load_shedding","wind_curtailment"]].mean()
        summary.to_csv(output / "table5_suc.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
