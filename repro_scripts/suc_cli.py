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

    root = Path(args.root)
    scenario_root = root / "scenarios"
    selected: list[tuple[str, Path]] = []
    for model in args.models:
        matches = sorted(scenario_root.glob(f"{model}_test*.npz"))
        if matches:
            selected.append((model, matches[0]))
        else:
            print(f"skip {model}: no scenario archive")
    if not selected:
        raise FileNotFoundError("No requested scenario archive was found")

    lengths = []
    for _, path in selected:
        with np.load(path, allow_pickle=False) as archive:
            lengths.append(len(archive["observations"]))
    common_length = min(lengths)
    common_days = np.random.default_rng(args.seed).choice(
        common_length, size=min(args.days, common_length), replace=False
    )

    records = []
    archive_map: dict[str, str] = {}
    for model, path in selected:
        archive_map[model] = str(path.resolve())
        archive = np.load(path, allow_pickle=False)
        for day_index in common_days:
            reduced, probability = reduce_scenarios(
                archive["scenarios"][day_index], args.clusters, seed=args.seed + int(day_index)
            )
            planned = solve_suc(reduced, probability, time_limit=args.time_limit)
            actual_wind = archive["observations"][day_index][None] * 1200
            realized = solve_suc(
                actual_wind,
                np.ones(1),
                fixed_commitment=planned["commitment"],
                time_limit=args.time_limit,
            )
            record = {
                "model": model,
                "day_index": int(day_index),
                "total_cost": realized["total_cost"],
                "penalty_cost": realized["penalty_cost"],
                "load_shedding": realized["load_shedding"],
                "wind_curtailment": realized["wind_curtailment"],
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    frame = pd.DataFrame(records)
    output = root / "suc"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_results.csv", index=False)
    summary = frame.groupby("model", as_index=False)[
        ["total_cost", "penalty_cost", "load_shedding", "wind_curtailment"]
    ].mean()
    summary.to_csv(output / "table5_suc.csv", index=False)
    protocol = {
        "seed": args.seed,
        "common_day_indices": [int(value) for value in common_days],
        "clusters": args.clusters,
        "time_limit_seconds": args.time_limit,
        "wind_capacity_mw": 1200,
        "archives": archive_map,
        "fair_comparison": "all models use identical day indices",
        "paper_ambiguities": [
            "methods text says seven days while Table 5 caption says 100-day average",
            "K-Means cluster count is not disclosed",
            "mapping from GEFCom zones to six RTS-24 wind farms is not disclosed",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
