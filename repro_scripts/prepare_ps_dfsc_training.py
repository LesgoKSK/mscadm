from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ps_dfsc.exact_suc import evaluate_realized, solve_two_stage_suc
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.reduction import fit_fixed_assignments, weighted_cluster_reduction


def load_mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value.get("wind_buses", (3, 5, 7, 16, 21, 23))),
        zone_capacity_mw=float(value.get("zone_capacity_mw", 120.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare identity baseline, commitments and cost normalizers"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--scenario-key", default="scenarios")
    parser.add_argument("--observation-key", default="observations")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--identity-output", required=True)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    scenarios = source[args.scenario_key].astype(np.float64)
    observations = source[args.observation_key].astype(np.float64)
    days = source["day"] if "day" in source else np.arange(len(scenarios))
    if scenarios.ndim != 5 or scenarios.shape[2:] != (10, 24):
        raise ValueError("input scenarios must have shape [day,member,10,24]")
    if observations.shape != (len(scenarios), 10, 24):
        raise ValueError("observations do not align with scenarios")
    mapping = load_mapping(args.mapping)
    members = scenarios.shape[1]
    uniform = np.full(members, 1.0 / members)
    assignments = []
    reduced_scenarios = []
    reduced_probabilities = []
    commitments = []
    baseline_costs = []
    planned_gaps = []
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for day in range(len(scenarios)):
        mapped = mapping.transform(scenarios[day])
        labels = fit_fixed_assignments(mapped, args.clusters)
        reduced, probability = weighted_cluster_reduction(
            mapped, uniform, labels
        )
        assignments.append(labels)
        reduced_scenarios.append(reduced)
        reduced_probabilities.append(probability)
        cache_path = cache / f"day_{day:03d}.npz"
        if cache_path.exists():
            stored = np.load(cache_path)
            commitment = stored["commitment"]
            realized_cost = float(stored["realized_total_cost"])
            gap = float(stored["planned_mip_gap"])
        else:
            planned = solve_two_stage_suc(
                reduced,
                probability,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
            truth_wind = mapping.transform(observations[day])
            realized = evaluate_realized(
                planned.first_stage,
                truth_wind,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
            commitment = planned.first_stage.commitment
            realized_cost = realized.total_cost
            gap = planned.mip_gap
            np.savez_compressed(
                cache_path,
                commitment=commitment,
                realized_total_cost=np.asarray(realized_cost),
                planned_total_cost=np.asarray(planned.total_cost),
                planned_mip_gap=np.asarray(gap),
                planned_solve_time=np.asarray(planned.solve_time_seconds),
                realized_mip_gap=np.asarray(realized.mip_gap),
                realized_solve_time=np.asarray(realized.solve_time_seconds),
            )
        commitments.append(commitment)
        baseline_costs.append(realized_cost)
        planned_gaps.append(gap)
        print(
            json.dumps(
                {
                    "day_index": day,
                    "days": len(scenarios),
                    "realized_total_cost": realized_cost,
                    "planned_mip_gap": gap,
                }
            ),
            flush=True,
        )
    training_output = Path(args.training_output)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        training_output,
        base_scenarios=scenarios.astype(np.float32),
        observations=observations.astype(np.float32),
        assignments=np.stack(assignments).astype(np.int64),
        commitments=np.stack(commitments).astype(np.float32),
        baseline_realized_cost=np.asarray(baseline_costs, dtype=np.float64),
        planned_mip_gap=np.asarray(planned_gaps, dtype=np.float64),
        days=days,
    )
    identity_output = Path(args.identity_output)
    identity_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        identity_output,
        full_scenarios=scenarios.astype(np.float32),
        probabilities=np.broadcast_to(uniform, scenarios.shape[:2]).copy(),
        suc_scenarios=np.stack(reduced_scenarios),
        suc_probabilities=np.stack(reduced_probabilities),
        ess=np.full(len(scenarios), members, dtype=np.float64),
        entropy=np.full(len(scenarios), np.log(members), dtype=np.float64),
        transport_cost=np.zeros(len(scenarios), dtype=np.float64),
        used_fallback=np.zeros(len(scenarios), dtype=bool),
        fallback_reason=np.full(len(scenarios), "", dtype="<U1"),
        days=days,
    )
    print(
        json.dumps(
            {
                "training_output": str(training_output.resolve()),
                "identity_output": str(identity_output.resolve()),
                "days": len(scenarios),
                "baseline_mean_cost": float(np.mean(baseline_costs)),
            }
        )
    )


if __name__ == "__main__":
    main()
