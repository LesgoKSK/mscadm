from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.evaluation import relaxed_exact_mismatch
from ps_dfsc.exact_suc import solve_two_stage_suc
from ps_dfsc.fast_differentiable_suc import FastDifferentiableSUC
from ps_dfsc.inference import calibrate
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.model import PSDFSCNetwork


def _mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value["wind_buses"]),
        zone_capacity_mw=float(value["zone_capacity_mw"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fixed-commitment QP versus exact MILP mismatch"
    )
    parser.add_argument("--training-archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive = np.load(args.training_archive)
    scenarios = archive["base_scenarios"]
    commitments = archive["commitments"]
    assignments = archive["assignments"]
    days = archive["days"]
    count = len(scenarios) if args.limit is None else min(args.limit, len(scenarios))
    mapping = _mapping(args.mapping)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = PSDFSCNetwork().to(args.device)
    model.load_state_dict(payload["model_state"])
    layer = FastDifferentiableSUC(scenarios=args.clusters)
    grid = layer.system
    energy = np.asarray([item.energy_cost for item in grid.generators])
    startup_cost = np.asarray([item.startup_cost for item in grid.generators])
    rows = []
    for day in range(count):
        distribution = calibrate(
            model,
            scenarios[day],
            mapping,
            clusters=args.clusters,
            assignments=assignments[day],
            device=args.device,
        )
        commitment = commitments[day].astype(np.float64)
        startup, shutdown = commitment_transitions(commitment)
        tensors = [
            torch.as_tensor(value, dtype=torch.float64)
            for value in (commitment, startup, shutdown)
        ]
        planning = layer.plan(
            torch.as_tensor(distribution.suc_scenarios, dtype=torch.float64),
            torch.as_tensor(distribution.suc_probabilities, dtype=torch.float64),
            *tensors,
        )
        pda = planning.day_ahead_dispatch.detach().numpy()
        reserve_up = planning.reserve_up.detach().numpy()
        reserve_down = planning.reserve_down.detach().numpy()
        dispatch = planning.scenario_dispatch.detach().numpy()
        used = planning.used_wind.detach().numpy()
        shed = planning.load_shedding.detach().numpy()
        probability = distribution.suc_probabilities
        relaxed_objective = float(
            np.sum(startup * startup_cost[:, None])
            + np.sum(reserve_up * (0.10 * energy[:, None]))
            + np.sum(reserve_down * (0.05 * energy[:, None]))
            + 500.0
            * (
                planning.reserve_shortage_up.detach().numpy().sum()
                + planning.reserve_shortage_down.detach().numpy().sum()
            )
            + np.sum(
                probability[:, None, None]
                * dispatch
                * energy[None, :, None]
            )
            + 1000.0 * np.sum(probability[:, None] * shed)
            + 80.0
            * np.sum(
                probability[:, None]
                * (
                    distribution.suc_scenarios.sum(axis=1)
                    - used.sum(axis=1)
                )
            )
        )
        exact = solve_two_stage_suc(
            distribution.suc_scenarios,
            probability,
            mip_gap=args.mip_gap,
            time_limit=args.time_limit,
        )
        rows.append(
            {
                "date": str(days[day]),
                **relaxed_exact_mismatch(
                    commitment,
                    pda,
                    reserve_up,
                    reserve_down,
                    exact,
                    relaxed_objective=relaxed_objective,
                ),
                "exact_mip_gap": exact.mip_gap,
                "exact_solve_time": exact.solve_time_seconds,
            }
        )
        print(json.dumps({"day_index": day, **rows[-1]}), flush=True)
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "cases": len(frame),
        **{
            f"{column}_mean": float(frame[column].mean())
            for column in frame.columns
            if column != "date"
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
