"""Smoke-test the Clarabel QP on actual prepared PS-DFSC days."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.fast_differentiable_suc_publication_v3 import (
    ClarabelPublicationDifferentiableSUC,
)
from repro_scripts.run_ps_dfsc import _load_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()
    source = np.load(args.input, allow_pickle=False)
    mapping = _load_mapping(args.mapping)
    scenarios = source["base_scenarios"]
    observations = source["observations"]
    assignments = source["assignments"]
    commitments = source["commitments"]
    clusters = int(assignments.max()) + 1
    layer = ClarabelPublicationDifferentiableSUC(
        scenarios=clusters, strong_convexity=1e-4
    )
    records = []
    for day in range(min(args.days, len(scenarios))):
        farm = mapping.transform(scenarios[day])
        reduced = np.stack(
            [farm[assignments[day] == cluster].mean(axis=0)
             for cluster in range(clusters)]
        )
        counts = np.bincount(assignments[day], minlength=clusters)
        probability = torch.tensor(
            counts / counts.sum(), dtype=torch.double, requires_grad=True
        )
        wind = torch.tensor(reduced, dtype=torch.double, requires_grad=True)
        commitment = commitments[day]
        startup, shutdown = commitment_transitions(commitment)
        commitment_t = torch.tensor(commitment, dtype=torch.double)
        startup_t = torch.tensor(startup, dtype=torch.double)
        shutdown_t = torch.tensor(shutdown, dtype=torch.double)
        planned = layer.plan(
            wind,
            probability,
            commitment_t,
            startup_t,
            shutdown_t,
        )
        truth = torch.tensor(
            mapping.transform(observations[day]), dtype=torch.double
        )
        realized = layer.realize(
            truth,
            commitment_t,
            startup_t,
            shutdown_t,
            planned.day_ahead_dispatch,
            planned.reserve_up,
            planned.reserve_down,
        )
        loss = (
            realized.dispatch.sum()
            + 0.1 * planned.reserve_up.sum()
            + 0.1 * planned.reserve_down.sum()
            + 10.0 * realized.load_shedding.sum()
        )
        loss.backward()
        record = {
            "day": day,
            "finite": bool(
                torch.isfinite(loss)
                and torch.isfinite(probability.grad).all()
                and torch.isfinite(wind.grad).all()
            ),
            "probability_gradient_l1": float(
                probability.grad.abs().sum()
            ),
            "wind_gradient_l1": float(wind.grad.abs().sum()),
            "maximum_dispatch_mw": float(
                planned.day_ahead_dispatch.max()
            ),
        }
        if (
            not record["finite"]
            or record["probability_gradient_l1"] <= 0.0
            or record["wind_gradient_l1"] <= 0.0
            or record["maximum_dispatch_mw"] <= 1.0
        ):
            raise RuntimeError(f"actual-data QP smoke failed: {record}")
        records.append(record)
        print(json.dumps(record), flush=True)
    print(json.dumps({"passed_days": len(records)}))


if __name__ == "__main__":
    main()

