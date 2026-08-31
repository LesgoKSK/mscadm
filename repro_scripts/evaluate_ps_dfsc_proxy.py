"""Evaluate the common validation proxy objective for candidate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.fast_differentiable_suc_publication import (
    PublicationDifferentiableSUC,
)
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.selection import empirical_cvar
from ps_dfsc.training import _realized_cost


def _mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value["wind_buses"]),
        zone_capacity_mw=float(value["zone_capacity_mw"]),
    )


def _costs(
    archive,
    prepared,
    mapping,
    *,
    clusters: int,
    strong_convexity: float,
) -> tuple[np.ndarray, list[str]]:
    layer = PublicationDifferentiableSUC(
        scenarios=clusters, strong_convexity=strong_convexity
    )
    observations = prepared["observations"]
    commitments = prepared["commitments"]
    values = []
    statuses = []
    for day in range(len(observations)):
        commitment = commitments[day].astype(np.float64)
        startup, shutdown = commitment_transitions(commitment)
        try:
            planning = layer.plan(
                torch.as_tensor(
                    archive["suc_scenarios"][day], dtype=torch.float64
                ),
                torch.as_tensor(
                    archive["suc_probabilities"][day], dtype=torch.float64
                ),
                torch.as_tensor(commitment, dtype=torch.float64),
                torch.as_tensor(startup, dtype=torch.float64),
                torch.as_tensor(shutdown, dtype=torch.float64),
            )
            truth_wind = torch.as_tensor(
                mapping.transform(observations[day]), dtype=torch.float64
            )
            realized = layer.realize(
                truth_wind,
                torch.as_tensor(commitment, dtype=torch.float64),
                torch.as_tensor(startup, dtype=torch.float64),
                torch.as_tensor(shutdown, dtype=torch.float64),
                planning.day_ahead_dispatch,
                planning.reserve_up,
                planning.reserve_down,
            )
            cost = _realized_cost(
                layer,
                planning,
                realized,
                truth_wind,
                torch.as_tensor(startup, dtype=torch.float64),
            )
            numeric = float(cost.detach().cpu())
            if not np.isfinite(numeric):
                raise FloatingPointError("non-finite proxy cost")
            values.append(numeric)
            statuses.append("ok")
        except Exception as error:
            values.append(np.nan)
            statuses.append(f"failed:{type(error).__name__}:{error}")
        print(
            json.dumps(
                {
                    "day_index": day,
                    "days": len(observations),
                    "status": statuses[-1],
                    "cost": values[-1],
                }
            ),
            flush=True,
        )
    return np.asarray(values, dtype=np.float64), statuses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paired candidate/identity QP validation proxy"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--strong-convexity", type=float, default=1e-4)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = np.load(args.candidate, allow_pickle=False)
    identity = np.load(args.identity, allow_pickle=False)
    prepared = np.load(args.prepared, allow_pickle=False)
    mapping = _mapping(args.mapping)
    if not (
        len(candidate["suc_scenarios"])
        == len(identity["suc_scenarios"])
        == len(prepared["observations"])
    ):
        raise ValueError("candidate, identity and prepared dates do not align")
    candidate_cost, candidate_status = _costs(
        candidate,
        prepared,
        mapping,
        clusters=args.clusters,
        strong_convexity=args.strong_convexity,
    )
    identity_cost, identity_status = _costs(
        identity,
        prepared,
        mapping,
        clusters=args.clusters,
        strong_convexity=args.strong_convexity,
    )
    paired = (
        np.isfinite(candidate_cost)
        & np.isfinite(identity_cost)
        & (np.asarray(candidate_status) == "ok")
        & (np.asarray(identity_status) == "ok")
    )
    if not np.all(paired):
        raise RuntimeError(
            f"proxy evaluation lacks {int((~paired).sum())} paired dates"
        )
    candidate_mean = float(candidate_cost.mean())
    identity_mean = float(identity_cost.mean())
    candidate_cvar = empirical_cvar(candidate_cost, 0.90)
    identity_cvar = empirical_cvar(identity_cost, 0.90)
    objective = (
        candidate_mean / identity_mean
        + args.risk_weight * candidate_cvar / identity_cvar
    )
    days = (
        prepared["days"]
        if "days" in prepared
        else np.arange(len(candidate_cost))
    )
    frame = pd.DataFrame(
        {
            "date": np.asarray(days).astype(str),
            "candidate_cost": candidate_cost,
            "identity_cost": identity_cost,
            "candidate_status": candidate_status,
            "identity_status": identity_status,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "schema": "ps_dfsc_validation_proxy_v1",
        "cases": int(len(frame)),
        "paired_cases": int(paired.sum()),
        "strong_convexity": float(args.strong_convexity),
        "risk_weight": float(args.risk_weight),
        "candidate_mean_cost": candidate_mean,
        "identity_mean_cost": identity_mean,
        "candidate_CVaR90": candidate_cvar,
        "identity_CVaR90": identity_cvar,
        "proxy_objective": float(objective),
        "test_truth_accessed": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output.resolve()), **summary}))


if __name__ == "__main__":
    main()
