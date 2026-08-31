"""Publication runtime with fixed pre-transport scenario assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.inference import calibrate
from ps_dfsc.reduction import fit_fixed_assignments
from repro_scripts.ps_dfsc_publication_runtime import _dates, safety_gate


def calibrate_archive(args) -> None:
    source = np.load(args.input, allow_pickle=False)
    scenarios = source[args.scenario_key]
    days = _dates(source, len(scenarios))
    mapping = pipeline._mapping_from_args(args)
    model = pipeline._load_model(args.checkpoint, args.device)
    assignments = np.stack(
        [
            fit_fixed_assignments(mapping.transform(values), args.clusters)
            for values in scenarios
        ]
    )
    distributions = [
        calibrate(
            model,
            values,
            mapping,
            clusters=args.clusters,
            assignments=assignments[day],
            ess_floor=args.ess_floor,
            normalized_entropy_floor=args.entropy_floor,
            transport_budget=args.transport_budget,
            device=args.device,
        )
        for day, values in enumerate(scenarios)
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        full_scenarios=np.stack([item.full_scenarios for item in distributions]),
        probabilities=np.stack([item.probabilities for item in distributions]),
        suc_scenarios=np.stack([item.suc_scenarios for item in distributions]),
        suc_probabilities=np.stack(
            [item.suc_probabilities for item in distributions]
        ),
        assignments=assignments.astype(np.int64),
        ess=np.asarray([item.ess for item in distributions]),
        entropy=np.asarray([item.entropy for item in distributions]),
        transport_cost=np.asarray(
            [item.transport_cost for item in distributions]
        ),
        used_fallback=np.asarray(
            [item.used_fallback for item in distributions]
        ),
        fallback_reason=np.asarray(
            [item.fallback_reason for item in distributions]
        ),
        groups=np.asarray(
            [
                list(group) + [-1] * (2 - len(group))
                for group in mapping.groups
            ],
            dtype=np.int64,
        ),
        days=days,
        assignment_source=np.asarray("untransported_base_scenarios"),
    )
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "days": len(distributions),
                "fallback_days": sum(
                    item.used_fallback for item in distributions
                ),
                "date_field_preserved": True,
                "assignment_source": "untransported_base_scenarios",
            }
        )
    )


__all__ = ["calibrate_archive", "safety_gate"]
