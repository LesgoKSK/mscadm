"""Run a real decision batch and verify the updated model stays finite."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ps_dfsc.metrics import weighted_per_date_metrics
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.training import TrainingConfig
from ps_dfsc.training_v5 import train_calibrator
from repro_scripts.run_ps_dfsc import _load_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = np.load(args.input, allow_pickle=False)
    count = min(args.days, len(source["base_scenarios"]))
    scenarios = source["base_scenarios"][:count]
    observations = source["observations"][:count]
    probability = np.full(scenarios.shape[:2], 1.0 / scenarios.shape[1])
    metrics = weighted_per_date_metrics(
        scenarios, probability, observations
    )
    scores = {
        key: float(np.mean(metrics[key]))
        for key in ("CRPS", "ES", "VS", "ramp_CRPS", "zero_Brier")
    }
    costs = source["baseline_realized_cost"][:count]
    model = PSDFSCNetwork()
    history = train_calibrator(
        model,
        scenarios,
        observations,
        source["assignments"][:count],
        source["commitments"][:count],
        _load_mapping(args.mapping),
        baseline_scores=scores,
        baseline_mean_cost=float(np.mean(costs)),
        baseline_cvar90=float(np.max(costs)),
        config=TrainingConfig(
            beta=0.0,
            epochs=1,
            proper_warmup_epochs=0,
            batch_size=count,
            strong_convexity=1.0e-4,
            seed=91504,
        ),
        device=args.device,
        commitment_refresh=None,
    )
    record = history.epochs[-1]
    records_finite = all(np.isfinite(value) for value in record.values())
    parameters_finite = all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )
    if not records_finite or not parameters_finite:
        raise RuntimeError(
            "decision batch produced a non-finite record or parameter"
        )
    print(
        json.dumps(
            {
                "passed_days": count,
                "records_finite": records_finite,
                "parameters_finite": parameters_finite,
                "record": record,
            }
        )
    )


if __name__ == "__main__":
    main()
