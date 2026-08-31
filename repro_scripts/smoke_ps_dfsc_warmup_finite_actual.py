"""Verify two real-data warm-up epochs keep every parameter finite."""

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
            epochs=2,
            proper_warmup_epochs=2,
            batch_size=count,
            strong_convexity=1.0e-4,
            seed=91502,
        ),
        device=args.device,
        commitment_refresh=None,
    )
    records_finite = all(
        np.isfinite(value)
        for record in history.epochs
        for value in record.values()
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )
    if not records_finite or not parameters_finite:
        raise RuntimeError(
            "warm-up produced a non-finite record or model parameter"
        )
    print(
        json.dumps(
            {
                "passed_days": count,
                "epochs": len(history.epochs),
                "records_finite": records_finite,
                "parameters_finite": parameters_finite,
                "last_record": history.epochs[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
