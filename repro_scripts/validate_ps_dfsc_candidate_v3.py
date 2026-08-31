"""Validate formal stochastic-day candidate and exact refresh artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ps_dfsc.resumable_training_runtime_v4 import (
    DECISION_BATCHES_PER_EPOCH,
)
from ps_dfsc.stable_training_metrics import (
    SMOOTHING_EPSILON,
    SURROGATE_SCHEMA,
)
from repro_scripts.schedule_ps_dfsc_candidates_v16 import (
    _validate_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    evidence = _validate_candidate(
        checkpoint,
        outer=args.outer,
        beta=args.beta,
        seed=args.seed,
    )
    resume_path = checkpoint.with_suffix(".epoch_resume.pt")
    resume = torch.load(
        resume_path, map_location="cpu", weights_only=False
    )
    metadata = resume["metadata"]
    state = resume["state"]
    if metadata.get("schema") != "ps_dfsc_epoch_resume_metadata_v3":
        raise ValueError("candidate resume metadata schema is invalid")
    if metadata.get("proper_score_surrogate") != SURROGATE_SCHEMA:
        raise ValueError("candidate proper-score surrogate is invalid")
    if not math.isclose(
        float(metadata.get("proper_score_smoothing_epsilon")),
        SMOOTHING_EPSILON,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("candidate smoothing epsilon is invalid")
    expected = {
        "beta": args.beta,
        "seed": args.seed,
        "epochs": 50,
        "warmup_epochs": 20,
        "batch_size": 4,
        "decision_batches_per_epoch": DECISION_BATCHES_PER_EPOCH,
        "midpoint_time_limit_seconds": 600.0,
    }
    for key, expected_value in expected.items():
        actual = metadata[key]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(actual),
                expected_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"candidate metadata mismatch: {key}")
        elif int(actual) != expected_value:
            raise ValueError(f"candidate metadata mismatch: {key}")
    if (
        int(state["next_epoch"]) != 50
        or len(state["history_epochs"]) != 50
        or not bool(resume["midpoint_refresh_completed"])
    ):
        raise ValueError("candidate resume state is incomplete")
    if not all(
        bool(torch.isfinite(value).all())
        for value in state["model_state"].values()
    ):
        raise FloatingPointError(
            "candidate resume model contains a non-finite value"
        )
    final = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    sampling = final.get("training_sampling", {})
    if (
        sampling.get("schema")
        != "ps_dfsc_stochastic_day_training_v1"
        or int(sampling.get("decision_days_per_epoch", -1)) != 8
        or int(sampling.get("midpoint_refresh_days", -1)) != 100
    ):
        raise ValueError("candidate sampling audit is invalid")
    if set(final["model_state"]) != set(state["model_state"]):
        raise ValueError("final/resume model-state keys differ")
    if not all(
        torch.equal(final["model_state"][key], state["model_state"][key])
        for key in final["model_state"]
    ):
        raise ValueError("final and epoch50 model states differ")
    evidence.update(
        {
            "resume_checkpoint": str(resume_path),
            "resume_next_epoch": int(state["next_epoch"]),
            "resume_midpoint_refresh_completed": True,
            "proper_score_surrogate": SURROGATE_SCHEMA,
            "proper_score_smoothing_epsilon": SMOOTHING_EPSILON,
            "decision_batches_per_epoch": DECISION_BATCHES_PER_EPOCH,
            "decision_days_per_epoch": 8,
            "midpoint_refresh_days": 100,
            "midpoint_time_limit_seconds": 600.0,
            "final_resume_model_identical": True,
        }
    )
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
