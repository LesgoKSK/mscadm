"""Start candidates after baselines with final/resume dual validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    required = [
        ROOT
        / "outputs"
        / "ps_dfsc"
        / f"outer{outer}"
        / name
        for outer in (1, 2)
        for name in (
            "development_train_prepared.npz",
            "identity_train.npz",
        )
    ]
    while not all(path.is_file() for path in required):
        time.sleep(args.poll_seconds)
    import repro_scripts.schedule_ps_dfsc_candidates_v19 as scheduler

    original_validate = scheduler._validate_command

    def dual_artifact_validation(
        checkpoint,
        *,
        outer,
        beta,
        seed,
        output,
    ):
        command = original_validate(
            checkpoint,
            outer=outer,
            beta=beta,
            seed=seed,
            output=output,
        )
        index = command.index(
            "repro_scripts.validate_ps_dfsc_candidate_v1"
        )
        command[index] = (
            "repro_scripts.validate_ps_dfsc_candidate_v2"
        )
        return command

    scheduler._validate_command = dual_artifact_validation
    sys.argv = [
        sys.argv[0],
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]
    scheduler.main()


if __name__ == "__main__":
    main()
