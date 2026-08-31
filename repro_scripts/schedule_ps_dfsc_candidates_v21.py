"""Corrected sequential scheduler for stochastic-day formal candidates."""

from __future__ import annotations

import repro_scripts.schedule_ps_dfsc_candidates_v19 as scheduler


ORIGINAL_TRAIN = scheduler._train_command
ORIGINAL_VALIDATE = scheduler._validate_command


def train_command(
    outer: int, name: str, beta: float, seed: int
) -> list[str]:
    command = ORIGINAL_TRAIN(outer, name, beta, seed)
    module_index = command.index(
        "repro_scripts.run_ps_dfsc_canonical_v17"
    )
    command[module_index] = "repro_scripts.run_ps_dfsc_canonical_v18"
    time_index = command.index("--time-limit") + 1
    command[time_index] = "600"
    return command


def validate_command(
    checkpoint,
    *,
    outer: int,
    beta: float,
    seed: int,
    output,
) -> list[str]:
    command = ORIGINAL_VALIDATE(
        checkpoint,
        outer=outer,
        beta=beta,
        seed=seed,
        output=output,
    )
    module_index = command.index(
        "repro_scripts.validate_ps_dfsc_candidate_v1"
    )
    command[module_index] = (
        "repro_scripts.validate_ps_dfsc_candidate_v3"
    )
    return command


def main() -> None:
    scheduler._train_command = train_command
    scheduler._validate_command = validate_command
    scheduler.main()


if __name__ == "__main__":
    main()
