"""Start the memory-light candidate scheduler after both baselines finish."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
    log = (
        ROOT
        / "outputs"
        / "ps_dfsc"
        / "candidate_after_baselines_v1.jsonl"
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "event": "waiting_for_all_training_baselines",
                    "required": [str(path) for path in required],
                },
                sort_keys=True,
            )
            + "\n"
        )
    while not all(path.is_file() for path in required):
        time.sleep(args.poll_seconds)
    sys.argv = [
        sys.argv[0],
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]
    from repro_scripts.schedule_ps_dfsc_candidates_v19 import main as run

    run()


if __name__ == "__main__":
    main()
