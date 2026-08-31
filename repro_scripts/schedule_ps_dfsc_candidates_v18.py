"""Lightweight sequential scheduler using the CPU-safe v17 trainer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    log = ROOT / "outputs" / "ps_dfsc" / "candidate_scheduler_v18.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "event": "lightweight_wait_started",
                    "active_pid": args.active_pid,
                },
                sort_keys=True,
            )
            + "\n"
        )
    while args.active_pid and psutil.pid_exists(args.active_pid):
        time.sleep(args.poll_seconds)
    import repro_scripts.schedule_ps_dfsc_candidates_v16 as scheduler

    original_command = scheduler._command

    def cpu_safe_command(outer, name, beta, seed):
        command = original_command(outer, name, beta, seed)
        index = command.index(
            "repro_scripts.run_ps_dfsc_canonical_v16"
        )
        command[index] = "repro_scripts.run_ps_dfsc_canonical_v17"
        return command

    scheduler._command = cpu_safe_command
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
