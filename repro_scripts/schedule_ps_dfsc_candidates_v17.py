"""Lightweight waiter before loading the fail-closed v16 scheduler."""

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
    log = ROOT / "outputs" / "ps_dfsc" / "candidate_scheduler_v17.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "lightweight_wait_started",
        "active_pid": args.active_pid,
    }
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    while args.active_pid and psutil.pid_exists(args.active_pid):
        time.sleep(args.poll_seconds)
    sys.argv = [
        sys.argv[0],
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]
    from repro_scripts.schedule_ps_dfsc_candidates_v16 import main as run

    run()


if __name__ == "__main__":
    main()
