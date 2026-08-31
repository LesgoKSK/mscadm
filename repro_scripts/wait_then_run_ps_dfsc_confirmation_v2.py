"""Memory-light wait followed by failure-reporting locked confirmation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    marker = ROOT / "outputs" / "ps_dfsc" / "validation_locks.complete.json"
    while not marker.is_file():
        time.sleep(args.poll_seconds)
    import repro_scripts.run_ps_dfsc_locked_confirmation_v2 as confirmation

    sys.argv = [
        sys.argv[0],
        "--poll-seconds",
        str(args.poll_seconds),
        "--attempts",
        str(args.attempts),
    ]
    confirmation.main()


if __name__ == "__main__":
    main()
