"""Wait for validation locks without importing PyTorch, then confirm."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    marker = (
        ROOT / "outputs" / "ps_dfsc" / "validation_locks.complete.json"
    )
    while not marker.is_file():
        time.sleep(args.poll_seconds)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "repro_scripts.run_ps_dfsc_locked_confirmation_v1",
            "--poll-seconds",
            str(args.poll_seconds),
            "--attempts",
            str(args.attempts),
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("locked confirmation pipeline failed closed")


if __name__ == "__main__":
    main()
