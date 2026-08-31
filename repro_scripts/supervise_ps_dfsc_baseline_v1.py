"""Memory-light restart supervisor for one strict identity baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


def emit(path: Path, event: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def command(outer: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "repro_scripts.prepare_ps_dfsc_publication_strict_v3",
        "--input",
        f"outputs/ps_dfsc/base/outer{outer}/pooled/"
        "development_train_M100.npz",
        "--mapping",
        f"repro_configs/ps_dfsc_mapping_outer{outer}.json",
        "--clusters",
        "20",
        "--mip-gap",
        "0.001",
        "--time-limit",
        "600",
        "--cache-dir",
        f"outputs/ps_dfsc/outer{outer}/baseline_cache/train",
        "--training-output",
        f"outputs/ps_dfsc/outer{outer}/development_train_prepared.npz",
        "--identity-output",
        f"outputs/ps_dfsc/outer{outer}/identity_train.npz",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--active-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()
    outer_dir = ROOT / "outputs" / "ps_dfsc" / f"outer{args.outer}"
    prepared = outer_dir / "development_train_prepared.npz"
    identity = outer_dir / "identity_train.npz"
    log = outer_dir / "baseline_supervisor_v1.jsonl"
    emit(
        log,
        {
            "event": "supervisor_started",
            "outer": args.outer,
            "active_pid": args.active_pid,
        },
    )
    while args.active_pid and psutil.pid_exists(args.active_pid):
        time.sleep(args.poll_seconds)
    for attempt in range(1, args.max_attempts + 1):
        if prepared.is_file() and identity.is_file():
            emit(
                log,
                {
                    "event": "baseline_completed",
                    "outer": args.outer,
                    "attempt": attempt - 1,
                },
            )
            return
        stdout = outer_dir / "baseline_train_supervised.stdout.log"
        stderr = outer_dir / "baseline_train_supervised.stderr.log"
        emit(
            log,
            {
                "event": "baseline_attempt_started",
                "outer": args.outer,
                "attempt": attempt,
            },
        )
        with stdout.open("ab") as out, stderr.open("ab") as err:
            result = subprocess.run(
                command(args.outer),
                cwd=ROOT,
                stdout=out,
                stderr=err,
                check=False,
            )
        emit(
            log,
            {
                "event": "baseline_attempt_finished",
                "outer": args.outer,
                "attempt": attempt,
                "returncode": result.returncode,
                "prepared": prepared.is_file(),
                "identity": identity.is_file(),
            },
        )
    raise RuntimeError(
        f"outer{args.outer} baseline failed closed after "
        f"{args.max_attempts} attempts"
    )


if __name__ == "__main__":
    main()
