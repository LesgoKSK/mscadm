"""Memory-light sequential scheduler for all formal PS-DFSC candidates."""

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
BETA_SPECS = (
    ("beta000", 0.0, 0),
    ("beta025", 0.25, 25),
    ("beta050", 0.5, 50),
    ("beta100", 1.0, 100),
)


def _emit(path: Path, event: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def _wait_for(path: Path, poll_seconds: float, log: Path) -> None:
    last_notice = 0.0
    while not path.is_file():
        now = time.monotonic()
        if now - last_notice >= 600.0:
            _emit(
                log,
                {"event": "waiting_for_prerequisite", "path": str(path)},
            )
            last_notice = now
        time.sleep(poll_seconds)


def _train_command(
    outer: int, name: str, beta: float, seed: int
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "repro_scripts.run_ps_dfsc_canonical_v17",
        "train",
        "--input",
        f"outputs/ps_dfsc/outer{outer}/development_train_prepared.npz",
        "--mapping",
        f"repro_configs/ps_dfsc_mapping_outer{outer}.json",
        "--clusters",
        "20",
        "--beta",
        str(beta),
        "--epochs",
        "50",
        "--warmup-epochs",
        "20",
        "--batch-size",
        "4",
        "--strong-convexity",
        "1e-4",
        "--mip-gap",
        "0.001",
        "--time-limit",
        "600",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--output",
        f"outputs/ps_dfsc/outer{outer}/candidates/{name}.pt",
    ]


def _validate_command(
    checkpoint: Path,
    *,
    outer: int,
    beta: float,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "repro_scripts.validate_ps_dfsc_candidate_v1",
        "--checkpoint",
        str(checkpoint),
        "--outer",
        str(outer),
        "--beta",
        str(beta),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]


def _validate(
    checkpoint: Path,
    *,
    outer: int,
    beta: float,
    seed: int,
    candidate_dir: Path,
) -> tuple[bool, dict | None]:
    validation = candidate_dir / f"{checkpoint.stem}.validation.json"
    stdout = candidate_dir / f"{checkpoint.stem}.validation.stdout.log"
    stderr = candidate_dir / f"{checkpoint.stem}.validation.stderr.log"
    with stdout.open("ab") as out, stderr.open("ab") as err:
        result = subprocess.run(
            _validate_command(
                checkpoint,
                outer=outer,
                beta=beta,
                seed=seed,
                output=validation,
            ),
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    if result.returncode != 0 or not validation.is_file():
        return False, None
    return True, json.loads(validation.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    log = (
        ROOT / "outputs" / "ps_dfsc" / "candidate_scheduler_v19.jsonl"
    )
    _emit(
        log,
        {
            "event": "memory_light_scheduler_started",
            "active_pid": args.active_pid,
        },
    )
    while args.active_pid and psutil.pid_exists(args.active_pid):
        time.sleep(args.poll_seconds)
    for outer in (3, 1, 2):
        outer_dir = ROOT / "outputs" / "ps_dfsc" / f"outer{outer}"
        prepared = outer_dir / "development_train_prepared.npz"
        _wait_for(prepared, args.poll_seconds, log)
        candidate_dir = outer_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for name, beta, seed_offset in BETA_SPECS:
            seed = outer * 10_000 + seed_offset
            checkpoint = candidate_dir / f"{name}.pt"
            if checkpoint.is_file():
                valid, evidence = _validate(
                    checkpoint,
                    outer=outer,
                    beta=beta,
                    seed=seed,
                    candidate_dir=candidate_dir,
                )
                if not valid:
                    raise RuntimeError(
                        f"existing candidate failed validation: {checkpoint}"
                    )
                _emit(
                    log,
                    {
                        "event": "candidate_validated_existing",
                        "outer": outer,
                        "candidate": name,
                        **evidence,
                    },
                )
                continue
            train_stdout = candidate_dir / f"{name}_v17.stdout.log"
            train_stderr = candidate_dir / f"{name}_v17.stderr.log"
            completed = False
            for attempt in range(1, args.max_attempts + 1):
                _emit(
                    log,
                    {
                        "event": "candidate_attempt_started",
                        "outer": outer,
                        "candidate": name,
                        "beta": beta,
                        "seed": seed,
                        "attempt": attempt,
                    },
                )
                with train_stdout.open("ab") as out, train_stderr.open(
                    "ab"
                ) as err:
                    result = subprocess.run(
                        _train_command(outer, name, beta, seed),
                        cwd=ROOT,
                        stdout=out,
                        stderr=err,
                        check=False,
                    )
                valid = False
                evidence = None
                if result.returncode == 0 and checkpoint.is_file():
                    valid, evidence = _validate(
                        checkpoint,
                        outer=outer,
                        beta=beta,
                        seed=seed,
                        candidate_dir=candidate_dir,
                    )
                if valid:
                    _emit(
                        log,
                        {
                            "event": "candidate_completed",
                            "outer": outer,
                            "candidate": name,
                            "attempt": attempt,
                            **evidence,
                        },
                    )
                    completed = True
                    break
                _emit(
                    log,
                    {
                        "event": "candidate_attempt_failed",
                        "outer": outer,
                        "candidate": name,
                        "attempt": attempt,
                        "returncode": result.returncode,
                    },
                )
            if not completed:
                raise RuntimeError(
                    f"candidate failed closed after {args.max_attempts} "
                    f"attempts: outer{outer}/{name}"
                )
    _emit(log, {"event": "all_candidates_completed"})


if __name__ == "__main__":
    main()
