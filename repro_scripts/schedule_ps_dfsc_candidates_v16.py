"""Sequential, fail-closed scheduler for all formal PS-DFSC candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ps_dfsc.manifest import file_sha256
from ps_dfsc.pipeline_runtime_publication import _model_sha256


ROOT = Path(__file__).resolve().parents[1]
BETA_SPECS = (
    ("beta000", 0.0, 0),
    ("beta025", 0.25, 25),
    ("beta050", 0.5, 50),
    ("beta100", 1.0, 100),
)


def _emit(log_path: Path, event: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_path(path: Path, poll_seconds: float, log_path: Path) -> None:
    last_notice = 0.0
    while not path.is_file():
        now = time.monotonic()
        if now - last_notice >= 600.0:
            _emit(
                log_path,
                {"event": "waiting_for_prerequisite", "path": str(path)},
            )
            last_notice = now
        time.sleep(poll_seconds)


def _validate_candidate(
    checkpoint: Path,
    *,
    outer: int,
    beta: float,
    seed: int,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ps_dfsc_checkpoint_v3":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint}")
    config = payload["training_config"]
    expected = {
        "beta": beta,
        "epochs": 50,
        "proper_warmup_epochs": 20,
        "batch_size": 4,
        "strong_convexity": 1.0e-4,
        "seed": seed,
    }
    for key, value in expected.items():
        actual = config[key]
        if isinstance(value, float):
            if not math.isclose(
                float(actual), value, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    f"checkpoint config mismatch for {key}: {actual}"
                )
        elif int(actual) != value:
            raise ValueError(
                f"checkpoint config mismatch for {key}: {actual}"
            )
    if not all(
        bool(torch.isfinite(value).all())
        for value in payload["model_state"].values()
    ):
        raise FloatingPointError(f"non-finite model: {checkpoint}")
    computed_model_hash = _model_sha256(
        _model_from_checkpoint(payload)
    )
    if computed_model_hash != payload["model_state_sha256"]:
        raise ValueError(f"model-state hash mismatch: {checkpoint}")
    history_path = checkpoint.with_suffix(".history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = history.get("epochs", [])
    if len(epochs) != 50:
        raise ValueError(
            f"candidate history does not contain 50 epochs: {checkpoint}"
        )
    if not all(
        math.isfinite(float(value))
        for record in epochs
        for value in record.values()
    ):
        raise FloatingPointError(f"non-finite history: {history_path}")
    summary = payload["commitment_refresh_summary"]
    if (
        int(payload["commitment_refresh_calls"]) < 1
        or int(summary["cases"]) != 100
    ):
        raise ValueError(
            f"candidate lacks complete midpoint refresh: {checkpoint}"
        )
    prepared = (
        ROOT
        / "outputs"
        / "ps_dfsc"
        / f"outer{outer}"
        / "development_train_prepared.npz"
    )
    mapping = (
        ROOT / "repro_configs" / f"ps_dfsc_mapping_outer{outer}.json"
    )
    if payload["training_input_sha256"] != file_sha256(prepared):
        raise ValueError(f"training-input hash mismatch: {checkpoint}")
    if payload["mapping_sha256"] != file_sha256(mapping):
        raise ValueError(f"mapping hash mismatch: {checkpoint}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_state_sha256": payload["model_state_sha256"],
        "history_sha256": file_sha256(history_path),
        "refresh_cases": int(summary["cases"]),
        "refresh_target_gap_pass_rate": float(
            summary["target_gap_pass_rate"]
        ),
    }


def _model_from_checkpoint(payload):
    from ps_dfsc.model import PSDFSCNetwork

    model = PSDFSCNetwork()
    model.load_state_dict(payload["model_state"])
    return model


def _command(outer: int, name: str, beta: float, seed: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "repro_scripts.run_ps_dfsc_canonical_v16",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    log_path = (
        ROOT / "outputs" / "ps_dfsc" / "candidate_scheduler_v16.jsonl"
    )
    if args.active_pid:
        _emit(
            log_path,
            {
                "event": "waiting_for_active_candidate",
                "pid": args.active_pid,
            },
        )
        while _pid_exists(args.active_pid):
            time.sleep(args.poll_seconds)
    for outer in (3, 1, 2):
        prepared = (
            ROOT
            / "outputs"
            / "ps_dfsc"
            / f"outer{outer}"
            / "development_train_prepared.npz"
        )
        _wait_for_path(prepared, args.poll_seconds, log_path)
        candidate_dir = prepared.parent / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for name, beta, seed_offset in BETA_SPECS:
            seed = outer * 10_000 + seed_offset
            checkpoint = candidate_dir / f"{name}.pt"
            if checkpoint.is_file():
                evidence = _validate_candidate(
                    checkpoint, outer=outer, beta=beta, seed=seed
                )
                _emit(
                    log_path,
                    {
                        "event": "candidate_validated_existing",
                        "outer": outer,
                        "candidate": name,
                        **evidence,
                    },
                )
                continue
            stdout_path = candidate_dir / f"{name}_v16.stdout.log"
            stderr_path = candidate_dir / f"{name}_v16.stderr.log"
            completed = False
            for attempt in range(1, args.max_attempts + 1):
                _emit(
                    log_path,
                    {
                        "event": "candidate_attempt_started",
                        "outer": outer,
                        "candidate": name,
                        "beta": beta,
                        "seed": seed,
                        "attempt": attempt,
                    },
                )
                with stdout_path.open("ab") as stdout, stderr_path.open(
                    "ab"
                ) as stderr:
                    result = subprocess.run(
                        _command(outer, name, beta, seed),
                        cwd=ROOT,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                if result.returncode == 0 and checkpoint.is_file():
                    evidence = _validate_candidate(
                        checkpoint,
                        outer=outer,
                        beta=beta,
                        seed=seed,
                    )
                    _emit(
                        log_path,
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
                    log_path,
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
    _emit(log_path, {"event": "all_candidates_completed"})


if __name__ == "__main__":
    main()
