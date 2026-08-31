"""Run confirmation safety metrics with verified lock provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ps_dfsc.manifest import file_sha256
from ps_dfsc.manifest_publication import verify_lock_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--regime-reference", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock_manifest(lock)
    if int(lock["outer"]) != args.outer:
        raise ValueError("confirmation safety lock outer mismatch")
    command = [
        sys.executable,
        "-u",
        "-m",
        "repro_scripts.run_ps_dfsc_canonical_v17",
        "safety-gate",
        "--candidate",
        args.candidate,
        "--baseline",
        args.baseline,
        "--truth",
        args.truth,
        "--regime-reference",
        args.regime_reference,
        "--bootstrap-samples",
        "10000",
        "--seed",
        str(args.seed),
        "--output",
        args.output,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("confirmation safety command failed")
    output = Path(args.output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": "ps_dfsc_confirmation_safety_v1",
            "split_role": "confirmation",
            "test_truth_accessed": True,
            "lock_path": str(lock_path.resolve()),
            "lock_sha256": file_sha256(lock_path),
        }
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "passed": bool(payload["passed"]),
                "lock_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
