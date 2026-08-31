"""Add locked confirmation provenance to the restartable exact evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ps_dfsc.manifest import file_sha256
from ps_dfsc.manifest_publication import verify_lock_manifest
import repro_scripts.run_ps_dfsc_exact_validation_cached_v1 as cached


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--split-role",
        choices=("validation", "confirmation"),
        required=True,
    )
    parser.add_argument("--lock")
    known, remaining = parser.parse_known_args()
    lock = None
    lock_path = None
    if known.split_role == "confirmation":
        if known.lock is None:
            raise ValueError("confirmation exact evaluation requires a lock")
        lock_path = Path(known.lock)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        verify_lock_manifest(lock)
    elif known.lock is not None:
        raise ValueError("validation exact evaluation must not receive a lock")
    output_index = remaining.index("--output") + 1
    output = Path(remaining[output_index])
    cache_index = remaining.index("--cache-dir") + 1
    cache_dir = Path(remaining[cache_index])
    outer_index = remaining.index("--outer") + 1
    outer = int(remaining[outer_index])
    if lock is not None and int(lock["outer"]) != outer:
        raise ValueError("confirmation lock outer mismatch")
    sys.argv = [sys.argv[0], *remaining]
    error = None
    try:
        cached.main()
    except Exception as caught:
        error = caught
    provenance = {
        "split_role": known.split_role,
        "test_truth_accessed": known.split_role == "confirmation",
        "lock_path": (
            None if lock_path is None else str(lock_path.resolve())
        ),
        "lock_sha256": (
            None if lock_path is None else file_sha256(lock_path)
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "schema": "ps_dfsc_cached_exact_evaluation_v2",
                **provenance,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    for path in cache_dir.glob("day_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evaluation_provenance"] = provenance
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
