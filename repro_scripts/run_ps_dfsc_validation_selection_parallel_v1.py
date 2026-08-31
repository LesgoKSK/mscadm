"""Run the audited validation/locking pipeline for three outers in parallel.

Each outer still executes the original fail-closed sequence serially.  The
parallelism is only across mutually exclusive outer directories, so it does
not change candidate ranking, exact-MILP settings, or lock contents.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import repro_scripts.run_ps_dfsc_validation_selection_v2 as audited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    # Install the v3-config substitutions used by the audited v2 entry point
    # before starting workers.  No worker mutates these globals afterwards.
    audited.pipeline._python = audited.python_with_v3_config
    audited.pipeline._run_step = audited.run_step_with_v3_config
    audited.pipeline._wait_for_candidates(args.poll_seconds)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(audited.pipeline._run_outer, outer, args.attempts): outer
            for outer in (1, 2, 3)
        }
        for future in as_completed(futures):
            outer = futures[future]
            future.result()
            print(
                json.dumps(
                    {
                        "event": "outer_validation_completed",
                        "outer": outer,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    root = audited.pipeline.ROOT
    completion = root / "outputs" / "ps_dfsc" / "validation_locks.complete.json"
    completion.write_text(
        json.dumps(
            {
                "schema": "ps_dfsc_validation_locks_complete_v1",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "locks": {
                    f"outer{outer}": audited.pipeline._sha256(
                        root
                        / "outputs"
                        / "ps_dfsc"
                        / f"outer{outer}"
                        / "lock_manifest.json"
                    )
                    for outer in (1, 2, 3)
                },
                "test_truth_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
