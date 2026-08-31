"""Run locked confirmation for three outers concurrently.

The original confirmation sequence remains serial inside each outer.  Only
the mutually exclusive outer directories are scheduled in parallel; the
absence audit, lock verification, exact solver settings, and final report are
unchanged.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import repro_scripts.run_ps_dfsc_locked_confirmation_v3 as audited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    # Apply the same v3 fallback-reuse and failed-case reporting patches as the
    # canonical entry point before workers are started.
    audited.pipeline._run_step = audited.run_step_with_fallback_reuse
    audited.failure_reporting.pipeline._python_original = (
        audited.failure_reporting.pipeline._python
    )
    audited.failure_reporting.pipeline._python = (
        audited.failure_reporting.confirmation_python
    )
    audited.failure_reporting.pipeline._audit_final = (
        audited.failure_reporting.audit_final_allow_explicit_failures
    )

    marker = audited.failure_reporting._wait_for_locks(args.poll_seconds)
    locks = audited.failure_reporting._load_locks()
    absence = audited.failure_reporting._absence_audit(marker, locks)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                audited.failure_reporting._run_outer,
                outer,
                lock_path,
                lock,
                absence,
                args.attempts,
            ): outer
            for outer, (lock_path, lock) in locks.items()
        }
        outer_artifacts = {}
        for future in as_completed(futures):
            outer = futures[future]
            outer_artifacts[outer] = future.result()
            print(
                json.dumps(
                    {
                        "event": "outer_confirmation_completed",
                        "outer": outer,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    root = audited.failure_reporting.ROOT
    report = root / "outputs" / "ps_dfsc" / "confirmation" / "report.json"
    markdown = report.with_suffix(".md")
    exact_inputs = [
        root
        / "outputs"
        / "ps_dfsc"
        / "confirmation"
        / f"outer{outer}"
        / name
        for outer in (1, 2, 3)
        for name in (
            "exact_ps_dfsc.csv",
            "exact_identity.csv",
            "confirmation_safety.json",
        )
    ]
    audited.failure_reporting._run_step(
        label="final_report",
        command=audited.failure_reporting._python(
            "repro_scripts.ps_dfsc_final_report_publication",
            "--confirmation-root",
            str(root / "outputs" / "ps_dfsc" / "confirmation"),
            "--output-json",
            str(report),
            "--output-markdown",
            str(markdown),
        ),
        inputs=exact_inputs,
        outputs=[report, markdown],
        attempts=args.attempts,
    )
    audited.failure_reporting._audit_final(
        locks, absence, report, markdown, outer_artifacts
    )
    print(
        json.dumps(
            {
                "schema": "ps_dfsc_parallel_confirmation_complete_v1",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "outers": [1, 2, 3],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
