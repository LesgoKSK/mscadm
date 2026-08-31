"""Run locked confirmation for three outers concurrently.

This is the audited v3 confirmation sequence with only the independent outer
directories scheduled in parallel.  All steps inside an outer remain serial,
and the v3 identity-fallback exact-output reuse patch is retained.
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

    # v3 wraps the v2 failure-reporting module, which wraps the v1 pipeline.
    # Keep these aliases explicit so the parallel entry point uses the same
    # audited functions as the canonical v3 entry point.
    audited.pipeline._run_step = audited.run_step_with_fallback_reuse
    fr = audited.failure_reporting
    p = fr.pipeline
    p._python_original = p._python
    p._python = fr.confirmation_python
    p._audit_final = fr.audit_final_allow_explicit_failures

    marker = p._wait_for_locks(args.poll_seconds)
    locks = p._load_locks()
    absence = p._absence_audit(marker, locks)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                p._run_outer,
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

    root = p.ROOT
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
    p._run_step(
        label="final_report",
        command=p._python(
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
    fr.audit_final_allow_explicit_failures(
        locks, absence, report, markdown, outer_artifacts
    )
    print(
        json.dumps(
            {
                "schema": "ps_dfsc_parallel_confirmation_complete_v2",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "outers": [1, 2, 3],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
