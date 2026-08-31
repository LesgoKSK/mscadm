"""Confirmation exact evaluator that reports, rather than hides, failures.

Validation candidate selection remains fail-closed in
``run_ps_dfsc_exact_validation_cached_v1``.  For locked confirmation, however,
the pre-registered protocol requires unsuccessful days to remain explicit in
the main table so paired-solution rates and worst-case-penalty sensitivity can
be reported.  This wrapper suppresses only the final "incomplete" sentinel
raised after the CSV and audited summary have already been written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import repro_scripts.run_ps_dfsc_exact_cached_v2 as exact_v2


def main() -> None:
    original_argv = list(sys.argv)
    output = Path(original_argv[original_argv.index("--output") + 1])
    try:
        exact_v2.main()
    except RuntimeError as error:
        if "exact validation remains incomplete after registered retries" not in str(
            error
        ):
            raise
        summary_path = output.with_suffix(".summary.json")
        if not output.is_file() or not summary_path.is_file():
            raise
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = int(summary["cases"])
        successful = int(summary["successful_cases"])
        if successful >= cases:
            raise
        summary.update(
            {
                "schema": "ps_dfsc_cached_exact_evaluation_v3",
                "confirmation_failure_policy": (
                    "failed days retained explicitly; no replacement"
                ),
                "incomplete_exact_evaluation_reported": True,
                "failed_cases": cases - successful,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(output.resolve()),
                    "cases": cases,
                    "successful_cases": successful,
                    "failed_cases": cases - successful,
                    "failure_policy": (
                        "explicit rows plus worst-case sensitivity"
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
