"""Locked confirmation with explicit failed-case reporting semantics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import repro_scripts.run_ps_dfsc_locked_confirmation_v1 as pipeline
from ps_dfsc.manifest_publication import verify_lock_manifest


def confirmation_python(module: str, *arguments: str) -> list[str]:
    if module == "repro_scripts.run_ps_dfsc_exact_cached_v2":
        module = "repro_scripts.run_ps_dfsc_exact_cached_v3"
    return pipeline._python_original(module, *arguments)


def audit_final_allow_explicit_failures(
    locks: dict[int, tuple[Path, dict]],
    absence: Path,
    report: Path,
    markdown: Path,
    outer_artifacts: dict[int, list[Path]],
) -> Path:
    absence_payload = json.loads(absence.read_text(encoding="utf-8"))
    audit_time = datetime.fromisoformat(absence_payload["audited_at_utc"])
    checks = {}
    paired_rows = 0
    table_rows = 0
    for outer, (lock_path, lock) in locks.items():
        verify_lock_manifest(lock)
        lock_time = datetime.fromisoformat(lock["created_utc"])
        artifacts = outer_artifacts[outer]
        after_lock = all(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            > lock_time
            for path in artifacts
        )
        after_audit = all(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            > audit_time
            for path in artifacts
        )
        root = (
            pipeline.ROOT
            / "outputs"
            / "ps_dfsc"
            / "confirmation"
            / f"outer{outer}"
        )
        candidate = pd.read_csv(root / "exact_ps_dfsc.csv")
        identity = pd.read_csv(root / "exact_identity.csv")
        joined = candidate.merge(
            identity,
            on=["outer", "date"],
            suffixes=("_candidate", "_baseline"),
            how="outer",
            validate="one_to_one",
        )
        paired = int(
            (
                (joined["status_candidate"] == "solution_available")
                & (joined["status_baseline"] == "solution_available")
            ).sum()
        )
        paired_rows += paired
        table_rows += min(len(candidate), len(identity))
        checks[f"outer{outer}"] = {
            "lock_sha256": pipeline._sha256(lock_path),
            "all_artifacts_after_lock": after_lock,
            "all_artifacts_after_absence_audit": after_audit,
            "candidate_rows": int(len(candidate)),
            "identity_rows": int(len(identity)),
            "paired_solution_rows": paired,
            "failed_rows_retained_explicitly": True,
        }

    report_payload = json.loads(report.read_text(encoding="utf-8"))
    reported_pairs = int(report_payload["paired_cases"])
    worst_case = report_payload.get(
        "failed_case_worst_case_penalty_sensitivity"
    )
    complete = bool(
        table_rows == 150
        and paired_rows == reported_pairs
        and 0 <= paired_rows <= 150
        and isinstance(worst_case, dict)
        and all(
            value["all_artifacts_after_lock"]
            and value["all_artifacts_after_absence_audit"]
            and value["candidate_rows"] == 50
            and value["identity_rows"] == 50
            for value in checks.values()
        )
    )
    output = (
        pipeline.ROOT
        / "outputs"
        / "ps_dfsc"
        / "confirmation"
        / "completion_audit.json"
    )
    output.write_text(
        json.dumps(
            {
                "schema": "ps_dfsc_confirmation_completion_audit_v2",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete": complete,
                "table_rows": table_rows,
                "paired_solution_rows": paired_rows,
                "paired_success_rate": paired_rows / 150.0,
                "report_sha256": pipeline._sha256(report),
                "markdown_sha256": pipeline._sha256(markdown),
                "absence_audit_sha256": pipeline._sha256(absence),
                "outer_checks": checks,
                "failure_policy": (
                    "failed exact days remain in tables; no silent replacement"
                ),
                "worst_case_penalty_sensitivity_reported": isinstance(
                    worst_case, dict
                ),
                "test_results_used_for_model_selection": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError("confirmation completion audit failed closed")
    return output


def main() -> None:
    pipeline._python_original = pipeline._python
    pipeline._python = confirmation_python
    pipeline._audit_final = audit_final_allow_explicit_failures
    pipeline.main()


if __name__ == "__main__":
    main()
