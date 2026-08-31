"""Locked confirmation with audited exact reuse for identity fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import repro_scripts.run_ps_dfsc_locked_confirmation_v1 as pipeline
import repro_scripts.run_ps_dfsc_locked_confirmation_v2 as failure_reporting
from ps_dfsc.manifest import file_sha256


ORIGINAL_RUN_STEP = pipeline._run_step
EXACT_ARRAYS = (
    "suc_scenarios",
    "suc_probabilities",
)


def _exact_inputs_identical(candidate: Path, identity: Path) -> bool:
    with np.load(candidate, allow_pickle=False) as left, np.load(
        identity, allow_pickle=False
    ) as right:
        return all(
            key in left
            and key in right
            and np.array_equal(left[key], right[key])
            for key in EXACT_ARRAYS
        )


def _write_fallback_identity_reuse(**kwargs) -> None:
    outputs = kwargs["outputs"]
    identity_csv, identity_summary = outputs
    candidate_csv = identity_csv.with_name("exact_ps_dfsc.csv")
    candidate_summary = candidate_csv.with_suffix(".summary.json")
    identity_distribution = Path(kwargs["inputs"][1])
    candidate_distribution = identity_distribution.with_name(
        "ps_dfsc_test.npz"
    )
    if not candidate_csv.is_file() or not candidate_summary.is_file():
        raise RuntimeError(
            "fallback identity reuse requires completed candidate exact output"
        )
    if not _exact_inputs_identical(
        candidate_distribution, identity_distribution
    ):
        raise RuntimeError(
            "identity fallback distributions are not exact-SUC identical"
        )
    frame = pd.read_csv(candidate_csv)
    frame["method"] = "MM-JDWind identity"
    frame["used_fallback"] = False
    frame["fallback_reason"] = ""
    frame.to_csv(identity_csv, index=False)
    summary = json.loads(
        candidate_summary.read_text(encoding="utf-8")
    )
    summary.update(
        {
            "schema": "ps_dfsc_cached_exact_evaluation_v3",
            "method": "MM-JDWind identity",
            "calibrated_sha256": file_sha256(identity_distribution),
            "cache_reused_cases": int(summary["cases"]),
            "newly_solved_cases": 0,
            "identity_fallback_exact_reuse": True,
            "exact_reuse_source": str(candidate_csv.resolve()),
            "exact_reuse_source_sha256": file_sha256(candidate_csv),
            "reuse_condition": (
                "suc_scenarios and suc_probabilities array_equal"
            ),
            "test_truth_accessed": True,
        }
    )
    identity_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logs = pipeline.ROOT / "outputs" / "ps_dfsc" / "confirmation_automation"
    logs.mkdir(parents=True, exist_ok=True)
    done = logs / f"{kwargs['label']}.done.json"
    payload = {
        "schema": "ps_dfsc_confirmation_step_v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": kwargs["label"],
        "attempt": 0,
        "command": kwargs["command"],
        "input_sha256": pipeline._hashes(kwargs["inputs"]),
        "output_sha256": pipeline._hashes(outputs),
        "identity_fallback_exact_reuse": True,
        "no_solver_result_substitution": True,
    }
    temporary = done.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(done)


def run_step_with_fallback_reuse(**kwargs) -> None:
    if kwargs.get("label", "").endswith("_identity_exact"):
        lock_path = Path(kwargs["inputs"][0])
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if bool(lock["used_identity_fallback"]):
            _write_fallback_identity_reuse(**kwargs)
            return
    ORIGINAL_RUN_STEP(**kwargs)


def main() -> None:
    pipeline._run_step = run_step_with_fallback_reuse
    failure_reporting.main()


if __name__ == "__main__":
    main()
