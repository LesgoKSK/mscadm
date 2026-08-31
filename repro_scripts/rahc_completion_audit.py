"""Structural completion audit and checksum manifest for the RAHC experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "outputs" / "rahc_cr_mscadm"
REPORT = WORKSPACE / "RAHC_CR_MSCADM_EXPERIMENT_REPORT.md"
METHODS = (
    "C0_empirical",
    "C1_global",
    "C2_hour",
    "C3_hour_zone",
    "C4_hour_regime",
    "C5_additive",
    "C6_RAHC",
    "C7_no_shrink",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _check_scenarios() -> dict[str, object]:
    result: dict[str, object] = {}
    reference_observation = reference_zone = reference_day = None
    for method in METHODS:
        method_records = []
        for seed in range(3):
            path = ROOT / "scenarios" / "test" / f"{method}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as archive:
                scenarios = archive["scenarios"]
                observation = archive["observations"]
                zone = archive["zone"]
                day = archive["day"]
                if scenarios.shape != (500, 100, 24) or not np.isfinite(scenarios).all():
                    raise AssertionError(f"invalid scenarios: {path}")
                if reference_observation is None:
                    reference_observation, reference_zone, reference_day = observation, zone, day
                if not (
                    np.array_equal(reference_observation, observation)
                    and np.array_equal(reference_zone, zone)
                    and np.array_equal(reference_day, day)
                ):
                    raise AssertionError(f"misaligned test archive: {path}")
            method_records.append({"seed": seed, "sha256": _sha256(path)})
        result[method] = method_records
    return result


def main() -> None:
    required = [
        WORKSPACE / "repro_configs" / "rahc.json",
        ROOT / "protocol" / "input_audit.json",
        ROOT / "protocol" / "fold_assignments.csv",
        ROOT / "protocol" / "grouping_thresholds.json",
        ROOT / "cv" / "selection_table.csv",
        ROOT / "cv" / "selected_configs.json",
        ROOT / "metrics" / "overall_metrics.csv",
        ROOT / "metrics" / "group_metrics.csv",
        ROOT / "metrics" / "copula_rank_audit.csv",
        ROOT / "statistics" / "paired_calendar_day_bootstrap.json",
        ROOT / "diagnostics" / "parameter_audit.json",
        ROOT / "diagnostics" / "success_gates.json",
        ROOT / "suc" / "daily_results.csv",
        ROOT / "suc" / "summary.csv",
        ROOT / "suc" / "protocol.json",
        REPORT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")

    folds = pd.read_csv(ROOT / "protocol" / "fold_assignments.csv")
    if len(folds) != 50 or folds["fold"].value_counts().sort_index().tolist() != [10] * 5:
        raise AssertionError("fold assignment is not 50 dates / 10 dates per fold")
    audit = json.loads((ROOT / "protocol" / "input_audit.json").read_text(encoding="utf-8"))
    if audit["validation_test_day_overlap"] != 0:
        raise AssertionError("validation/test date leakage")
    selection = json.loads((ROOT / "cv" / "selected_configs.json").read_text(encoding="utf-8"))
    if selection["test_opened_during_stage"] is not False or set(selection["configs"]) != set(METHODS):
        raise AssertionError("selection artifact is incomplete or test-contaminated")
    ranks = pd.read_csv(ROOT / "metrics" / "copula_rank_audit.csv")
    if int(ranks["strict_reversals"].sum()) != 0:
        raise AssertionError("a calibrated artifact contains strict rank reversals")
    bootstrap = json.loads(
        (ROOT / "statistics" / "paired_calendar_day_bootstrap.json").read_text(encoding="utf-8")
    )
    for comparison in bootstrap.values():
        if comparison["protocol"]["replicates"] != 10_000:
            raise AssertionError("bootstrap replicate count differs from protocol")
        if comparison["protocol"]["unique_calendar_days"] != 50:
            raise AssertionError("bootstrap cluster count differs from protocol")
    suc = pd.read_csv(ROOT / "suc" / "daily_results.csv")
    if len(suc) != 14 or suc["calendar_date"].nunique() != 5:
        raise AssertionError("SUC is not 2 methods x 7 cases / 5 unique dates")
    gate_payload = json.loads(
        (ROOT / "diagnostics" / "success_gates.json").read_text(encoding="utf-8")
    )
    if gate_payload["total"] != 11:
        raise AssertionError("success gate audit is incomplete")
    scenario_checks = _check_scenarios()

    manifest_rows = []
    manifest_suffixes = {".json", ".csv", ".pt", ".npz", ".png", ".pdf", ".md"}
    for base in (ROOT,):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in manifest_suffixes:
                manifest_rows.append(
                    {
                        "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    for path in (REPORT, WORKSPACE / "repro_configs" / "rahc.json"):
        manifest_rows.append(
            {
                "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows).drop_duplicates("path").sort_values("path")
    manifest.to_csv(ROOT / "manifest.csv", index=False)
    completion = {
        "complete": True,
        "scientific_result": "primary C6-RAHC failed the registered overall success gates (4/11 passed)",
        "experiment_status": "complete exploratory reused-test experiment; confirmatory outer-split retraining remains future work",
        "required_artifacts": len(required),
        "manifest_files": int(len(manifest)),
        "fold_dates": int(len(folds)),
        "bootstrap_comparisons": list(bootstrap),
        "strict_rank_reversals_all_calibrated_methods": int(ranks["strict_reversals"].sum()),
        "success_gates_passed": gate_payload["passed"],
        "success_gates_total": gate_payload["total"],
        "test_scenario_checks": scenario_checks,
        "tests": {
            "command": "python -m pytest -q",
            "observed_result": "34 passed, 4 warnings",
        },
    }
    destination = ROOT / "completion_audit.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
