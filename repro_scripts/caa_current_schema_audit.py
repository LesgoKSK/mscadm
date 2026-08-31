"""Fail-closed completion audit for the current frozen CAA-RAHC schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "outputs" / "caa_rahc"
FAMILIES = tuple(f"A{i}" for i in range(7))
EXPECTED_ANALYTIC = {
    "structural_pi0", "structural_pi1", "A2_pi0", "A2_pi1",
    "A4_pi0", "A4_pi1", "A5_pi0", "A5_pi1", "A6_pi0", "A6_pi1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def numeric_leaves(value: object) -> list[float]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in numeric_leaves(child)]
    if isinstance(value, list):
        return [item for child in value for item in numeric_leaves(child)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, **details: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    lock_path = ROOT / "selection.lock.json"
    analysis_path = ROOT / "calibration" / "selection.audit.json"
    absence_path = ROOT / "calibration" / "preselection_test_absence.audit.json"
    lock = read_json(lock_path)
    absence = read_json(absence_path)
    lock_sha = sha256(lock_path)
    analysis_sha = sha256(analysis_path)
    locked_at = stamp(lock["locked_at_utc"])
    check(
        "selection_hash_chain",
        lock["analysis"]["sha256"] == analysis_sha,
        lock_sha256=lock_sha,
        analysis_sha256=analysis_sha,
    )
    check(
        "preselection_test_artifact_absence",
        absence["matching_count"] == 0
        and not absence["selection_lock_exists"]
        and stamp(absence["generated_at_utc"]) < locked_at,
        scope=absence["scope"],
        generated_at_utc=absence["generated_at_utc"],
        locked_at_utc=lock["locked_at_utc"],
    )

    checkpoint_count = 0
    checkpoint_ok = True
    history_ok = True
    for outer in range(1, 4):
        for seed in range(3):
            run = ROOT / f"outer{outer}" / "runs" / "full" / f"seed{seed}"
            model = run / "final.pt"
            audit = read_json(run / "final.pt.audit.json")
            history = read_json(run / "history.json")
            checkpoint_count += 1
            checkpoint_ok &= (
                sha256(model) == audit["checkpoint_sha256"]
                and audit["training_step"] == 18000
                and audit["resume"] is False
                and audit["outer"] == outer
                and audit["model_seed"] == seed
            )
            leaves = np.asarray(numeric_leaves(history), dtype=float)
            history_ok &= leaves.size > 0 and np.isfinite(leaves).all()
    check("nine_final_checkpoints", checkpoint_ok and checkpoint_count == 9,
          count=checkpoint_count, training_step=18000, resume=False)
    check("nine_finite_training_histories", history_ok, count=checkpoint_count)

    raw_count = 0
    raw_ok = True
    raw_hashes: dict[tuple[int, int, str], str] = {}
    test_after_lock = True
    for outer in range(1, 4):
        for seed in range(3):
            for split in ("calibration", "test"):
                path = ROOT / f"outer{outer}" / "scenarios" / f"full_seed{seed}_{split}_raw.npz"
                audit = read_json(Path(str(path) + ".audit.json"))
                digest = sha256(path)
                raw_hashes[(outer, seed, split)] = digest
                with np.load(path, allow_pickle=False) as stored:
                    scenarios = stored["scenarios"]
                    observations = stored["observations"]
                    raw_ok &= (
                        scenarios.shape == (500, 100, 24)
                        and scenarios.dtype == np.float32
                        and observations.shape == (500, 24)
                        and np.isfinite(scenarios).all()
                        and np.isfinite(observations).all()
                        and float(scenarios.min()) >= 0.0
                        and float(scenarios.max()) <= 1.0
                    )
                raw_ok &= digest == audit["archive_sha256"]
                if split == "test":
                    test_after_lock &= datetime.fromtimestamp(
                        path.stat().st_mtime, tz=locked_at.tzinfo
                    ) > locked_at
                raw_count += 1
    check("eighteen_raw_archives", raw_ok and raw_count == 18, count=raw_count)
    check("test_raw_created_after_lock", test_after_lock, test_count=9)

    final_count = 0
    final_ok = True
    exact_fallback_ok = True
    atom_count = 0
    atom_ok = True
    manifest_ok = True
    diagnostic_ok = True
    for outer in range(1, 4):
        scenarios_dir = ROOT / f"outer{outer}" / "scenarios"
        manifest = read_json(scenarios_dir / "caa_test_final_manifest.json")
        manifest_ok &= (
            manifest["families"] == list(FAMILIES)
            and manifest["selection_lock_sha256"] == lock_sha
            and manifest["selection_analysis_sha256"] == analysis_sha
            and manifest["generated_after_selection_lock"] is True
            and stamp(manifest["generated_at_utc"]) > locked_at
        )
        diagnostics = read_json(scenarios_dir / "caa_test_final_diagnostics.json")
        for record in diagnostics["records"]:
            for key, value in record.items():
                if key.startswith("rank_audit"):
                    diagnostic_ok &= value["strict_reversals"] == 0
            transform = record.get("transformation", {})
            diagnostic_ok &= transform.get("central_values_changed_by_tail", 0) == 0
            diagnostic_ok &= transform.get("nonfinite_input_values", 0) == 0
            diagnostic_ok &= transform.get("nonfinite_intermediate_values", 0) == 0
        for seed in range(3):
            cached: dict[str, np.ndarray] = {}
            for family in FAMILIES:
                path = scenarios_dir / f"{family}_seed{seed}_test_final.npz"
                audit = read_json(Path(str(path) + ".audit.json"))
                with np.load(path, allow_pickle=False) as stored:
                    scenarios = stored["scenarios"]
                    cached[family] = scenarios.copy() if family in {"A0", "A1"} else np.empty(0)
                    final_ok &= (
                        scenarios.shape == (500, 100, 24)
                        and np.isfinite(scenarios).all()
                        and float(scenarios.min()) >= 0.0
                        and float(scenarios.max()) <= 1.0
                    )
                final_ok &= (
                    sha256(path) == audit["archive_sha256"]
                    and audit["selection_lock_sha256"] == lock_sha
                    and audit["selection_analysis_sha256"] == analysis_sha
                    and audit["source_test_raw_sha256"] == raw_hashes[(outer, seed, "test")]
                    and audit["generated_after_selection_lock"] is True
                )
                final_count += 1
            exact_fallback_ok &= np.array_equal(cached["A0"], cached["A1"])

            atom_path = scenarios_dir / f"analytic_atoms_seed{seed}_test_final.npz"
            atom_audit = read_json(Path(str(atom_path) + ".audit.json"))
            with np.load(atom_path, allow_pickle=False) as stored:
                atom_ok &= EXPECTED_ANALYTIC.issubset(stored.files)
                for key in EXPECTED_ANALYTIC:
                    values = stored[key]
                    atom_ok &= (
                        values.shape == (500, 24)
                        and np.isfinite(values).all()
                        and float(values.min()) >= 0.0
                        and float(values.max()) <= 1.0
                    )
            atom_ok &= (
                sha256(atom_path) == atom_audit["archive_sha256"]
                and atom_audit["selection_lock_sha256"] == lock_sha
                and atom_audit["source_test_raw_sha256"] == raw_hashes[(outer, seed, "test")]
            )
            atom_count += 1
    check("three_locked_final_manifests", manifest_ok, count=3)
    check("sixty_three_final_archives", final_ok and final_count == 63, count=final_count)
    check("nine_analytic_atom_archives", atom_ok and atom_count == 9, count=atom_count)
    check("A1_exact_A0_fallback", exact_fallback_ok, comparisons=9)
    check("rank_monotonicity_and_finiteness", diagnostic_ok,
          strict_reversals=0, central_tail_changes=0, nonfinite=0)

    required = [
        ROOT / "metrics" / "overall_metrics.csv",
        ROOT / "metrics" / "conditional_metrics.csv",
        ROOT / "metrics" / "atom_scores.csv",
        ROOT / "metrics" / "rank_audit.csv",
        ROOT / "statistics" / "paired_calendar_day_bootstrap.json",
        ROOT / "statistics" / "noninferiority.json",
        ROOT / "statistics" / "outer_consistency.json",
        ROOT / "statistics" / "success_gates.json",
        ROOT / "suc" / "daily_results.csv",
        ROOT / "suc" / "summary.csv",
        ROOT / "figures" / "figure_manifest.json",
        ROOT / "experiment_manifest.json",
        WORKSPACE / "CAA_RAHC_EXPERIMENT_REPORT.md",
    ]
    check("evaluation_suc_figures_report", all(path.is_file() and path.stat().st_size for path in required),
          required_count=len(required), hashes={str(path.relative_to(WORKSPACE)): sha256(path) for path in required})

    success = read_json(ROOT / "statistics" / "success_gates.json")
    check("frozen_success_gate_recorded", success["total"] == 10 and success["passed"] == 9,
          gate_passed=success["passed"], total=success["total"],
          all_passed=success["all_passed"])

    output = {
        "schema": "caa_rahc_current_schema_completion_audit_v1",
        "complete": all(item["passed"] for item in checks),
        "scientific_success_gates_all_passed": success["all_passed"],
        "note": "Artifact completeness is distinct from scientific success-gate outcome.",
        "checks": checks,
    }
    destination = ROOT / "current_schema_completion_audit.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"complete": output["complete"], "checks": len(checks),
                      "failed": [item["name"] for item in checks if not item["passed"]],
                      "destination": str(destination)}, indent=2))
    raise SystemExit(0 if output["complete"] else 1)


if __name__ == "__main__":
    main()
