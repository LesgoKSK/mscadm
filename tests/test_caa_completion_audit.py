from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from repro_scripts import caa_completion_audit as audit


LOCKED_AT = "2025-01-01T00:00:00Z"
GENERATED_AT = "2025-01-01T01:00:00Z"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dates(start: str) -> list[str]:
    values = np.datetime64(start, "D") + np.arange(50).astype("timedelta64[D]")
    return values.astype(str).tolist()


def _archive_arrays(dates: list[str], value: float) -> dict[str, np.ndarray]:
    cases = len(dates)
    return {
        "scenarios": np.full((cases, 3, 2), value, dtype=np.float32),
        "observations": np.full((cases, 2), 0.4, dtype=np.float32),
        "zone": np.ones(cases, dtype=np.int64),
        "day": np.asarray(dates, dtype="datetime64[D]"),
    }


def _write_archive(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    selected_config_sha256: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    sidecar: dict[str, object] = {
        "schema": "caa_rahc_scenario_archive_audit_v1",
        "archive_sha256": audit.sha256_file(path),
    }
    if "selection_lock_sha256" in metadata:
        sidecar["selection_lock_sha256"] = metadata["selection_lock_sha256"]
    if selected_config_sha256 is not None:
        sidecar["selected_config_sha256"] = selected_config_sha256
    _write_json(audit.sidecar_path(path), sidecar)


def _full_code_fingerprint(workspace: Path) -> dict[str, object]:
    sources = {
        "caa_rahc/candidates.py": "# prototype streaming scorer\n",
        "caa_rahc/candidates_nested.py": "# authoritative strict nested scorer\n",
        "repro_scripts/caa_finalize.py": "# frozen finalizer\n",
    }
    records = []
    for relative, content in sources.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": audit.sha256_file(path),
            }
        )
    return {
        "algorithm": "sha256",
        "combined_sha256": audit.canonical_json_sha256(records),
        "files": records,
    }


def _refresh_experiment_manifest(workspace: Path, root: Path, report: Path) -> None:
    path = root / audit.EXPERIMENT_MANIFEST
    excluded = {
        path.resolve(),
        (root / "completion_audit.json").resolve(),
        (root / "manifest.csv").resolve(),
    }
    files = [
        item.resolve()
        for item in root.rglob("*")
        if item.is_file() and item.resolve() not in excluded
    ]
    files.extend(
        [
            report.resolve(),
            (workspace / "caa_rahc/candidates.py").resolve(),
            (workspace / "caa_rahc/candidates_nested.py").resolve(),
            (workspace / "repro_scripts/caa_finalize.py").resolve(),
        ]
    )
    records = []
    for item in sorted(set(files)):
        records.append(
            {
                "path": item.relative_to(workspace).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": audit.sha256_file(item),
            }
        )
    _write_json(
        path,
        {"schema": "caa_rahc_experiment_manifest_v1", "files": records},
    )


def _build_complete_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "outputs" / "caa_rahc"
    root.mkdir(parents=True)
    report = workspace / "CAA_RAHC_EXPERIMENT_REPORT.md"
    report.write_text("# CAA-RAHC\n\n" + "完整确认性实验报告。" * 30, encoding="utf-8")

    config_path = workspace / "repro_configs" / "caa_rahc_frozen.json"
    _write_json(config_path, {"outer_splits": [1, 2, 3], "model_seeds": [0, 1, 2]})
    code = _full_code_fingerprint(workspace)

    test_dates = {
        1: _dates("2020-01-01"),
        2: _dates("2020-04-01"),
        3: _dates("2020-07-01"),
    }
    calibration_dates = {
        1: _dates("2019-01-01"),
        2: _dates("2019-04-01"),
        3: _dates("2019-07-01"),
    }
    frozen = {
        "schema": "caa_rahc_frozen_outer_manifest_v1",
        "config_path": config_path.relative_to(workspace).as_posix(),
        "config_sha256": audit.sha256_file(config_path),
        "code": {"combined_sha256": "a" * 64, "files": []},
        "outer_splits": [1, 2, 3],
        "model_seeds": [0, 1, 2],
        "protocols": {
            f"outer{outer}": {
                "protocol_sha256": f"{outer}" * 64,
                "calendar_day_counts": {
                    "calibration": 50,
                    "test": 50,
                },
                "dates": {
                    "calibration": calibration_dates[outer],
                    "test": test_dates[outer],
                },
            }
            for outer in audit.OUTERS
        },
    }
    frozen_path = root / audit.FROZEN_MANIFEST
    _write_json(frozen_path, frozen)

    selected_configs = {
        family: {"family": family, "registered_strength": index / 10.0}
        for index, family in enumerate(audit.FAMILIES)
    }
    selection_analysis = {
        "schema": "caa_rahc_selection_analysis_v1",
        "authoritative_module": audit.AUTHORITATIVE_MODULE,
        "prototype_role": (
            "caa_rahc/candidates.py supplies the guarded streaming score engine only"
        ),
        "full_code": code,
        "selected_configs": selected_configs,
    }
    analysis_path = root / audit.SELECTION_ANALYSIS
    _write_json(analysis_path, selection_analysis)
    lock = {
        "schema": "caa_rahc_selection_lock_v1",
        "analysis_path": audit.SELECTION_ANALYSIS,
        "analysis_sha256": audit.sha256_file(analysis_path),
        "full_code_sha256": code["combined_sha256"],
        "manifest_sha256": audit.sha256_file(frozen_path),
        "locked_at_utc": LOCKED_AT,
        "authoritative_module": audit.AUTHORITATIVE_MODULE,
        "selected_configs": selected_configs,
    }
    lock_path = root / audit.SELECTION_LOCK
    _write_json(lock_path, lock)
    lock_sha = audit.sha256_file(lock_path)
    analysis_sha = audit.sha256_file(analysis_path)

    for outer in audit.OUTERS:
        for seed in audit.SEEDS:
            checkpoint = audit.checkpoint_path(root, outer, seed)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint outer={outer} seed={seed}".encode())
            _write_json(
                audit.sidecar_path(checkpoint),
                {
                    "schema": "caa_rahc_checkpoint_audit_v1",
                    "checkpoint_sha256": audit.sha256_file(checkpoint),
                    "outer": outer,
                    "model_seed": seed,
                    "resume": False,
                },
            )

            calibration_path = audit.raw_archive_path(
                root, outer, seed, "calibration"
            )
            _write_archive(
                calibration_path,
                _archive_arrays(calibration_dates[outer], 0.35 + seed * 0.01),
                {
                    "split": "calibration",
                    "outer": outer,
                    "training_seed": seed,
                },
            )

            test_path = audit.raw_archive_path(root, outer, seed, "test")
            _write_archive(
                test_path,
                _archive_arrays(test_dates[outer], 0.45 + seed * 0.01),
                {
                    "split": "test",
                    "outer": outer,
                    "training_seed": seed,
                    "selection_lock_sha256": lock_sha,
                    "selection_analysis_sha256": analysis_sha,
                    "generated_after_selection_lock": True,
                    "generated_at_utc": GENERATED_AT,
                },
            )
            test_hash = audit.sha256_file(test_path)

            for family_index, family in enumerate(audit.FAMILIES):
                selected = selected_configs[family]
                selected_hash = audit.canonical_json_sha256(selected)
                final_path = audit.final_archive_path(root, outer, seed, family)
                _write_archive(
                    final_path,
                    _archive_arrays(
                        test_dates[outer], 0.40 + family_index * 0.01
                    ),
                    {
                        "split": "test",
                        "outer": outer,
                        "training_seed": seed,
                        "family": family,
                        "selection_lock_sha256": lock_sha,
                        "selection_analysis_sha256": analysis_sha,
                        "generated_after_selection_lock": True,
                        "generated_at_utc": GENERATED_AT,
                        "source_test_raw_sha256": test_hash,
                        "selected_config": selected,
                        "selected_config_sha256": selected_hash,
                        "authoritative_module": audit.AUTHORITATIVE_MODULE,
                        "prototype_role": (
                            "caa_rahc/candidates.py guarded streaming dependency"
                        ),
                        "transform_diagnostics": {
                            "strict_reversals": 0,
                            "nonfinite": 0,
                            "central_changed": 0,
                        },
                    },
                    selected_config_sha256=selected_hash,
                )

    metric_rows = [
        {"method": family, "outer": outer, "value": 0.1}
        for outer in audit.OUTERS
        for family in audit.FAMILIES
    ]
    _write_csv(root / "metrics/overall_metrics.csv", metric_rows)
    _write_csv(root / "metrics/conditional_metrics.csv", metric_rows)
    _write_json(root / "metrics/atom_diagnostics.json", {"complete": True})
    _write_json(root / "metrics/quantization_audit.json", {"complete": True})
    _write_json(
        root / "statistics/paired_calendar_day_bootstrap.json",
        {
            "protocol": {
                "replicates": 5000,
                "unique_calendar_days": 150,
                "cluster": "calendar date",
            },
            "comparisons": {"A4_vs_A0": {"estimate": 0.0}},
        },
    )
    _write_json(
        root / "statistics/noninferiority.json",
        {"A4": {"complete": True, "eligible": True}},
    )
    _write_json(
        root / "statistics/success_gates.json",
        {
            "passed": 1,
            "total": 2,
            "all_passed": False,
            "gates": {"coverage": True, "conditional": False},
        },
    )
    _write_json(
        root / "statistics/outer_consistency.json",
        {"outers": [1, 2, 3], "complete": True},
    )
    _write_csv(root / "suc/daily_results.csv", [{"method": "A0", "cost": 1.0}])
    _write_csv(root / "suc/summary.csv", [{"method": "A0", "mean_cost": 1.0}])
    _write_json(root / "suc/protocol.json", {"complete": True, "cases": 7})

    figure = root / "figures/overview.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-unit-test-figure")
    _write_json(
        root / "figures/figure_manifest.json",
        {
            "files": [
                {
                    "path": "figures/overview.png",
                    "bytes": figure.stat().st_size,
                    "sha256": audit.sha256_file(figure),
                }
            ]
        },
    )
    _refresh_experiment_manifest(workspace, root, report)
    return workspace, root, report


def _rewrite_archive_metadata(path: Path, changes: dict[str, object]) -> None:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files if name != "metadata"}
        metadata = json.loads(str(stored["metadata"].item()))
    metadata.update(changes)
    _write_archive(path, arrays, metadata)


def test_complete_synthetic_tree_passes_and_writes_both_outputs(tmp_path: Path) -> None:
    workspace, root, report = _build_complete_tree(tmp_path)

    result = audit.audit_completion(root, workspace=workspace, report=report)

    assert result["complete"] is True
    assert result["missing"] == []
    assert result["failed"] == []
    assert result["counts"] == {
        "audited_checkpoints": 9,
        "calibration_raw_archives": 9,
        "sealed_test_raw_archives": 9,
        "final_scenario_archives": 63,
        "manifest_rows": result["counts"]["manifest_rows"],
    }
    assert result["counts"]["manifest_rows"] > 100
    assert (root / "completion_audit.json").is_file()
    assert (root / "manifest.csv").is_file()
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["counts"]["manifest_rows"]
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_missing_artifact_is_listed_and_main_returns_nonzero(
    tmp_path: Path, capsys
) -> None:
    workspace, root, report = _build_complete_tree(tmp_path)
    missing = audit.final_archive_path(root, 3, 2, "A6")
    missing.unlink()

    exit_code = audit.main(
        [
            "--root",
            str(root),
            "--workspace",
            str(workspace),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 1
    payload = json.loads((root / "completion_audit.json").read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert any("A6_seed2_test_final.npz" in value for value in payload["missing"])
    assert "sixty_three_hashed_final_archives" in payload["failed"]
    printed = capsys.readouterr().out
    assert "A6_seed2_test_final.npz" in printed


def test_test_raw_timestamp_must_be_strictly_after_selection_lock(tmp_path: Path) -> None:
    workspace, root, report = _build_complete_tree(tmp_path)
    path = audit.raw_archive_path(root, 1, 0, "test")
    _rewrite_archive_metadata(path, {"generated_at_utc": LOCKED_AT})
    _refresh_experiment_manifest(workspace, root, report)

    result = audit.audit_completion(root, workspace=workspace, report=report)

    assert result["complete"] is False
    check = result["checks"]["nine_sealed_test_raw_archives"]
    assert check["passed"] is False
    assert "not generated after selection lock" in check["detail"]["message"]


def test_hash_drift_is_rejected_even_when_artifact_still_exists(tmp_path: Path) -> None:
    workspace, root, report = _build_complete_tree(tmp_path)
    path = audit.final_archive_path(root, 2, 1, "A4")
    with path.open("ab") as handle:
        handle.write(b"post-audit-drift")

    result = audit.audit_completion(root, workspace=workspace, report=report)

    assert result["complete"] is False
    check = result["checks"]["sixty_three_hashed_final_archives"]
    assert check["passed"] is False
    assert "hash drift" in check["detail"]["message"]
    assert (root / "completion_audit.json").is_file()
    assert (root / "manifest.csv").is_file()
