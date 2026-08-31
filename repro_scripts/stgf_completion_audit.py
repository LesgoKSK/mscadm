from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stgf_flow.data import load_stgf_confirmation_splits


ROOT = Path(__file__).resolve().parents[1]
STGF_CONFIG = ROOT / "repro_configs" / "stgf_confirmation_v2.json"
DDPM_CONFIG = ROOT / "repro_configs" / "ddpm_stgf_confirmation_v2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite_state_dict(payload: dict[str, Any]) -> bool:
    return all(
        torch.isfinite(value).all().item()
        for value in payload["model"].values()
    )


def main() -> None:
    config = json.loads(STGF_CONFIG.read_text(encoding="utf-8"))
    ddpm_config = json.loads(DDPM_CONFIG.read_text(encoding="utf-8"))
    root = ROOT / config["output_root"]
    ddpm_root = ROOT / ddpm_config["output_root"]
    issues: list[str] = []
    checks: dict[str, Any] = {}
    registry_path = ROOT / config["split_registry"]
    registry = load_stgf_confirmation_splits(registry_path)
    checks["split_registry"] = {
        "valid": True,
        "all_outer_test_sha256": registry["all_outer_test_sha256"],
        "calendar_day_counts": registry["calendar_day_counts"],
    }
    lock_path = ROOT / config["development_selection_lock"]
    lock_hash = sha256_file(lock_path)
    if lock_hash != config["development_selection_lock_sha256"]:
        issues.append("selection lock hash mismatch")
    checks["selection_lock"] = {
        "path": str(lock_path.resolve()),
        "sha256": lock_hash,
        "matches_config": not issues,
    }
    selected = config["locked_selection"]["transform_mode"]
    expected_models = [
        (mode, seed)
        for mode in config["transform_modes"]
        for seed in (
            config["primary_model_seeds"]
            if mode == selected
            else config["ablation_model_seeds"]
        )
    ]
    checkpoint_records = []
    for mode, seed in expected_models:
        path = (
            root
            / "outer1"
            / "runs"
            / mode
            / f"seed{seed}"
            / "final.pt"
        )
        if not path.exists():
            issues.append(f"missing checkpoint: {path}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        finite = finite_state_dict(payload)
        if not finite:
            issues.append(f"non-finite checkpoint: {path}")
        checkpoint_records.append(
            {
                "mode": mode,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "selected_step": int(payload["step"]),
                "finite": finite,
                "config_matches": payload["config"] == config,
            }
        )
        if payload["config"] != config:
            issues.append(f"checkpoint config mismatch: {path}")
    checks["unique_stgf_models"] = checkpoint_records
    scenario_records = []
    for outer in config["outer_splits"]:
        for mode, seed in expected_models:
            path = (
                root
                / f"outer{outer}"
                / "scenarios"
                / f"{mode}_seed{seed}_test.npz"
            )
            metrics = path.with_suffix(".common.metrics.json")
            if not path.exists() or not metrics.exists():
                issues.append(f"missing scenario/metric pair: {path}")
                continue
            with np.load(path, allow_pickle=False) as stored:
                scenarios = stored["scenarios"]
                observations = stored["observations"]
                metadata = json.loads(str(stored["metadata"]))
                valid_shape = scenarios.shape == (50, 100, 10, 24)
                finite = np.isfinite(scenarios).all()
                bounded = bool(
                    scenarios.min() >= 0.0 and scenarios.max() <= 1.0
                )
                observation_shape = observations.shape == (50, 10, 24)
                lock_matches = (
                    metadata["selection_lock_sha256"]
                    == config["development_selection_lock_sha256"]
                )
            metric_record = json.loads(metrics.read_text(encoding="utf-8"))
            metric_finite = all(
                np.isfinite(value)
                for value in metric_record["metrics"].values()
            )
            valid = all(
                (
                    valid_shape,
                    finite,
                    bounded,
                    observation_shape,
                    lock_matches,
                    metric_finite,
                )
            )
            if not valid:
                issues.append(f"invalid scenario/metrics: {path}")
            scenario_records.append(
                {
                    "outer": outer,
                    "mode": mode,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "metrics_sha256": sha256_file(metrics),
                    "shape": list(scenarios.shape),
                    "finite": bool(finite),
                    "bounded_0_1": bounded,
                    "metric_finite": bool(metric_finite),
                    "selection_lock_matches": lock_matches,
                }
            )
    checks["stgf_evaluation_cells"] = scenario_records
    ddpm_checkpoint = (
        ddpm_root / "outer1" / "runs" / "seed0" / "final.pt"
    )
    if not ddpm_checkpoint.exists():
        issues.append("missing DDPM checkpoint")
        ddpm_finite = False
    else:
        ddpm_payload = torch.load(
            ddpm_checkpoint, map_location="cpu", weights_only=False
        )
        ddpm_finite = finite_state_dict(ddpm_payload)
        if not ddpm_finite:
            issues.append("non-finite DDPM checkpoint")
    ddpm_records = []
    for outer in ddpm_config["outer_splits"]:
        path = (
            ddpm_root
            / f"outer{outer}"
            / "scenarios"
            / "ddpm_seed0_test.npz"
        )
        metrics = path.with_suffix(".metrics.json")
        if not path.exists() or not metrics.exists():
            issues.append(f"missing DDPM scenario/metrics: outer {outer}")
            continue
        with np.load(path, allow_pickle=False) as stored:
            shape = stored["scenarios"].shape
            finite = np.isfinite(stored["scenarios"]).all()
        metric_record = json.loads(metrics.read_text(encoding="utf-8"))
        metric_finite = all(
            np.isfinite(value)
            for value in metric_record["metrics"].values()
        )
        if shape != (50, 100, 10, 24) or not finite or not metric_finite:
            issues.append(f"invalid DDPM evaluation: outer {outer}")
        ddpm_records.append(
            {
                "outer": outer,
                "archive_sha256": sha256_file(path),
                "metrics_sha256": sha256_file(metrics),
                "shape": list(shape),
                "finite": bool(finite),
                "metric_finite": bool(metric_finite),
            }
        )
    checks["ddpm"] = {
        "checkpoint": str(ddpm_checkpoint.resolve()),
        "checkpoint_sha256": (
            sha256_file(ddpm_checkpoint)
            if ddpm_checkpoint.exists()
            else None
        ),
        "checkpoint_finite": bool(ddpm_finite),
        "evaluation_cells": ddpm_records,
    }
    analysis = root / "final_analysis" / "stgf_final_analysis.json"
    report = root / "final_analysis" / "STGF_FLOW_FINAL_REPORT.md"
    if not analysis.exists() or not report.exists():
        issues.append("missing final analysis/report")
    checks["final_outputs"] = {
        "analysis": str(analysis.resolve()),
        "analysis_sha256": sha256_file(analysis) if analysis.exists() else None,
        "report": str(report.resolve()),
        "report_sha256": sha256_file(report) if report.exists() else None,
    }
    result = {
        "schema": "stgf_completion_audit_v1",
        "status": "complete" if not issues else "incomplete",
        "issues": issues,
        "counts": {
            "unique_stgf_models": len(checkpoint_records),
            "stgf_evaluation_cells": len(scenario_records),
            "ddpm_models": int(ddpm_checkpoint.exists()),
            "ddpm_evaluation_cells": len(ddpm_records),
        },
        "checks": checks,
    }
    destination = root / "final_analysis" / "completion_audit.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "issues": issues,
                "counts": result["counts"],
                "audit": str(destination.resolve()),
            }
        )
    )
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
