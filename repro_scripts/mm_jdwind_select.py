from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and lock one MM-JDWind candidate using calibration only"
    )
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_development.json"
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    lock_path = output / "selection.lock.json"
    if lock_path.exists():
        raise RuntimeError(f"selection is already locked: {lock_path}")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_hashes: list[dict[str, str]] = []
    for outer in config["outer_splits"]:
        for seed in config["model_seeds"]:
            directory = output / f"outer{outer}" / "candidates"
            for stage in ("flow", "proper"):
                for mode in config["sampling"]["state_modes"]:
                    path = directory / f"{stage}_{mode}_seed{seed}_calibration.metrics.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("split") != "calibration_only":
                        raise RuntimeError("candidate metric is not calibration-only")
                    records[f"{stage}_{mode}"].append(value)
                    source_hashes.append(
                        {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    )
    expected = len(config["outer_splits"]) * len(config["model_seeds"])
    if any(len(values) != expected for values in records.values()):
        raise RuntimeError("candidate replications are incomplete")
    summaries: dict[str, dict[str, Any]] = {}
    for name, values in records.items():
        metrics = values[0]["metrics"]
        summaries[name] = {
            metric: float(np.mean([item["metrics"][metric] for item in values]))
            for metric in metrics
        }
    baseline = summaries["flow_none"]
    feasible: list[str] = []
    for name, metrics in summaries.items():
        checks = {
            "CRPS": metrics["CRPS"] <= 1.005 * baseline["CRPS"],
            "MAE": metrics["MAE"] <= 1.01 * baseline["MAE"],
            "VS": metrics["VS"] <= 1.01 * baseline["VS"],
            "ramp_CRPS": metrics["ramp_CRPS"] <= 1.01 * baseline["ramp_CRPS"],
            "width_90": metrics["width_90"] <= baseline["width_90"] + 0.05,
        }
        summaries[name]["constraint_checks"] = checks
        if all(checks.values()):
            feasible.append(name)
    if not feasible:
        selected = "flow_none"
        fallback = True
    else:
        selected = min(
            feasible,
            key=lambda name: (
                summaries[name]["CRPS"]
                + 0.1 * abs(summaries[name]["coverage_90"] - 0.9)
                + 0.05 * summaries[name]["adjacency_VS"],
                name,
            ),
        )
        fallback = False
    stage, mode = selected.split("_", 1)
    lock = {
        "schema": "mm_jdwind_development_selection_lock_v1",
        "selection_data": "calibration only; no MM-JDWind test archive accessed",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_metric_hashes": source_hashes,
        "baseline": "flow_none",
        "constraint_rule": {
            "CRPS_relative_degradation": 0.005,
            "MAE_VS_ramp_relative_degradation": 0.01,
            "width_90_absolute_increase": 0.05,
        },
        "objective": "CRPS + 0.1*abs(coverage90-0.9) + 0.05*adjacency_VS",
        "selected": selected,
        "selected_stage": stage,
        "selected_state_mode": mode,
        "fallback": fallback,
        "feasible_candidates": feasible,
        "summaries": summaries,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"locked": str(lock_path.resolve()), "selected": selected}))


if __name__ == "__main__":
    main()
