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


def candidate_name(mode: str, anchor: float, temperature: float) -> str:
    return f"{mode}_pa{anchor:.2f}_t{temperature:.2f}"


def objective(metrics: dict[str, float]) -> float:
    return float(
        metrics["CRPS"]
        + 0.05 * abs(metrics["coverage_90"] - 0.90)
        + 0.01 * metrics["joint_ES_240"]
        + 0.10 * metrics["adjacency_VS"]
        + 0.02 * metrics["spectral_energy_MAE"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and lock STGF representation on calibration only"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_development_v2.json"
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = ROOT / config["output_root"]
    lock_path = root / "selection.lock.json"
    if lock_path.exists():
        raise RuntimeError(f"selection is already locked: {lock_path}")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hashes: list[dict[str, str]] = []
    for outer in config["outer_splits"]:
        directory = root / f"outer{outer}" / "candidates_v2"
        for mode in config["transform_modes"]:
            for anchor in config["sampling_selection"][
                "physical_mean_anchor"
            ]:
                for temperature in config["sampling_selection"][
                    "residual_temperature"
                ]:
                    name = candidate_name(
                        mode, float(anchor), float(temperature)
                    )
                    path = directory / f"{name}_calibration.metrics.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("split") != "calibration_only":
                        raise RuntimeError("non-calibration record encountered")
                    records[name].append(value)
                    hashes.append(
                        {
                            "path": str(path.resolve()),
                            "sha256": sha256_file(path),
                        }
                    )
    expected = len(config["outer_splits"])
    if any(len(values) != expected for values in records.values()):
        raise RuntimeError("candidate outer replications are incomplete")
    summaries: dict[str, dict[str, Any]] = {}
    feasible: list[str] = []
    for name, values in records.items():
        metrics = {
            metric: float(
                np.mean([value["metrics"][metric] for value in values])
            )
            for metric in values[0]["metrics"]
        }
        checks = {
            "coverage_90": 0.82 <= metrics["coverage_90"] <= 0.93,
            "width_90": metrics["width_90"] <= 0.55,
            "finite": all(np.isfinite(value) for value in metrics.values()),
        }
        summaries[name] = {
            **metrics,
            "objective": objective(metrics),
            "constraint_checks": checks,
        }
        if all(checks.values()):
            feasible.append(name)
    if not feasible:
        raise RuntimeError("no STGF candidate satisfies calibration constraints")
    selected = min(
        feasible, key=lambda name: (summaries[name]["objective"], name)
    )
    selected_record = records[selected][0]
    mode = selected_record["transform_mode"]
    anchor = float(selected_record["physical_mean_anchor"])
    temperature = float(selected_record["residual_temperature"])
    best_by_mode = {}
    for current_mode in config["transform_modes"]:
        names = [
            name
            for name in feasible
            if records[name][0]["transform_mode"] == current_mode
        ]
        best_by_mode[current_mode] = (
            min(names, key=lambda name: summaries[name]["objective"])
            if names
            else None
        )
    lock = {
        "schema": "stgf_development_selection_lock_v2",
        "selection_data": (
            "calibration only across three development outers; no STGF "
            "confirmation test archive accessed"
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_metric_hashes": hashes,
        "constraints": {
            "coverage_90": [0.82, 0.93],
            "width_90_maximum": 0.55,
            "all_metrics_finite": True,
        },
        "objective": (
            "CRPS + 0.05*abs(coverage90-0.90) + 0.01*joint_ES_240 "
            "+ 0.10*adjacency_VS + 0.02*spectral_energy_MAE"
        ),
        "selected": selected,
        "selected_transform_mode": mode,
        "selected_physical_mean_anchor": anchor,
        "selected_residual_temperature": temperature,
        "best_by_transform_mode": best_by_mode,
        "feasible_candidates": feasible,
        "summaries": summaries,
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "locked": str(lock_path.resolve()),
                "selected": selected,
                "best_by_transform_mode": best_by_mode,
            }
        )
    )


if __name__ == "__main__":
    main()
