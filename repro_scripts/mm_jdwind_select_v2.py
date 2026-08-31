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
        description="Lock one real MM-JDWind v2 state mechanism on calibration data"
    )
    parser.add_argument(
        "--config",
        default="repro_configs/mm_jdwind_development_v2_selection.json",
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    lock_path = output / "selection.lock.json"
    if lock_path.exists():
        raise RuntimeError(f"selection already locked: {lock_path}")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_hashes: list[dict[str, str]] = []
    stability: dict[str, list[float]] = defaultdict(list)
    for outer in config["outer_splits"]:
        for seed in config["model_seeds"]:
            directory = output / f"outer{outer}" / "candidates"
            for mode in config["sampling"]["state_modes"]:
                primary = (
                    directory / f"flow_{mode}_seed{seed}_calibration.metrics.json"
                )
                repeat = (
                    directory / f"proper_{mode}_seed{seed}_calibration.metrics.json"
                )
                if not primary.exists() or not repeat.exists():
                    raise FileNotFoundError(primary if not primary.exists() else repeat)
                value = json.loads(primary.read_text(encoding="utf-8"))
                repeated = json.loads(repeat.read_text(encoding="utf-8"))
                if value.get("split") != "calibration_only":
                    raise RuntimeError("non-calibration metric encountered")
                records[mode].append(value)
                stability[mode].append(
                    repeated["metrics"]["CRPS"] - value["metrics"]["CRPS"]
                )
                source_hashes.extend(
                    [
                        {"path": str(primary.resolve()), "sha256": sha256_file(primary)},
                        {"path": str(repeat.resolve()), "sha256": sha256_file(repeat)},
                    ]
                )
    expected = len(config["outer_splits"]) * len(config["model_seeds"])
    if any(len(values) != expected for values in records.values()):
        raise RuntimeError("incomplete calibration replications")
    summaries: dict[str, dict[str, Any]] = {}
    for mode, values in records.items():
        summaries[mode] = {
            metric: float(
                np.mean([item["metrics"][metric] for item in values])
            )
            for metric in values[0]["metrics"]
        }
        summaries[mode]["crps_std_across_replications"] = float(
            np.std([item["metrics"]["CRPS"] for item in values], ddof=1)
        )
        summaries[mode]["mc_repeat_crps_mean_delta"] = float(
            np.mean(stability[mode])
        )
        summaries[mode]["mc_repeat_crps_max_abs_delta"] = float(
            np.max(np.abs(stability[mode]))
        )
    baseline = summaries["none"]
    feasible: list[str] = []
    for mode, metrics in summaries.items():
        checks = {
            "CRPS": metrics["CRPS"] <= 1.005 * baseline["CRPS"],
            "MAE": metrics["MAE"] <= 1.01 * baseline["MAE"],
            "VS": metrics["VS"] <= 1.01 * baseline["VS"],
            "ramp_CRPS": metrics["ramp_CRPS"]
            <= 1.01 * baseline["ramp_CRPS"],
            "width_90": metrics["width_90"] <= baseline["width_90"] + 0.05,
        }
        summaries[mode]["constraint_checks"] = checks
        if all(checks.values()):
            feasible.append(mode)
    if feasible:
        selected = min(
            feasible,
            key=lambda mode: (
                summaries[mode]["CRPS"]
                + 0.1 * abs(summaries[mode]["coverage_90"] - 0.9)
                + 0.05 * summaries[mode]["adjacency_VS"],
                mode,
            ),
        )
        fallback = False
    else:
        selected = "none"
        fallback = True
    lock = {
        "schema": "mm_jdwind_development_selection_lock_v2",
        "selection_data": "calibration only; no MM-JDWind test archive accessed",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_metric_hashes": source_hashes,
        "baseline_state_mode": "none",
        "selected_stage": "flow",
        "selected_state_mode": selected,
        "selected": f"flow_{selected}",
        "fallback": fallback,
        "feasible_state_modes": feasible,
        "constraint_rule": {
            "CRPS_relative_degradation": 0.005,
            "MAE_VS_ramp_relative_degradation": 0.01,
            "width_90_absolute_increase": 0.05,
        },
        "objective": (
            "mean CRPS + 0.1*abs(mean coverage90-0.9) "
            "+ 0.05*mean adjacency_VS"
        ),
        "proper_branch_status": "failed_nonfinite_in_prelock_v1_pilot",
        "proper_prefixed_archives_use": (
            "Monte Carlo repeat only; excluded from candidate identities"
        ),
        "summaries": summaries,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "locked": str(lock_path.resolve()),
                "selected": lock["selected"],
                "fallback": fallback,
                "summaries": {
                    mode: {
                        "CRPS": summaries[mode]["CRPS"],
                        "coverage_90": summaries[mode]["coverage_90"],
                        "joint_ES_240": summaries[mode]["joint_ES_240"],
                    }
                    for mode in summaries
                },
            }
        )
    )


if __name__ == "__main__":
    main()
