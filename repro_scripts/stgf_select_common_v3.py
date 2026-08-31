from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stgf_flow.data import build_stgf_development_gefcom2014
from stgf_flow.graph import fit_spectral_artifacts
from stgf_flow.metrics import spectral_structure_scores


ROOT = Path(__file__).resolve().parents[1]
SPECTRAL_KEYS = {
    "spectral_energy_MAE",
    "low_graph_low_time_energy_MAE",
    "high_graph_energy_MAE",
    "high_time_frequency_energy_MAE",
    "cross_zone_correlation_Frobenius",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
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
        description=(
            "Re-evaluate all STGF candidates in one training-only GFT+DCT "
            "basis and create a corrected lock"
        )
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_development_v2.json"
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = ROOT / config["output_root"]
    lock_path = root / "selection.common_basis.lock.json"
    if lock_path.exists():
        raise RuntimeError(f"selection is already locked: {lock_path}")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_hashes: list[dict[str, str]] = []
    common_basis: dict[str, dict[str, Any]] = {}
    for outer in config["outer_splits"]:
        data = build_stgf_development_gefcom2014(
            ROOT / config["data_dir"], outer=int(outer)
        )
        graph = config["graph"]
        artifacts = fit_spectral_artifacts(
            data.train,
            transform_mode="stgf",
            neighbors=int(graph["neighbors"]),
            correlation_power=float(graph["correlation_power"]),
            epsilon=float(graph["logit_epsilon"]),
        )
        basis_hash = sha256_arrays(
            artifacts.adjacency,
            artifacts.graph_basis,
            artifacts.graph_eigenvalues,
            artifacts.temporal_basis,
        )
        common_basis[str(outer)] = {
            "fit_split": "train only",
            "transform": "normalized-Laplacian GFT x orthonormal DCT-II",
            "basis_sha256": basis_hash,
            "metadata": artifacts.metadata,
        }
        source_dir = root / f"outer{outer}" / "candidates_v2"
        corrected_dir = root / f"outer{outer}" / "candidates_common_basis_v3"
        corrected_dir.mkdir(parents=True, exist_ok=True)
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
                    archive = source_dir / f"{name}_calibration.npz"
                    original = (
                        source_dir / f"{name}_calibration.metrics.json"
                    )
                    if not archive.exists() or not original.exists():
                        raise FileNotFoundError(name)
                    original_record = json.loads(
                        original.read_text(encoding="utf-8")
                    )
                    if original_record.get("split") != "calibration_only":
                        raise RuntimeError("non-calibration input encountered")
                    with np.load(archive, allow_pickle=False) as stored:
                        spectral = spectral_structure_scores(
                            stored["scenarios"],
                            stored["observations"],
                            artifacts,
                        )
                    if set(spectral) != SPECTRAL_KEYS:
                        raise RuntimeError("unexpected spectral metric schema")
                    metrics = dict(original_record["metrics"])
                    metrics.update(spectral)
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError(name)
                    corrected = {
                        **{
                            key: value
                            for key, value in original_record.items()
                            if key != "metrics"
                        },
                        "spectral_evaluation_basis": (
                            "common training-only GFT x DCT basis"
                        ),
                        "spectral_evaluation_basis_sha256": basis_hash,
                        "metrics": metrics,
                    }
                    corrected_path = (
                        corrected_dir
                        / f"{name}_calibration.common.metrics.json"
                    )
                    corrected_path.write_text(
                        json.dumps(corrected, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    records[name].append(corrected)
                    evidence_hashes.extend(
                        [
                            {
                                "path": str(original.resolve()),
                                "sha256": sha256_file(original),
                            },
                            {
                                "path": str(archive.resolve()),
                                "sha256": sha256_file(archive),
                            },
                            {
                                "path": str(corrected_path.resolve()),
                                "sha256": sha256_file(corrected_path),
                            },
                        ]
                    )
    expected = len(config["outer_splits"])
    if any(len(values) != expected for values in records.values()):
        raise RuntimeError("incomplete candidate replication")
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
        raise RuntimeError("no candidate satisfies calibration constraints")
    selected = min(
        feasible, key=lambda name: (summaries[name]["objective"], name)
    )
    selected_record = records[selected][0]
    best_by_mode: dict[str, str | None] = {}
    for mode in config["transform_modes"]:
        names = [
            name
            for name in feasible
            if records[name][0]["transform_mode"] == mode
        ]
        best_by_mode[mode] = (
            min(names, key=lambda name: summaries[name]["objective"])
            if names
            else None
        )
    lock = {
        "schema": "stgf_development_selection_common_basis_v3",
        "correction": (
            "All candidates are evaluated in the same per-outer "
            "training-only GFT x DCT basis. This corrects the incomparable "
            "candidate-native spectral term in the v2 lock."
        ),
        "selection_data": (
            "calibration only across three development outers; no STGF "
            "confirmation test archive accessed"
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "common_spectral_bases": common_basis,
        "evidence_hashes": evidence_hashes,
        "constraints": {
            "coverage_90": [0.82, 0.93],
            "width_90_maximum": 0.55,
            "all_metrics_finite": True,
        },
        "objective": (
            "CRPS + 0.05*abs(coverage90-0.90) + 0.01*joint_ES_240 "
            "+ 0.10*adjacency_VS + 0.02*common_basis_spectral_energy_MAE"
        ),
        "selected": selected,
        "selected_transform_mode": selected_record["transform_mode"],
        "selected_physical_mean_anchor": float(
            selected_record["physical_mean_anchor"]
        ),
        "selected_residual_temperature": float(
            selected_record["residual_temperature"]
        ),
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
                "sha256": sha256_file(lock_path),
            }
        )
    )


if __name__ == "__main__":
    main()
