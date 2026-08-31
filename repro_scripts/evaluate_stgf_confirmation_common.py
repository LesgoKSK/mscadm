from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stgf_flow.data import build_stgf_confirmation_gefcom2014
from stgf_flow.graph import fit_spectral_artifacts
from stgf_flow.metrics import evaluate_stgf


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score STGF confirmation in one common GFT x DCT basis"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_confirmation_v2.json"
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest: list[dict[str, object]] = []
    for outer in config["outer_splits"]:
        data = build_stgf_confirmation_gefcom2014(
            ROOT / config["data_dir"],
            outer=int(outer),
            split_path=ROOT / config["split_registry"],
        )
        graph = config["graph"]
        artifacts = fit_spectral_artifacts(
            data.train,
            transform_mode="stgf",
            neighbors=int(graph["neighbors"]),
            correlation_power=float(graph["correlation_power"]),
            epsilon=float(graph["logit_epsilon"]),
        )
        scenarios_dir = (
            ROOT / config["output_root"] / f"outer{outer}" / "scenarios"
        )
        selected_mode = config["locked_selection"]["transform_mode"]
        for mode in config["transform_modes"]:
            seeds = (
                config["primary_model_seeds"]
                if mode == selected_mode
                else config["ablation_model_seeds"]
            )
            for seed in seeds:
                archive = scenarios_dir / f"{mode}_seed{seed}_test.npz"
                if not archive.exists():
                    raise FileNotFoundError(archive)
                with np.load(archive, allow_pickle=False) as stored:
                    scenarios = stored["scenarios"]
                    if scenarios.shape != (50, 100, 10, 24):
                        raise ValueError(
                            f"unexpected scenario shape: {archive}"
                        )
                    if not np.array_equal(
                        stored["observations"], data.test.target
                    ):
                        raise RuntimeError(
                            f"observation mismatch: {archive}"
                        )
                    metadata = json.loads(str(stored["metadata"]))
                    if (
                        metadata["protocol_sha256"]
                        != data.protocol["protocol_sha256"]
                    ):
                        raise RuntimeError(
                            f"protocol hash mismatch: {archive}"
                        )
                    if metadata["selection_lock_sha256"] != config[
                        "development_selection_lock_sha256"
                    ]:
                        raise RuntimeError(
                            f"selection lock mismatch: {archive}"
                        )
                    metrics = evaluate_stgf(
                        scenarios, data.test, artifacts
                    )
                destination = archive.with_suffix(".common.metrics.json")
                record = {
                    "outer": int(outer),
                    "seed": int(seed),
                    "model": "STGF-Flow",
                    "transform_mode": mode,
                    "split": "new_frozen_test",
                    "spectral_evaluation_basis": (
                        "common training-only GFT x DCT basis"
                    ),
                    "protocol_sha256": data.protocol["protocol_sha256"],
                    "archive_sha256": sha256_file(archive),
                    "metrics": metrics,
                }
                destination.write_text(
                    json.dumps(record, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                manifest.append(
                    {
                        "outer": int(outer),
                        "seed": int(seed),
                        "transform_mode": mode,
                        "archive": str(archive.resolve()),
                        "archive_sha256": record["archive_sha256"],
                        "metrics": str(destination.resolve()),
                        "metrics_sha256": sha256_file(destination),
                    }
                )
                print(
                    json.dumps(
                        {
                            "evaluated": str(destination),
                            "outer": outer,
                            "mode": mode,
                            "seed": seed,
                        }
                    )
                )
    manifest_path = (
        ROOT / config["output_root"] / "confirmation_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "stgf_confirmation_manifest_v2",
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "selection_lock_sha256": config[
                    "development_selection_lock_sha256"
                ],
                "records": manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path.resolve())}))


if __name__ == "__main__":
    main()
