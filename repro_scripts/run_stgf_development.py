from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stgf_flow.data import build_stgf_development_gefcom2014
from stgf_flow.experiment import (
    generate_stgf_scenarios,
    save_stgf_archive,
)
from stgf_flow.graph import SpectralArtifacts
from stgf_flow.metrics import evaluate_stgf
from stgf_flow.training import STGFTrainer


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return json.loads(source.read_text(encoding="utf-8"))


def run_dir(
    config: dict, outer: int, seed: int, transform_mode: str
) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "runs"
        / transform_mode
        / f"seed{seed}"
    )


def scenario_path(
    config: dict, outer: int, seed: int, transform_mode: str
) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "candidates"
        / f"{transform_mode}_seed{seed}_calibration.npz"
    )


def train_one(
    config: dict,
    *,
    outer: int,
    seed: int,
    transform_mode: str,
    device: str | None,
) -> Path:
    destination = run_dir(config, outer, seed, transform_mode)
    final = destination / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if (
            payload.get("config") != config
            or payload.get("transform_mode") != transform_mode
            or int(payload.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"checkpoint/config mismatch: {final}")
        return final
    data = build_stgf_development_gefcom2014(
        ROOT / config["data_dir"], outer=outer
    )
    return STGFTrainer(
        config,
        data,
        destination,
        seed=seed,
        transform_mode=transform_mode,
        device=device,
    ).fit()


def calibration_one(
    config: dict,
    *,
    outer: int,
    seed: int,
    transform_mode: str,
    device: str | None,
) -> Path:
    data = build_stgf_development_gefcom2014(
        ROOT / config["data_dir"], outer=outer
    )
    checkpoint = run_dir(config, outer, seed, transform_mode) / "final.pt"
    destination = scenario_path(config, outer, seed, transform_mode)
    metrics_path = destination.with_suffix(".metrics.json")
    if not destination.exists():
        sampling = config["sampling"]
        arrays, metadata = generate_stgf_scenarios(
            checkpoint,
            data,
            split_name="calibration",
            members=int(sampling["members"]),
            flow_steps=int(sampling["flow_steps"]),
            day_batch=int(sampling["day_batch"]),
            member_chunk=int(sampling["member_chunk"]),
            seed=1_500_000
            + outer * 10_000
            + seed * 100
            + config["transform_modes"].index(transform_mode) * 10,
            device=device,
        )
        metadata["evidence_scope"] = config["evidence_label"]
        save_stgf_archive(destination, arrays, metadata)
    if not metrics_path.exists():
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        artifacts = SpectralArtifacts.from_torch_payload(payload["artifacts"])
        with np.load(destination, allow_pickle=False) as stored:
            metrics = evaluate_stgf(
                stored["scenarios"], data.calibration, artifacts
            )
        metrics_path.write_text(
            json.dumps(
                {
                    "outer": outer,
                    "seed": seed,
                    "transform_mode": transform_mode,
                    "split": "calibration_only",
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return metrics_path


def selected(configured: list, requested):
    if requested is None:
        return configured
    if requested not in configured:
        raise ValueError(f"{requested} is not registered in {configured}")
    return [requested]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and screen STGF-Flow development variants"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_development_v1.json"
    )
    parser.add_argument(
        "--phase", required=True, choices=["train", "calibration", "all"]
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--transform-mode",
        choices=[
            "time_domain",
            "graph_only",
            "time_frequency",
            "stgf",
        ],
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    outers = selected(config["outer_splits"], args.outer)
    seeds = selected(config["model_seeds"], args.seed)
    modes = selected(config["transform_modes"], args.transform_mode)
    for outer in outers:
        for mode in modes:
            for seed in seeds:
                if args.phase in {"train", "all"}:
                    path = train_one(
                        config,
                        outer=int(outer),
                        seed=int(seed),
                        transform_mode=mode,
                        device=args.device,
                    )
                    print(
                        json.dumps(
                            {
                                "trained": str(path),
                                "outer": outer,
                                "seed": seed,
                                "transform_mode": mode,
                            }
                        )
                    )
                if args.phase in {"calibration", "all"}:
                    path = calibration_one(
                        config,
                        outer=int(outer),
                        seed=int(seed),
                        transform_mode=mode,
                        device=args.device,
                    )
                    print(
                        json.dumps(
                            {
                                "evaluated": str(path),
                                "outer": outer,
                                "seed": seed,
                                "transform_mode": mode,
                            }
                        )
                    )


if __name__ == "__main__":
    main()
