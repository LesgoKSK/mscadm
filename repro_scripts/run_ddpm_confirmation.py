from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mm_jdwind.confirmation_data import build_confirmation_gefcom2014
from mm_jdwind.metrics import evaluate_joint
from repro.sampling import generate_torch_scenarios
from repro.training import ExperimentTrainer


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return json.loads(source.read_text(encoding="utf-8"))


def build_data(config: dict, outer: int):
    return build_confirmation_gefcom2014(
        ROOT / config["data_dir"],
        outer=outer,
        split_path=ROOT / config["split_registry"],
    )


def model_config(config: dict, seed: int) -> dict:
    return {
        "experiment_name": config["experiment_name"],
        "protocol_revision": config["protocol_revision"],
        "seed": seed,
        "device": config["device"],
        "model": deepcopy(config["model"]),
        "diffusion": deepcopy(config["diffusion"]),
        "training": deepcopy(config["training"]),
        "sampling": deepcopy(config["sampling"]),
        "evidence_label": config["evidence_label"],
    }


def run_dir(config: dict, outer: int, seed: int) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "runs"
        / f"seed{seed}"
    )


def jointify_generated(
    scenarios: np.ndarray, day: np.ndarray, zone: np.ndarray
) -> np.ndarray:
    days = np.sort(np.unique(day.astype("datetime64[D]")))
    chunks: list[np.ndarray] = []
    for current in days:
        indices = np.flatnonzero(day.astype("datetime64[D]") == current)
        if len(indices) != 10:
            raise ValueError("DDPM split does not contain ten zones per day")
        selected = indices[np.argsort(zone[indices])]
        if not np.array_equal(zone[selected], np.arange(1, 11)):
            raise ValueError("DDPM split zones are not 1..10")
        chunks.append(scenarios[selected].transpose(1, 0, 2))
    return np.ascontiguousarray(np.stack(chunks), dtype=np.float32)


def train(config: dict, outer: int, seed: int, device: str | None) -> Path:
    data = build_data(config, outer)
    destination = run_dir(config, outer, seed)
    final = destination / "final.pt"
    selected = model_config(config, seed)
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if payload.get("name") != "ddpm" or payload.get("config") != selected:
            raise RuntimeError(f"DDPM checkpoint/config mismatch: {final}")
        return final
    return ExperimentTrainer(
        "ddpm", selected, data.source, destination, device=device
    ).fit()


def test(config: dict, outer: int, seed: int, device: str | None) -> Path:
    data = build_data(config, outer)
    output = ROOT / config["output_root"] / f"outer{outer}" / "scenarios"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"ddpm_seed{seed}_test.npz"
    metrics_path = archive.with_suffix(".metrics.json")
    if not archive.exists():
        sampling = config["sampling"]
        generated, metadata = generate_torch_scenarios(
            run_dir(config, outer, seed) / "final.pt",
            data.source,
            split_name="test",
            scenarios=int(sampling["members"]),
            sampling_steps=int(sampling["steps"]),
            sampler=sampling["sampler"],
            eta=float(sampling["eta"]),
            day_batch=int(sampling["day_batch"]),
            device=device,
            seed=1_300_000 + outer * 10_000 + seed * 100,
        )
        joint = jointify_generated(
            generated, data.source.test.day, data.source.test.zone
        )
        metadata.update(
            {
                "joint_comparison_scope": config["comparison_scope"],
                "evidence_scope": config["evidence_label"],
                "confirmation_protocol_sha256": data.protocol["protocol_sha256"],
            }
        )
        np.savez_compressed(
            archive,
            scenarios=joint,
            observations=data.test.target.astype(np.float32),
            day=data.test.day,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    if not metrics_path.exists():
        with np.load(archive, allow_pickle=False) as stored:
            metrics = evaluate_joint(stored["scenarios"], data.test)
        metrics_path.write_text(
            json.dumps(
                {
                    "outer": outer,
                    "seed": seed,
                    "model": "DDPM",
                    "split": "new_frozen_test",
                    "comparison_scope": config["comparison_scope"],
                    "protocol_sha256": data.protocol["protocol_sha256"],
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate DDPM on MM-JDWind confirmation splits"
    )
    parser.add_argument(
        "--config", default="repro_configs/ddpm_confirmation_v1.json"
    )
    parser.add_argument("--phase", required=True, choices=["train", "test", "all"])
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    for outer in outers:
        for seed in seeds:
            if args.phase in {"train", "all"}:
                path = train(config, int(outer), int(seed), args.device)
                print(json.dumps({"trained": str(path), "outer": outer, "seed": seed}))
            if args.phase in {"test", "all"}:
                path = test(config, int(outer), int(seed), args.device)
                print(json.dumps({"evaluated": str(path), "outer": outer, "seed": seed}))


if __name__ == "__main__":
    main()
