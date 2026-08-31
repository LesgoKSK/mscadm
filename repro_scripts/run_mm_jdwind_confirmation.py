from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mm_jdwind.confirmation_data import build_confirmation_gefcom2014
from mm_jdwind.experiment import generate_joint_scenarios, save_joint_archive
from mm_jdwind.metrics import evaluate_joint
from mm_jdwind.training_v2 import StableMMTrainer


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> tuple[Path, dict]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return source.resolve(), json.loads(source.read_text(encoding="utf-8"))


def verify_lock(config: dict) -> tuple[Path, dict]:
    lock_path = ROOT / config["development_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "mm_jdwind_development_selection_lock_v2":
        raise RuntimeError("unexpected development lock schema")
    if lock.get("selected_stage") != "flow":
        raise RuntimeError("confirmation protocol requires flow checkpoint")
    if lock.get("selected_state_mode") != config["sampling"]["locked_state_mode"]:
        raise RuntimeError("confirmation mode differs from development lock")
    return lock_path, lock


def build_data(config: dict, outer: int):
    return build_confirmation_gefcom2014(
        ROOT / config["data_dir"],
        outer=outer,
        split_path=ROOT / config["split_registry"],
    )


def run_dir(config: dict, outer: int, seed: int) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "runs"
        / f"seed{seed}"
    )


def train(config: dict, outer: int, seed: int, device: str | None) -> Path:
    destination = run_dir(config, outer, seed)
    final = destination / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if payload.get("config") != config or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"checkpoint/config mismatch: {final}")
        return final
    data = build_data(config, outer)
    return StableMMTrainer(
        config, data, destination, seed=seed, device=device
    ).fit()


def generate_and_evaluate(
    config: dict,
    outer: int,
    seed: int,
    mode: str,
    device: str | None,
) -> Path:
    lock_path, lock = verify_lock(config)
    if mode not in {
        config["sampling"]["locked_state_mode"],
        config["sampling"]["ablation_state_mode"],
    }:
        raise ValueError("only locked mode and predeclared no-jump ablation are allowed")
    data = build_data(config, outer)
    destination = (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "scenarios"
        / f"flow_{mode}_seed{seed}_test.npz"
    )
    metrics_path = destination.with_suffix(".metrics.json")
    if not destination.exists():
        sampling = config["sampling"]
        arrays, metadata = generate_joint_scenarios(
            run_dir(config, outer, seed) / "flow_best.pt",
            data,
            split_name="test",
            members=int(sampling["members"]),
            jump_steps=int(sampling["jump_steps"]),
            flow_steps=int(sampling["flow_steps"]),
            state_mode=mode,
            day_batch=int(sampling["day_batch"]),
            member_chunk=int(sampling["member_chunk"]),
            seed=1_100_000
            + outer * 10_000
            + seed * 100
            + (0 if mode == sampling["locked_state_mode"] else 50),
            device=device,
        )
        metadata.update(
            {
                "evidence_scope": config["evidence_label"],
                "development_lock": str(lock_path.resolve()),
                "development_lock_sha256": sha256_file(lock_path),
                "locked_candidate": lock["selected"],
                "confirmation_protocol_sha256": data.protocol["protocol_sha256"],
            }
        )
        save_joint_archive(destination, arrays, metadata)
    if not metrics_path.exists():
        with np.load(destination, allow_pickle=False) as stored:
            metrics = evaluate_joint(
                stored["scenarios"],
                data.test,
                zero_probability=stored["zero_probability"],
                one_probability=stored["one_probability"],
            )
        metrics_path.write_text(
            json.dumps(
                {
                    "outer": outer,
                    "seed": seed,
                    "split": "new_frozen_test",
                    "state_mode": mode,
                    "evidence_scope": config["evidence_label"],
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
        description="Run MM-JDWind on post-freeze internal confirmation splits"
    )
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_confirmation_v1.json"
    )
    parser.add_argument(
        "--phase", choices=["train", "test", "all"], required=True
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["mass_preserving", "none"])
    parser.add_argument("--device")
    args = parser.parse_args()
    _, config = load_config(args.config)
    verify_lock(config)
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    modes = (
        [args.mode]
        if args.mode
        else [
            config["sampling"]["locked_state_mode"],
            config["sampling"]["ablation_state_mode"],
        ]
    )
    for outer in outers:
        for seed in seeds:
            if args.phase in {"train", "all"}:
                path = train(config, int(outer), int(seed), args.device)
                print(json.dumps({"trained": str(path), "outer": outer, "seed": seed}))
            if args.phase in {"test", "all"}:
                for mode in modes:
                    path = generate_and_evaluate(
                        config, int(outer), int(seed), mode, args.device
                    )
                    print(
                        json.dumps(
                            {
                                "evaluated": str(path),
                                "outer": outer,
                                "seed": seed,
                                "mode": mode,
                            }
                        )
                    )


if __name__ == "__main__":
    main()
