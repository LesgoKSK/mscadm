from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mm_jdwind.data import build_joint_nested_gefcom2014
from mm_jdwind.experiment import generate_joint_scenarios, save_joint_archive
from mm_jdwind.metrics import evaluate_joint
from mm_jdwind.training import MMTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "repro_configs" / "mm_jdwind_development.json"


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    return resolved, config


def run_dir(config: dict[str, Any], outer: int, seed: int) -> Path:
    return ROOT / config["output_root"] / f"outer{outer}" / "runs" / f"seed{seed}"


def archive_path(
    config: dict[str, Any], outer: int, seed: int, split: str, state_mode: str
) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "scenarios"
        / f"{state_mode}_seed{seed}_{split}.npz"
    )


def train_one(
    config: dict[str, Any], *, outer: int, seed: int, device: str | None
) -> Path:
    data = build_joint_nested_gefcom2014(ROOT / config["data_dir"], outer=outer)
    destination = run_dir(config, outer, seed)
    final = destination / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if payload.get("config") != config or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"existing checkpoint does not match config/seed: {final}")
        print(json.dumps({"skip": "train", "checkpoint": str(final.resolve())}), flush=True)
        return final
    return MMTrainer(config, data, destination, seed=seed, device=device).fit()


def generate_one(
    config: dict[str, Any],
    *,
    outer: int,
    seed: int,
    split: str,
    state_mode: str,
    device: str | None,
) -> Path:
    destination = archive_path(config, outer, seed, split, state_mode)
    if destination.exists():
        print(
            json.dumps({"skip": "generate", "archive": str(destination.resolve())}),
            flush=True,
        )
        return destination
    checkpoint = run_dir(config, outer, seed) / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    data = build_joint_nested_gefcom2014(ROOT / config["data_dir"], outer=outer)
    sampling = config["sampling"]
    arrays, metadata = generate_joint_scenarios(
        checkpoint,
        data,
        split_name=split,
        members=int(sampling["members"]),
        jump_steps=int(sampling["jump_steps"]),
        flow_steps=int(sampling["flow_steps"]),
        state_mode=state_mode,
        day_batch=int(sampling["day_batch"]),
        member_chunk=int(sampling["member_chunk"]),
        seed=730_000
        + 10_000 * outer
        + 100 * seed
        + (0 if split == "calibration" else 1),
        device=device,
    )
    save_joint_archive(destination, arrays, metadata)
    print(json.dumps({"generated": str(destination.resolve()), **metadata}), flush=True)
    return destination


def evaluate_one(
    config: dict[str, Any],
    *,
    outer: int,
    seed: int,
    split: str,
    state_mode: str,
) -> Path:
    source = archive_path(config, outer, seed, split, state_mode)
    if not source.exists():
        raise FileNotFoundError(source)
    data = build_joint_nested_gefcom2014(ROOT / config["data_dir"], outer=outer)
    selected = getattr(data, split)
    with np.load(source, allow_pickle=False) as stored:
        metrics = evaluate_joint(
            stored["scenarios"],
            selected,
            zero_probability=stored["zero_probability"],
            one_probability=stored["one_probability"],
        )
    destination = source.with_suffix(".metrics.json")
    destination.write_text(
        json.dumps(
            {
                "outer": outer,
                "seed": seed,
                "split": split,
                "state_mode": state_mode,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"evaluated": str(destination.resolve()), **metrics}), flush=True)
    return destination


def _selected(configured: list[int], requested: int | None) -> list[int]:
    values = [int(value) for value in configured]
    if requested is None:
        return values
    if requested not in values:
        raise ValueError(f"{requested} is not registered in {values}")
    return [requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MM-JDWind stages")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--phase", required=True, choices=["train", "generate", "evaluate", "all"]
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument(
        "--state-mode",
        choices=["none", "independent", "correlated", "mass_preserving"],
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    _, config = load_config(args.config)
    outers = _selected(config["outer_splits"], args.outer)
    seeds = _selected(config["model_seeds"], args.seed)
    modes = (
        [args.state_mode]
        if args.state_mode is not None
        else list(config["sampling"]["state_modes"])
    )
    for outer in outers:
        for seed in seeds:
            if args.phase in {"train", "all"}:
                train_one(config, outer=outer, seed=seed, device=args.device)
            if args.phase in {"generate", "all"}:
                for mode in modes:
                    generate_one(
                        config,
                        outer=outer,
                        seed=seed,
                        split=args.split,
                        state_mode=mode,
                        device=args.device,
                    )
            if args.phase in {"evaluate", "all"}:
                for mode in modes:
                    evaluate_one(
                        config,
                        outer=outer,
                        seed=seed,
                        split=args.split,
                        state_mode=mode,
                    )


if __name__ == "__main__":
    main()
