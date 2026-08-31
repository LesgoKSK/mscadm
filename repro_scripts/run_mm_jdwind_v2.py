from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mm_jdwind.data import build_joint_nested_gefcom2014
from mm_jdwind.training_v2 import StableMMTrainer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train finite-guarded MM-JDWind v2 development runs"
    )
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_development_v2.json"
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    for outer in outers:
        for seed in seeds:
            destination = (
                ROOT
                / config["output_root"]
                / f"outer{outer}"
                / "runs"
                / f"seed{seed}"
            )
            final = destination / "final.pt"
            if final.exists():
                payload = torch.load(final, map_location="cpu", weights_only=False)
                if payload.get("config") != config or int(payload.get("seed", -1)) != seed:
                    raise RuntimeError(f"checkpoint/config mismatch: {final}")
                print(json.dumps({"skip": "train", "checkpoint": str(final)}))
                continue
            data = build_joint_nested_gefcom2014(
                ROOT / config["data_dir"], outer=int(outer)
            )
            trainer = StableMMTrainer(
                config, data, destination, seed=int(seed), device=args.device
            )
            final = trainer.fit()
            print(json.dumps({"trained": str(final), "outer": outer, "seed": seed}))


if __name__ == "__main__":
    main()
