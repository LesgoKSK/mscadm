from __future__ import annotations

import argparse
from pathlib import Path

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.training import ExperimentTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one full-reproduction model")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--model", required=True, choices=["mscadm", "ddpm", "vae", "nf", "wgan"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    selected = model_config(config, args.model)
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    output = Path(config.get("output_root", "outputs/full_reproduction")) / args.model
    checkpoint = ExperimentTrainer(
        args.model, selected, data, output, device=args.device
    ).fit()
    print(checkpoint.resolve())


if __name__ == "__main__":
    main()
