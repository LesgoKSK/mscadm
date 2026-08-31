from __future__ import annotations

import argparse
from pathlib import Path

from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wind scenarios from a checkpoint")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--model", required=True, choices=["mscadm", "ddpm", "vae", "nf", "wgan"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sampler", choices=["ddim", "ancestral"], default="ddim")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--day-batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    checkpoint = root / args.model / "final.pt"
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    scenarios, metadata = generate_torch_scenarios(
        checkpoint,
        data,
        split_name=args.split,
        scenarios=args.scenarios,
        sampling_steps=args.steps,
        sampler=args.sampler,
        eta=args.eta,
        day_batch=args.day_batch,
        device=args.device,
    )
    suffix = f"_{args.steps}steps" if args.steps is not None else ""
    path = root / "scenarios" / f"{args.model}_{args.split}{suffix}.npz"
    save_scenarios(path, scenarios, getattr(data, args.split), metadata)
    print(path.resolve())


if __name__ == "__main__":
    main()
