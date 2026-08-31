from __future__ import annotations

import argparse
import time
from pathlib import Path

from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MS-CADM scenarios at all paper denoising counts")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50, 100, 250])
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    checkpoint = root / "mscadm" / "final.pt"
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    for steps in args.steps:
        started = time.perf_counter()
        scenarios, metadata = generate_torch_scenarios(
            checkpoint,
            data,
            scenarios=args.scenarios,
            sampling_steps=steps,
            sampler="ddim",
            eta=1.0,
            device=args.device,
        )
        metadata["elapsed_seconds"] = time.perf_counter() - started
        path = root / "scenarios" / "sampling_steps" / f"mscadm_{steps}steps.npz"
        save_scenarios(path, scenarios, data.test, metadata)
        print(path.resolve())


if __name__ == "__main__":
    main()
