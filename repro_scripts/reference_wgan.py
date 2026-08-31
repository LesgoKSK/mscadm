from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.training import ExperimentTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the public Dumas WGAN-GP schedule")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    selected = deepcopy(model_config(config, "wgan"))
    # Public wind baseline: 300 epochs, 10 minibatches/epoch, batch=10% of LS.
    selected["training"].update({"steps": 3_000, "batch_size": 631, "resume": False})
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    checkpoint = ExperimentTrainer("wgan", selected, data, root / "wgan_reference", device=args.device).fit()
    scenarios, metadata = generate_torch_scenarios(
        checkpoint, data, scenarios=args.scenarios, day_batch=64, device=args.device
    )
    metadata["baseline_variant"] = "dumas_reference_schedule"
    archive = root / "scenarios" / "wgan_reference_test.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
