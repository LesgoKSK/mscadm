from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.training_featuremask import FeatureMaskTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the feature-wise RCM + linear-schedule reconstruction")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    selected = deepcopy(model_config(config, "mscadm"))
    selected["diffusion"]["schedule"] = "linear"
    selected["training"]["condition_mask_mode"] = "elementwise"
    selected["training"]["resume"] = True
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    run = root / "mscadm_featuremask_linear"
    checkpoint = FeatureMaskTrainer("mscadm", selected, data, run, device=args.device).fit()
    scenarios, metadata = generate_torch_scenarios(
        checkpoint,
        data,
        scenarios=args.scenarios,
        sampling_steps=50,
        sampler="ddim",
        eta=0.0,
        day_batch=16,
        device=args.device,
    )
    metadata["reconstruction_variant"] = "featurewise_rcm_linear_schedule"
    archive = root / "scenarios" / "mscadm_featuremask_linear_test.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
