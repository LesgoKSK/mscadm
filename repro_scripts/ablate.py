from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.training import ExperimentTrainer


VARIANTS = {
    "full": {},
    "no_ce": {"model.use_multiscale_embedding": False},
    "no_adaln": {"model.use_adaln": False},
    "no_lv": {"model.learn_variance": False, "diffusion.learn_variance": False},
    "no_rcm": {"training.condition_mask_probability": 0.0},
}


def assign(config: dict, dotted: str, value: object) -> None:
    target = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a single-zone MS-CADM ablation")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--zone", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root_config = load_config(args.config)
    selected = deepcopy(model_config(root_config, "mscadm"))
    for key, value in VARIANTS[args.variant].items():
        assign(selected, key, value)
    data = build_gefcom2014(
        root_config.get("data_dir", "Data"), zones=(args.zone,), seed=int(root_config.get("split_seed", 0))
    )
    root = Path(root_config.get("output_root", "outputs/full_reproduction")) / "ablations" / f"zone{args.zone}" / args.variant
    checkpoint = ExperimentTrainer("mscadm", selected, data, root, device=args.device).fit()
    sampling = selected.get("sampling", {})
    scenarios, metadata = generate_torch_scenarios(
        checkpoint,
        data,
        scenarios=args.scenarios,
        sampling_steps=int(sampling.get("steps", 50)),
        sampler=sampling.get("sampler", "ddim"),
        eta=float(sampling.get("eta", 1.0)),
        device=args.device,
    )
    metadata.update({"variant": args.variant, "zone": args.zone})
    output = root_config.get("output_root", "outputs/full_reproduction")
    archive = Path(output) / "scenarios" / "ablations" / f"zone{args.zone}_{args.variant}.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
