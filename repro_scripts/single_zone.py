from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from repro.classical import QRGBMConfig, QuantileRegressionGBM, random_scenarios
from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.training import ExperimentTrainer


TRAINABLE = {"mscadm", "ddpm", "vae", "nf", "wgan"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and generate a paper Table-2 single-zone model")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--model", required=True, choices=sorted(TRAINABLE | {"rand", "rand_train", "qrgbm"}))
    parser.add_argument("--zone", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data = build_gefcom2014(config.get("data_dir", "Data"), zones=(args.zone,), seed=int(config.get("split_seed", 0)))
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    if args.model in TRAINABLE:
        selected = model_config(config, args.model)
        run = root / "single_zone" / f"zone{args.zone}" / args.model
        checkpoint = ExperimentTrainer(args.model, selected, data, run, device=args.device).fit()
        sampling = selected.get("sampling", {})
        scenarios, metadata = generate_torch_scenarios(
            checkpoint,
            data,
            scenarios=args.scenarios,
            sampling_steps=sampling.get("steps"),
            sampler=sampling.get("sampler", "ddim"),
            eta=float(sampling.get("eta", 1.0)),
            device=args.device,
        )
    elif args.model in {"rand", "rand_train"}:
        source = None if args.model == "rand" else data.train
        scenarios = random_scenarios(data.test, scenarios=args.scenarios, source=source, seed=int(config.get("seed", 0)))
        metadata = {"model": args.model, "protocol": "evaluation" if source is None else "training"}
    else:
        selected = model_config(config, "qrgbm")
        qrgbm = QuantileRegressionGBM(QRGBMConfig(**selected.get("model", {}))).fit(data.train)
        run = root / "single_zone" / f"zone{args.zone}" / "qrgbm"
        run.mkdir(parents=True, exist_ok=True)
        with (run / "model.pkl").open("wb") as handle:
            pickle.dump(qrgbm, handle)
        scenarios = qrgbm.sample(data.test, scenarios=args.scenarios, seed=int(selected.get("seed", 0)))
        metadata = {"model": "qrgbm", "coupling": "ecc"}
    metadata.update({"zone": args.zone, "training_scope": "single_zone", "scenarios": args.scenarios})
    path = root / "scenarios" / "single_zone" / f"zone{args.zone}_{args.model}.npz"
    save_scenarios(path, scenarios, data.test, metadata)
    print(path.resolve())


if __name__ == "__main__":
    main()
