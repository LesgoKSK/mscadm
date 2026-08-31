from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from repro.classical import QRGBMConfig, QuantileRegressionGBM, random_scenarios
from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import save_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/generate RAND and QRGBM baselines")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--model", required=True, choices=["rand", "rand_train", "qrgbm"])
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    if args.model in {"rand", "rand_train"}:
        source = None if args.model == "rand" else data.train
        scenarios = random_scenarios(
            data.test, scenarios=args.scenarios, source=source, seed=int(config.get("seed", 0))
        )
        metadata = {
            "model": args.model,
            "protocol": "evaluation-split sampling" if source is None else "training-split sampling",
            "scenarios": args.scenarios,
        }
    else:
        selected = model_config(config, "qrgbm")
        qrgbm = QuantileRegressionGBM(QRGBMConfig(**selected.get("model", {}))).fit(data.train)
        model_path = root / "qrgbm" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(qrgbm, handle)
        scenarios = qrgbm.sample(
            data.test,
            scenarios=args.scenarios,
            seed=int(selected.get("seed", 0)),
            coupling=selected.get("sampling", {}).get("coupling", "ecc"),
        )
        metadata = {
            "model": "qrgbm",
            "scenarios": args.scenarios,
            "model_path": str(model_path.resolve()),
            "coupling": selected.get("sampling", {}).get("coupling", "ecc"),
        }
    path = root / "scenarios" / f"{args.model}_test.npz"
    save_scenarios(path, scenarios, data.test, metadata)
    print(path.resolve())


if __name__ == "__main__":
    main()
