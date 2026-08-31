from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from repro.classical import QRGBMConfig
from repro.classical_gpu import GPUQuantileRegressionGBM
from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import save_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the formal multi-quantile QRGBM on CUDA")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config); selected = model_config(config, "qrgbm")
    data = build_gefcom2014(config.get("data_dir", "Data"), seed=int(config.get("split_seed", 0)))
    model = GPUQuantileRegressionGBM(QRGBMConfig(**selected.get("model", {}))).fit(data.train)
    root = Path(config.get("output_root", "outputs/full_reproduction"))
    model_path = root / "qrgbm" / "model_gpu.pkl"; model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    scenarios = model.sample(data.test, scenarios=args.scenarios, seed=int(selected.get("seed", 0)))
    metadata = {"model": "qrgbm", "device": "cuda", "coupling": "ecc", "scenarios": args.scenarios, "model_path": str(model_path.resolve())}
    archive = root / "scenarios" / "qrgbm_test.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
