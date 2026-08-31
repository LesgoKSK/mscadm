from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from repro.classical import QRGBMConfig, QuantileRegressionGBM
from repro.classical_gpu import GPUQuantileRegressionGBM
from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import save_scenarios


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def sample_from_quantiles(
    quantiles: np.ndarray,
    levels: np.ndarray,
    split,
    residual_rank_templates: dict[int, np.ndarray],
    scenarios: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    result = np.empty((len(split), scenarios, split.target.shape[1]), dtype=np.float32)
    for index, zone in enumerate(split.zone):
        templates = residual_rank_templates[int(zone)]
        selected = templates[generator.integers(0, len(templates), scenarios)]
        probabilities = (
            selected + generator.uniform(-0.5 / 24, 0.5 / 24, selected.shape)
        ).clip(0, 1)
        for hour in range(split.target.shape[1]):
            result[index, :, hour] = QuantileRegressionGBM._interpolate_levels(
                quantiles[index, hour], levels, probabilities[:, hour]
            )
    return np.clip(result, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-bounded, resumable CUDA QRGBM")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--zone", type=int, default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    selected = model_config(config, "qrgbm")
    zones = None if args.zone is None else (args.zone,)
    data = build_gefcom2014(
        config.get("data_dir", "Data"), zones=zones, seed=int(config.get("split_seed", 0))
    )
    qconfig = QRGBMConfig(**selected.get("model", {}))
    factory = GPUQuantileRegressionGBM(qconfig)
    levels = factory.levels
    median_index = int(np.argmin(np.abs(levels - 0.5)))

    output_root = Path(config.get("output_root", "outputs/full_reproduction"))
    if args.zone is None:
        run_root = output_root / "qrgbm_sharded"
        archive_path = output_root / "scenarios" / "qrgbm_test.npz"
    else:
        run_root = output_root / "single_zone" / f"zone{args.zone}" / "qrgbm_sharded"
        archive_path = output_root / "scenarios" / "single_zone" / f"zone{args.zone}_qrgbm.npz"
    run_root.mkdir(parents=True, exist_ok=True)

    train_median = np.empty_like(data.train.target, dtype=np.float32)
    test_quantiles = np.empty(
        (len(data.test), data.test.target.shape[1], len(levels)), dtype=np.float32
    )
    for hour in range(data.train.target.shape[1]):
        model_path = run_root / f"hour_{hour:02d}.ubj"
        train_path = run_root / f"hour_{hour:02d}_train_median.npy"
        test_path = run_root / f"hour_{hour:02d}_test_quantiles.npy"
        if model_path.exists() and train_path.exists() and test_path.exists():
            train_median[:, hour] = np.load(train_path, allow_pickle=False)
            test_quantiles[:, hour] = np.load(test_path, allow_pickle=False)
            print(json.dumps({"hour": hour, "status": "resume"}), flush=True)
            continue
        model = factory._new_model(hour)
        model.fit(data.train.flat_condition, data.train.target[:, hour], verbose=False)
        train_prediction = np.asarray(model.predict(data.train.flat_condition), dtype=np.float32)
        test_prediction = np.asarray(model.predict(data.test.flat_condition), dtype=np.float32)
        if train_prediction.ndim == 1:
            train_prediction = train_prediction[:, None]
        if test_prediction.ndim == 1:
            test_prediction = test_prediction[:, None]
        train_median[:, hour] = train_prediction[:, median_index]
        test_quantiles[:, hour] = np.maximum.accumulate(test_prediction, axis=-1)
        model.save_model(model_path)
        atomic_save_npy(train_path, train_median[:, hour])
        atomic_save_npy(test_path, test_quantiles[:, hour])
        print(
            json.dumps({"hour": hour, "status": "trained", "model_bytes": model_path.stat().st_size}),
            flush=True,
        )
        del model, train_prediction, test_prediction
        gc.collect()

    residuals = data.train.target - train_median
    residual_rank_templates = {}
    for zone in np.unique(data.train.zone):
        zone_residuals = residuals[data.train.zone == zone]
        residual_rank_templates[int(zone)] = np.stack(
            [rankdata(row, method="average") / (len(row) + 1) for row in zone_residuals]
        ).astype(np.float32)
    scenarios = sample_from_quantiles(
        test_quantiles,
        levels,
        data.test,
        residual_rank_templates,
        args.scenarios,
        int(selected.get("seed", 0)),
    )
    metadata = {
        "model": "qrgbm",
        "device": "cuda",
        "coupling": "ecc",
        "scenarios": args.scenarios,
        "zone": args.zone,
        "training_scope": "all_zones" if args.zone is None else "single_zone",
        "storage": "24 independently checkpointed XGBoost UBJ shards",
        "model_directory": str(run_root.resolve()),
    }
    save_scenarios(archive_path, scenarios, data.test, metadata)
    print(archive_path.resolve())


if __name__ == "__main__":
    main()
