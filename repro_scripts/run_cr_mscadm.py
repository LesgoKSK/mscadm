from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cr_mscadm.experiment import (
    calibrate_and_save,
    extended_scores,
    generate_cr_scenarios,
    paired_bootstrap,
    per_day_scores,
)
from cr_mscadm.training import CRTrainer, seed_everything
from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios, save_scenarios


def run_dir(root: Path, variant: str, seed: int) -> Path:
    return root / "runs" / variant / f"seed{seed}"


def archive_path(root: Path, variant: str, seed: int, split: str, state: str) -> Path:
    return root / "scenarios" / f"{variant}_seed{seed}_{split}_{state}.npz"


def variants(config: dict) -> list[tuple[str, int]]:
    values = [("full", int(seed)) for seed in config["experiment"]["full_seeds"]]
    values.extend((str(name), 0) for name in config["experiment"]["ablations"])
    return values


def train_all(config: dict, data, *, selected: list[tuple[str, int]], device: str | None) -> None:
    root = Path(config["output_root"])
    for variant, seed in selected:
        final = run_dir(root, variant, seed) / "final.pt"
        if final.exists():
            print(json.dumps({"skip": "training", "checkpoint": str(final)}), flush=True)
            continue
        checkpoint = CRTrainer(
            config, data, run_dir(root, variant, seed), variant=variant, seed=seed, device=device
        ).fit()
        print(json.dumps({"trained": str(checkpoint.resolve())}), flush=True)


def generate_all(config: dict, data, *, selected: list[tuple[str, int]], device: str | None) -> None:
    root = Path(config["output_root"])
    sampling = config["sampling"]
    for variant, seed in selected:
        checkpoint = run_dir(root, variant, seed) / "final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for split in ("validation", "test"):
            path = archive_path(root, variant, seed, split, "raw")
            if path.exists():
                print(json.dumps({"skip": "generation", "archive": str(path)}), flush=True)
                continue
            scenarios, metadata = generate_cr_scenarios(
                checkpoint,
                data,
                split_name=split,
                scenarios=int(sampling["scenarios"]),
                steps=int(sampling["steps"]),
                eta=float(sampling["eta"]),
                day_batch=int(sampling.get("day_batch", 8)),
                seed=10_000 + seed + (0 if split == "validation" else 1_000),
                device=device,
            )
            save_scenarios(path, scenarios, getattr(data, split), metadata)
            print(json.dumps({"generated": str(path.resolve())}), flush=True)


def generate_controlled_baseline(config: dict, data, *, device: str | None) -> None:
    root = Path(config["output_root"])
    checkpoint = Path("outputs/full_reproduction/mscadm/final.pt")
    sampling = config["sampling"]
    for split in ("validation", "test"):
        path = root / "scenarios" / f"baseline_mscadm_{split}_raw.npz"
        if path.exists():
            continue
        seed_everything(20_000 + (0 if split == "validation" else 1_000))
        scenarios, metadata = generate_torch_scenarios(
            checkpoint,
            data,
            split_name=split,
            scenarios=int(sampling["scenarios"]),
            sampling_steps=int(sampling["steps"]),
            sampler="ddim",
            eta=float(sampling["eta"]),
            day_batch=int(sampling.get("day_batch", 8)),
            device=device,
        )
        metadata["controlled_resampling"] = True
        save_scenarios(path, scenarios, getattr(data, split), metadata)
        print(json.dumps({"generated_baseline": str(path.resolve())}), flush=True)


def calibrate_all(config: dict) -> None:
    root = Path(config["output_root"])
    calibration_root = root / "calibration"
    targets = [("baseline_mscadm", None)]
    targets.extend(("full", int(seed)) for seed in config["experiment"]["full_seeds"])
    for variant, seed in targets:
        if seed is None:
            validation = root / "scenarios" / "baseline_mscadm_validation_raw.npz"
            test = root / "scenarios" / "baseline_mscadm_test_raw.npz"
            output = root / "scenarios" / "baseline_mscadm_test_calibrated.npz"
            calibration = calibration_root / "baseline_mscadm.json"
        else:
            validation = archive_path(root, variant, seed, "validation", "raw")
            test = archive_path(root, variant, seed, "test", "raw")
            output = archive_path(root, variant, seed, "test", "calibrated")
            calibration = calibration_root / f"{variant}_seed{seed}.json"
        calibrated, metadata = calibrate_and_save(validation, test, output, calibration)
        print(
            json.dumps(
                {
                    "calibrated": str(output.resolve()),
                    "shape": list(calibrated.shape),
                    "strength": metadata["strength"],
                    "rank_inversions": metadata["rank_inversions"],
                }
            ),
            flush=True,
        )


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return archive["scenarios"], archive["observations"]


def evaluate_all(config: dict) -> None:
    root = Path(config["output_root"])
    table_root = root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {
        "MS-CADM controlled raw": root / "scenarios" / "baseline_mscadm_test_raw.npz",
        "MS-CADM + calibration": root / "scenarios" / "baseline_mscadm_test_calibrated.npz",
        "CR fixed-scale": archive_path(root, "fixed", 0, "test", "raw"),
        "CR hetero w/o CRPS": archive_path(root, "hetero_no_crps", 0, "test", "raw"),
    }
    for seed in config["experiment"]["full_seeds"]:
        sources[f"CR full raw seed{seed}"] = archive_path(root, "full", int(seed), "test", "raw")
        sources[f"CR full calibrated seed{seed}"] = archive_path(
            root, "full", int(seed), "test", "calibrated"
        )
    reference = Path(config["experiment"]["secondary_baseline"])
    if reference.exists():
        sources["MS-CADM reproduced 250-step"] = reference
    records: list[dict[str, float | str]] = []
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(path)
        scenarios, observations = _load(path)
        loaded[name] = (scenarios, observations)
        records.append({"method": name, **extended_scores(scenarios, observations)})
    frame = pd.DataFrame(records)
    frame.to_csv(table_root / "all_metrics.csv", index=False)

    seed_rows = frame[frame["method"].str.startswith("CR full")].copy()
    seed_rows["state"] = np.where(seed_rows["method"].str.contains("calibrated"), "calibrated", "raw")
    numeric = [column for column in seed_rows.columns if column not in {"method", "state"}]
    seed_summary = seed_rows.groupby("state")[numeric].agg(["mean", "std"])
    seed_summary.to_csv(table_root / "seed_robustness.csv")

    baseline_day = per_day_scores(*loaded["MS-CADM controlled raw"])
    bootstrap: dict[str, dict[str, dict[str, float]]] = {}
    for seed in config["experiment"]["full_seeds"]:
        name = f"CR full calibrated seed{seed}"
        method_day = per_day_scores(*loaded[name])
        bootstrap[name] = {}
        for metric in ("MAE", "CRPS", "VS"):
            bootstrap[name][metric] = paired_bootstrap(
                baseline_day[metric],
                method_day[metric],
                seed=2026 + int(seed),
                replicates=int(config["experiment"]["bootstrap_replicates"]),
            )
        bootstrap[name]["coverage_90"] = paired_bootstrap(
            baseline_day["coverage_90"],
            method_day["coverage_90"],
            seed=3026 + int(seed),
            replicates=int(config["experiment"]["bootstrap_replicates"]),
            higher_is_better=True,
        )
    (table_root / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )

    ablation_order = [
        "MS-CADM controlled raw",
        "CR fixed-scale",
        "CR hetero w/o CRPS",
        "CR full raw seed0",
        "CR full calibrated seed0",
        "MS-CADM + calibration",
    ]
    frame.set_index("method").loc[ablation_order].reset_index().to_csv(
        table_root / "ablation.csv", index=False
    )
    print(frame.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete CR-MS-CADM experiment")
    parser.add_argument("--config", default="repro_configs/cr_mscadm.json")
    parser.add_argument(
        "--phase", choices=["train", "generate", "calibrate", "evaluate", "all"], default="all"
    )
    parser.add_argument("--variant", choices=["fixed", "hetero_no_crps", "full"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    data = build_gefcom2014(config["data_dir"], seed=int(config.get("split_seed", 0)))
    selected = variants(config)
    if args.variant is not None:
        selected = [(args.variant, int(args.seed or 0))]
    elif args.seed is not None:
        selected = [("full", args.seed)]
    if args.phase in {"train", "all"}:
        train_all(config, data, selected=selected, device=args.device)
    if args.phase in {"generate", "all"}:
        generate_all(config, data, selected=selected, device=args.device)
        if args.variant is None and args.seed is None:
            generate_controlled_baseline(config, data, device=args.device)
    if args.phase in {"calibrate", "all"}:
        calibrate_all(config)
    if args.phase in {"evaluate", "all"}:
        evaluate_all(config)


if __name__ == "__main__":
    main()
