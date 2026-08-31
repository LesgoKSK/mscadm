from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cr_mscadm.experiment import calibrate_and_save, extended_scores, generate_cr_scenarios
from cr_mscadm.training import CRTrainer
from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.sampling import save_scenarios


class SampleMaskContext:
    """Make CRTrainer's mask draw one uniform value per sample.

    CRTrainer calls ``torch.rand_like(condition)`` exactly once per diffusion
    update for RCM. Replacing only tensors shaped [B,24,20] by an expanded
    [B,1,1] draw reproduces the original reconstruction's whole-condition
    masking without changing any other random draw or training code.
    """

    def __enter__(self):
        self.original = torch.rand_like

        def samplewise(value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if value.ndim == 3 and value.shape[1:] == (24, 20):
                draw = self.original(value[:, :1, :1], *args, **kwargs)
                return draw.expand_as(value)
            return self.original(value, *args, **kwargs)

        torch.rand_like = samplewise
        return self

    def __exit__(self, *_):
        torch.rand_like = self.original


def main() -> None:
    config = deepcopy(load_config("repro_configs/cr_mscadm.json"))
    config["training"]["condition_mask_probability"] = 0.1
    data = build_gefcom2014(config["data_dir"], seed=int(config.get("split_seed", 0)))
    root = Path(config["output_root"])
    run = root / "runs" / "full_samplemask" / "seed0"
    checkpoint = run / "final.pt"
    if not checkpoint.exists():
        with SampleMaskContext():
            checkpoint = CRTrainer(config, data, run, variant="full", seed=0, device="cuda").fit()
    sampling = config["sampling"]
    paths = {}
    for split in ("validation", "test"):
        path = root / "scenarios" / f"full_samplemask_seed0_{split}_raw.npz"
        paths[split] = path
        if not path.exists():
            scenarios, metadata = generate_cr_scenarios(
                checkpoint,
                data,
                split_name=split,
                scenarios=int(sampling["scenarios"]),
                steps=int(sampling["steps"]),
                eta=float(sampling["eta"]),
                day_batch=int(sampling["day_batch"]),
                seed=40_000 + (0 if split == "validation" else 1_000),
                device="cuda",
            )
            metadata["condition_mask_control"] = "whole sample, probability 0.1"
            save_scenarios(path, scenarios, getattr(data, split), metadata)
    calibrated_path = root / "scenarios" / "full_samplemask_seed0_test_calibrated.npz"
    calibrate_and_save(
        paths["validation"],
        paths["test"],
        calibrated_path,
        root / "calibration" / "full_samplemask_seed0.json",
    )
    rows = []
    for name, path in [
        ("full sample-mask raw", paths["test"]),
        ("full sample-mask calibrated", calibrated_path),
    ]:
        archive = np.load(path, allow_pickle=False)
        rows.append({"method": name, **extended_scores(archive["scenarios"], archive["observations"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "tables" / "samplemask_control.csv", index=False)
    (run / "control_definition.json").write_text(
        json.dumps(
            {
                "purpose": "match the reconstructed baseline's whole-condition RCM exactly",
                "mask_shape": "[batch,1,1], expanded to [batch,24,20]",
                "condition_mask_probability": 0.1,
                "all_other_hyperparameters": "identical to CR-MS-CADM full seed0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
