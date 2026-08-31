from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from cr_mscadm.experiment import calibrate_and_save, extended_scores, generate_cr_scenarios
from cr_mscadm.training import CRTrainer
from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.sampling import save_scenarios


def main() -> None:
    config = deepcopy(load_config("repro_configs/cr_mscadm.json"))
    config["training"]["condition_mask_probability"] = 0.0
    data = build_gefcom2014(config["data_dir"], seed=int(config.get("split_seed", 0)))
    root = Path(config["output_root"])
    run = root / "runs" / "full_nomask" / "seed0"
    checkpoint = run / "final.pt"
    if not checkpoint.exists():
        checkpoint = CRTrainer(config, data, run, variant="full", seed=0, device="cuda").fit()
    sampling = config["sampling"]
    paths = {}
    for split in ("validation", "test"):
        path = root / "scenarios" / f"full_nomask_seed0_{split}_raw.npz"
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
                seed=30_000 + (0 if split == "validation" else 1_000),
                device="cuda",
            )
            metadata["condition_mask_control"] = "none"
            save_scenarios(path, scenarios, getattr(data, split), metadata)
    calibrated_path = root / "scenarios" / "full_nomask_seed0_test_calibrated.npz"
    calibrate_and_save(
        paths["validation"],
        paths["test"],
        calibrated_path,
        root / "calibration" / "full_nomask_seed0.json",
    )
    rows = []
    for name, path in [("full no-mask raw", paths["test"]), ("full no-mask calibrated", calibrated_path)]:
        import numpy as np

        archive = np.load(path, allow_pickle=False)
        rows.append({"method": name, **extended_scores(archive["scenarios"], archive["observations"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "tables" / "mask_control.csv", index=False)
    (run / "control_definition.json").write_text(
        json.dumps(
            {
                "purpose": "exclude feature-wise condition masking as the main source of improvement",
                "condition_mask_probability": 0.0,
                "all_other_hyperparameters": "identical to CR-MS-CADM full seed0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
