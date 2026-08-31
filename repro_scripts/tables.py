from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro.metrics import all_scores


PAPER_TABLE1 = {
    "RAND": [0.2583, 0.3012, 0.1692, 0.0855, 0.9615, 23.21],
    "QRGBM": [0.1314, 0.1759, 0.1036, 0.0524, 0.6255, 20.09],
    "WGAN": [0.1331, 0.1789, 0.0979, 0.0495, 0.6052, 19.87],
    "VAE": [0.1244, 0.1677, 0.0880, 0.0445, 0.5482, 17.87],
    "NF": [0.1267, 0.1753, 0.0907, 0.0458, 0.5671, 18.54],
    "DDPM": [0.1281, 0.1815, 0.0981, 0.0486, 0.5985, 19.61],
    "MS-CADM": [0.1191, 0.1645, 0.0873, 0.0441, 0.5380, 18.14],
}


def score_archive(path: Path, zone: int | None = None) -> dict[str, float]:
    archive = np.load(path, allow_pickle=False)
    scenarios, observations = archive["scenarios"], archive["observations"]
    if zone is not None:
        mask = archive["zone"] == zone
        scenarios, observations = scenarios[mask], observations[mask]
    return all_scores(scenarios, observations)


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tables 1-4 and paper deltas")
    parser.add_argument("--root", default="outputs/full_reproduction")
    parser.add_argument("--zone", type=int, default=1)
    args = parser.parse_args()
    root, scenario_root = Path(args.root), Path(args.root) / "scenarios"
    main_records, zone_records = [], []
    for path in sorted(scenario_root.glob("*_test*.npz")):
        main_records.append({"model": path.stem, **score_archive(path)})
        zone_records.append({"model": path.stem, **score_archive(path, args.zone)})
    if main_records:
        table1 = pd.DataFrame(main_records)
        save(table1, root / "tables" / "table1_all_zones.csv")
        save(pd.DataFrame(zone_records), root / "tables" / f"table2_zone{args.zone}.csv")
        columns = ["MAE", "RMSE", "CRPS", "QS", "ES", "VS"]
        comparison = []
        for record in main_records:
            label = record["model"].split("_")[0].upper()
            aliases = {"MSCADM": "MS-CADM", "RAND": "RAND", "QRGBM": "QRGBM"}
            paper_label = aliases.get(label, label)
            if paper_label in PAPER_TABLE1:
                comparison.append({
                    "model": record["model"],
                    **{f"delta_{metric}": record[metric] - PAPER_TABLE1[paper_label][i] for i, metric in enumerate(columns)},
                })
        save(pd.DataFrame(comparison), root / "tables" / "table1_delta_from_paper.csv")
    step_records = []
    for path in sorted((scenario_root / "sampling_steps").glob("*.npz")):
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        step_records.append({
            "steps": metadata["sampling_steps"],
            "seconds": metadata.get("elapsed_seconds"),
            **all_scores(archive["scenarios"], archive["observations"]),
        })
    if step_records:
        save(pd.DataFrame(step_records).sort_values("steps"), root / "tables" / "table3_sampling_steps.csv")
    ablation_records = []
    for path in sorted((scenario_root / "ablations").glob(f"zone{args.zone}_*.npz")):
        ablation_records.append({"variant": path.stem.split("_", 1)[1], **score_archive(path)})
    if ablation_records:
        save(pd.DataFrame(ablation_records), root / "tables" / "table4_ablations.csv")


if __name__ == "__main__":
    main()
