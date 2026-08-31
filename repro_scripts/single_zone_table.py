from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from repro.metrics import all_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper Table 2 from separately trained single-zone models")
    parser.add_argument("--root", default="outputs/full_reproduction")
    parser.add_argument("--zone", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    for path in sorted((root / "scenarios" / "single_zone").glob(f"zone{args.zone}_*.npz")):
        archive = np.load(path, allow_pickle=False)
        records.append({"model": path.stem.split("_", 1)[1], **all_scores(archive["scenarios"], archive["observations"])})
    if not records:
        raise FileNotFoundError("No independently trained single-zone scenario archives")
    frame = pd.DataFrame(records)
    destination = root / "tables" / f"table2_zone{args.zone}_retrained.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
