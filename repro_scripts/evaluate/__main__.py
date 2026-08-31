from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro.metrics import all_scores, interval_scores


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved scenario archives")
    parser.add_argument("--scenario-dir", default="outputs/full_reproduction/scenarios")
    parser.add_argument("--output-dir", default="outputs/full_reproduction/tables")
    args = parser.parse_args()
    source, output = Path(args.scenario_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, intervals = [], {}
    for path in sorted(source.glob("*.npz")):
        archive = np.load(path, allow_pickle=False)
        scenarios, observations = archive["scenarios"], archive["observations"]
        records.append({"model": path.stem, **all_scores(scenarios, observations)})
        intervals[path.stem] = interval_scores(scenarios, observations)
    if not records:
        raise FileNotFoundError(f"No .npz scenario files found in {source}")
    frame = pd.DataFrame(records).sort_values("model")
    frame.to_csv(output / "table1_metrics.csv", index=False)
    (output / "table1_metrics.md").write_text(markdown_table(frame), encoding="utf-8")
    (output / "interval_metrics.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
