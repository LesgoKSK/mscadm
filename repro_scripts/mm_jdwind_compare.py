from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_jdwind.analysis import (
    load_caa_family,
    metric_record,
    paired_date_bootstrap,
    split_for_archive,
    summarize_metric_records,
    write_json,
)
from mm_jdwind.data import build_joint_nested_gefcom2014


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MM-JDWind with frozen CAA A4")
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_development.json"
    )
    parser.add_argument("--baseline-family", default="A4")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    caa_root = ROOT / "outputs" / "caa_rahc"
    baseline_records: list[dict] = []
    method_records: dict[str, list[dict]] = {
        mode: [] for mode in config["sampling"]["state_modes"]
    }
    paired: dict[str, list[dict]] = {
        mode: [] for mode in config["sampling"]["state_modes"]
    }
    for outer in config["outer_splits"]:
        data = build_joint_nested_gefcom2014(
            ROOT / config["data_dir"], outer=int(outer)
        )
        for seed in config["model_seeds"]:
            baseline = load_caa_family(
                caa_root,
                outer=int(outer),
                seed=int(seed),
                family=args.baseline_family,
            )
            if not np.array_equal(baseline["day"], data.test.day):
                raise RuntimeError("CAA and MM test dates do not match")
            baseline_split = split_for_archive(
                data.test.condition, baseline["observations"], baseline["day"]
            )
            baseline_metrics, baseline_daily = metric_record(
                baseline["scenarios"],
                baseline_split,
                baseline["zero_probability"],
                baseline["one_probability"],
            )
            baseline_records.append(
                {"outer": outer, "seed": seed, "metrics": baseline_metrics}
            )
            for mode in config["sampling"]["state_modes"]:
                archive = (
                    output_root
                    / f"outer{outer}"
                    / "scenarios"
                    / f"{mode}_seed{seed}_test.npz"
                )
                if not archive.exists():
                    raise FileNotFoundError(archive)
                with np.load(archive, allow_pickle=False) as stored:
                    if not np.array_equal(stored["day"], data.test.day):
                        raise RuntimeError("MM archive dates do not match frozen split")
                    method_metrics, method_daily = metric_record(
                        stored["scenarios"],
                        data.test,
                        stored["zero_probability"],
                        stored["one_probability"],
                    )
                method_records[mode].append(
                    {"outer": outer, "seed": seed, "metrics": method_metrics}
                )
                paired[mode].append(
                    {
                        "outer": outer,
                        "seed": seed,
                        "dates": data.test.day,
                        "baseline": baseline_daily,
                        "method": method_daily,
                    }
                )
    summary = {
        "schema": "mm_jdwind_development_comparison_v1",
        "evidence_label": config["evidence_label"],
        "baseline_family": args.baseline_family,
        "baseline": summarize_metric_records(baseline_records),
        "methods": {
            mode: summarize_metric_records(records)
            for mode, records in method_records.items()
        },
        "paired_calendar_day_bootstrap": {
            mode: {
                metric: paired_date_bootstrap(
                    records,
                    metric=metric,
                    higher_is_better=metric == "coverage_90",
                    replicates=args.bootstrap_replicates,
                )
                for metric in ("CRPS", "MAE", "coverage_90", "aggregate_CRPS")
            }
            for mode, records in paired.items()
        },
    }
    destination = output_root / "comparison_summary.json"
    write_json(destination, summary)
    print(json.dumps({"written": str(destination.resolve())}, indent=2))


if __name__ == "__main__":
    main()
