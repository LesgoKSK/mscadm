from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ps_dfsc.data import build_ps_dfsc_outer_data
from ps_dfsc.mapping import all_cyclic_mappings, fit_wind_farm_mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one training-only PS-DFSC wind-farm mapping"
    )
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument(
        "--splits", default="repro_configs/ps_dfsc_splits.json"
    )
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = build_ps_dfsc_outer_data(
        args.data_dir, outer=args.outer, split_registry=args.splits
    )
    mapping = fit_wind_farm_mapping(data.base_generator.train.target)
    value = {
        "schema": "ps_dfsc_wind_farm_mapping_v1",
        "outer": args.outer,
        "fit_role": "base_generator_train_only",
        "groups": [list(group) for group in mapping.groups],
        "wind_buses": list(mapping.wind_buses),
        "zone_capacity_mw": mapping.zone_capacity_mw,
        "cyclic_sensitivity_wind_buses": [
            list(item.wind_buses) for item in all_cyclic_mappings(mapping)
        ],
        "split_registry": str(Path(args.splits).resolve()),
    }
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    value["mapping_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "outer": args.outer,
                "groups": value["groups"],
                "mapping_sha256": value["mapping_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
