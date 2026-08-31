from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_jdwind.confirmation_data import build_confirmation_gefcom2014
from mm_jdwind.metrics import evaluate_joint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved DDPM confirmation archives with empirical atoms"
    )
    parser.add_argument(
        "--config", default="repro_configs/ddpm_confirmation_v1.json"
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    for outer in outers:
        data = build_confirmation_gefcom2014(
            ROOT / config["data_dir"],
            outer=int(outer),
            split_path=ROOT / config["split_registry"],
        )
        archive = (
            ROOT
            / config["output_root"]
            / f"outer{outer}"
            / "scenarios"
            / f"ddpm_seed{args.seed}_test.npz"
        )
        if not archive.exists():
            raise FileNotFoundError(archive)
        destination = archive.with_suffix(".metrics.json")
        with np.load(archive, allow_pickle=False) as stored:
            scenarios = stored["scenarios"]
            zero_probability = (scenarios == 0.0).mean(axis=1)
            one_probability = (scenarios == 1.0).mean(axis=1)
            metrics = evaluate_joint(
                scenarios,
                data.test,
                zero_probability=zero_probability,
                one_probability=one_probability,
            )
        destination.write_text(
            json.dumps(
                {
                    "outer": int(outer),
                    "seed": args.seed,
                    "model": "DDPM",
                    "split": "new_frozen_test",
                    "atom_probability_source": "empirical M=100 ensemble frequency",
                    "comparison_scope": config["comparison_scope"],
                    "protocol_sha256": data.protocol["protocol_sha256"],
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"evaluated": str(destination), "outer": outer}))


if __name__ == "__main__":
    main()
