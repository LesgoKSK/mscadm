from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stgf_flow.data import build_stgf_confirmation_gefcom2014
from stgf_flow.graph import fit_spectral_artifacts
from stgf_flow.metrics import evaluate_stgf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DDPM on STGF blocks with empirical atoms"
    )
    parser.add_argument(
        "--config",
        default="repro_configs/ddpm_stgf_confirmation_v2.json",
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(
        (ROOT / args.config).read_text(encoding="utf-8")
    )
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    for outer in outers:
        data = build_stgf_confirmation_gefcom2014(
            ROOT / config["data_dir"],
            outer=int(outer),
            split_path=ROOT / config["split_registry"],
        )
        graph = json.loads(
            (ROOT / config["stgf_confirmation_config"]).read_text(
                encoding="utf-8"
            )
        )["graph"]
        artifacts = fit_spectral_artifacts(
            data.train,
            transform_mode="stgf",
            neighbors=int(graph["neighbors"]),
            correlation_power=float(graph["correlation_power"]),
            epsilon=float(graph["logit_epsilon"]),
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
        with np.load(archive, allow_pickle=False) as stored:
            metrics = evaluate_stgf(
                stored["scenarios"], data.test, artifacts
            )
        destination = archive.with_suffix(".metrics.json")
        destination.write_text(
            json.dumps(
                {
                    "outer": int(outer),
                    "seed": args.seed,
                    "model": "DDPM",
                    "split": "new_frozen_test",
                    "atom_probability_source": (
                        "empirical M=100 ensemble frequency"
                    ),
                    "spectral_evaluation_basis": (
                        "common training-only GFT x DCT basis"
                    ),
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
