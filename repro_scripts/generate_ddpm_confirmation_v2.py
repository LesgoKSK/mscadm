from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_jdwind.confirmation_data import build_confirmation_gefcom2014
from repro.sampling import generate_torch_scenarios
from repro_scripts.run_ddpm_confirmation import jointify_generated, run_dir


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DDPM confirmation archive without inline evaluation"
    )
    parser.add_argument(
        "--config", default="repro_configs/ddpm_confirmation_v1.json"
    )
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    data = build_confirmation_gefcom2014(
        ROOT / config["data_dir"],
        outer=args.outer,
        split_path=ROOT / config["split_registry"],
    )
    output = ROOT / config["output_root"] / f"outer{args.outer}" / "scenarios"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"ddpm_seed{args.seed}_test.npz"
    if archive.exists():
        print(json.dumps({"skip": str(archive)}))
        return
    sampling = config["sampling"]
    generated, metadata = generate_torch_scenarios(
        run_dir(config, args.outer, args.seed) / "final.pt",
        data.source,
        split_name="test",
        scenarios=int(sampling["members"]),
        sampling_steps=int(sampling["steps"]),
        sampler=sampling["sampler"],
        eta=float(sampling["eta"]),
        day_batch=int(sampling["day_batch"]),
        device=args.device,
        seed=1_300_000 + args.outer * 10_000 + args.seed * 100,
    )
    joint = jointify_generated(
        generated, data.source.test.day, data.source.test.zone
    )
    metadata.update(
        {
            "joint_comparison_scope": config["comparison_scope"],
            "evidence_scope": config["evidence_label"],
            "confirmation_protocol_sha256": data.protocol["protocol_sha256"],
        }
    )
    np.savez_compressed(
        archive,
        scenarios=joint,
        observations=data.test.target.astype(np.float32),
        day=data.test.day,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({"generated": str(archive)}))


if __name__ == "__main__":
    main()
