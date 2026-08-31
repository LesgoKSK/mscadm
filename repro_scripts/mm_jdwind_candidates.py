from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mm_jdwind.data import build_joint_nested_gefcom2014
from mm_jdwind.experiment import generate_joint_scenarios, save_joint_archive
from mm_jdwind.metrics import evaluate_joint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration-only MM-JDWind ablation candidates"
    )
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_development.json"
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    sampling = config["sampling"]
    for outer in outers:
        data = build_joint_nested_gefcom2014(
            ROOT / config["data_dir"], outer=int(outer)
        )
        for seed in seeds:
            run = output / f"outer{outer}" / "runs" / f"seed{seed}"
            checkpoints = {
                "flow": run / "flow_best.pt",
                "proper": run / "final.pt",
            }
            for stage, checkpoint in checkpoints.items():
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                for mode in sampling["state_modes"]:
                    destination = (
                        output
                        / f"outer{outer}"
                        / "candidates"
                        / f"{stage}_{mode}_seed{seed}_calibration.npz"
                    )
                    metrics_path = destination.with_suffix(".metrics.json")
                    if not destination.exists():
                        arrays, metadata = generate_joint_scenarios(
                            checkpoint,
                            data,
                            split_name="calibration",
                            members=int(sampling["members"]),
                            jump_steps=int(sampling["jump_steps"]),
                            flow_steps=int(sampling["flow_steps"]),
                            state_mode=mode,
                            day_batch=int(sampling["day_batch"]),
                            member_chunk=int(sampling["member_chunk"]),
                            seed=810_000
                            + int(outer) * 10_000
                            + int(seed) * 100
                            + (0 if stage == "flow" else 50),
                            device=args.device,
                        )
                        metadata["candidate_stage"] = stage
                        save_joint_archive(destination, arrays, metadata)
                    if not metrics_path.exists():
                        with np.load(destination, allow_pickle=False) as stored:
                            metrics = evaluate_joint(
                                stored["scenarios"],
                                data.calibration,
                                zero_probability=stored["zero_probability"],
                                one_probability=stored["one_probability"],
                            )
                        metrics_path.write_text(
                            json.dumps(
                                {
                                    "outer": int(outer),
                                    "seed": int(seed),
                                    "stage": stage,
                                    "state_mode": mode,
                                    "split": "calibration_only",
                                    "metrics": metrics,
                                },
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    print(
                        json.dumps(
                            {
                                "candidate": f"{stage}_{mode}",
                                "outer": outer,
                                "seed": seed,
                                "archive": str(destination.resolve()),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
