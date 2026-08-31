from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mm_jdwind.data import build_joint_nested_gefcom2014
from mm_jdwind.experiment import generate_joint_scenarios, save_joint_archive


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the calibration-locked MM-JDWind candidate to development test"
    )
    parser.add_argument(
        "--config", default="repro_configs/mm_jdwind_development.json"
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    lock_path = output / "selection.lock.json"
    if not lock_path.exists():
        raise RuntimeError("test generation is forbidden before selection.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("selection lock configuration hash mismatch")
    if lock.get("selection_data") != (
        "calibration only; no MM-JDWind test archive accessed"
    ):
        raise RuntimeError("selection lock does not certify calibration-only selection")
    stage = lock["selected_stage"]
    mode = lock["selected_state_mode"]
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    sampling = config["sampling"]
    for outer in outers:
        data = build_joint_nested_gefcom2014(
            ROOT / config["data_dir"], outer=int(outer)
        )
        for seed in seeds:
            run = output / f"outer{outer}" / "runs" / f"seed{seed}"
            checkpoint = run / ("flow_best.pt" if stage == "flow" else "final.pt")
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            destination = (
                output
                / f"outer{outer}"
                / "scenarios"
                / f"locked_{stage}_{mode}_seed{seed}_test.npz"
            )
            if destination.exists():
                print(json.dumps({"skip": str(destination.resolve())}), flush=True)
                continue
            arrays, metadata = generate_joint_scenarios(
                checkpoint,
                data,
                split_name="test",
                members=int(sampling["members"]),
                jump_steps=int(sampling["jump_steps"]),
                flow_steps=int(sampling["flow_steps"]),
                state_mode=mode,
                day_batch=int(sampling["day_batch"]),
                member_chunk=int(sampling["member_chunk"]),
                seed=910_000 + int(outer) * 10_000 + int(seed) * 100,
                device=args.device,
            )
            metadata.update(
                {
                    "selection_lock": str(lock_path.resolve()),
                    "selection_lock_sha256": sha256_file(lock_path),
                    "selected_candidate": lock["selected"],
                }
            )
            save_joint_archive(destination, arrays, metadata)
            print(json.dumps({"generated": str(destination.resolve())}), flush=True)


if __name__ == "__main__":
    main()
