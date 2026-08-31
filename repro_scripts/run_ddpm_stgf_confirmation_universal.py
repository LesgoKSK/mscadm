from __future__ import annotations

from pathlib import Path

from repro_scripts import run_ddpm_confirmation as runner
from stgf_flow.data import build_stgf_confirmation_gefcom2014


ROOT = Path(__file__).resolve().parents[1]


def build_data(config: dict, outer: int):
    return build_stgf_confirmation_gefcom2014(
        ROOT / config["data_dir"],
        outer=outer,
        split_path=ROOT / config["split_registry"],
    )


def universal_run_dir(config: dict, outer: int, seed: int) -> Path:
    return (
        ROOT
        / config["output_root"]
        / "outer1"
        / "runs"
        / f"seed{seed}"
    )


if __name__ == "__main__":
    runner.build_data = build_data
    runner.run_dir = universal_run_dir
    runner.main()
