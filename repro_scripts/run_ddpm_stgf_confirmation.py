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


if __name__ == "__main__":
    runner.build_data = build_data
    runner.main()
