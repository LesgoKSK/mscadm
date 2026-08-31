from __future__ import annotations

from pathlib import Path

from repro_scripts import run_stgf_confirmation_v1 as runner
from repro_scripts.run_stgf_confirmation_common import load_config


ROOT = Path(__file__).resolve().parents[1]


def universal_run_dir(
    config: dict, outer: int, seed: int, transform_mode: str
) -> Path:
    """Use the one model fitted to the universal 481-day training split."""
    return (
        ROOT
        / config["output_root"]
        / "outer1"
        / "runs"
        / transform_mode
        / f"seed{seed}"
    )


if __name__ == "__main__":
    runner.load_config = load_config
    runner.run_dir = universal_run_dir
    runner.main()
