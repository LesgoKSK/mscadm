from __future__ import annotations

from pathlib import Path

import numpy as np

from mm_jdwind.metrics import evaluate_joint as strict_evaluate_joint
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


def empirical_evaluate_joint(
    scenarios: np.ndarray,
    split,
    *,
    zero_probability: np.ndarray | None = None,
    one_probability: np.ndarray | None = None,
):
    values = np.asarray(scenarios)
    if zero_probability is None:
        zero_probability = (values == 0.0).mean(axis=1)
    if one_probability is None:
        one_probability = (values == 1.0).mean(axis=1)
    return strict_evaluate_joint(
        values,
        split,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )


if __name__ == "__main__":
    runner.build_data = build_data
    runner.run_dir = universal_run_dir
    runner.evaluate_joint = empirical_evaluate_joint
    runner.main()
