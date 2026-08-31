from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro_scripts import run_stgf_confirmation_v1 as runner


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    config = json.loads(source.read_text(encoding="utf-8"))
    lock_path = (ROOT / config["development_selection_lock"]).resolve()
    if (
        runner.sha256_file(lock_path)
        != config["development_selection_lock_sha256"]
    ):
        raise RuntimeError("development selection lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "stgf_development_selection_common_basis_v3":
        raise RuntimeError("unexpected common-basis selection lock")
    selected = config["locked_selection"]
    expected = {
        "transform_mode": lock["selected_transform_mode"],
        "physical_mean_anchor": lock["selected_physical_mean_anchor"],
        "residual_temperature": lock["selected_residual_temperature"],
    }
    if selected != expected:
        raise RuntimeError("confirmation configuration does not match lock")
    if config.get("test_access_policy") != (
        "selection lock required before any frozen test archive is generated"
    ):
        raise RuntimeError("missing frozen-test access policy")
    return config


if __name__ == "__main__":
    runner.load_config = load_config
    runner.main()
