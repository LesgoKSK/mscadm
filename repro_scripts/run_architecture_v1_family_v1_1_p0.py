#!/usr/bin/env python3
"""Run the frozen family-v1.1 train-only P0 with v-prediction D0."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.training import canonical_sha256, file_sha256
import repro_scripts.run_architecture_v1_family_v1 as base_runner


CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_1.json"
BASE_CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1.json"
_base_load_config = base_runner._load_config


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    revision = base_runner._read(path)
    if revision.get("schema") != "architecture_v1_family_v1_1":
        raise ValueError("unexpected family-v1.1 schema")
    if revision.get("status") != "frozen_before_family_v1_1_implementation_P0_or_retained_training":
        raise RuntimeError("family-v1.1 was not prospectively frozen")
    sidecar = path.with_name(path.name + ".sha256")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError("family-v1.1 config sidecar is malformed")
    base_runner._verified(path, pieces[0])
    base_path = ROOT / revision["base_family_v1_config"]
    base_runner._verified(base_path, revision["base_family_v1_config_sha256"])
    freeze_path = ROOT / revision["family_v1_no_go_freeze"]
    base_runner._verified(freeze_path, revision["family_v1_no_go_freeze_sha256"])
    freeze = base_runner._read(freeze_path)
    if freeze.get("status") != revision["required_family_v1_status"]:
        raise RuntimeError("family-v1 No-Go freeze does not authorize v1.1")
    base, model_config = _base_load_config(base_path)
    scope = revision["revision_scope"]
    required_false = (
        "train_only_latent_normalization", "SNR_weighting", "predicted_x0_clipping"
    )
    if scope["only_algorithmic_change"] != "epsilon_prediction_to_v_prediction_and_exact_inverse_sampler_algebra":
        raise RuntimeError("family-v1.1 revision scope drifted")
    if any(scope[name] is not False for name in required_false):
        raise RuntimeError("family-v1.1 accidentally combined multiple D0 remedies")
    merged = dict(base)
    merged.update(revision)
    # The base runner consumes these names; values remain exactly frozen in v1.
    merged["D0_diffusion"] = dict(base["D0_diffusion"])
    merged["D0_diffusion"].update(
        {
            "prediction_parameterization": "v",
            "simple_loss": "mean_squared_v_error_over_observed_interior_cells_only",
        }
    )
    return merged, model_config


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(), config_path.resolve(), BASE_CONFIG,
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "family_diffusion_v1_1.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "repro_scripts" / "run_architecture_v1_family_v1.py",
        ROOT / "tests" / "test_architecture_v1_family_diffusion_v1_1.py",
    ]
    records = [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in paths
    ]
    core = {"schema": "architecture_v1_family_v1_1_P0_code_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


_base_sampling_audit = base_runner._sampling_audit


def _strict_sampling_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = _base_sampling_audit(*args, **kwargs)
    # VPredictionJointDDPM.sample_ddim raises before returning if any active
    # interior value decodes to an exact boundary.
    report["strict_interior_contract_enforced_by_sampler"] = True
    report["interior_decoded_boundary_count"] = 0
    report["interior_decoded_boundary_count_max"] = 0
    report["passed"] = bool(report["passed"] and report["interior_decoded_boundary_count"] == 0)
    return report


def main() -> int:
    base_runner.DEFAULT_CONFIG = CONFIG
    base_runner.P0_SCHEMA = "architecture_v1_family_v1_1_P0_result_v1"
    base_runner.P0_FAMILY_SCHEMA = "architecture_v1_family_v1_1_P0_family_v1"
    base_runner.MaskedJointDDPM = VPredictionJointDDPM
    base_runner._load_config = _load_config
    base_runner._code_manifest = _code_manifest
    base_runner._sampling_audit = _strict_sampling_audit
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
