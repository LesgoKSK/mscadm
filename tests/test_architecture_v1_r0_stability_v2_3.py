"""Self-contained tests for the R0 stability-v2.3 attribution runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "repro_scripts" / "run_architecture_v1_r0_stability_v2_3_pilot.py"
CONFIG_PATH = ROOT / "repro_configs" / "architecture_v1_r0_stability_v2_3_pilot.json"


def _runner():
    spec = importlib.util.spec_from_file_location("r0_stability_v2_3", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_keeps_forbidden_roles_sealed() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    access = config["role_access"]
    assert access["allowed_target_roles"] == ["train", "validation"]
    assert set(access["forbidden_target_roles"]) == {
        "calibration", "selection", "r_seen", "final"
    }
    assert access["selection_state"] == "sealed"
    assert access["calibration_state"] == "sealed"
    assert config["attribution_decision"]["candidate_D_is_not_frozen_by_this_file"]


def test_arm_seed_contract_is_exact() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    B = config["arms"]["B_shared_ea_independent_training_path"]
    C = config["arms"]["C_shared_ea_common_training_path"]
    assert B["training_seeds"] == C["training_seeds"] == [0, 1, 2]
    assert B["flow_initialization_seed"] == C["flow_initialization_seed"]
    assert B["epoch_shuffle_seed"] != C["epoch_shuffle_seed"]
    assert B["flow_path_seed"] != C["flow_path_seed"]
    assert config["reference_seed0"]["reused_as"] == ["B_seed0", "C_seed0"]


def test_spread_and_gate_semantics() -> None:
    module = _runner()
    metrics = {
        0: {name: 1.0 for name in module.PRIMARY},
        1: {name: 1.001 for name in module.PRIMARY},
        2: {name: 1.002 for name in module.PRIMARY},
    }
    spread = module._spread(metrics)
    assert abs(spread["absolute_spreads"]["level_CRPS"] - 0.002) < 1e-12
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = module._target_checks(spread, config)
    assert checks == {
        "level_CRPS": True,
        "ramp_CRPS": False,
        "normalized_joint_ES": True,
    }


def test_dry_run_is_nonmutating() -> None:
    module = _runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registered = module._resolve(config["output_root"])
    before = {path: path.stat().st_mtime_ns for path in ROOT.rglob("*") if path.is_file()}
    result = module.dry_run(CONFIG_PATH)
    after = {path: path.stat().st_mtime_ns for path in ROOT.rglob("*") if path.is_file()}
    assert result["mode"] == "predictor_only_no_targets_loaded_no_files_created"
    assert result["target_roles_if_executed"] == ["train", "validation"]
    assert before == after


def main() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
