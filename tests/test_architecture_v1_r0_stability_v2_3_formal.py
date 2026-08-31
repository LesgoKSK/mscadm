"""Self-contained tests for the prospective R0 stability formal-D runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "repro_scripts" / "run_architecture_v1_r0_stability_v2_3_formal.py"
CONFIG = ROOT / "repro_configs" / "architecture_v1_r0_stability_v2_3_formal.json"


def _module():
    spec = importlib.util.spec_from_file_location("r0_stability_formal", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_D_is_minimal_and_selection_stays_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["candidate_id"] == "R0-D-shared-EA-common-path"
    assert config["flow_training"]["common_epoch_shuffle_seed"] == 13000
    assert config["flow_training"]["common_flow_path_seed"] == 14000
    assert config["role_access"]["allowed_target_roles"] == ["train", "validation"]
    assert config["role_access"]["selection_state"] == "sealed"
    assert config["decision"]["selection_authorized"] is False


def test_new_sampling_gate_keeps_exact_and_score_checks_strict() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    exact = {
        name: True
        for name in (
            "state_exact", "active_mask_exact", "analytic_probability_exact",
            "realized_probability_exact", "atom_probability_exact",
            "zero_probability_exact", "one_probability_exact",
            "same_seed_primary_replay_bitwise_exact", "atom_latent_strictly_zero",
            "atom_boundary_values_exact",
        )
    }
    pair = {
        "audit": {
            **exact,
            "value_allclose": False,
            "value_absolute_difference": {"maximum": 1.5e-6},
        },
        "exact_metric_checks": {
            "coverage90": True,
            "zero_Brier": True,
            "one_Brier": True,
            "atom_state_Brier": True,
        },
        "proper_score_absolute_deltas": {
            "level_CRPS": 1e-10,
            "ramp_CRPS": 1e-10,
            "normalized_joint_ES": 1e-10,
        },
    }
    gate = module._new_sampling_gate(pair, config)
    assert gate["passed"] is True
    assert gate["per_cell_allclose_diagnostic"] is False
    pair["audit"]["state_exact"] = False
    assert module._new_sampling_gate(pair, config)["passed"] is False


def test_new_sampling_gate_rejects_value_or_score_excess() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    names = (
        "state_exact", "active_mask_exact", "analytic_probability_exact",
        "realized_probability_exact", "atom_probability_exact",
        "zero_probability_exact", "one_probability_exact",
        "same_seed_primary_replay_bitwise_exact", "atom_latent_strictly_zero",
        "atom_boundary_values_exact",
    )
    pair = {
        "audit": {
            **{name: True for name in names},
            "value_allclose": True,
            "value_absolute_difference": {"maximum": 2.1e-6},
        },
        "exact_metric_checks": {name: True for name in ("coverage90", "zero_Brier", "one_Brier", "atom_state_Brier")},
        "proper_score_absolute_deltas": {"level_CRPS": 0.0, "ramp_CRPS": 0.0, "normalized_joint_ES": 0.0},
    }
    assert module._new_sampling_gate(pair, config)["passed"] is False
    pair["audit"]["value_absolute_difference"]["maximum"] = 1e-6
    pair["proper_score_absolute_deltas"]["ramp_CRPS"] = 2e-7
    assert module._new_sampling_gate(pair, config)["passed"] is False


def test_dry_run_is_predictor_only() -> None:
    module = _module()
    output = ROOT / "outputs" / "architecture_v1_r0_stability_v2_3_formal"
    before = {
        path.relative_to(output).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    } if output.exists() else {}
    result = module.dry_run(CONFIG)
    assert result["mode"] == "predictor_only_no_targets_loaded_no_files_created"
    assert result["target_roles_if_executed"] == ["train", "validation"]
    after = {
        path.relative_to(output).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    } if output.exists() else {}
    assert after == before


def main() -> int:
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
            print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
