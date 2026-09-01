"""Fail-closed tests for the frozen family-v1.2 probe protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.family_v1_2_probe import (
    TemporalContextResidualAdapter,
    permutation_bank_sha256,
    validate_permutation_bank,
    validate_stage_registry,
)
from repro_scripts.run_architecture_v1_family_v1_2_probe import (
    _load_data_provenance_amendment,
    _nested_equal,
    _validate_train_data_provenance_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_2_probe.json"
AMENDMENT = (
    ROOT
    / "repro_configs"
    / "architecture_v1_family_v1_2_data_provenance_amendment.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_and_sidecar_are_frozen() -> None:
    value = _config()
    assert value["schema"] == "architecture_v1_family_v1_2_temporal_utility_probe"
    assert value["status"] == (
        "frozen_before_family_v1_2_implementation_P0_or_retained_adapter_training"
    )
    assert CONFIG.with_name(CONFIG.name + ".sha256").read_text().split() == [
        _sha(CONFIG),
        CONFIG.name,
    ]


def test_lineage_binds_unresolved_family_result_and_three_D0_v_checkpoints() -> None:
    value = _config()
    lineage = value["lineage"]
    for path_key, sha_key in (
        ("base_family_v1_1_config", "base_family_v1_1_config_sha256"),
        ("family_v1_1_training_freeze", "family_v1_1_training_freeze_sha256"),
        ("family_v1_1_comparison", "family_v1_1_comparison_sha256"),
        ("data_protocol_config", "data_protocol_config_sha256"),
    ):
        path = ROOT / lineage[path_key]
        assert path.is_file()
        assert _sha(path) == lineage[sha_key]
    comparison = json.loads(
        (ROOT / lineage["family_v1_1_comparison"]).read_text(encoding="utf-8")
    )
    assert comparison["status"] == lineage["required_family_v1_1_status"]
    assert comparison["selected_family"] is None
    for seed in (3, 4, 5):
        record = value["frozen_D0_v_backbones"][f"seed{seed}"]
        assert _sha(ROOT / record["path"]) == record["sha256"]


def test_data_provenance_amendment_reconstructs_legacy_train_identity() -> None:
    value = _config()
    amendment = _load_data_provenance_amendment(CONFIG)
    assert AMENDMENT.with_name(AMENDMENT.name + ".sha256").read_text().split() == [
        _sha(AMENDMENT),
        AMENDMENT.name,
    ]
    bundle = build_architecture_v1_train_data(
        config_path=ROOT / value["lineage"]["data_protocol_config"]
    )
    assert bundle.materialized_roles == ("train",)
    audit = _validate_train_data_provenance_bridge(bundle, value, amendment)
    assert audit["passed"] is True
    assert audit["changed_model_facing_arrays"] == 0
    assert audit["train_split_array_sha256"] == amendment[
        "legacy_D0_v_lineage"
    ]["train_split_array_sha256"]
    assert audit["projected_legacy_train_only_data_bundle_sha256"] == amendment[
        "legacy_D0_v_lineage"
    ]["train_only_data_bundle_sha256"]


def test_nested_resume_state_comparison_is_device_independent() -> None:
    left = {"state": {0: {"step": torch.tensor(26), "exp_avg": torch.arange(4)}}}
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    right = {
        "state": {
            0: {
                "step": left["state"][0]["step"].to(device),
                "exp_avg": left["state"][0]["exp_avg"].to(device),
            }
        }
    }
    assert _nested_equal(left, right)
    right["state"][0]["exp_avg"][-1] += 1
    assert not _nested_equal(left, right)


def test_stage_registry_is_exact_and_named_as_DDIM_blocks() -> None:
    value = _config()
    contract = value["diffusion_contract"]
    diffusion = VPredictionJointDDPM(
        timesteps=contract["training_timesteps"],
        cosine_offset=contract["cosine_offset"],
        beta_min=contract["beta_clip"][0],
        beta_max=contract["beta_clip"][1],
    )
    blocks = validate_stage_registry(
        diffusion,
        steps=contract["DDIM_steps"],
        records=value["DDIM_stage_blocks"],
    )
    assert len(blocks) == 8
    assert [len(block.timesteps) for block in blocks] == [4] * 7 + [3]
    assert [item for block in blocks for item in block.timesteps] == contract[
        "reverse_grid"
    ]
    assert contract["training_timesteps"] == 250
    assert contract["DDIM_steps"] == 31 and contract["DDIM_eta"] == 0.0


def test_adapter_and_shuffle_contract_are_fixed() -> None:
    value = _config()
    adapter_config = value["adapter"]
    adapter = TemporalContextResidualAdapter(
        adapter_config["context_dimension"],
        hidden_dim=adapter_config["hidden_dimension"],
        hours=adapter_config["hours"],
        residual_scale=adapter_config["residual_scale"],
    )
    assert adapter.parameter_count() == adapter_config["trainable_parameters_expected"]
    assert adapter.output_is_exactly_zero()
    assert adapter_config["learned_gate"] is False
    shuffle = value["shuffle_control"]
    bank = validate_permutation_bank(
        shuffle["permutation_bank"],
        maximum_adjacent_edges=shuffle[
            "true_or_reverse_adjacent_edges_allowed_per_permutation"
        ],
    )
    assert len(bank) == 8
    assert set(shuffle["training_permutation_indices"]).isdisjoint(
        shuffle["same_checkpoint_inference_only_indices"]
    )
    assert len(permutation_bank_sha256(bank)) == 64


def test_matrix_roles_statistics_and_decision_tree_are_preregistered() -> None:
    value = _config()
    training = value["retained_adapter_training"]
    assert len(training["backbone_seeds"]) == 3
    assert len(training["stage_block_indices"]) == 8
    assert training["order_modes"] == ["chronological", "shuffle"]
    assert training["total_runs"] == 48
    assert training["only_adapter_trainable"] is True
    assert training["checkpoint_selection"] == "none"
    assert training["validation_targets_used_for_training_or_checkpoint_selection"] is False
    roles = value["role_access"]
    assert roles["P0_allowed_target_roles"] == ["train"]
    assert roles["adapter_training_allowed_target_roles"] == ["train"]
    assert roles["selection_state"] == roles["calibration_state"] == "sealed"
    evaluation = value["formal_probe_evaluation"]
    assert set(evaluation["primary_contrasts"]) == {
        "chronological_minus_shuffle",
        "chronological_minus_D0_v",
        "shuffle_minus_D0_v",
    }
    assert "max-T" in evaluation["multiple_comparison_control"]
    decisions = value["decision_tree"]
    assert decisions["chronological_equivalent_to_shuffle"].endswith("No_Go")
    assert decisions["supported_stage_by_NWP_interaction"].startswith("family_v1_3")
    assert decisions["selection_access_authorized"] is False
