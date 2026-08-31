"""Fail-closed checks for the frozen Flow-vs-Joint-DDPM protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_and_sidecar_are_frozen() -> None:
    value = _config()
    assert value["schema"] == "architecture_v1_family_v1"
    assert value["status"] == "frozen_before_family_v1_implementation_smoke_or_training"
    assert CONFIG.with_name(CONFIG.name + ".sha256").read_text().split() == [_sha(CONFIG), CONFIG.name]


def test_lineage_and_roles_are_exact() -> None:
    value = _config()
    lineage = value["lineage"]
    for key in ("v3_2_result", "v3_2_freeze", "shared_EA_checkpoint"):
        path = ROOT / lineage[key]
        assert path.is_file()
        assert _sha(path) == lineage[f"{key}_sha256"]
    assert json.loads((ROOT / lineage["v3_2_result"]).read_text())["status"] == lineage["v3_2_required_status"]
    roles = value["role_access"]
    assert roles["p0_allowed_target_roles"] == ["train"]
    assert roles["formal_allowed_target_roles"] == ["train", "validation"]
    assert roles["selection_state"] == roles["calibration_state"] == "sealed"
    assert roles["selection_access_authorized"] is False


def test_capacity_latent_and_atom_contracts_are_matched() -> None:
    value = _config()
    model = value["common_model"]
    assert model["exact_parameter_keys_shapes_counts_and_initial_tensors_within_seed"] is True
    assert model["diffusion_adds_trainable_parameters"] is False
    assert model["total_parameters_expected"] == 796837
    assert value["common_latent_contract"]["extra_DDPM_standardization"] is False
    assert value["common_latent_contract"]["internal_predicted_x0_clipping"] is False
    assert value["common_latent_contract"]["inactive_latent_after_every_network_or_sampler_step"] == 0.0


def test_primary_compute_and_fresh_seeds_are_matched() -> None:
    value = _config()
    assert value["fresh_seed_protocol"]["model_seeds"] == [3, 4, 5]
    assert value["fresh_seed_protocol"]["within_seed_F0_D0_initial_tensor_sha256_must_match"] is True
    evaluation = value["evaluation"]
    assert evaluation["primary_F0"]["NFE"] == evaluation["primary_D0"]["NFE"] == 31
    assert evaluation["primary_D0"]["eta"] == 0.0
    assert evaluation["members"] == 100
    assert value["D0_diffusion"]["variance"] == "fixed_posterior_not_learned"


def test_ema_applies_symmetrically_and_selection_stays_sealed() -> None:
    value = _config()
    assert value["common_EMA_amendment"]["applies_to"] == ["F0", "D0"]
    assert value["common_EMA_amendment"]["decay"] == 0.999
    assert value["decision_tree"]["selection_access_authorized_by_family_v1_runner"] is False
    assert value["freeze_and_resume"]["selection_and_calibration"] == "remain_sealed"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
