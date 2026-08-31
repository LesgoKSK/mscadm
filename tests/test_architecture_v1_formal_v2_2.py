"""Self-contained CPU contracts for the formal-v2.2 engineering revision."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.formal_training import (
    FormalEpochTrainer,
    FormalValidationBank,
    common_random_chunk_samples,
    evaluate_sampling_chunk_gate,
    run_discarded_train_only_stage,
    summarize_preclip_gradient_norms,
)
from architecture_v1.model import R0JointRectifiedFlow
from architecture_v1.training import ArchitectureBatch
from repro_scripts.run_architecture_v1_formal_r0_v2_2 import (
    DEFAULT_CONFIG,
    _load_config,
    dry_run,
)


def _model(seed: int) -> R0JointRectifiedFlow:
    torch.manual_seed(seed)
    return R0JointRectifiedFlow(
        condition_dim=20,
        zones=10,
        hours=24,
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
        atom_location_auxiliary_weight=0.1,
    )


def _batch(days: int = 4, seed: int = 7) -> ArchitectureBatch:
    generator = torch.Generator().manual_seed(seed)
    condition = torch.randn(days, 10, 24, 20, generator=generator)
    target = 0.05 + 0.9 * torch.rand(days, 10, 24, generator=generator)
    target[:, 0, 0] = 0.0
    target[:, 1, 1] = 1.0
    observed = torch.ones(days, 10, 24, dtype=torch.bool)
    observed[:, 2, 2] = False
    target[:, 2, 2] = 0.5
    state = torch.ones(days, 10, 24, dtype=torch.long)
    state[target == 0.0] = 0
    state[target == 1.0] = 2
    state[~observed] = 1
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.arange(20_000, 20_000 + days, dtype=torch.long),
    )


def test_train_only_bundle_blocks_every_nontrain_role() -> None:
    bundle = build_architecture_v1_train_data(config_path=DEFAULT_CONFIG)
    assert bundle.materialized_roles == ("train",)
    assert len(bundle.train) == 267
    access = bundle.manifest["formal_train_only_target_access"]
    assert access["materialized_roles"] == ["train"]
    assert access["validation_bank_constructed"] is False
    assert access["forbidden_target_arrays_materialized"] is False
    for role in ("validation", "calibration", "selection", "r_seen", "final"):
        try:
            bundle.role(role)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"forbidden P0 role was accessible: {role}")


def test_gradient_summary_uses_update_level_ratio_and_fixed_window() -> None:
    records = [
        {
            "epoch_number": epoch,
            "preclip_total_l2_norm": float(1.0 + update / 100.0),
        }
        for epoch in range(1, 41)
        for update in range(17)
    ]
    result = summarize_preclip_gradient_norms(
        records,
        audit_epoch_start=31,
        audit_epoch_end=40,
        gradient_clip=5.0,
        gate={
            "minimum_updates": 170,
            "gradient_clip_fraction_max": 0.25,
            "preclip_gradient_norm_p99_to_median_max": 5.0,
            "preclip_gradient_norm_max_over_all_680_flow_updates": 50.0,
            "require_all_finite": True,
        },
    )
    assert result["passed"] is True
    assert result["distribution"]["count"] == 170
    assert result["all_update_distribution"]["count"] == 680
    assert result["p99_to_median"] is not None


def test_discarded_stage_has_no_validation_contract_or_checkpoint() -> None:
    result = run_discarded_train_only_stage(
        _model(3),
        _batch(),
        data_role="train",
        stage="flow",
        epochs=2,
        batch_days=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=50.0,
        training_seed=0,
        shuffle_seed=11,
        path_seed=12,
        device="cpu",
        audit_epoch_start=1,
        audit_epoch_end=2,
        gate={"minimum_updates": 4, "gradient_clip_fraction_max": 1.0},
    )
    assert result["updates"] == 4
    assert result["checkpoint_written"] is False
    assert result["weights_retained"] is False
    assert result["materialized_target_roles"] == ["train"]
    assert result["gradient_audit"]["passed"] is True


def test_common_random_chunk_gate_separates_exact_and_tolerant_semantics() -> None:
    model = _model(5)
    condition = _batch(days=1).condition
    samples = common_random_chunk_samples(
        model,
        condition,
        members=4,
        steps=2,
        seed=21,
        method="heun",
        primary_member_chunk=2,
        alternate_member_chunk=4,
        value_allclose_atol=5e-7,
        value_allclose_rtol=1e-6,
    )
    result = evaluate_sampling_chunk_gate(
        samples.audit,
        score_deltas={
            "level_CRPS": 0.0,
            "ramp_CRPS": 0.0,
            "normalized_joint_ES": 0.0,
        },
        gate={
            "value_abs_max": 1e-6,
            "value_allclose_atol": 5e-7,
            "value_allclose_rtol": 1e-6,
            "score_abs_delta_max": 1e-7,
        },
    )
    assert result["passed"] is True
    assert samples.audit["state_exact"] is True
    assert samples.audit["atom_boundary_values_exact"] is True
    assert samples.audit["same_seed_primary_replay_bitwise_exact"] is True


def test_formal_stage_persists_update_gradient_audit() -> None:
    train = _batch(days=4, seed=31)
    validation = _batch(days=2, seed=32)
    bank = FormalValidationBank.from_explicit_seeds(
        validation.day_index,
        noise_seeds=(41, 42),
        time_seeds=(51, 52),
        plan_id="formal_v2_2_gradient_test",
        evaluation_batch_days=2,
    )
    with tempfile.TemporaryDirectory() as directory:
        trainer = FormalEpochTrainer(
            _model(33),
            directory,
            resolved_config={"schema": "formal_v2_2_gradient_test"},
            protocol_sha256="a" * 64,
            data_bundle_sha256="b" * 64,
            code_sha256="c" * 64,
            validation_bank=bank,
            training_seed=0,
            device="cpu",
        )
        completion = trainer.fit_stage(
            "flow",
            train,
            validation,
            max_epochs=2,
            batch_days=2,
            learning_rate=1e-3,
            gradient_clip=50.0,
            validate_every_epochs=1,
            early_stopping_patience=4,
            minimum_epochs_before_early_stop=2,
            record_update_preclip_gradient_norms=True,
            gradient_audit_epoch_start=1,
            gradient_audit_epoch_end=2,
            gradient_audit_gate={
                "minimum_updates": 4,
                "gradient_clip_fraction_max": 1.0,
                "preclip_gradient_norm_p99_to_median_max": 100.0,
                "preclip_gradient_norm_max_over_all_flow_updates": 1000.0,
            },
        )
        audit = completion["preclip_gradient_audit"]
        assert audit["passed"] is True
        assert audit["distribution"]["count"] == 4
        assert Path(completion["history"]).is_file()


def test_v2_2_dry_run_is_predictor_only_and_nonmutating() -> None:
    config_path, config = _load_config(DEFAULT_CONFIG)
    output = ROOT / config["output_root"]
    before = (
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
        if output.exists()
        else []
    )
    result = dry_run(config_path, config)
    assert result["target_file_opened"] is False
    assert result["P0_target_roles_if_executed"] == ["train"]
    assert result["selection_state"] == "sealed"
    assert result["calibration_state"] == "sealed"
    assert result["retained_training_implemented"] is False
    after = (
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
        if output.exists()
        else []
    )
    assert after == before


if __name__ == "__main__":
    tests = [
        test_train_only_bundle_blocks_every_nontrain_role,
        test_gradient_summary_uses_update_level_ratio_and_fixed_window,
        test_discarded_stage_has_no_validation_contract_or_checkpoint,
        test_common_random_chunk_gate_separates_exact_and_tolerant_semantics,
        test_formal_stage_persists_update_gradient_audit,
        test_v2_2_dry_run_is_predictor_only_and_nonmutating,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
