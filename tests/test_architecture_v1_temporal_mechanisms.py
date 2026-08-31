"""Self-contained tests for the matched T0/T1 temporal mechanism family."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from architecture_v1.formal_evaluation import lagged_increment_variogram_score
from architecture_v1.formal_training import FormalValidationBank
from architecture_v1.mechanism_evaluation import chronological_vs_control_gate
from architecture_v1.model import (
    T0MemorylessRectifiedFlow,
    T1FeatureRectifiedFlow,
    T1ShuffleRectifiedFlow,
    T1SourceRectifiedFlow,
)
from architecture_v1.training import ArchitectureBatch, configure_stage, tensor_state_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_0.json"
RUNNER = ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanism_smoke.py"
FORMAL_RUNNER = ROOT / "repro_scripts" / "run_architecture_v1_temporal_mechanisms_v3_0.py"
SMOKE_RESULT = ROOT / "outputs" / "architecture_v1_temporal_mechanism_smoke_v3_0" / "SMOKE_RESULT.json"


def _kwargs() -> dict:
    return {
        "condition_dim": 5,
        "zones": 3,
        "hours": 4,
        "encoder_dim": 8,
        "encoder_depth": 1,
        "flow_dim": 8,
        "flow_depth": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "dropout": 0.0,
        "atom_hidden_dim": 8,
        "atom_initial_probabilities": (0.25, 0.50, 0.25),
    }


def _models() -> list:
    kwargs = _kwargs()
    torch.manual_seed(73)
    t0 = T0MemorylessRectifiedFlow(**kwargs)
    initial = t0.state_dict()
    models = [
        t0,
        T1FeatureRectifiedFlow(**kwargs),
        T1SourceRectifiedFlow(**kwargs),
        T1ShuffleRectifiedFlow(
            temporal_hour_order=(2, 0, 3, 1),
            shuffle_target="feature",
            **kwargs,
        ),
        T1ShuffleRectifiedFlow(
            temporal_hour_order=(2, 0, 3, 1),
            shuffle_target="source",
            **kwargs,
        ),
    ]
    for model in models[1:]:
        model.load_state_dict(initial, strict=True)
    return models


def test_config_is_fail_closed_and_shuffle_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_any_retained_T0_T1_training"
    assert config["role_access"]["allowed_target_roles"] == ["train", "validation"]
    assert config["role_access"]["selection_state"] == "sealed"
    assert config["role_access"]["calibration_state"] == "sealed"
    assert config["go_no_go"]["decision"]["selection_access_authorized"] is False
    order = config["shuffle_control"]["hour_order_zero_based"]
    assert sorted(order) == list(range(24))
    assert order != list(range(24))


def test_smoke_dry_run_is_nonmutating_and_persisted_result_is_verified() -> None:
    spec = importlib.util.spec_from_file_location("temporal_smoke", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = SMOKE_RESULT.parent
    before = {
        path.relative_to(output).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    result = module.dry_run(CONFIG)
    after = {
        path.relative_to(output).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert result["real_target_roles_accessed"] == []
    persisted = json.loads(SMOKE_RESULT.read_text(encoding="utf-8"))
    assert persisted["status"] == "SMOKE_PASS"
    assert all(persisted["checks"].values())
    digest = hashlib.sha256(SMOKE_RESULT.read_bytes()).hexdigest()
    assert SMOKE_RESULT.with_name(SMOKE_RESULT.name + ".sha256").read_text().split()[0] == digest


def test_formal_runner_dry_run_does_not_create_output_or_open_roles() -> None:
    spec = importlib.util.spec_from_file_location("temporal_formal", FORMAL_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = ROOT / "outputs" / "architecture_v1_temporal_mechanisms_v3_0"
    before = output.exists()
    result = module.dry_run(CONFIG)
    assert result["mode"] == "predictor_only_no_targets_loaded_no_files_created"
    assert result["target_roles_if_executed"] == ["train", "validation"]
    assert result["selection_state"] == "sealed"
    assert result["calibration_state"] == "sealed"
    assert output.exists() is before


def test_all_mechanism_candidates_have_identical_parameter_topology() -> None:
    models = _models()
    reference_names = list(models[0].state_dict())
    reference_shapes = [tuple(value.shape) for value in models[0].state_dict().values()]
    reference_count = models[0].parameter_count()
    initial_hash = tensor_state_sha256(models[0].state_dict())
    for model in models:
        assert list(model.state_dict()) == reference_names
        assert [tuple(value.shape) for value in model.state_dict().values()] == reference_shapes
        assert model.parameter_count() == reference_count
        assert tensor_state_sha256(model.state_dict()) == initial_hash
    assert models[0].model_spec()["feature_recurrent"] is False
    assert models[0].model_spec()["source_recurrent"] is False
    assert models[1].model_spec()["feature_recurrent"] is True
    assert models[2].model_spec()["source_recurrent"] is True


def test_feature_recurrence_reads_history_but_T0_does_not() -> None:
    t0, feature, *_ = _models()
    first = torch.randn(2, 3, 4, 8)
    changed = first.clone()
    changed[:, :, 0] += 5.0
    t0_a, _ = t0.feature_cell.scan(first)
    t0_b, _ = t0.feature_cell.scan(changed)
    feature_a, _ = feature.feature_cell.scan(first)
    feature_b, _ = feature.feature_cell.scan(changed)
    assert torch.equal(t0_a[:, :, 1:], t0_b[:, :, 1:])
    assert not torch.allclose(feature_a[:, :, 1], feature_b[:, :, 1])


def test_source_recurrence_is_interior_only_and_reads_previous_noise() -> None:
    t0, _, source, *_ = _models()
    # Activate the otherwise zero-initialized common residual head so this unit
    # test can observe the already-wired recurrent edge before training.
    with torch.no_grad():
        t0.source_adapter.output_projection.weight.zero_()
        source.source_adapter.output_projection.weight.zero_()
        t0.source_adapter.output_projection.weight[0, 0] = 0.3
        source.source_adapter.output_projection.weight[0, 0] = 0.3
    encoded = torch.randn(2, 3, 4, 8)
    active = torch.ones(2, 3, 4, dtype=torch.bool)
    active[:, 1, 2] = False
    first = torch.randn(2, 3, 4)
    changed = first.clone()
    changed[:, :, 0] += 4.0
    t0_a = t0.prepare_source_noise(first, encoded, active)
    t0_b = t0.prepare_source_noise(changed, encoded, active)
    source_a = source.prepare_source_noise(first, encoded, active)
    source_b = source.prepare_source_noise(changed, encoded, active)
    assert torch.equal(t0_a[:, :, 1], t0_b[:, :, 1])
    assert not torch.allclose(source_a[:, :, 1], source_b[:, :, 1])
    assert torch.all(source_a[~active] == 0.0)
    assert torch.all(source_b[~active] == 0.0)


def test_shuffle_breaks_recurrent_adjacency_and_restores_hour_layout() -> None:
    _, feature, source, feature_shuffle, source_shuffle = _models()
    for natural, shuffled in (
        (feature, feature_shuffle),
        (source, source_shuffle),
    ):
        shuffled.load_state_dict(natural.state_dict(), strict=True)
    encoded = torch.randn(2, 3, 4, 8)
    feature_natural = feature.condition_context(encoded)
    feature_negative = feature_shuffle.condition_context(encoded)
    assert feature_natural.shape == feature_negative.shape == encoded.shape
    assert not torch.allclose(feature_natural, feature_negative)

    with torch.no_grad():
        source.source_adapter.output_projection.weight.zero_()
        source_shuffle.source_adapter.output_projection.weight.zero_()
        source.source_adapter.output_projection.weight[0, 0] = 0.3
        source_shuffle.source_adapter.output_projection.weight[0, 0] = 0.3
    noise = torch.randn(2, 3, 4)
    active = torch.ones_like(noise, dtype=torch.bool)
    source_natural = source.prepare_source_noise(noise, encoded, active)
    source_negative = source_shuffle.prepare_source_noise(noise, encoded, active)
    assert source_natural.shape == source_negative.shape == noise.shape
    assert not torch.allclose(source_natural, source_negative)


def test_all_candidates_train_and_sample_with_exact_atom_semantics() -> None:
    models = _models()
    generator = torch.Generator().manual_seed(101)
    condition = torch.randn(2, 3, 4, 5, generator=generator)
    target = torch.rand(2, 3, 4, generator=generator) * 0.8 + 0.1
    target[:, 0, 0] = 0.0
    target[:, 1, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    for model in models:
        loss = model.flow_loss(
            condition,
            target,
            observed_mask=observed,
            generator=torch.Generator().manual_seed(303),
        )["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        scenario = model.sample(
            condition,
            members=4,
            steps=2,
            seed=404,
            member_chunk=2,
        )
        assert scenario.values.shape == (2, 4, 3, 4)
        assert torch.all(scenario.interior_latent[~scenario.active_mask] == 0.0)
        assert torch.all(scenario.values[scenario.states == 0] == 0.0)
        assert torch.all(scenario.values[scenario.states == 2] == 1.0)
        assert scenario.per_path_nfe == 3


def test_lagged_increment_variogram_is_masked_and_zero_for_perfect_paths() -> None:
    truth = torch.linspace(0.1, 0.9, 24).repeat(2, 10, 1).numpy()
    scenarios = truth[:, None].repeat(4, axis=1)
    mask = torch.ones(2, 10, 24, dtype=torch.bool).numpy()
    perfect = lagged_increment_variogram_score(scenarios, truth, mask)
    assert perfect.shape == (2,)
    assert (perfect == 0.0).all()
    changed_truth = truth.copy()
    changed_truth[:, 0, 5] = 100.0
    mask[:, 0, 5] = False
    changed = lagged_increment_variogram_score(scenarios, changed_truth, mask)
    assert (changed == 0.0).all()


def test_temporal_gate_requires_practical_proper_score_and_shuffle_effects() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    n = 50
    t0 = {
        "ramp_CRPS": torch.full((n,), 0.052).numpy(),
        "lagged_increment_variogram_score": torch.full((n,), 0.020).numpy(),
        "level_CRPS": torch.full((n,), 0.080).numpy(),
        "normalized_joint_ES": torch.full((n,), 0.110).numpy(),
        "coverage90": torch.full((n,), 0.86).numpy(),
        "width90": torch.full((n,), 0.38).numpy(),
    }
    chronological = {key: value.copy() for key, value in t0.items()}
    chronological["ramp_CRPS"] -= 0.0012
    chronological["lagged_increment_variogram_score"] *= 0.90
    shuffled = {key: value.copy() for key, value in chronological.items()}
    shuffled["ramp_CRPS"] += 0.0007
    shuffled["lagged_increment_variogram_score"] += 0.001
    gate = chronological_vs_control_gate(
        t0,
        chronological,
        shuffled,
        rules=config["go_no_go"],
        repetitions=200,
        seed=1,
    )
    assert gate["passed"] is True
    shuffled["ramp_CRPS"] = chronological["ramp_CRPS"].copy()
    assert chronological_vs_control_gate(
        t0,
        chronological,
        shuffled,
        rules=config["go_no_go"],
        repetitions=200,
        seed=1,
    )["passed"] is False


def test_fixed_validation_bank_uses_candidate_source_transformation() -> None:
    kwargs = {
        "condition_dim": 20,
        "zones": 10,
        "hours": 24,
        "encoder_dim": 8,
        "encoder_depth": 1,
        "flow_dim": 8,
        "flow_depth": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "atom_hidden_dim": 8,
    }
    torch.manual_seed(808)
    t0 = T0MemorylessRectifiedFlow(**kwargs)
    source = T1SourceRectifiedFlow(**kwargs)
    source.load_state_dict(t0.state_dict(), strict=True)
    with torch.no_grad():
        for model in (t0, source):
            model.source_adapter.output_projection.weight.zero_()
            model.source_adapter.output_projection.weight[0, 0] = 0.3
    generator = torch.Generator().manual_seed(809)
    condition = torch.randn(2, 10, 24, 20, generator=generator)
    target = torch.rand(2, 10, 24, generator=generator) * 0.8 + 0.1
    observed = torch.ones_like(target, dtype=torch.bool)
    batch = ArchitectureBatch(
        condition=condition,
        target=target,
        state=torch.ones_like(target, dtype=torch.long),
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.tensor([0, 1], dtype=torch.long),
    )
    bank = FormalValidationBank.from_explicit_seeds(
        [0, 1],
        noise_seeds=[810],
        time_seeds=[811],
        plan_id="temporal-source-validation-test",
        evaluation_batch_days=2,
    )
    t0_loss = bank.evaluate(t0, batch, stage="flow", device="cpu")["loss"]
    source_loss = bank.evaluate(source, batch, stage="flow", device="cpu")["loss"]
    assert t0_loss != source_loss


def test_formal_flow_stage_owns_both_temporal_cells_and_source_adapter() -> None:
    for model in _models():
        selected = configure_stage(model, "flow")
        selected_ids = {id(parameter) for parameter in selected}
        expected_modules = (model.flow, model.feature_cell, model.source_cell, model.source_adapter)
        expected_ids = {
            id(parameter)
            for module in expected_modules
            for parameter in module.parameters()
        }
        assert selected_ids == expected_ids
        assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
        assert all(not parameter.requires_grad for parameter in model.atom.parameters())
        assert all(parameter.requires_grad for parameter in selected)


def main() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
