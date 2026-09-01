"""Semantic tests for the family-v1.2 frozen-backbone probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from architecture_v1.atom import (
    AtomAllocation,
    INTERIOR_STATE,
    NUM_STATES,
    ONE_STATE,
    ZERO_STATE,
)
from architecture_v1.family_diffusion import ddim_timestep_grid
from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.family_v1_2_probe import (
    ProbeEMATrainer,
    TemporalContextResidualAdapter,
    TemporalUtilityProbe,
    assign_nwp_dynamicity,
    build_ddim_stage_blocks,
    fit_nwp_dynamicity_registry,
    permutation_bank_sha256,
    validate_permutation_bank,
)
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import ArchitectureBatch, tensor_state_sha256


def _tiny_model(seed: int = 23) -> T0StableSourceRectifiedFlow:
    torch.manual_seed(seed)
    model = T0StableSourceRectifiedFlow(
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
    )
    # Retained D0-v checkpoints have a trained, non-zero output projection.
    # The base constructor intentionally zero-initializes it, which would make
    # d(flow)/d(context) exactly zero and would not represent the probe input.
    with torch.no_grad():
        model.flow.output.weight.fill_(0.05)
    return model


def _batch(batch_size: int = 2) -> ArchitectureBatch:
    condition = torch.linspace(
        -1.0,
        1.0,
        batch_size * 10 * 24 * 20,
        dtype=torch.float32,
    ).reshape(batch_size, 10, 24, 20)
    target = torch.full((batch_size, 10, 24), 0.42, dtype=torch.float32)
    target[:, 0, 0] = 0.0
    target[:, 0, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, 0, 5] = False
    state = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    state[observed & (target == 0.0)] = ZERO_STATE
    state[observed & (target == 1.0)] = ONE_STATE
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.arange(batch_size, dtype=torch.long) + 31_000,
    )


def _forced_allocation(
    model: T0StableSourceRectifiedFlow,
    condition: torch.Tensor,
    members: int,
) -> AtomAllocation:
    model.eval()
    with torch.no_grad():
        statistics = model.atom(model.encode_condition(condition))
    states = torch.full(
        (len(condition), members, model.zones, model.hours),
        INTERIOR_STATE,
        dtype=torch.long,
    )
    states[:, 0, 0, 0] = ZERO_STATE
    states[:, 1, 0, 1] = ONE_STATE
    realized = F.one_hot(states, num_classes=NUM_STATES).float().mean(dim=1)
    return AtomAllocation(
        states=states,
        active_mask=states == INTERIOR_STATE,
        analytic_probabilities=statistics.probabilities,
        realized_probabilities=realized,
    )


def _state_hash(module: torch.nn.Module) -> str:
    return tensor_state_sha256(
        {name: value.detach().cpu() for name, value in module.state_dict().items()}
    )


def test_stage_blocks_exactly_partition_registered_reverse_grid() -> None:
    diffusion = VPredictionJointDDPM(timesteps=12)
    blocks = build_ddim_stage_blocks(
        diffusion, steps=4, block_sizes=(2, 2), labels=("early", "late")
    )
    reverse = tuple(ddim_timestep_grid(12, 4).flip(0).tolist())
    assert tuple(value for block in blocks for value in block.timesteps) == reverse
    assert blocks[0].index == 0 and blocks[1].index == 1
    assert len(blocks[0].logsnr) == len(blocks[0].timesteps) == 2
    assert max(blocks[0].logsnr) < min(blocks[1].logsnr)


def test_adapter_zero_init_and_shuffle_only_change_recurrent_adjacency() -> None:
    torch.manual_seed(7)
    adapter = TemporalContextResidualAdapter(64, hidden_dim=32, hours=24)
    assert adapter.parameter_count() == 11712
    assert adapter.output_is_exactly_zero()
    context = torch.randn((2, 3, 24, 64), generator=torch.Generator().manual_seed(8))
    chronological = adapter(context)
    assert torch.equal(chronological, torch.zeros_like(chronological))
    with torch.no_grad():
        adapter.output_projection.weight.fill_(0.03)
        adapter.output_projection.bias.fill_(0.01)
    order = tuple(range(0, 24, 2)) + tuple(range(1, 24, 2))
    chronological = adapter(context)
    shuffled = adapter(context, hour_order=order)
    assert chronological.shape == shuffled.shape == context.shape
    assert torch.isfinite(chronological).all() and torch.isfinite(shuffled).all()
    assert not torch.equal(chronological, shuffled)


def test_loss_rejects_out_of_block_timestep_and_trains_adapter_only() -> None:
    model = _tiny_model()
    diffusion = VPredictionJointDDPM(timesteps=12)
    stage = build_ddim_stage_blocks(diffusion, steps=4, block_sizes=(2, 2))[0]
    torch.manual_seed(101)
    probe = TemporalUtilityProbe(
        model,
        diffusion,
        stage,
        adapter_hidden_dim=3,
        registered_ddim_steps=4,
    )
    batch = _batch()
    outside = torch.full((len(batch.condition),), stage.timesteps[-1] - 1, dtype=torch.long)
    while int(outside[0]) in stage.timesteps:
        outside -= 1
    with pytest.raises(ValueError, match="outside its registered block"):
        probe.loss(
            batch.condition,
            batch.target,
            states=batch.state,
            observed_mask=batch.observed_mask,
            timestep=outside,
            noise=torch.zeros_like(batch.target),
        )
    backbone_before = probe.backbone_tensor_sha256()
    adapter_before = _state_hash(probe.adapter)
    trainer = ProbeEMATrainer(probe, learning_rate=1e-3, ema_decay=0.9)
    timestep = torch.tensor(
        [stage.timesteps[0], stage.timesteps[1]], dtype=torch.long
    )
    noise = torch.randn(batch.target.shape, generator=torch.Generator().manual_seed(41))
    metrics = trainer.train_step(
        batch,
        hour_order=None,
        timestep=timestep,
        noise=noise,
    )
    assert metrics["output_projection_gradient_norm"] > 0.0
    assert _state_hash(probe.adapter) != adapter_before
    assert probe.backbone_tensor_sha256() == backbone_before
    probe.assert_backbone_unchanged()
    assert all(parameter.grad is None for parameter in probe.backbone.parameters())
    assert {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    } == {id(parameter) for parameter in probe.adapter.parameters()}


def test_zero_initialized_probe_matches_base_v_ddim_and_counts_stage_calls() -> None:
    model = _tiny_model(seed=37)
    diffusion = VPredictionJointDDPM(timesteps=12)
    stage = build_ddim_stage_blocks(diffusion, steps=4, block_sizes=(2, 2))[0]
    torch.manual_seed(77)
    probe = TemporalUtilityProbe(
        model,
        diffusion,
        stage,
        adapter_hidden_dim=3,
        registered_ddim_steps=4,
    )
    condition = _batch(batch_size=1).condition
    members = 3
    allocation = _forced_allocation(model, condition, members)
    initial = torch.linspace(
        -1.0, 1.0, members * 10 * 24, dtype=torch.float32
    ).reshape(1, members, 10, 24)
    baseline = diffusion.sample_ddim(
        model,
        condition,
        members=members,
        steps=4,
        seed=80,
        member_chunk=1,
        allocation=allocation,
        initial_noise=initial,
    )
    sampled = probe.sample_ddim(
        condition,
        members=members,
        steps=4,
        seed=80,
        member_chunk=1,
        allocation=allocation,
        initial_noise=initial,
    )
    assert torch.equal(sampled.values, baseline.values)
    assert torch.equal(sampled.interior_latent, baseline.interior_latent)
    assert torch.equal(
        sampled.atom_statistics.probabilities,
        baseline.atom_statistics.probabilities,
    )
    assert sampled.context_residual_abs_max == 0.0
    assert sampled.intervention_timesteps == stage.timesteps
    assert sampled.context_intervention_calls == len(stage.timesteps) * members
    assert sampled.context_baseline_calls == (4 - len(stage.timesteps)) * members


def test_adapter_checkpoint_resume_restores_only_adapter_state(tmp_path: Path) -> None:
    diffusion = VPredictionJointDDPM(timesteps=12)
    stage = build_ddim_stage_blocks(diffusion, steps=4, block_sizes=(2, 2))[1]
    torch.manual_seed(303)
    first = TemporalUtilityProbe(
        _tiny_model(seed=43),
        diffusion,
        stage,
        adapter_hidden_dim=3,
        registered_ddim_steps=4,
    )
    trainer = ProbeEMATrainer(
        first,
        learning_rate=1e-3,
        ema_decay=0.9,
        backbone_identity={"seed": 3, "checkpoint": "synthetic"},
    )
    batch = _batch()
    timestep = torch.tensor(stage.timesteps, dtype=torch.long)
    noise = torch.randn(batch.target.shape, generator=torch.Generator().manual_seed(72))
    trainer.train_step(batch, hour_order=None, timestep=timestep, noise=noise)
    identity = {"protocol": "family-v1.2-test", "block": stage.index}
    path = tmp_path / "adapter.pt"
    trainer.save_checkpoint(
        path,
        identity=identity,
        runner_state={"next_update_zero_based": 1},
    )

    torch.manual_seed(303)
    second = TemporalUtilityProbe(
        _tiny_model(seed=43),
        VPredictionJointDDPM(timesteps=12),
        stage,
        adapter_hidden_dim=3,
        registered_ddim_steps=4,
    )
    restored = ProbeEMATrainer(
        second,
        learning_rate=1e-3,
        ema_decay=0.9,
        backbone_identity={"seed": 3, "checkpoint": "synthetic"},
    )
    payload = restored.load_checkpoint(path, expected_identity=identity)
    assert payload["runner_state"]["next_update_zero_based"] == 1
    assert restored.optimizer_updates == trainer.optimizer_updates == 1
    assert _state_hash(second.adapter) == _state_hash(first.adapter)
    assert restored.ema.tensor_sha256() == trainer.ema.tensor_sha256()
    assert second.backbone_tensor_sha256() == first.backbone_tensor_sha256()
    second.assert_backbone_unchanged()


def test_train_only_nwp_dynamicity_registry_is_deterministic_and_target_free() -> None:
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(120, 10, 24, 10)).astype(np.float32)
    registry = fit_nwp_dynamicity_registry(raw)
    replay = fit_nwp_dynamicity_registry(raw.copy())
    assert registry == replay
    assert registry["target_power_used"] is False
    assert registry["fit_days"] == 120
    score, labels = assign_nwp_dynamicity(raw, registry)
    assert score.shape == labels.shape == (120,)
    assert set(labels.tolist()) == {"stable", "moderate", "dynamic"}
    assert sum(registry["train_label_counts"].values()) == 120


def test_permutation_bank_rejects_true_adjacency_and_hashes_valid_bank() -> None:
    valid = [
        (
            3, 12, 20, 6, 8, 10, 19, 9, 16, 22, 11, 23,
            1, 15, 5, 2, 7, 18, 14, 0, 4, 13, 21, 17,
        )
    ]
    bank = validate_permutation_bank(valid)
    assert len(bank) == 1
    assert len(permutation_bank_sha256(bank)) == 64
    with pytest.raises(ValueError, match="adjacent"):
        validate_permutation_bank([tuple(range(23, -1, -1))], require_derangement=False)
