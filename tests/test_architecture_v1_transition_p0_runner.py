from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from architecture_v1.family_diffusion import cosine_betas
from architecture_v1.g0b_tiny_denoiser import (
    TinyEMA,
    module_state_sha256,
    tensor_mapping_sha256,
)
from architecture_v1.transition_object import (
    MaskConditionedOperator,
    fit_operator_scale,
)
from architecture_v1.transition_probe import TransitionDenoisingSystem
from repro_scripts.run_architecture_v1_transition_object_p0 import (
    DEFAULT_CONFIG,
    _atom_semantic_audit,
    _checkpoint_blob,
    _ema_hash,
    _one_update,
    _operator_audit,
    _optimizer_finite,
    _random_bank,
    _restore_checkpoint,
    _semantic_state_sha256,
    dry_run,
)


def _training_contract() -> dict[str, object]:
    return {
        "learning_rate": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "EMA_decay": 0.995,
    }


def _synthetic_level_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7301)
    level = torch.randn((4, 10, 24), generator=generator, dtype=torch.float64)
    active = torch.ones_like(level, dtype=torch.bool)
    # Exercise multiple segments and inactive isolation without removing every
    # long segment needed by the wrong-adjacency mechanism audit.
    active[0, 0, 5:7] = False
    active[1, 3, 12] = False
    active[2, 7, 2:4] = False
    return level, active


def _operators(
    level: torch.Tensor, active: torch.Tensor
) -> dict[str, MaskConditionedOperator]:
    return {
        kind: MaskConditionedOperator(
            kind, scale=fit_operator_scale(level, active, kind=kind)
        )
        for kind in (
            "transition_true",
            "transition_wrong",
            "orthogonal_dct",
        )
    }


def test_target_free_dry_run_validates_frozen_protocol_without_arrays() -> None:
    result = dry_run(Path(DEFAULT_CONFIG))
    assert result["mode"] == "target_free_dry_run"
    assert result["probe_id"] == "TGO_V1"
    assert result["retained_runs"] == 84
    assert result["target_arrays_materialized"] is False
    assert result["weights_written"] is False
    assert len(result["paths"]) == 7


def test_p0_random_bank_is_deterministic_and_seed_sensitive() -> None:
    first, first_hash = _random_bank(
        seed=63003,
        updates=3,
        batch_size=4,
        train_days=19,
        timesteps=250,
    )
    replay, replay_hash = _random_bank(
        seed=63003,
        updates=3,
        batch_size=4,
        train_days=19,
        timesteps=250,
    )
    other, other_hash = _random_bank(
        seed=63004,
        updates=3,
        batch_size=4,
        train_days=19,
        timesteps=250,
    )
    assert first_hash == replay_hash == tensor_mapping_sha256(first)
    assert all(torch.equal(first[key], replay[key]) for key in first)
    assert first_hash != other_hash
    assert any(not torch.equal(first[key], other[key]) for key in first)


def test_operator_audit_covers_inverse_noise_and_full_ddim_grid() -> None:
    level, active = _synthetic_level_and_mask()
    operators = _operators(level, active)
    alpha_bar = torch.cumprod(1.0 - cosine_betas(250), dim=0).numpy()
    audit = _operator_audit(level, active, alpha_bar, operators)
    assert audit["inactive_value_max_influence"] == 0.0
    assert audit["matched_noise_forward_max_abs_error"] <= 1e-10
    assert audit["oracle_full_grid_DDIM_pullback_max_abs_error"] <= 1e-9
    assert audit["wrong_adjacency"]["retained_true_edge_fraction"] <= 0.30
    assert set(audit["per_operator"]) == set(operators)


def test_in_memory_checkpoint_replays_the_next_update_exactly() -> None:
    level_double, active = _synthetic_level_and_mask()
    level = level_double.float()
    operators = _operators(level_double, active)
    generator = torch.Generator(device="cpu").manual_seed(7302)
    tensors = {
        "level": level,
        "active": active,
        "condition": torch.randn((4, 47), generator=generator),
    }
    bank, _hash = _random_bank(
        seed=63003,
        updates=2,
        batch_size=4,
        train_days=4,
        timesteps=250,
    )
    alpha_bar = torch.cumprod(1.0 - cosine_betas(250), dim=0)
    training = _training_contract()
    system = TransitionDenoisingSystem("TRANSITION_TRUE", model_seed=3)
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema = TinyEMA(system, decay=float(training["EMA_decay"]))
    _one_update(
        system,
        optimizer,
        ema,
        tensors,
        bank,
        0,
        torch.device("cpu"),
        alpha_bar,
        operators["transition_true"],
        operators["transition_wrong"],
        operators["orthogonal_dct"],
        gradient_clip=5.0,
    )
    checkpoint = _checkpoint_blob(system, optimizer, ema, next_update=1)
    reference_loss, _ = _one_update(
        system,
        optimizer,
        ema,
        tensors,
        bank,
        1,
        torch.device("cpu"),
        alpha_bar,
        operators["transition_true"],
        operators["transition_wrong"],
        operators["orthogonal_dct"],
        gradient_clip=5.0,
    )
    reference = (
        reference_loss,
        module_state_sha256(system),
        _semantic_state_sha256(optimizer.state_dict()),
        _ema_hash(ema),
    )

    restored, restored_optimizer, restored_ema, next_update = _restore_checkpoint(
        checkpoint,
        "TRANSITION_TRUE",
        3,
        torch.device("cpu"),
        training,
    )
    assert next_update == 1
    replay_loss, _ = _one_update(
        restored,
        restored_optimizer,
        restored_ema,
        tensors,
        bank,
        next_update,
        torch.device("cpu"),
        alpha_bar,
        operators["transition_true"],
        operators["transition_wrong"],
        operators["orthogonal_dct"],
        gradient_clip=5.0,
    )
    replay = (
        replay_loss,
        module_state_sha256(restored),
        _semantic_state_sha256(restored_optimizer.state_dict()),
        _ema_hash(restored_ema),
    )
    assert replay == reference
    assert _optimizer_finite(restored_optimizer)


def test_atom_semantic_audit_allocates_without_held_out_state_input() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7303)
    condition = torch.randn((3, 10, 24, 20), generator=generator)
    observation = torch.rand((3, 10, 24), generator=generator)
    observation[:, :, 0] = 0.0
    observation[0, 0, 1] = 1.0
    observed = torch.ones_like(observation, dtype=torch.bool)
    result = _atom_semantic_audit(
        condition, observation, observed, seed=51000
    )
    assert result["target_state_argument_used_during_allocate"] is False
    assert result["paths_with_exact_common_allocation"] == 7
    assert result["contract"]["upper_atom_fixed"] is True
    assert math.isfinite(float(result["contract"]["location_std"]))
