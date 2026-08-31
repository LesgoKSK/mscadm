from __future__ import annotations

import torch

from architecture_v1.atom import (
    INTERIOR_STATE,
    ONE_STATE,
    ZERO_STATE,
    ExactAtomModule,
)
from architecture_v1.model import (
    ParameterMatchedTemporalCell,
    R0JointRectifiedFlow,
    T0MemorylessRectifiedFlow,
)


def _model_kwargs() -> dict:
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
        "atom_shared_priority_weight": 0.7,
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    condition = torch.randn(2, 3, 4, 5, generator=generator)
    observation = torch.rand(2, 3, 4, generator=generator) * 0.8 + 0.1
    observation[:, 0, 0] = 0.0
    observation[:, 1, 1] = 1.0
    states = ExactAtomModule.states_from_observation(observation)
    return condition, observation, states


def test_r0_t0_shapes_losses_and_direct_time_domain_contract() -> None:
    condition, observation, states = _inputs()
    observed_mask = torch.ones_like(states, dtype=torch.bool)
    for model_type, variant in (
        (R0JointRectifiedFlow, "R0"),
        (T0MemorylessRectifiedFlow, "T0"),
    ):
        torch.manual_seed(7)
        model = model_type(**_model_kwargs())
        encoded = model.encode_condition(condition)
        context = model.condition_context(encoded)
        statistics = model.atom(encoded)
        assert encoded.shape == (2, 3, 4, 8)
        assert context.shape == encoded.shape
        assert statistics.logits.shape == (2, 3, 4, 3)
        assert torch.allclose(
            statistics.probabilities.sum(dim=-1),
            torch.ones(2, 3, 4),
            atol=1e-6,
        )

        generator = torch.Generator().manual_seed(101)
        flow_losses = model.flow_loss(
            condition,
            observation,
            states=states,
            observed_mask=observed_mask,
            generator=generator,
        )
        atom_losses = model.atom_loss(
            condition, states=states, observed_mask=observed_mask
        )
        assert torch.isfinite(flow_losses["loss"])
        assert torch.isfinite(atom_losses["loss"])
        assert flow_losses["inactive_velocity_max"].detach().item() == 0.0
        (flow_losses["loss"] + atom_losses["loss"]).backward()
        assert model.model_spec()["variant"] == variant
        assert model.model_spec()["continuous_latent_domain"] == "interior_only"


def test_seeded_atom_allocation_and_sampling_are_exact_and_deterministic() -> None:
    torch.manual_seed(3)
    model = T0MemorylessRectifiedFlow(**_model_kwargs())
    condition, _, _ = _inputs()
    first = model.sample(condition, members=8, steps=2, seed=29)
    second = model.sample(condition, members=8, steps=2, seed=29)
    different = model.sample(condition, members=8, steps=2, seed=30)

    assert first.values.shape == (2, 8, 3, 4)
    assert first.states.shape == first.values.shape
    assert first.active_mask.dtype == torch.bool
    assert torch.equal(first.active_mask, first.states == INTERIOR_STATE)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.interior_latent, second.interior_latent)
    assert torch.equal(first.values, second.values)
    assert not torch.equal(first.interior_latent, different.interior_latent)

    # With probabilities (.25,.50,.25) and M=8, allocation is exactly 2/4/2.
    assert torch.all((first.states == ZERO_STATE).sum(dim=1) == 2)
    assert torch.all((first.states == INTERIOR_STATE).sum(dim=1) == 4)
    assert torch.all((first.states == ONE_STATE).sum(dim=1) == 2)
    assert torch.all(first.values[first.states == ZERO_STATE] == 0.0)
    assert torch.all(first.values[first.states == ONE_STATE] == 1.0)
    assert torch.all(first.interior_latent[~first.active_mask] == 0.0)
    assert torch.all(first.values[first.active_mask] > 0.0)
    assert torch.all(first.values[first.active_mask] < 1.0)
    assert first.per_path_nfe == 3
    assert first.batched_forward_calls == 3
    assert first.velocity_calls == first.batched_forward_calls

    chunked = model.sample(
        condition, members=8, steps=2, seed=29, member_chunk=3
    )
    assert torch.equal(first.values, chunked.values)
    assert chunked.per_path_nfe == 3
    assert chunked.batched_forward_calls == 9

    sixteen_step = model.sample(
        condition[:1], members=1, steps=16, seed=29, member_chunk=1
    )
    assert sixteen_step.per_path_nfe == 31
    assert sixteen_step.batched_forward_calls == 31


def test_external_noise_is_masked_before_flow_and_velocity_is_zero_on_atoms() -> None:
    torch.manual_seed(11)
    model = R0JointRectifiedFlow(**_model_kwargs())
    condition, _, _ = _inputs()
    encoded = model.encode_condition(condition)
    statistics = model.atom(encoded)
    allocation = model.atom.allocate(statistics, members=4, seed=8)
    deliberately_nonzero = torch.full((2, 4, 3, 4), 99.0)
    result = model.sample(
        condition,
        members=4,
        steps=1,
        seed=8,
        allocation=allocation,
        initial_noise=deliberately_nonzero,
    )
    assert torch.all(result.interior_latent[~result.active_mask] == 0.0)

    states = allocation.states[:, 0]
    active = states == INTERIOR_STATE
    value = torch.randn(2, 3, 4)
    time = torch.tensor([0.2, 0.8])
    velocity = model.velocity(
        value,
        time,
        condition,
        states,
        observed_mask=torch.ones_like(states, dtype=torch.bool),
    )
    assert velocity.shape == value.shape
    assert torch.all(velocity[~active] == 0.0)


def test_memoryless_cell_is_parameter_matched_but_disconnects_previous_state() -> None:
    torch.manual_seed(23)
    memoryless = ParameterMatchedTemporalCell(8, use_memory=False)
    recurrent = ParameterMatchedTemporalCell(8, use_memory=True)
    recurrent.load_state_dict(memoryless.state_dict(), strict=True)
    assert memoryless.parameter_signature() == recurrent.parameter_signature()
    assert sum(p.numel() for p in memoryless.parameters()) == sum(
        p.numel() for p in recurrent.parameters()
    )

    input_t = torch.randn(2, 3, 8)
    previous_a = torch.randn(2, 3, 8, requires_grad=True)
    previous_b = torch.randn(2, 3, 8)
    memoryless_a, _ = memoryless(input_t, previous_a)
    memoryless_b, _ = memoryless(input_t, previous_b)
    recurrent_a, _ = recurrent(input_t, previous_a.detach())
    recurrent_b, _ = recurrent(input_t, previous_b)
    assert torch.equal(memoryless_a, memoryless_b)
    assert not torch.allclose(recurrent_a, recurrent_b)

    memoryless_a.square().sum().backward()
    assert previous_a.grad is None
    assert memoryless.recurrent_projection.weight.grad is None

    sequence = torch.randn(2, 3, 4, 8)
    initial_a = torch.randn(2, 3, 8)
    initial_b = torch.randn(2, 3, 8)
    scan_a, _ = memoryless.scan(sequence, initial_a)
    scan_b, _ = memoryless.scan(sequence, initial_b)
    assert torch.equal(scan_a, scan_b)


def test_t0_recurrent_reference_matches_parameters_and_shared_shell_can_freeze() -> None:
    torch.manual_seed(31)
    r0 = R0JointRectifiedFlow(**_model_kwargs())
    t0 = T0MemorylessRectifiedFlow(**_model_kwargs())
    t0.load_shared_from(r0, freeze=True)

    for r0_value, t0_value in zip(
        r0.encoder.state_dict().values(), t0.encoder.state_dict().values()
    ):
        assert torch.equal(r0_value, t0_value)
        assert r0_value.data_ptr() != t0_value.data_ptr()
    assert all(not parameter.requires_grad for parameter in t0.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in t0.atom.parameters())
    assert all(parameter.requires_grad for parameter in t0.flow.parameters())
    assert t0.temporal_cell is not None
    assert all(parameter.requires_grad for parameter in t0.temporal_cell.parameters())

    recurrent = t0.recurrent_cell_reference()
    assert t0.temporal_cell.parameter_signature() == recurrent.parameter_signature()
    assert sum(p.numel() for p in t0.temporal_cell.parameters()) == sum(
        p.numel() for p in recurrent.parameters()
    )
    spec = t0.model_spec()
    assert spec["temporal_mode"] == "matched_memoryless"
    assert spec["feature_recurrent"] is False
    assert spec["source_recurrent"] is False
    assert spec["effective_active_parameters"] < spec["trainable_parameters"]
    assert spec["atom_allocation_semantics"] == "balanced_shared_priority_control"

    t0.train()
    assert not t0.encoder.training
    assert not t0.atom.training
    t0.unfreeze_shared()
    assert all(parameter.requires_grad for parameter in t0.encoder.parameters())
    assert all(parameter.requires_grad for parameter in t0.atom.parameters())
    assert t0.encoder.training
    assert t0.atom.training


def test_raw_missing_mask_excludes_filled_values_from_both_losses() -> None:
    torch.manual_seed(37)
    model = R0JointRectifiedFlow(**_model_kwargs())
    condition, observation, states = _inputs()
    observed_mask = torch.ones_like(states, dtype=torch.bool)
    observed_mask[:, 2, 3] = False

    # Deliberately change both filled target value and its derived state.  With
    # the same flow RNG, neither supervised objective may notice that change.
    changed_observation = observation.clone()
    changed_observation[:, 2, 3] = 0.0
    changed_states = states.clone()
    changed_states[:, 2, 3] = ZERO_STATE

    first_atom = model.atom_loss(
        condition, states=states, observed_mask=observed_mask
    )["loss"]
    second_atom = model.atom_loss(
        condition, states=changed_states, observed_mask=observed_mask
    )["loss"]
    first_flow = model.flow_loss(
        condition,
        observation,
        states=states,
        observed_mask=observed_mask,
        generator=torch.Generator().manual_seed(91),
    )["loss"]
    second_flow = model.flow_loss(
        condition,
        changed_observation,
        states=changed_states,
        observed_mask=observed_mask,
        generator=torch.Generator().manual_seed(91),
    )["loss"]
    assert torch.equal(first_atom, second_atom)
    assert torch.equal(first_flow, second_flow)

    no_active = torch.zeros_like(observed_mask)
    try:
        model.flow_loss(
            condition,
            observation,
            states=states,
            observed_mask=no_active,
            generator=torch.Generator().manual_seed(2),
        )
    except ValueError as error:
        assert "at least one observed interior" in str(error)
    else:
        raise AssertionError("an all-missing flow batch must be rejected")
