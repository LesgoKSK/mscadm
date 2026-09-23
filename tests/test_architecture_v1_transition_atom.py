from __future__ import annotations

import torch

from architecture_v1.transition_atom import (
    TransitionAtomNuisance,
    train_only_atom_contract,
)


def _synthetic_targets() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7201)
    observation = torch.rand((5, 10, 24), generator=generator)
    observation[:, :, 0] = 0.0
    observation[0, 0, 1] = 1.0
    observed = torch.ones_like(observation, dtype=torch.bool)
    observed[1, 2, 7] = False
    observation[~observed] = 0.5
    return observation, observed


def test_train_only_atom_contract_uses_jeffreys_upper_atom_when_rare() -> None:
    observation, observed = _synthetic_targets()
    contract = train_only_atom_contract(observation, observed)
    assert contract["upper_atom_fixed"] is True
    assert contract["one_count"] == 1
    expected = 1.5 / (int(observed.sum()) + 1.0)
    assert abs(float(contract["fixed_one_probability"]) - expected) < 1e-15
    assert float(contract["location_std"]) > 0.0


def test_shared_atom_model_loss_and_allocation_are_finite_and_replayable() -> None:
    observation, observed = _synthetic_targets()
    contract = train_only_atom_contract(observation, observed)
    torch.manual_seed(7202)
    model = TransitionAtomNuisance(
        fixed_one_probability=float(contract["fixed_one_probability"]),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    )
    condition = torch.randn((5, 10, 24, 20), generator=torch.Generator().manual_seed(7203))
    losses = model.loss(condition, observation, observed)
    assert set(losses) >= {
        "loss",
        "atom_nll",
        "interior_location_smooth_l1",
    }
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None for value in gradients)
    assert all(bool(torch.isfinite(value).all()) for value in gradients if value is not None)

    model.eval()
    first_statistics, first = model.allocate(condition[:2], members=11, seed=7204)
    second_statistics, second = model.allocate(condition[:2], members=11, seed=7204)
    assert torch.equal(first_statistics.probabilities, second_statistics.probabilities)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.active_mask, second.active_mask)
    assert first.states.shape == (2, 11, 10, 24)
    assert torch.equal(first.active_mask, first.states == 1)


def test_shared_atom_allocation_does_not_require_observation() -> None:
    observation, observed = _synthetic_targets()
    contract = train_only_atom_contract(observation, observed)
    model = TransitionAtomNuisance(
        fixed_one_probability=float(contract["fixed_one_probability"]),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    )
    condition = torch.zeros((1, 10, 24, 20))
    _statistics, allocation = model.allocate(condition, members=7, seed=7205)
    latent = torch.randn((1, 7, 10, 24), generator=torch.Generator().manual_seed(7206))
    values = model.atom.reconstruct(latent, allocation.states)
    assert bool(((values >= 0.0) & (values <= 1.0)).all())
    assert torch.equal(
        values[allocation.states == 0],
        torch.zeros_like(values[allocation.states == 0]),
    )
    assert torch.equal(
        values[allocation.states == 2],
        torch.ones_like(values[allocation.states == 2]),
    )
