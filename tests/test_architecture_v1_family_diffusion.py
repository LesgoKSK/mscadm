"""Synthetic, train-only semantic checks for family-v1 diffusion and EMA."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.nn import functional as F

from architecture_v1.atom import (
    AtomAllocation,
    INTERIOR_STATE,
    NUM_STATES,
    ONE_STATE,
    ZERO_STATE,
)
from architecture_v1.family_diffusion import (
    CommonEMA,
    FamilyEMATrainer,
    MaskedJointDDPM,
    cosine_betas,
    ddim_timestep_grid,
)
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import ArchitectureBatch, shared_ea_state_sha256


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert re.search(pattern, str(error)), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def _tiny_model(seed: int = 19) -> T0StableSourceRectifiedFlow:
    torch.manual_seed(seed)
    return T0StableSourceRectifiedFlow(
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
    )


def _batch(*, missing_fill: float = 0.37, batch_size: int = 2) -> ArchitectureBatch:
    condition = torch.linspace(
        -1.0,
        1.0,
        batch_size * 10 * 24 * 20,
        dtype=torch.float32,
    ).reshape(batch_size, 10, 24, 20)
    target = torch.full((batch_size, 10, 24), 0.4, dtype=torch.float32)
    target[:, 0, 0] = 0.0
    target[:, 0, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, 0, 5] = False
    target[:, 0, 5] = missing_fill
    state = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    state[observed & (target == 0.0)] = ZERO_STATE
    state[observed & (target == 1.0)] = ONE_STATE
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.arange(batch_size, dtype=torch.long) + 30_000,
    )


def test_cosine_schedule_and_registered_ddim_grid() -> None:
    betas = cosine_betas(250, offset=0.008, beta_min=1e-8, beta_max=0.999)
    assert betas.shape == (250,)
    assert betas.dtype == torch.float32
    assert bool(((betas >= 1e-8) & (betas <= 0.999)).all())
    assert bool((torch.cumprod(1.0 - betas, dim=0)[1:] < torch.cumprod(1.0 - betas, dim=0)[:-1]).all())
    grid = ddim_timestep_grid(250, 31)
    assert len(grid) == len(torch.unique(grid)) == 31
    assert grid[0] == 0 and grid[-1] == 249


def test_masked_q_sample_matches_closed_form_and_zeroes_inactive() -> None:
    diffusion = MaskedJointDDPM(timesteps=12)
    clean = torch.linspace(-2.0, 2.0, 2 * 10 * 24).reshape(2, 10, 24)
    active = torch.ones_like(clean, dtype=torch.bool)
    active[:, 0, :3] = False
    timestep = torch.tensor([0, 11], dtype=torch.long)
    noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    noisy, masked_noise = diffusion.q_sample(
        clean, timestep, active, noise=noise
    )
    alpha = diffusion.alpha_bar[timestep].reshape(2, 1, 1)
    expected = torch.sqrt(alpha) * torch.where(active, clean, 0.0)
    expected += torch.sqrt(1.0 - alpha) * torch.where(active, noise, 0.0)
    assert torch.equal(masked_noise[~active], torch.zeros_like(masked_noise[~active]))
    assert torch.allclose(noisy, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(noisy[~active], torch.zeros_like(noisy[~active]))


def test_epsilon_loss_ignores_missing_fill_and_uses_one_network_call() -> None:
    model = _tiny_model()
    diffusion = MaskedJointDDPM(timesteps=12)
    first = _batch(missing_fill=0.1)
    second = _batch(missing_fill=0.9)
    timestep = torch.tensor([2, 9], dtype=torch.long)
    noise = torch.linspace(-1.0, 1.0, first.target.numel()).reshape_as(first.target)
    calls = 0

    def count_call(*_args):
        nonlocal calls
        calls += 1

    hook = model.flow.register_forward_hook(count_call)
    first_loss, sample = diffusion.loss(
        model,
        first.condition,
        first.target,
        states=first.state,
        observed_mask=first.observed_mask,
        timestep=timestep,
        noise=noise,
        return_sample=True,
    )
    second_loss = diffusion.loss(
        model,
        second.condition,
        second.target,
        states=second.state,
        observed_mask=second.observed_mask,
        timestep=timestep,
        noise=noise,
    )
    hook.remove()
    assert calls == 2
    assert torch.equal(first_loss["loss"], second_loss["loss"])
    assert torch.equal(
        sample.noisy_latent[~sample.active_mask],
        torch.zeros_like(sample.noisy_latent[~sample.active_mask]),
    )
    assert first_loss["inactive_epsilon_max"] == 0.0


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


def test_ddim_replay_atoms_nfe_and_member_chunk_equivalence() -> None:
    model = _tiny_model()
    diffusion = MaskedJointDDPM(timesteps=12)
    condition = _batch(batch_size=1).condition
    members = 3
    allocation = _forced_allocation(model, condition, members)
    initial = torch.linspace(
        -1.0, 1.0, members * 10 * 24, dtype=torch.float32
    ).reshape(1, members, 10, 24)
    first = diffusion.sample_ddim(
        model,
        condition,
        members=members,
        steps=4,
        seed=80,
        member_chunk=1,
        allocation=allocation,
        initial_noise=initial,
    )
    second = diffusion.sample_ddim(
        model,
        condition,
        members=members,
        steps=4,
        seed=999,
        member_chunk=3,
        allocation=allocation,
        initial_noise=initial,
    )
    replay = diffusion.sample_ddim(
        model,
        condition,
        members=members,
        steps=4,
        seed=80,
        member_chunk=1,
        allocation=allocation,
        initial_noise=initial,
    )
    assert torch.equal(first.values, replay.values)
    assert torch.allclose(first.interior_latent, second.interior_latent, atol=2e-6, rtol=2e-6)
    assert first.per_path_nfe == 4
    assert first.batched_forward_calls == 12
    assert second.batched_forward_calls == 4
    assert first.values[0, 0, 0, 0] == 0.0
    assert first.values[0, 1, 0, 1] == 1.0
    assert torch.equal(
        first.interior_latent[~first.active_mask],
        torch.zeros_like(first.interior_latent[~first.active_mask]),
    )
    # With the zero-initialized epsilon head this also proves x0 was not clipped.
    assert first.interior_latent.abs().max() > initial.abs().max()


def test_common_ema_tracks_transport_only_and_restores_online_exactly() -> None:
    model = _tiny_model()
    from architecture_v1.training import configure_stage

    configure_stage(model, "flow")
    shared_before = shared_ea_state_sha256(model)
    ema = CommonEMA(model, decay=0.9)
    name = ema.parameter_names[0]
    parameter = dict(model.named_parameters())[name]
    initial = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(2.0)
    online = parameter.detach().clone()
    ema.update(model)
    assert torch.allclose(ema.shadow[name], initial + 0.2)
    with ema.average_parameters(model):
        assert torch.equal(parameter, ema.shadow[name])
        assert shared_ea_state_sha256(model) == shared_before
    assert torch.equal(parameter, online)
    assert shared_ea_state_sha256(model) == shared_before
    assert all(not name.startswith(("encoder.", "atom.")) for name in ema.parameter_names)


def test_common_trainer_one_call_per_update_and_symmetric_ema() -> None:
    batch = _batch(batch_size=1)
    for family in ("F0", "D0"):
        model = _tiny_model(seed=29)
        shared_before = shared_ea_state_sha256(model)
        trainer = FamilyEMATrainer(
            model,
            family=family,
            diffusion=MaskedJointDDPM(timesteps=12),
            learning_rate=1e-3,
            ema_decay=0.9,
        )
        calls = 0

        def count_call(*_args):
            nonlocal calls
            calls += 1

        hook = model.flow.register_forward_hook(count_call)
        metrics = trainer.train_step(
            batch, generator=torch.Generator().manual_seed(101)
        )
        hook.remove()
        assert calls == 1
        assert metrics["optimizer_updates"] == 1.0
        assert trainer.ema.num_updates == trainer.optimizer_updates == 1
        assert shared_ea_state_sha256(model) == shared_before
        assert len(diffusion_parameters := list(trainer.diffusion.parameters())) == 0
        del diffusion_parameters


def test_checkpoint_restores_online_ema_optimizer_and_rng(tmp_path: Path) -> None:
    batch = _batch(batch_size=1)
    model = _tiny_model(seed=31)
    trainer = FamilyEMATrainer(
        model,
        family="D0",
        diffusion=MaskedJointDDPM(timesteps=12),
        learning_rate=1e-3,
        ema_decay=0.9,
    )
    trainer.train_step(batch, generator=torch.Generator().manual_seed(404))
    path = tmp_path / "latest_safe.pt"
    identity = {"protocol": "family-v1-test", "seed": 3}
    digest = trainer.save_checkpoint(path, identity=identity)
    assert digest == path.with_name(path.name + ".sha256").read_text().split()[0]
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    restored_model = _tiny_model(seed=999)
    restored = FamilyEMATrainer(
        restored_model,
        family="D0",
        diffusion=MaskedJointDDPM(timesteps=12),
        learning_rate=1e-3,
        ema_decay=0.9,
    )
    restored.load_checkpoint(path, expected_identity=identity)
    assert restored.optimizer_updates == trainer.optimizer_updates == 1
    assert restored.ema.tensor_sha256() == trainer.ema.tensor_sha256()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    with _raises(ValueError, "identity"):
        restored.load_checkpoint(path, expected_identity={"protocol": "wrong"})


def main() -> None:
    import tempfile

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        if "tmp_path" in test.__annotations__:
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
        else:
            test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
