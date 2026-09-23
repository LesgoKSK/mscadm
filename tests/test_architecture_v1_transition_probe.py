from __future__ import annotations

import torch

from architecture_v1.family_diffusion import cosine_betas
from architecture_v1.g0b_tiny_denoiser import (
    TINY_DENOISER_PARAMETERS,
    module_state_sha256,
)
from architecture_v1.transition_object import (
    MaskConditionedOperator,
    fit_operator_scale,
)
from architecture_v1.transition_probe import (
    PATH_IDS,
    TransitionDenoisingSystem,
    sample_transition_ddim,
    transition_denoising_loss,
)


def _schedule() -> torch.Tensor:
    return torch.cumprod(1.0 - cosine_betas(250), dim=0)


def _batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7101)
    clean = torch.randn((4, 10, 24), generator=generator)
    active = torch.rand((4, 10, 24), generator=generator) > 0.12
    condition = torch.randn((4, 47), generator=generator)
    timestep = torch.tensor([0, 49, 127, 249], dtype=torch.long)
    noise = torch.randn((4, 10, 24), generator=generator)
    return clean, active, condition, timestep, noise


def _operators(
    clean: torch.Tensor, active: torch.Tensor
) -> tuple[MaskConditionedOperator, MaskConditionedOperator, MaskConditionedOperator]:
    true_scale = fit_operator_scale(clean.double(), active, kind="transition_true")
    wrong_scale = fit_operator_scale(clean.double(), active, kind="transition_wrong")
    dct_scale = fit_operator_scale(clean.double(), active, kind="orthogonal_dct")
    return (
        MaskConditionedOperator("transition_true", scale=true_scale),
        MaskConditionedOperator("transition_wrong", scale=wrong_scale),
        MaskConditionedOperator("orthogonal_dct", scale=dct_scale),
    )


def test_all_paths_share_exact_initial_parameter_state() -> None:
    systems = [
        TransitionDenoisingSystem(path_id, model_seed=3) for path_id in PATH_IDS
    ]
    hashes = {module_state_sha256(system) for system in systems}
    assert len(hashes) == 1
    assert all(
        sum(parameter.numel() for parameter in system.parameters())
        == TINY_DENOISER_PARAMETERS
        for system in systems
    )


def test_all_seven_losses_are_finite_masked_and_one_forward_call() -> None:
    clean, active, condition, timestep, noise = _batch()
    true_operator, wrong_operator, dct_operator = _operators(clean, active)
    schedule = _schedule()
    losses: dict[str, torch.Tensor] = {}
    for path_id in PATH_IDS:
        system = TransitionDenoisingSystem(path_id, model_seed=3)
        calls = 0

        def count_call(_module: torch.nn.Module, _args: tuple[object, ...], _out: object) -> None:
            nonlocal calls
            calls += 1

        handle = system.denoiser.register_forward_hook(count_call)
        try:
            sample = transition_denoising_loss(
                system,
                clean,
                active,
                condition,
                timestep,
                noise,
                schedule,
                true_operator=true_operator,
                wrong_operator=wrong_operator,
                dct_operator=dct_operator,
            )
        finally:
            handle.remove()
        assert calls == 1
        assert bool(torch.isfinite(sample.loss))
        assert sample.prediction_level.shape == clean.shape
        assert sample.prediction_native.shape == clean.shape
        assert sample.noisy_native.shape == clean.shape
        assert torch.equal(
            sample.prediction_level[~active],
            torch.zeros_like(sample.prediction_level[~active]),
        )
        losses[path_id] = sample.loss.detach()

    # All systems are zero-output initialized.  Train-only RMS matching makes
    # every native clean-target energy identical at update zero.
    reference = losses["LEVEL_IID"]
    for path_id, value in losses.items():
        assert torch.allclose(value, reference, atol=2e-6, rtol=2e-6), (
            path_id,
            float(value),
            float(reference),
        )


def test_every_path_supports_a_finite_optimizer_update() -> None:
    clean, active, condition, timestep, noise = _batch()
    true_operator, wrong_operator, dct_operator = _operators(clean, active)
    schedule = _schedule()
    for path_id in PATH_IDS:
        system = TransitionDenoisingSystem(path_id, model_seed=4)
        optimizer = torch.optim.AdamW(
            system.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8
        )
        optimizer.zero_grad(set_to_none=True)
        sample = transition_denoising_loss(
            system,
            clean,
            active,
            condition,
            timestep,
            noise,
            schedule,
            true_operator=true_operator,
            wrong_operator=wrong_operator,
            dct_operator=dct_operator,
        )
        sample.loss.backward()
        gradients = [
            parameter.grad
            for parameter in system.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        torch.nn.utils.clip_grad_norm_(system.parameters(), 5.0)
        optimizer.step()
        assert all(
            bool(torch.isfinite(parameter).all()) for parameter in system.parameters()
        )


def test_all_paths_sample_replay_exactly_and_preserve_inactive_zero() -> None:
    clean, active, condition, _timestep, noise = _batch()
    true_operator, wrong_operator, dct_operator = _operators(clean, active)
    schedule = _schedule()
    for path_id in PATH_IDS:
        system = TransitionDenoisingSystem(path_id, model_seed=5)
        with torch.no_grad():
            system.denoiser.output.bias.fill_(0.025)
        system.train()
        first = sample_transition_ddim(
            system,
            condition,
            active,
            noise,
            schedule,
            true_operator=true_operator,
            wrong_operator=wrong_operator,
            dct_operator=dct_operator,
            steps=5,
        )
        second = sample_transition_ddim(
            system,
            condition,
            active,
            noise,
            schedule,
            true_operator=true_operator,
            wrong_operator=wrong_operator,
            dct_operator=dct_operator,
            steps=5,
        )
        assert system.training
        assert first.path_id == path_id
        assert first.per_path_nfe == 5
        assert first.level.shape == clean.shape
        assert torch.equal(first.level, second.level)
        assert torch.equal(first.native, second.native)
        assert bool(torch.isfinite(first.level).all())
        assert torch.equal(
            first.level[~active], torch.zeros_like(first.level[~active])
        )


def test_different_model_seeds_change_initial_state() -> None:
    first = TransitionDenoisingSystem("LEVEL_IID", model_seed=3)
    second = TransitionDenoisingSystem("LEVEL_IID", model_seed=4)
    assert module_state_sha256(first) != module_state_sha256(second)
