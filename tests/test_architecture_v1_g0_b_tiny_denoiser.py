"""Synthetic fail-closed tests for the G0-B tiny-denoiser implementation."""

from __future__ import annotations

import numpy as np
import torch

from architecture_v1.family_diffusion import MaskedJointDDPM
from architecture_v1.g0_predictability import build_mode_groups
from architecture_v1.g0b_allocation import reserve_then_water_fill
from architecture_v1.g0b_tiny_denoiser import (
    MULAN_LITE_PARAMETERS,
    PATH_IDS,
    TINY_DENOISER_PARAMETERS,
    ModeProjectorBank,
    TinyDenoisingSystem,
    TinyEMA,
    allocate_torch_schedule,
    module_state_sha256,
    proxy_reserve_then_water_fill,
    tiny_denoising_loss,
)


def _groups():
    return build_mode_groups(
        {"low": (0, 4), "mid": (4, 12), "high": (12, 24)},
        (
            "low_common",
            "low_local",
            "mid_common",
            "mid_local",
            "high_common",
            "high_local",
        ),
    )


def _schedule() -> torch.Tensor:
    return MaskedJointDDPM().alpha_bar.detach().clone()


def test_parameter_counts_zero_output_and_paired_base_initialization() -> None:
    hashes = []
    for path in PATH_IDS:
        system = TinyDenoisingSystem(path, model_seed=3)
        base_count = sum(parameter.numel() for parameter in system.denoiser.parameters())
        total_count = sum(parameter.numel() for parameter in system.parameters())
        assert base_count == TINY_DENOISER_PARAMETERS
        assert total_count == TINY_DENOISER_PARAMETERS + (
            MULAN_LITE_PARAMETERS if path == "MULAN_LITE" else 0
        )
        hashes.append(module_state_sha256(system.denoiser))
        output = system.denoiser(
            torch.randn(2, 10, 24),
            torch.ones(2, 10, 24, dtype=torch.bool),
            torch.randn(2, 47),
            torch.randn(2, 6),
            torch.tensor([0, 249], dtype=torch.long),
        )
        assert torch.equal(output, torch.zeros_like(output))
    assert len(set(hashes)) == 1


def test_mulan_starts_exactly_at_fixed_and_respects_bounds() -> None:
    system = TinyDenoisingSystem("MULAN_LITE", model_seed=3)
    assert system.scheduler is not None
    fixed = torch.tensor([0.2, 0.5, 0.8, 1.2, 2.0, 4.0])
    condition = torch.randn(7, 47)
    initial = system.scheduler(condition, fixed)
    assert torch.equal(initial, fixed[None].expand_as(initial))
    with torch.no_grad():
        system.scheduler.output.weight.fill_(100.0)
        system.scheduler.output.bias.fill_(-100.0)
    changed = system.scheduler(condition, fixed)
    assert torch.all(changed >= fixed[None] / 4.0 - 1e-6)
    assert torch.all(changed <= fixed[None] * 4.0 + 1e-5)


def test_proxy_true_identity_matches_g0b0_reference() -> None:
    generator = np.random.default_rng(7)
    variance = np.exp(generator.normal(size=(5, 6)))
    weights = generator.uniform(0.1, 1.0, size=(5, 6))
    weights /= weights.sum(axis=1, keepdims=True)
    for snr in (1e-7, 0.5, 2e4):
        reference = reserve_then_water_fill(
            variance, weights, snr, eta=0.5
        )
        actual = proxy_reserve_then_water_fill(
            variance, weights, variance, snr, eta=0.5
        )
        assert np.array_equal(actual.information, reference.information)
        assert np.array_equal(actual.alpha_bar, reference.alpha_bar)


def test_torch_paths_conserve_budget_and_mulan_schedule_is_differentiable() -> None:
    alpha = _schedule()
    true = torch.tensor(
        [[0.2, 0.4, 0.8, 1.5, 3.0, 6.0], [0.5, 0.7, 1.0, 2.0, 4.0, 8.0]]
    )
    weight = torch.tensor(
        [[4.0, 36.0, 8.0, 72.0, 12.0, 108.0]]
    ).expand_as(true)
    weight = weight / weight.sum(dim=1, keepdim=True)
    timestep = torch.tensor([0, 249], dtype=torch.long)
    proxy = true.detach().clone().requires_grad_(True)
    for path in PATH_IDS:
        path_proxy = proxy if path not in {"IID", "CW_GROUP"} else None
        schedule = allocate_torch_schedule(
            path,
            true,
            weight,
            timestep,
            alpha,
            proxy_variance=path_proxy,
        )
        baseline_snr = alpha[timestep] / (1.0 - alpha[timestep])
        iid_information = 0.5 * torch.log1p(true * baseline_snr[:, None])
        expected = torch.sum(weight * iid_information, dim=1)
        actual = torch.sum(weight * schedule.information, dim=1)
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-6)
        assert torch.isfinite(schedule.alpha_bar).all()
        assert torch.all(schedule.alpha_bar > 0.0)
        assert torch.all(schedule.alpha_bar < 1.0)
    allocate_torch_schedule(
        "MULAN_LITE",
        true,
        weight,
        torch.tensor([40, 170]),
        alpha,
        proxy_variance=proxy,
    ).alpha_bar.sum().backward()
    assert proxy.grad is not None
    assert torch.isfinite(proxy.grad).all()


def test_projector_partition_cw_inverse_and_all_six_losses() -> None:
    torch.manual_seed(9)
    projectors = ModeProjectorBank(_groups())
    residual = torch.randn(4, 10, 24)
    active = torch.rand(4, 10, 24) > 0.15
    residual = torch.where(active, residual, torch.zeros_like(residual))
    variance = torch.exp(torch.randn(4, 6) * 0.3)
    weight = torch.rand(4, 6) + 0.1
    weight = weight / weight.sum(dim=1, keepdim=True)
    condition = torch.randn(4, 47)
    timestep = torch.tensor([0, 31, 143, 249], dtype=torch.long)
    noise = torch.randn_like(residual)
    fixed = variance.mean(dim=0)
    alpha = _schedule()
    reconstructed = projectors.reconstruct(projectors.project(residual))
    assert torch.allclose(reconstructed, residual, atol=2e-6, rtol=1e-6)
    restored = projectors.unwhiten(
        projectors.whiten(residual, variance), variance
    )
    assert torch.allclose(restored, residual, atol=3e-6, rtol=2e-6)

    base_hashes = []
    permutation = torch.tensor([1, 2, 3, 0])
    for path in PATH_IDS:
        system = TinyDenoisingSystem(path, model_seed=3)
        base_hashes.append(module_state_sha256(system.denoiser))
        sample = tiny_denoising_loss(
            system,
            projectors,
            residual,
            active,
            condition,
            variance,
            weight,
            fixed,
            timestep,
            noise,
            alpha,
            shuffle_proxy=variance[permutation] if path == "PA_SHUFFLE" else None,
        )
        sample.loss.backward()
        assert torch.isfinite(sample.loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in system.parameters()
        )
        ema = TinyEMA(system, decay=0.999)
        ema.update(system)
        state = ema.state_dict()
        restored_ema = TinyEMA(system, decay=0.999)
        restored_ema.load_state_dict(state, system)
        assert restored_ema.num_updates == 1
    assert len(set(base_hashes)) == 1
