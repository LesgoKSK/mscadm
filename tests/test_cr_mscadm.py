from __future__ import annotations

import numpy as np
import torch

from cr_mscadm.calibration import CopulaPITCalibrator, count_rank_inversions, pit_values
from cr_mscadm.model import ResidualMSCADM, gaussian_crps, statistics_loss


def small_model(mode: str = "hetero") -> ResidualMSCADM:
    return ResidualMSCADM(
        head_config={
            "condition_dim": 20,
            "model_dim": 16,
            "bottleneck": 8,
            "depth": 1,
            "heads": 2,
        },
        denoiser_config={
            "condition_dim": 20,
            "sequence_length": 24,
            "model_dim": 16,
            "condition_bottleneck": 8,
            "depth": 1,
            "heads": 2,
            "ff_multiplier": 2,
            "learn_variance": True,
        },
        mode=mode,
    )


def test_residual_model_roundtrip_and_shapes() -> None:
    torch.manual_seed(0)
    model = small_model()
    condition = torch.randn(3, 24, 20)
    target = torch.randn(3, 24, 1)
    location, scale = model.predict_statistics(condition)
    residual = model.standardize(target, condition)
    reconstructed = model.reconstruct(residual, condition)
    epsilon, variance = model(residual, torch.tensor([1, 2, 3]), condition)
    assert location.shape == scale.shape == target.shape
    assert torch.all(scale > 0)
    assert torch.allclose(target, reconstructed, atol=1e-5)
    assert epsilon.shape == variance.shape == target.shape


def test_fixed_scale_is_condition_invariant() -> None:
    model = small_model("fixed")
    model.fixed_scale.copy_(torch.linspace(0.2, 1.0, 24)[None, :, None])
    _, first = model.predict_statistics(torch.randn(2, 24, 20))
    _, second = model.predict_statistics(torch.randn(2, 24, 20))
    assert torch.equal(first, second)


def test_gaussian_crps_and_statistics_loss_are_finite() -> None:
    observation = torch.tensor([[[0.0], [1.0]]])
    location = torch.zeros_like(observation, requires_grad=True)
    scale = torch.ones_like(observation, requires_grad=True)
    score = gaussian_crps(observation, location, scale)
    losses = statistics_loss(observation, location, scale, crps_weight=0.5)
    losses["loss"].backward()
    assert torch.all(score >= 0)
    assert torch.isfinite(losses["loss"])
    assert location.grad is not None and scale.grad is not None


def test_calibration_preserves_order_and_improves_undercoverage() -> None:
    rng = np.random.default_rng(7)
    days, members, hours = 300, 50, 24
    center = rng.normal(0.5, 0.08, size=(days, 1, hours))
    scenarios = center + rng.normal(0, 0.015, size=(days, members, hours))
    observations = center[:, 0] + rng.normal(0, 0.06, size=(days, hours))
    calibrator = CopulaPITCalibrator.fit(scenarios, observations, strength=1.0)
    calibrated = calibrator.transform(scenarios)
    assert calibrated.shape == scenarios.shape
    assert count_rank_inversions(scenarios, calibrated) == 0
    raw_pit = pit_values(scenarios, observations)
    calibrated_pit = pit_values(calibrated, observations)
    raw_extreme = np.mean((raw_pit < 0.1) | (raw_pit > 0.9))
    calibrated_extreme = np.mean((calibrated_pit < 0.1) | (calibrated_pit > 0.9))
    assert calibrated_extreme < raw_extreme
    assert np.mean(np.ptp(calibrated, axis=1)) > np.mean(np.ptp(scenarios, axis=1))
