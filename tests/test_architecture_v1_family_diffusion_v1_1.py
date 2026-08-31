from __future__ import annotations

import torch

from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM


def test_v_prediction_inverse_is_orthogonal_and_finite_at_schedule_endpoint() -> None:
    diffusion = VPredictionJointDDPM(timesteps=250)
    generator = torch.Generator().manual_seed(17)
    x0 = torch.randn((3, 4, 5), generator=generator)
    epsilon = torch.randn((3, 4, 5), generator=generator)
    timestep = torch.tensor([0, 123, 249], dtype=torch.long)
    a = diffusion.alpha_bar.index_select(0, timestep).reshape(3, 1, 1)
    sqrt_a = torch.sqrt(a)
    sqrt_one = torch.sqrt(1.0 - a)
    xt = sqrt_a * x0 + sqrt_one * epsilon
    v = sqrt_a * epsilon - sqrt_one * x0
    recovered_x0 = sqrt_a * xt - sqrt_one * v
    recovered_epsilon = sqrt_one * xt + sqrt_a * v
    assert torch.isfinite(recovered_x0).all()
    assert torch.isfinite(recovered_epsilon).all()
    assert torch.allclose(recovered_x0, x0, atol=2e-6, rtol=2e-6)
    assert torch.allclose(recovered_epsilon, epsilon, atol=2e-6, rtol=2e-6)


def test_v_prediction_wrapper_has_no_trainable_parameters() -> None:
    diffusion = VPredictionJointDDPM()
    assert sum(parameter.numel() for parameter in diffusion.parameters()) == 0
    assert diffusion.prediction_parameterization == "v"
