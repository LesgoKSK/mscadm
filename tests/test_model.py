import torch

from mscadm.diffusion import GaussianDiffusion
from mscadm.model import MSCADM


def tiny_model() -> MSCADM:
    return MSCADM(
        condition_dim=20,
        sequence_length=24,
        patch_size=1,
        model_dim=32,
        condition_bottleneck=16,
        depth=1,
        heads=2,
        ff_multiplier=2,
        learn_variance=True,
    )


def test_model_shapes_and_loss() -> None:
    model = tiny_model()
    diffusion = GaussianDiffusion(timesteps=8, learn_variance=True)
    clean = torch.rand(2, 24, 1) * 2 - 1
    condition = torch.randn(2, 24, 20)
    timesteps = torch.tensor([0, 7])
    noisy, _ = diffusion.q_sample(clean, timesteps)
    epsilon, variance = model(noisy, timesteps, condition)
    assert epsilon.shape == clean.shape
    assert variance is not None and variance.shape == clean.shape
    losses = diffusion.training_loss(model, clean, condition, timesteps)
    assert torch.isfinite(losses["loss"])


def test_ancestral_sampling_shape_and_range() -> None:
    model = tiny_model().eval()
    diffusion = GaussianDiffusion(timesteps=4, learn_variance=True)
    condition = torch.randn(2, 24, 20)
    sample = diffusion.sample(model, condition)
    assert sample.shape == (2, 24, 1)
    assert sample.min() >= -1
    assert sample.max() <= 1

