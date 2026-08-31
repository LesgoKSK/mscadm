import torch

from repro.diffusion import GaussianDiffusion
from repro.models import MSCADM


def make_model(**overrides) -> MSCADM:
    config = dict(
        condition_dim=20,
        sequence_length=24,
        model_dim=32,
        condition_bottleneck=16,
        depth=1,
        heads=2,
        ff_multiplier=2,
        learn_variance=True,
    )
    config.update(overrides)
    return MSCADM(**config)


def test_full_and_ablation_forward_paths() -> None:
    noisy = torch.randn(3, 24, 1)
    condition = torch.randn(3, 24, 20)
    timestep = torch.tensor([0, 3, 7])
    for model in (
        make_model(),
        make_model(use_multiscale_embedding=False),
        make_model(use_adaln=False),
    ):
        epsilon, variance = model(noisy, timestep, condition)
        assert epsilon.shape == noisy.shape
        assert variance is not None and variance.shape == noisy.shape


def test_learned_variance_loss_and_respaced_sampling() -> None:
    model = make_model()
    diffusion = GaussianDiffusion(timesteps=8, learn_variance=True)
    target = torch.randn(2, 24, 1)
    condition = torch.randn(2, 24, 20)
    losses = diffusion.loss(model, target, condition, torch.tensor([0, 7]))
    assert all(torch.isfinite(value) for value in losses.values())
    ancestral = diffusion.sample_ancestral(model, condition)
    respaced = diffusion.sample_ddim(model, condition, steps=4)
    assert ancestral.shape == target.shape
    assert respaced.shape == target.shape


def test_fixed_variance_ablation() -> None:
    model = make_model(learn_variance=False)
    diffusion = GaussianDiffusion(timesteps=4, learn_variance=False)
    target = torch.randn(2, 24, 1)
    condition = torch.randn(2, 24, 20)
    losses = diffusion.loss(model, target, condition)
    assert losses["vlb"] == 0

