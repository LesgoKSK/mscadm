import torch

from repro.models.baselines import (
    ConditionalDDPMDenoiser,
    ConditionalRealNVP,
    ConditionalVAE,
    WGANCritic,
    WGANGenerator,
)


def test_flat_conditional_generators() -> None:
    condition = torch.randn(4, 250)
    target = torch.randn(4, 24)
    vae = ConditionalVAE()
    assert torch.isfinite(vae.loss(target, condition)["loss"])
    assert vae.sample(condition, 3).shape == (4, 3, 24)
    flow = ConditionalRealNVP(steps=2, hidden_dim=32)
    assert torch.isfinite(flow.loss(target, condition)["loss"])
    assert flow.sample(condition, 3).shape == (4, 3, 24)


def test_wgan_components_and_gradient_penalty() -> None:
    condition = torch.randn(4, 250)
    real = torch.randn(4, 24)
    generator = WGANGenerator(hidden_dim=32)
    critic = WGANCritic(hidden_dim=32)
    fake = generator(torch.randn(4, generator.latent_dim), condition)
    assert critic(fake, condition).shape == (4,)
    assert torch.isfinite(critic.gradient_penalty(real, fake.detach(), condition))
    assert generator.sample(condition, 2).shape == (4, 2, 24)


def test_conditional_ddpm_shape() -> None:
    denoiser = ConditionalDDPMDenoiser(channels=16, time_dim=32, layers=2)
    output, variance = denoiser(
        torch.randn(3, 24, 1), torch.tensor([0, 1, 2]), torch.randn(3, 24, 20)
    )
    assert output.shape == (3, 24, 1)
    assert variance is None

