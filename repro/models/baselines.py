from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mscadm import SinusoidalEmbedding


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU, final_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(sizes) - 2:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    """Wind VAE hyperparameters from the public Dumas et al. baseline."""

    def __init__(
        self, condition_dim: int = 250, target_dim: int = 24, latent_dim: int = 20, hidden_dim: int = 200
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = mlp([condition_dim + target_dim, hidden_dim, 2 * latent_dim])
        self.decoder = mlp([condition_dim + latent_dim, hidden_dim, target_dim])

    def encode(self, target: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encoder(torch.cat([target, condition], dim=-1)).chunk(2, -1)
        return mean, log_variance.clamp(-12, 12)

    def loss(self, target: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_variance = self.encode(target, condition)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        prediction = self.decoder(torch.cat([latent, condition], dim=-1))
        reconstruction = (prediction - target).square().sum(dim=-1).mean()
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=-1).mean()
        return {"loss": reconstruction + kl, "reconstruction": reconstruction, "kl": kl}

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, scenarios: int) -> torch.Tensor:
        repeated = condition[:, None].expand(-1, scenarios, -1).reshape(-1, condition.shape[-1])
        latent = torch.randn(len(repeated), self.latent_dim, device=condition.device)
        generated = self.decoder(torch.cat([latent, repeated], dim=-1))
        return generated.reshape(len(condition), scenarios, -1)


class WGANGenerator(nn.Module):
    def __init__(
        self,
        condition_dim: int = 250,
        target_dim: int = 24,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network = mlp([condition_dim + latent_dim] + [hidden_dim] * layers + [target_dim])

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([noise, condition], dim=-1))

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, scenarios: int) -> torch.Tensor:
        repeated = condition[:, None].expand(-1, scenarios, -1).reshape(-1, condition.shape[-1])
        noise = torch.randn(len(repeated), self.latent_dim, device=condition.device)
        return self(noise, repeated).reshape(len(condition), scenarios, -1)


class WGANCritic(nn.Module):
    def __init__(
        self, condition_dim: int = 250, target_dim: int = 24, hidden_dim: int = 256, layers: int = 2
    ) -> None:
        super().__init__()
        self.network = mlp(
            [condition_dim + target_dim] + [hidden_dim] * layers + [1],
            activation=lambda: nn.LeakyReLU(0.01),
        )

    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([target, condition], dim=-1)).squeeze(-1)

    def gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.rand(len(real), 1, device=real.device)
        mixed = (weight * real + (1 - weight) * fake).requires_grad_(True)
        score = self(mixed, condition)
        gradient = torch.autograd.grad(
            score,
            mixed,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]
        return (gradient.flatten(1).norm(2, dim=1) - 1).square().mean()


class ConditionalCoupling(nn.Module):
    def __init__(self, dimension: int, condition_dim: int, mask: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.network = mlp([dimension + condition_dim, hidden_dim, hidden_dim, 2 * dimension])

    def parameters_for(self, value: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.network(torch.cat([value * self.mask, condition], dim=-1)).chunk(2, -1)
        inverse_mask = 1 - self.mask
        return shift * inverse_mask, torch.tanh(log_scale) * inverse_mask

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.parameters_for(value, condition)
        transformed = value * self.mask + (1 - self.mask) * (value * torch.exp(log_scale) + shift)
        return transformed, log_scale.sum(dim=-1)

    def inverse(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.parameters_for(value, condition)
        transformed = value * self.mask + (1 - self.mask) * ((value - shift) * torch.exp(-log_scale))
        return transformed, -log_scale.sum(dim=-1)


class ConditionalRealNVP(nn.Module):
    def __init__(
        self, dimension: int = 24, condition_dim: int = 250, steps: int = 8, hidden_dim: int = 300
    ) -> None:
        super().__init__()
        layers = []
        for index in range(steps):
            mask = ((torch.arange(dimension) + index) % 2).float()
            layers.append(ConditionalCoupling(dimension, condition_dim, mask, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.dimension = dimension

    def log_prob(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent = target
        log_determinant = torch.zeros(len(target), device=target.device)
        for layer in self.layers:
            latent, contribution = layer(latent, condition)
            log_determinant += contribution
        base = -0.5 * (latent.square() + math.log(2 * math.pi)).sum(dim=-1)
        return base + log_determinant

    def loss(self, target: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        negative_log_likelihood = -self.log_prob(target, condition).mean()
        return {"loss": negative_log_likelihood, "nll": negative_log_likelihood}

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, scenarios: int) -> torch.Tensor:
        repeated = condition[:, None].expand(-1, scenarios, -1).reshape(-1, condition.shape[-1])
        value = torch.randn(len(repeated), self.dimension, device=condition.device)
        for layer in reversed(self.layers):
            value, _ = layer.inverse(value, repeated)
        return value.reshape(len(condition), scenarios, self.dimension)


class DiffusionResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        self.time = nn.Linear(time_dim, channels)
        self.condition = nn.Conv1d(condition_dim, 2 * channels, kernel_size=1)
        self.dilated = nn.Conv1d(
            channels, 2 * channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.output = nn.Conv1d(channels, 2 * channels, kernel_size=1)

    def forward(
        self, hidden: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = hidden + self.time(time)[:, :, None]
        value = self.dilated(value) + self.condition(condition)
        gate, filter_value = value.chunk(2, dim=1)
        value = torch.sigmoid(gate) * torch.tanh(filter_value)
        residual, skip = self.output(value).chunk(2, dim=1)
        return (hidden + residual) / math.sqrt(2), skip


class ConditionalDDPMDenoiser(nn.Module):
    """WaveNet-like DDPM baseline based on the public energy DDPM code."""

    def __init__(
        self,
        condition_dim: int = 20,
        channels: int = 64,
        time_dim: int = 128,
        layers: int = 8,
        dilation_cycle: int = 4,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, kernel_size=1)
        self.time = nn.Sequential(
            SinusoidalEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.blocks = nn.ModuleList(
            [
                DiffusionResidualBlock(channels, time_dim, condition_dim, 2 ** (index % dilation_cycle))
                for index in range(layers)
            ]
        )
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_target: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        hidden = F.relu(self.input(noisy_target.transpose(1, 2)))
        time = self.time(timestep)
        condition_channels = condition.transpose(1, 2)
        skips = []
        for block in self.blocks:
            hidden, skip = block(hidden, time, condition_channels)
            skips.append(skip)
        hidden = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        output = self.output(F.relu(self.skip(hidden))).transpose(1, 2)
        return output, None

