from __future__ import annotations

import math

import torch
from torch import nn


def cosine_betas(steps: int, offset: float = 0.008) -> torch.Tensor:
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((time / steps) + offset) / (1 + offset) * math.pi / 2) ** 2
    cumulative /= cumulative[0]
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999).float()


def linear_betas(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


def extract(values: torch.Tensor, timestep: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values[timestep].reshape(len(timestep), *((1,) * (len(shape) - 1)))


def normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        -1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).square() * torch.exp(-logvar2)
    )


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        *,
        timesteps: int = 250,
        schedule: str = "cosine",
        learn_variance: bool = True,
        vlb_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        betas = cosine_betas(timesteps) if schedule == "cosine" else linear_betas(timesteps)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, 0)
        previous = torch.cat([torch.ones(1), cumulative[:-1]])
        posterior_variance = betas * (1 - previous) / (1 - cumulative)
        posterior_log_variance = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]).clamp_min(1e-20)
        )
        tensors = {
            "betas": betas,
            "alphas": alphas,
            "cumulative": cumulative,
            "previous_cumulative": previous,
            "sqrt_cumulative": torch.sqrt(cumulative),
            "sqrt_one_minus_cumulative": torch.sqrt(1 - cumulative),
            "sqrt_recip_cumulative": torch.sqrt(1 / cumulative),
            "sqrt_recipm1_cumulative": torch.sqrt(1 / cumulative - 1),
            "posterior_variance": posterior_variance,
            "posterior_log_variance": posterior_log_variance,
            "posterior_coef1": betas * torch.sqrt(previous) / (1 - cumulative),
            "posterior_coef2": (1 - previous) * torch.sqrt(alphas) / (1 - cumulative),
        }
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor.float())
        self.timesteps = timesteps
        self.learn_variance = learn_variance
        self.vlb_weight = vlb_weight

    def q_sample(
        self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = extract(self.sqrt_cumulative, timestep, clean.shape) * clean
        noisy += extract(self.sqrt_one_minus_cumulative, timestep, clean.shape) * noise
        return noisy, noise

    def predict_clean(self, noisy: torch.Tensor, timestep: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_cumulative, timestep, noisy.shape) * noisy - extract(
            self.sqrt_recipm1_cumulative, timestep, noisy.shape
        ) * epsilon

    def q_posterior(
        self, clean: torch.Tensor, noisy: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_coef1, timestep, noisy.shape) * clean
        mean += extract(self.posterior_coef2, timestep, noisy.shape) * noisy
        return mean, extract(self.posterior_log_variance, timestep, noisy.shape)

    def model_distribution(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        *,
        detach_mean: bool = False,
        clip_clean: bool = False,
    ) -> dict[str, torch.Tensor]:
        epsilon, variance_value = model(noisy, timestep, condition)
        mean_epsilon = epsilon.detach() if detach_mean else epsilon
        clean = self.predict_clean(noisy, timestep, mean_epsilon)
        if clip_clean:
            clean = clean.clamp(-5, 5)
        mean, minimum_log_variance = self.q_posterior(clean, noisy, timestep)
        if self.learn_variance:
            if variance_value is None:
                raise ValueError("Learned variance requested but model returned none")
            fraction = (torch.tanh(variance_value) + 1) / 2
            maximum_log_variance = torch.log(extract(self.betas, timestep, noisy.shape).clamp_min(1e-20))
            log_variance = fraction * maximum_log_variance + (1 - fraction) * minimum_log_variance
        else:
            log_variance = minimum_log_variance
        return {"mean": mean, "log_variance": log_variance, "clean": clean, "epsilon": epsilon}

    def loss(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if timestep is None:
            timestep = torch.randint(self.timesteps, (len(clean),), device=clean.device)
        noisy, noise = self.q_sample(clean, timestep)
        epsilon, _ = model(noisy, timestep, condition)
        simple = (epsilon - noise).square().flatten(1).mean(1)
        if not self.learn_variance:
            vlb = torch.zeros_like(simple)
        else:
            true_mean, true_logvar = self.q_posterior(clean, noisy, timestep)
            predicted = self.model_distribution(
                model, noisy, timestep, condition, detach_mean=True, clip_clean=False
            )
            kl = normal_kl(
                true_mean, true_logvar, predicted["mean"], predicted["log_variance"]
            ).flatten(1).mean(1) / math.log(2)
            nll = 0.5 * (
                math.log(2 * math.pi)
                + predicted["log_variance"]
                + (clean - predicted["mean"]).square() * torch.exp(-predicted["log_variance"])
            ).flatten(1).mean(1) / math.log(2)
            vlb = torch.where(timestep == 0, nll, kl)
        total = simple + self.vlb_weight * vlb
        return {"loss": total.mean(), "simple": simple.mean(), "vlb": vlb.mean()}

    @torch.no_grad()
    def sample_ancestral(self, model: nn.Module, condition: torch.Tensor) -> torch.Tensor:
        sample = torch.randn(len(condition), condition.shape[1], 1, device=condition.device)
        for step in reversed(range(self.timesteps)):
            timestep = torch.full((len(sample),), step, device=sample.device, dtype=torch.long)
            distribution = self.model_distribution(model, sample, timestep, condition, clip_clean=True)
            noise = torch.randn_like(sample) if step else torch.zeros_like(sample)
            sample = distribution["mean"] + torch.exp(distribution["log_variance"] / 2) * noise
        return sample

    @torch.no_grad()
    def sample_ddim(
        self, model: nn.Module, condition: torch.Tensor, *, steps: int, eta: float = 1.0
    ) -> torch.Tensor:
        if steps < 1 or steps > self.timesteps:
            raise ValueError("steps must be between 1 and the training timestep count")
        selected = torch.linspace(0, self.timesteps - 1, steps, device=condition.device).round().long()
        selected = torch.unique(selected, sorted=True)
        sample = torch.randn(len(condition), condition.shape[1], 1, device=condition.device)
        for index in reversed(range(len(selected))):
            step = selected[index]
            timestep = torch.full((len(sample),), int(step), device=sample.device, dtype=torch.long)
            epsilon, _ = model(sample, timestep, condition)
            clean = self.predict_clean(sample, timestep, epsilon).clamp(-5, 5)
            alpha = self.cumulative[step]
            alpha_previous = self.cumulative[selected[index - 1]] if index > 0 else torch.tensor(1.0, device=sample.device)
            sigma = eta * torch.sqrt(
                ((1 - alpha_previous) / (1 - alpha)) * (1 - alpha / alpha_previous)
            ).clamp_min(0)
            direction = torch.sqrt((1 - alpha_previous - sigma.square()).clamp_min(0)) * epsilon
            noise = torch.randn_like(sample) if index > 0 else torch.zeros_like(sample)
            sample = torch.sqrt(alpha_previous) * clean + direction + sigma * noise
        return sample

