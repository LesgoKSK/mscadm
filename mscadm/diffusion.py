from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps).clamp(max=0.999)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def _normal_kl(
    mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
    )


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 250,
        beta_schedule: str = "cosine",
        learn_variance: bool = True,
        vlb_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        cumulative_previous = torch.cat([torch.ones(1), cumulative[:-1]])

        posterior_variance = (
            betas * (1.0 - cumulative_previous) / (1.0 - cumulative)
        )
        posterior_log_variance = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]).clamp(min=1e-20)
        )
        posterior_mean_coef1 = (
            betas * torch.sqrt(cumulative_previous) / (1.0 - cumulative)
        )
        posterior_mean_coef2 = (
            (1.0 - cumulative_previous) * torch.sqrt(alphas) / (1.0 - cumulative)
        )

        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alpha_cumulative": cumulative,
            "alpha_cumulative_previous": cumulative_previous,
            "sqrt_alpha_cumulative": torch.sqrt(cumulative),
            "sqrt_one_minus_alpha_cumulative": torch.sqrt(1.0 - cumulative),
            "sqrt_recip_alpha_cumulative": torch.sqrt(1.0 / cumulative),
            "sqrt_recipm1_alpha_cumulative": torch.sqrt(1.0 / cumulative - 1.0),
            "posterior_variance": posterior_variance,
            "posterior_log_variance": posterior_log_variance,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }.items():
            self.register_buffer(name, value.float())
        self.timesteps = timesteps
        self.learn_variance = learn_variance
        self.vlb_weight = vlb_weight

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = (
            _extract(self.sqrt_alpha_cumulative, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_cumulative, timesteps, clean.shape) * noise
        )
        return noisy, noise

    def predict_clean_from_noise(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alpha_cumulative, timesteps, noisy.shape) * noisy
            - _extract(self.sqrt_recipm1_alpha_cumulative, timesteps, noisy.shape) * epsilon
        )

    def q_posterior(
        self, clean: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            _extract(self.posterior_mean_coef1, timesteps, noisy.shape) * clean
            + _extract(self.posterior_mean_coef2, timesteps, noisy.shape) * noisy
        )
        log_variance = _extract(self.posterior_log_variance, timesteps, noisy.shape)
        return mean, log_variance

    def model_distribution(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        *,
        clip_clean: bool = True,
        detach_mean: bool = False,
    ) -> dict[str, torch.Tensor]:
        epsilon, variance_value = model(noisy, timesteps, condition)
        epsilon_for_mean = epsilon.detach() if detach_mean else epsilon
        predicted_clean = self.predict_clean_from_noise(noisy, timesteps, epsilon_for_mean)
        if clip_clean:
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
        mean, fixed_log_variance = self.q_posterior(predicted_clean, noisy, timesteps)
        if self.learn_variance:
            if variance_value is None:
                raise ValueError("Model must return variance values when learn_variance=True")
            fraction = (torch.tanh(variance_value) + 1.0) * 0.5
            maximum = torch.log(_extract(self.betas, timesteps, noisy.shape).clamp(min=1e-20))
            log_variance = fraction * maximum + (1.0 - fraction) * fixed_log_variance
        else:
            log_variance = fixed_log_variance
        return {
            "mean": mean,
            "log_variance": log_variance,
            "predicted_clean": predicted_clean,
            "epsilon": epsilon,
        }

    def training_loss(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.timesteps, (batch,), device=clean.device)
        noisy, noise = self.q_sample(clean, timesteps)
        epsilon, _ = model(noisy, timesteps, condition)
        simple = (epsilon - noise).pow(2).flatten(1).mean(1)

        if self.learn_variance:
            true_mean, true_log_variance = self.q_posterior(clean, noisy, timesteps)
            predicted = self.model_distribution(
                model,
                noisy,
                timesteps,
                condition,
                clip_clean=False,
                detach_mean=True,
            )
            vlb = _normal_kl(
                true_mean,
                true_log_variance,
                predicted["mean"],
                predicted["log_variance"],
            ).flatten(1).mean(1) / math.log(2.0)
        else:
            vlb = torch.zeros_like(simple)
        total = simple + self.vlb_weight * vlb
        return {
            "loss": total.mean(),
            "simple": simple.mean(),
            "vlb": vlb.mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        target_shape: tuple[int, int, int] | None = None,
        progress_callback: Any | None = None,
    ) -> torch.Tensor:
        shape = target_shape or (condition.shape[0], condition.shape[1], 1)
        sample = torch.randn(shape, device=condition.device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (shape[0],), step, device=condition.device, dtype=torch.long
            )
            distribution = self.model_distribution(
                model, sample, timesteps, condition, clip_clean=True
            )
            noise = torch.randn_like(sample) if step > 0 else torch.zeros_like(sample)
            sample = distribution["mean"] + torch.exp(
                0.5 * distribution["log_variance"]
            ) * noise
            if progress_callback is not None:
                progress_callback(step)
        return sample.clamp(-1.0, 1.0)

