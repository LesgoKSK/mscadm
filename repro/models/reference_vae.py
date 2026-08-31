from __future__ import annotations

import torch

from .baselines import ConditionalVAE


class DumasReferenceVAE(ConditionalVAE):
    """Behavior-compatible VAE for the public Dumas wind baseline.

    The reference code exponentiates its encoder's ``log_sigma`` directly in
    the reparameterization while using ``exp(log_sigma)`` in the KL formula.
    This is not the conventional half-log-variance parameterization used by
    :class:`ConditionalVAE`, but preserving it is necessary for baseline audit.
    The architecture and state-dict keys remain identical.
    """

    def loss(self, target: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_sigma = self.encode(target, condition)
        latent = mean + torch.exp(log_sigma) * torch.randn_like(mean)
        prediction = self.decoder(torch.cat([latent, condition], dim=-1))
        reconstruction = (prediction - target).square().sum(dim=-1).mean()
        kl = -0.5 * (1 + log_sigma - mean.square() - log_sigma.exp()).sum(dim=-1).mean()
        return {"loss": reconstruction + kl, "reconstruction": reconstruction, "kl": kl}


__all__ = ["DumasReferenceVAE"]
