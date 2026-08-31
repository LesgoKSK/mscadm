from __future__ import annotations

from typing import Any

from torch import nn

from .diffusion import GaussianDiffusion
from .models.baselines import (
    ConditionalDDPMDenoiser,
    ConditionalRealNVP,
    ConditionalVAE,
    WGANCritic,
    WGANGenerator,
)
from .models.mscadm import MSCADM


def build_torch_model(name: str, config: dict[str, Any]) -> nn.Module | dict[str, nn.Module]:
    model_config = dict(config.get("model", {}))
    if name == "mscadm":
        return MSCADM(**model_config)
    if name == "ddpm":
        return ConditionalDDPMDenoiser(**model_config)
    if name == "vae":
        return ConditionalVAE(**model_config)
    if name == "nf":
        return ConditionalRealNVP(**model_config)
    if name == "wgan":
        generator_config = dict(model_config.get("generator", {}))
        critic_config = dict(model_config.get("critic", {}))
        return {
            "generator": WGANGenerator(**generator_config),
            "critic": WGANCritic(**critic_config),
        }
    raise ValueError(f"Unknown trainable model: {name}")


def build_diffusion(name: str, config: dict[str, Any]) -> GaussianDiffusion:
    diffusion_config = dict(config.get("diffusion", {}))
    if name == "ddpm":
        diffusion_config.setdefault("learn_variance", False)
    return GaussianDiffusion(**diffusion_config)


def uses_flat_condition(name: str) -> bool:
    return name in {"vae", "nf", "wgan"}


__all__ = ["build_torch_model", "build_diffusion", "uses_flat_condition"]
