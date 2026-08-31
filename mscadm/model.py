from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class MultiScaleConditionEmbedding(nn.Module):
    """Parallel 1-D conditional feature extraction matching Fig. 2."""

    def __init__(self, input_dim: int, model_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.down_projection = nn.Linear(model_dim, bottleneck)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(bottleneck, bottleneck, kernel_size=k, padding=k // 2),
                    nn.SiLU(),
                    nn.Conv1d(bottleneck, bottleneck, kernel_size=k, padding=k // 2),
                )
                for k in (3, 5, 7)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(bottleneck * 3, bottleneck, kernel_size=1),
            nn.SiLU(),
        )
        self.up_projection = nn.Linear(bottleneck, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(condition)
        compact = self.down_projection(hidden).transpose(1, 2)
        multiscale = torch.cat([branch(compact) for branch in self.branches], dim=1)
        fused = self.fusion(multiscale).transpose(1, 2)
        return self.output_norm(hidden + self.up_projection(fused))


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * ff_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_multiplier, dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, hidden: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(
            timestep_embedding
        ).chunk(6, dim=-1)
        attention_input = _modulate(self.norm1(hidden), shift1, scale1)
        attention_output, _ = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )
        hidden = hidden + gate1[:, None] * attention_output
        ff_input = _modulate(self.norm2(hidden), shift2, scale2)
        hidden = hidden + gate2[:, None] * self.feedforward(ff_input)
        return hidden


class MSCADM(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int = 20,
        sequence_length: int = 24,
        patch_size: int = 1,
        model_dim: int = 128,
        condition_bottleneck: int = 64,
        depth: int = 4,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        learn_variance: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length % patch_size:
            raise ValueError("sequence_length must be divisible by patch_size")
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.learn_variance = learn_variance
        self.num_patches = sequence_length // patch_size

        self.condition_embedding = MultiScaleConditionEmbedding(
            condition_dim, model_dim, condition_bottleneck
        )
        self.noisy_patch_embedding = nn.Conv1d(
            1, model_dim, kernel_size=patch_size, stride=patch_size
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, model_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)

        self.timestep_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(model_dim, heads, ff_multiplier, dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        output_channels = 2 if learn_variance else 1
        self.output = nn.Linear(model_dim, output_channels * patch_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_target: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if noisy_target.shape[1:] != (self.sequence_length, 1):
            raise ValueError(
                f"Expected noisy target [B,{self.sequence_length},1], got {tuple(noisy_target.shape)}"
            )
        noisy_tokens = self.noisy_patch_embedding(noisy_target.transpose(1, 2)).transpose(1, 2)
        condition_tokens = self.condition_embedding(condition)
        if self.patch_size > 1:
            condition_tokens = F.avg_pool1d(
                condition_tokens.transpose(1, 2),
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).transpose(1, 2)
        hidden = noisy_tokens + condition_tokens + self.position_embedding
        timestep_embedding = self.timestep_embedding(timesteps)
        for block in self.blocks:
            hidden = block(hidden, timestep_embedding)
        output = self.output(self.final_norm(hidden))
        batch = output.shape[0]
        output_channels = 2 if self.learn_variance else 1
        output = output.view(batch, self.num_patches, self.patch_size, output_channels)
        output = output.reshape(batch, self.sequence_length, output_channels)
        epsilon = output[..., :1]
        variance_value = output[..., 1:2] if self.learn_variance else None
        return epsilon, variance_value

