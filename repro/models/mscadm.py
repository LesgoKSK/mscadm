from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = timestep.float()[:, None] * frequencies[None]
        result = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return F.pad(result, (0, self.dim - result.shape[-1]))


class MultiScaleEmbedding(nn.Module):
    def __init__(self, condition_dim: int, model_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(condition_dim, model_dim, kernel_size=1)
        self.down = nn.Conv1d(model_dim, bottleneck, kernel_size=1)
        self.scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(bottleneck, bottleneck, kernel_size, padding=kernel_size // 2),
                    nn.SiLU(),
                    nn.Conv1d(bottleneck, bottleneck, kernel_size, padding=kernel_size // 2),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.merge = nn.Conv1d(3 * bottleneck, bottleneck, kernel_size=1)
        self.up = nn.Conv1d(bottleneck, model_dim, kernel_size=1)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        initial = self.input_conv(condition.transpose(1, 2))
        compact = self.down(initial)
        scales = torch.cat([branch(compact) for branch in self.scales], dim=1)
        output = initial + self.up(F.silu(self.merge(scales)))
        return self.norm(output.transpose(1, 2))


class LinearConditionEmbedding(nn.Module):
    """Single-scale replacement used by the w/o CE ablation."""

    def __init__(self, condition_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(condition_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.projection(condition)


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_multiplier, dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(time_embedding).chunk(6, -1)
        attention_input = _modulate(self.norm1(hidden), shift1, scale1)
        attended, _ = self.attention(attention_input, attention_input, attention_input, need_weights=False)
        hidden = hidden + gate1[:, None] * attended
        hidden = hidden + gate2[:, None] * self.ff(_modulate(self.norm2(hidden), shift2, scale2))
        return hidden


class StandardBlock(nn.Module):
    """Standard pre-norm Transformer used by the w/o AdaLN ablation."""

    def __init__(self, dim: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_multiplier, dim),
        )

    def forward(self, hidden: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        value = self.norm1(hidden)
        attended, _ = self.attention(value, value, value, need_weights=False)
        hidden = hidden + attended
        return hidden + self.ff(self.norm2(hidden))


class MSCADM(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int = 20,
        sequence_length: int = 24,
        model_dim: int = 128,
        condition_bottleneck: int = 64,
        depth: int = 4,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        learn_variance: bool = True,
        use_multiscale_embedding: bool = True,
        use_adaln: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.learn_variance = learn_variance
        if use_multiscale_embedding:
            self.condition_embedding = MultiScaleEmbedding(
                condition_dim, model_dim, condition_bottleneck
            )
        else:
            self.condition_embedding = LinearConditionEmbedding(condition_dim, model_dim)
        self.input_embedding = nn.Linear(1, model_dim)
        self.position = nn.Parameter(torch.randn(1, sequence_length, model_dim) * 0.02)
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(model_dim),
            nn.Linear(model_dim, 4 * model_dim),
            nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        block_type = AdaLNBlock if use_adaln else StandardBlock
        self.blocks = nn.ModuleList(
            [block_type(model_dim, heads, ff_multiplier, dropout) for _ in range(depth)]
        )
        self.use_adaln = use_adaln
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 2 if learn_variance else 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_target: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.input_embedding(noisy_target) + self.condition_embedding(condition) + self.position
        time_embedding = self.time_embedding(timestep)
        if not self.use_adaln:
            hidden = hidden + time_embedding[:, None]
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        output = self.output(self.final_norm(hidden))
        epsilon = output[..., :1]
        variance = output[..., 1:2] if self.learn_variance else None
        return epsilon, variance

