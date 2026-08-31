"""Common R0/T0 model skeleton for the architecture-v1 comparison.

The two public baselines share the same NWP encoder ``E``, exact-boundary atom
module ``A``, and joint time-domain rectified-flow velocity field:

``R0JointRectifiedFlow``
    A clean joint time-domain common-shell reference.  It is intentionally not
    presented as a numerical reproduction of the legacy STGF ``time_domain``
    checkpoint, which used a different center/standardization and calibration.

``T0MemorylessRectifiedFlow``
    The capacity-control baseline.  It adds the exact cell implementation that
    a future recurrent T1 model can use, but evaluates that cell with a zero
    recurrent transition and no graph connection to ``h[t-1]``.

No spectral or graph transform is used.  All tensors retain their physical
``[batch, zone, hour, ...]`` layout and one ensemble member is transported as a
single joint field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import nn

from .atom import (
    AtomAllocation,
    AtomStatistics,
    ExactAtomModule,
    INTERIOR_STATE,
)


class FourierTimeEmbedding(nn.Module):
    """Deterministic Fourier embedding of rectified-flow time ``t in [0,1]``."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("time embedding dimension must be at least four")
        half = dimension // 2
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * half, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("flow time must have shape [B]")
        angles = time[:, None] * self.frequencies[None] * (2.0 * math.pi)
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


class AxialJointBlock(nn.Module):
    """Temporal then cross-zone attention for a joint ``[B,Z,T,D]`` field."""

    def __init__(
        self,
        dimension: int,
        *,
        heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("model dimension must be divisible by attention heads")
        self.time_norm = nn.LayerNorm(dimension)
        self.time_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.zone_norm = nn.LayerNorm(dimension)
        self.zone_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, ff_multiplier * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("joint hidden field must have shape [B,Z,T,D]")
        batch, zones, hours, dimension = values.shape
        temporal = self.time_norm(values).reshape(batch * zones, hours, dimension)
        attended, _ = self.time_attention(
            temporal, temporal, temporal, need_weights=False
        )
        values = values + attended.reshape(batch, zones, hours, dimension)
        spatial = (
            self.zone_norm(values)
            .transpose(1, 2)
            .reshape(batch * hours, zones, dimension)
        )
        attended, _ = self.zone_attention(spatial, spatial, spatial, need_weights=False)
        values = values + attended.reshape(batch, hours, zones, dimension).transpose(1, 2)
        return values + self.feed_forward(self.ff_norm(values))


class JointNWPEncoder(nn.Module):
    """Shared encoder ``E`` used unchanged by all architecture-v1 candidates."""

    def __init__(
        self,
        condition_dim: int,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
        zones: int,
        hours: int,
    ) -> None:
        super().__init__()
        if zones < 1 or hours < 1:
            raise ValueError("zones and hours must be positive")
        self.condition_dim = int(condition_dim)
        self.model_dim = int(model_dim)
        self.zones = int(zones)
        self.hours = int(hours)
        self.input_projection = nn.Linear(condition_dim, model_dim)
        self.zone_position = nn.Parameter(
            torch.randn(1, zones, 1, model_dim) * 0.02
        )
        self.hour_position = nn.Parameter(
            torch.randn(1, 1, hours, model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                AxialJointBlock(
                    model_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        expected = (self.zones, self.hours, self.condition_dim)
        if condition.ndim != 4 or condition.shape[1:] != expected:
            raise ValueError(
                f"condition must have shape [B,{self.zones},{self.hours},"
                f"{self.condition_dim}], got {tuple(condition.shape)}"
            )
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("condition contains non-finite values")
        hidden = (
            self.input_projection(condition)
            + self.zone_position
            + self.hour_position
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden)


class ParameterMatchedTemporalCell(nn.Module):
    """One cell implementation for the memoryless T0 and future recurrent T1.

    Both modes instantiate exactly the same parameters and accept the same
    ``(input_t, previous_state)`` interface.  In memoryless mode the recurrent
    projection is not evaluated and the previous state is not inserted into the
    autograd graph.  Consequently ``h[t-1]`` cannot leak temporal information,
    while checkpoint keys and total parameter count remain exactly matched.
    """

    def __init__(self, dimension: int, *, use_memory: bool) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("cell dimension must be positive")
        self.dimension = int(dimension)
        self.use_memory = bool(use_memory)
        self.input_projection = nn.Linear(dimension, 3 * dimension)
        # No bias: a zero recurrent state means an exactly zero transition.
        self.recurrent_projection = nn.Linear(dimension, 3 * dimension, bias=False)
        self.state_norm = nn.LayerNorm(dimension)
        self.output_projection = nn.Linear(dimension, dimension)

    def forward(
        self,
        input_t: torch.Tensor,
        previous_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_t.ndim != 3 or input_t.shape[-1] != self.dimension:
            raise ValueError("cell input must have shape [B,Z,D]")
        if previous_state is not None and previous_state.shape != input_t.shape:
            raise ValueError("previous state does not align with cell input")

        input_reset, input_update, input_candidate = self.input_projection(
            input_t
        ).chunk(3, dim=-1)
        if self.use_memory:
            if previous_state is None:
                previous = torch.zeros_like(input_t)
            else:
                previous = previous_state
            hidden_reset, hidden_update, hidden_candidate = self.recurrent_projection(
                previous
            ).chunk(3, dim=-1)
        else:
            # Do not call recurrent_projection(previous_state): both the numeric
            # transition and the h[t-1] graph edge are absent in T0.
            previous = torch.zeros_like(input_t)
            hidden_reset = torch.zeros_like(input_reset)
            hidden_update = torch.zeros_like(input_update)
            hidden_candidate = torch.zeros_like(input_candidate)

        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset * hidden_candidate)
        next_state = self.state_norm(
            (1.0 - update) * candidate + update * previous
        )
        output = self.output_projection(next_state)
        return output, next_state

    def scan(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[-1] != self.dimension:
            raise ValueError("cell sequence must have shape [B,Z,T,D]")
        if initial_state is not None and initial_state.shape != (
            inputs.shape[0],
            inputs.shape[1],
            self.dimension,
        ):
            raise ValueError("initial state does not align with cell sequence")
        previous = initial_state
        outputs: list[torch.Tensor] = []
        for hour in range(inputs.shape[2]):
            output, previous = self(inputs[:, :, hour], previous)
            outputs.append(output)
        assert previous is not None
        return torch.stack(outputs, dim=2), previous

    def scan_ordered(
        self,
        inputs: torch.Tensor,
        *,
        hour_order: torch.Tensor | Sequence[int] | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan a sequence in a registered order and return chronological output.

        The ordered path is used only by the negative control.  Inputs are
        visited in ``hour_order`` and the resulting outputs are then scattered
        back to their physical hour positions.  Thus tensor layout and marginal
        hour labels remain unchanged while the recurrent adjacency is broken.
        """

        if hour_order is None:
            return self.scan(inputs, initial_state)
        order = torch.as_tensor(hour_order, dtype=torch.long, device=inputs.device)
        if order.ndim != 1 or len(order) != inputs.shape[2]:
            raise ValueError("hour order must contain one index per sequence hour")
        expected = torch.arange(inputs.shape[2], device=inputs.device)
        if not torch.equal(torch.sort(order).values, expected):
            raise ValueError("hour order must be a permutation")
        ordered, final_state = self.scan(
            inputs.index_select(2, order), initial_state
        )
        inverse = torch.empty_like(order)
        inverse[order] = expected
        return ordered.index_select(2, inverse), final_state

    def parameter_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((name, tuple(value.shape)) for name, value in self.state_dict().items())

    def extra_repr(self) -> str:
        mode = "recurrent" if self.use_memory else "memoryless"
        return f"dimension={self.dimension}, mode={mode}"


class TemporalSourceAdapter(nn.Module):
    """Condition and reshape the continuous RF source without creating atoms.

    Every T0/T1 mechanism candidate owns and evaluates the same adapter.  The
    only experimental switch is whether its matched temporal cell reads the
    previous hidden state, and (for the shuffle control) in which order hours
    are visited.  The residual output projection starts at zero, so all models
    begin from the same iid Gaussian source law.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("source dimension must be positive")
        self.dimension = int(dimension)
        self.noise_projection = nn.Linear(1, dimension, bias=False)
        self.condition_projection = nn.Linear(dimension, dimension)
        self.output_norm = nn.LayerNorm(dimension)
        self.output_projection = nn.Linear(dimension, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        iid_noise: torch.Tensor,
        encoded_condition: torch.Tensor,
        active_mask: torch.Tensor,
        temporal_cell: ParameterMatchedTemporalCell,
        *,
        hour_order: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if iid_noise.ndim != 3:
            raise ValueError("source noise must have shape [B,Z,T]")
        if encoded_condition.shape != (*iid_noise.shape, self.dimension):
            raise ValueError("encoded condition does not align with source noise")
        if active_mask.shape != iid_noise.shape or active_mask.dtype != torch.bool:
            raise ValueError("source active mask must be boolean and aligned")
        safe_noise = torch.where(active_mask, iid_noise, torch.zeros_like(iid_noise))
        inputs = (
            self.noise_projection(safe_noise[..., None])
            + self.condition_projection(encoded_condition)
        )
        # Atom coordinates have no continuous latent input.  Recurrent state may
        # pass across an atom hour, but no value is ever assigned at that hour.
        inputs = torch.where(active_mask[..., None], inputs, torch.zeros_like(inputs))
        hidden, _ = temporal_cell.scan_ordered(inputs, hour_order=hour_order)
        correction = self.output_projection(self.output_norm(hidden))[..., 0]
        source = safe_noise + correction
        return torch.where(active_mask, source, torch.zeros_like(source))


class VariancePreservingARSource(nn.Module):
    """Bounded conditional AR source with analytic unit marginal variance.

    Only a correlation coefficient is learned.  There is no source mean or
    marginal-scale head, preventing the width-collapse failure observed for the
    unconstrained residual source in temporal-mechanism v3.0.
    """

    def __init__(self, dimension: int, *, rho_max: float = 0.95) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("source dimension must be positive")
        if not 0.0 < rho_max < 1.0:
            raise ValueError("rho_max must lie in (0,1)")
        self.dimension = int(dimension)
        self.rho_max = float(rho_max)
        self.condition_norm = nn.LayerNorm(dimension)
        self.rho_head = nn.Linear(dimension, 1)
        nn.init.zeros_(self.rho_head.weight)
        nn.init.zeros_(self.rho_head.bias)

    def correlation(self, encoded_condition: torch.Tensor) -> torch.Tensor:
        if encoded_condition.ndim != 4 or encoded_condition.shape[-1] != self.dimension:
            raise ValueError("encoded condition must have shape [B,Z,T,D]")
        return self.rho_max * torch.tanh(
            self.rho_head(self.condition_norm(encoded_condition))[..., 0]
        )

    def forward(
        self,
        iid_noise: torch.Tensor,
        encoded_condition: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        use_correlation: bool,
        hour_order: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if iid_noise.ndim != 3:
            raise ValueError("source noise must have shape [B,Z,T]")
        if encoded_condition.shape != (*iid_noise.shape, self.dimension):
            raise ValueError("encoded condition does not align with source noise")
        if active_mask.shape != iid_noise.shape or active_mask.dtype != torch.bool:
            raise ValueError("source active mask must be boolean and aligned")
        safe_noise = torch.where(active_mask, iid_noise, torch.zeros_like(iid_noise))
        if not use_correlation:
            return safe_noise

        hours = iid_noise.shape[2]
        if hour_order is None:
            order = torch.arange(hours, dtype=torch.long, device=iid_noise.device)
        else:
            order = torch.as_tensor(hour_order, dtype=torch.long, device=iid_noise.device)
            if order.shape != (hours,) or not torch.equal(
                torch.sort(order).values,
                torch.arange(hours, device=iid_noise.device),
            ):
                raise ValueError("source hour order must be a full permutation")
        epsilon = safe_noise.index_select(2, order)
        active = active_mask.index_select(2, order)
        rho = self.correlation(encoded_condition).index_select(2, order)
        previous = torch.zeros_like(epsilon[:, :, 0])
        previous_active = torch.zeros_like(active[:, :, 0])
        outputs: list[torch.Tensor] = []
        for index in range(hours):
            current_active = active[:, :, index]
            contiguous = current_active & previous_active
            current_rho = torch.where(
                contiguous,
                rho[:, :, index],
                torch.zeros_like(rho[:, :, index]),
            )
            innovation_scale = torch.sqrt(
                torch.clamp(1.0 - current_rho.square(), min=1e-6)
            )
            current = current_rho * previous + innovation_scale * epsilon[:, :, index]
            current = torch.where(current_active, current, torch.zeros_like(current))
            outputs.append(current)
            previous = current
            previous_active = current_active
        ordered = torch.stack(outputs, dim=2)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(hours, device=order.device)
        result = ordered.index_select(2, inverse)
        return torch.where(active_mask, result, torch.zeros_like(result))


class JointTimeDomainVelocity(nn.Module):
    """Joint rectified-flow velocity field in the original time domain."""

    def __init__(
        self,
        condition_dim: int,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
        zones: int,
        hours: int,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.model_dim = int(model_dim)
        self.zones = int(zones)
        self.hours = int(hours)
        self.value_projection = nn.Linear(1, model_dim)
        self.condition_projection = nn.Linear(condition_dim, model_dim)
        self.active_projection = nn.Linear(1, model_dim, bias=False)
        self.flow_time = FourierTimeEmbedding(model_dim)
        self.zone_position = nn.Parameter(
            torch.randn(1, zones, 1, model_dim) * 0.02
        )
        self.hour_position = nn.Parameter(
            torch.randn(1, 1, hours, model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                AxialJointBlock(
                    model_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 1)
        # Zero velocity is a stable starting point for rectified-flow fitting.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        value: torch.Tensor,
        flow_time: torch.Tensor,
        encoded_condition: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (value.shape[0], self.zones, self.hours)
        if value.ndim != 3 or value.shape != expected:
            raise ValueError(
                f"flow value must have shape [B,{self.zones},{self.hours}]"
            )
        if encoded_condition.shape != (*value.shape, self.condition_dim):
            raise ValueError("encoded condition does not align with flow value")
        if active_mask.shape != value.shape or active_mask.dtype != torch.bool:
            raise ValueError("active mask must be boolean and align with flow value")
        if flow_time.shape != (len(value),):
            raise ValueError("flow time must have shape [B]")
        safe_value = torch.where(active_mask, value, torch.zeros_like(value))
        hidden = (
            self.value_projection(safe_value[..., None])
            + self.condition_projection(encoded_condition)
            + self.active_projection(active_mask[..., None].to(value.dtype))
            + self.flow_time(flow_time)[:, None, None]
            + self.zone_position
            + self.hour_position
        )
        for block in self.blocks:
            hidden = block(hidden)
        velocity = self.output(self.output_norm(hidden))[..., 0]
        # This is the central semantic invariant: atom coordinates never move.
        return torch.where(active_mask, velocity, torch.zeros_like(velocity))


@dataclass(frozen=True)
class ScenarioBatch:
    """Joint scenarios plus the state/latent audit trail used to create them."""

    values: torch.Tensor
    states: torch.Tensor
    active_mask: torch.Tensor
    interior_latent: torch.Tensor
    atom_statistics: AtomStatistics
    atom_allocation: AtomAllocation
    per_path_nfe: int
    batched_forward_calls: int

    def __post_init__(self) -> None:
        if self.values.shape != self.states.shape:
            raise ValueError("scenario values and states do not align")
        if self.interior_latent.shape != self.states.shape:
            raise ValueError("interior latent and states do not align")
        if self.active_mask.shape != self.states.shape:
            raise ValueError("scenario active mask and states do not align")
        if not torch.equal(self.active_mask, self.states == INTERIOR_STATE):
            raise ValueError("scenario active mask has invalid semantics")
        if bool((self.interior_latent[~self.active_mask] != 0.0).any()):
            raise ValueError("atom coordinates must not contain continuous latent values")
        if self.per_path_nfe < 1:
            raise ValueError("per-path NFE must be positive")
        if self.batched_forward_calls < self.per_path_nfe:
            raise ValueError("batched forward calls cannot be smaller than per-path NFE")
        if self.batched_forward_calls % self.per_path_nfe:
            raise ValueError("batched forward calls must contain complete path evaluations")

    @property
    def velocity_calls(self) -> int:
        """Backward-compatible alias for actual batched network invocations."""

        return self.batched_forward_calls


TemporalMode = Literal[
    "none",
    "matched_memoryless",
    "feature_recurrent",
    "feature_recurrent_shuffled",
    "source_recurrent",
    "source_recurrent_shuffled",
    "stable_memoryless",
    "stable_feature_recurrent",
    "stable_feature_recurrent_shuffled",
    "stable_source_recurrent",
    "stable_source_recurrent_shuffled",
]


class JointRectifiedFlowBase(nn.Module):
    """Shared implementation for R0 and matched T0/T1 mechanism controls."""

    variant = "base"

    def __init__(
        self,
        *,
        condition_dim: int = 20,
        zones: int = 10,
        hours: int = 24,
        encoder_dim: int = 64,
        encoder_depth: int = 2,
        flow_dim: int = 96,
        flow_depth: int = 4,
        heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        atom_hidden_dim: int | None = None,
        atom_initial_probabilities: tuple[float, float, float] = (
            0.05,
            0.949,
            0.001,
        ),
        atom_fixed_one_probability: float | None = None,
        atom_shared_priority_weight: float = 0.75,
        atom_location_auxiliary_weight: float = 0.0,
        atom_location_mean: float = 0.0,
        atom_location_std: float = 1.0,
        atom_location_smooth_l1_beta: float = 1.0,
        temporal_mode: TemporalMode = "none",
        temporal_hour_order: Sequence[int] | None = None,
        stable_source_rho_max: float = 0.95,
        logit_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if not 0.0 < logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must lie in (0,0.5)")
        self.condition_dim = int(condition_dim)
        self.zones = int(zones)
        self.hours = int(hours)
        self.encoder_dim = int(encoder_dim)
        self.temporal_mode = str(temporal_mode)
        self.logit_epsilon = float(logit_epsilon)
        if atom_location_auxiliary_weight < 0.0:
            raise ValueError("atom location auxiliary weight must be non-negative")
        if atom_location_std <= 0.0:
            raise ValueError("atom location standard deviation must be positive")
        self.atom_location_auxiliary_weight = float(atom_location_auxiliary_weight)
        self.atom_location_mean = float(atom_location_mean)
        self.atom_location_std = float(atom_location_std)
        self.atom_location_smooth_l1_beta = float(atom_location_smooth_l1_beta)
        self.encoder = JointNWPEncoder(
            condition_dim,
            model_dim=encoder_dim,
            depth=encoder_depth,
            heads=heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            zones=zones,
            hours=hours,
        )
        self.atom = ExactAtomModule(
            encoder_dim,
            hidden_dim=atom_hidden_dim,
            initial_probabilities=atom_initial_probabilities,
            shared_priority_weight=atom_shared_priority_weight,
            fixed_one_probability=atom_fixed_one_probability,
        )
        if temporal_mode == "none":
            if temporal_hour_order is not None:
                raise ValueError("R0 cannot register a temporal hour order")
            self.feature_cell: ParameterMatchedTemporalCell | None = None
            self.source_cell: ParameterMatchedTemporalCell | None = None
            self.source_adapter: TemporalSourceAdapter | None = None
            self.stable_source: VariancePreservingARSource | None = None
            order = torch.arange(hours, dtype=torch.long)
        elif temporal_mode.startswith("stable_"):
            feature_memory = temporal_mode in (
                "stable_feature_recurrent",
                "stable_feature_recurrent_shuffled",
            )
            self.feature_cell = ParameterMatchedTemporalCell(
                encoder_dim, use_memory=feature_memory
            )
            self.source_cell = None
            self.source_adapter = None
            self.stable_source = VariancePreservingARSource(
                encoder_dim, rho_max=stable_source_rho_max
            )
            if temporal_mode in (
                "stable_feature_recurrent_shuffled",
                "stable_source_recurrent_shuffled",
            ):
                if temporal_hour_order is None:
                    raise ValueError("stable shuffle control requires an explicit hour order")
                order = torch.as_tensor(tuple(temporal_hour_order), dtype=torch.long)
                if order.shape != (hours,) or not torch.equal(
                    torch.sort(order).values, torch.arange(hours)
                ):
                    raise ValueError("temporal hour order must be a full permutation")
                if torch.equal(order, torch.arange(hours)):
                    raise ValueError("shuffle control cannot use chronological order")
            else:
                if temporal_hour_order is not None:
                    raise ValueError("only a stable shuffle control accepts an hour order")
                order = torch.arange(hours, dtype=torch.long)
        else:
            feature_memory = temporal_mode in (
                "feature_recurrent",
                "feature_recurrent_shuffled",
            )
            source_memory = temporal_mode in (
                "source_recurrent",
                "source_recurrent_shuffled",
            )
            self.feature_cell = ParameterMatchedTemporalCell(
                encoder_dim, use_memory=feature_memory
            )
            self.source_cell = ParameterMatchedTemporalCell(
                encoder_dim, use_memory=source_memory
            )
            self.source_adapter = TemporalSourceAdapter(encoder_dim)
            self.stable_source = None
            if temporal_mode in (
                "feature_recurrent_shuffled",
                "source_recurrent_shuffled",
            ):
                if temporal_hour_order is None:
                    raise ValueError("shuffle control requires an explicit hour order")
                order = torch.as_tensor(tuple(temporal_hour_order), dtype=torch.long)
                if order.shape != (hours,) or not torch.equal(
                    torch.sort(order).values, torch.arange(hours)
                ):
                    raise ValueError("temporal hour order must be a full permutation")
                if torch.equal(order, torch.arange(hours)):
                    raise ValueError("shuffle control cannot use chronological order")
            else:
                if temporal_hour_order is not None:
                    raise ValueError("only the shuffle control accepts an hour order")
                order = torch.arange(hours, dtype=torch.long)
        # The order is an experiment-config identity rather than a learned
        # checkpoint tensor.  Keeping it non-persistent preserves strict loading
        # of the already frozen R0-D checkpoints.
        self.register_buffer("temporal_hour_order", order, persistent=False)
        self.flow = JointTimeDomainVelocity(
            encoder_dim,
            model_dim=flow_dim,
            depth=flow_depth,
            heads=heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            zones=zones,
            hours=hours,
        )
        self._shared_frozen = False

    def train(self, mode: bool = True) -> JointRectifiedFlowBase:
        super().train(mode)
        if self._shared_frozen:
            self.encoder.eval()
            self.atom.eval()
        return self

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.encoder(condition)

    def condition_context(
        self,
        encoded_nwp: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.feature_cell is None:
            if initial_state is not None:
                raise ValueError("R0 has no temporal state")
            return encoded_nwp
        order: torch.Tensor | None = None
        if self.temporal_mode in (
            "feature_recurrent_shuffled",
            "stable_feature_recurrent_shuffled",
        ):
            order = self.temporal_hour_order
        context, _ = self.feature_cell.scan_ordered(
            encoded_nwp,
            hour_order=order,
            initial_state=initial_state,
        )
        return context

    @property
    def temporal_cell(self) -> ParameterMatchedTemporalCell | None:
        """Backward-compatible name for the feature-side matched cell."""

        return self.feature_cell

    def prepare_source_noise(
        self,
        iid_noise: torch.Tensor,
        encoded_nwp: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the candidate's continuous source on active coordinates only."""

        if self.stable_source is not None:
            order: torch.Tensor | None = None
            if self.temporal_mode == "stable_source_recurrent_shuffled":
                order = self.temporal_hour_order
            return self.stable_source(
                iid_noise,
                encoded_nwp,
                active_mask,
                use_correlation=self.temporal_mode in (
                    "stable_source_recurrent",
                    "stable_source_recurrent_shuffled",
                ),
                hour_order=order,
            )
        if self.source_cell is None or self.source_adapter is None:
            return torch.where(active_mask, iid_noise, torch.zeros_like(iid_noise))
        order: torch.Tensor | None = None
        if self.temporal_mode == "source_recurrent_shuffled":
            order = self.temporal_hour_order
        return self.source_adapter(
            iid_noise,
            encoded_nwp,
            active_mask,
            self.source_cell,
            hour_order=order,
        )

    def atom_statistics(self, condition: torch.Tensor) -> AtomStatistics:
        return self.atom(self.encode_condition(condition))

    def atom_loss(
        self,
        condition: torch.Tensor,
        *,
        observation: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        observed_mask: torch.Tensor,
        location_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (observation is None) == (states is None):
            raise ValueError("provide exactly one of observation or states")
        if states is None:
            assert observation is not None
            states = self.atom.states_from_observation(observation)
        encoded = self.encode_condition(condition)
        return self.atom.loss(
            self.atom(encoded),
            states,
            observed_mask=observed_mask,
            location_target=location_target,
            location_weight=self.atom_location_auxiliary_weight,
            location_mean=self.atom_location_mean,
            location_std=self.atom_location_std,
            location_logit_epsilon=self.logit_epsilon,
            location_smooth_l1_beta=self.atom_location_smooth_l1_beta,
        )

    def _encoded_context(self, condition: torch.Tensor) -> tuple[torch.Tensor, AtomStatistics]:
        encoded = self.encode_condition(condition)
        return self.condition_context(encoded), self.atom(encoded)

    def velocity(
        self,
        value: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        states: torch.Tensor,
        *,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = self.atom.active_mask(states, observed_mask)
        context = self.condition_context(self.encode_condition(condition))
        return self.flow(value, flow_time, context, active)

    def flow_loss(
        self,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        states: torch.Tensor | None = None,
        observed_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        expected = (len(condition), self.zones, self.hours)
        if observation.shape != expected:
            raise ValueError("observation does not align with condition")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains non-finite values")
        if bool(((observation < 0.0) | (observation > 1.0)).any()):
            raise ValueError("observation must lie in [0,1]")
        if states is None:
            states = self.atom.states_from_observation(observation)
        if states.shape != observation.shape:
            raise ValueError("states do not align with observation")
        if observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool:
            raise ValueError(
                "observed_mask must be boolean and align with observation"
            )
        active = self.atom.active_mask(states, observed_mask)
        if not bool(active.any()):
            raise ValueError("flow loss requires at least one observed interior target")

        # Values at exact atoms are replaced before logit, then masked to zero.
        # Thus they are never interpreted as censored/latent continuous values.
        safe_observation = torch.where(
            active, observation, torch.full_like(observation, 0.5)
        )
        target = torch.logit(
            safe_observation.clamp(self.logit_epsilon, 1.0 - self.logit_epsilon)
        )
        target = torch.where(active, target, torch.zeros_like(target))
        iid_noise = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
        iid_noise = torch.where(active, iid_noise, torch.zeros_like(iid_noise))
        flow_time = torch.rand(
            len(target),
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
        encoded = self.encode_condition(condition)
        noise = self.prepare_source_noise(iid_noise, encoded, active)
        path = (
            (1.0 - flow_time[:, None, None]) * noise
            + flow_time[:, None, None] * target
        )
        context = self.condition_context(encoded)
        prediction = self.flow(path, flow_time, context, active)
        desired = target - noise
        squared = (prediction - desired).square()
        active_float = active.to(squared.dtype)
        loss = (squared * active_float).sum() / active_float.sum().clamp_min(1.0)
        return {
            "loss": loss,
            "velocity_mse": loss,
            "active_fraction": active_float.mean(),
            "inactive_velocity_max": prediction[~active].abs().max()
            if bool((~active).any())
            else prediction.sum() * 0.0,
        }

    def freeze_shared(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the common E/A shell for mechanism comparisons."""

        self._shared_frozen = bool(frozen)
        for module in (self.encoder, self.atom):
            module.requires_grad_(not frozen)
            if frozen:
                module.eval()
            else:
                module.train(self.training)

    def unfreeze_shared(self) -> None:
        self.freeze_shared(False)

    def load_shared_from(
        self, other: JointRectifiedFlowBase, *, freeze: bool = False
    ) -> None:
        """Copy, rather than alias, the common E/A weights from another model."""

        self.encoder.load_state_dict(other.encoder.state_dict(), strict=True)
        self.atom.load_state_dict(other.atom.state_dict(), strict=True)
        self.freeze_shared(freeze)

    def shared_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.encoder, self.atom)
            for parameter in module.parameters()
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def effective_active_parameter_count(self) -> int:
        """Count trainable parameters that can participate in this variant.

        T0 retains the recurrent projection for exact checkpoint/parameter
        matching, but its forward graph deliberately never evaluates that
        projection.  Reporting this count alongside total/trainable counts makes
        the dormant control parameters explicit.
        """

        total = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("feature_cell.recurrent_projection.") and self.temporal_mode in (
                "matched_memoryless",
                "source_recurrent",
                "source_recurrent_shuffled",
                "stable_memoryless",
                "stable_source_recurrent",
                "stable_source_recurrent_shuffled",
            ):
                continue
            if name.startswith("source_cell.recurrent_projection.") and self.temporal_mode in (
                "matched_memoryless",
                "feature_recurrent",
                "feature_recurrent_shuffled",
            ):
                continue
            if name.startswith("stable_source.") and self.temporal_mode in (
                "stable_memoryless",
                "stable_feature_recurrent",
                "stable_feature_recurrent_shuffled",
            ):
                continue
            total += parameter.numel()
        return total

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        members: int,
        steps: int,
        seed: int,
        method: Literal["euler", "heun"] = "heun",
        member_chunk: int = 25,
        allocation: AtomAllocation | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> ScenarioBatch:
        """Generate a joint ensemble while keeping atom latents identically zero."""

        if members < 1:
            raise ValueError("members must be positive")
        if steps < 1:
            raise ValueError("flow steps must be positive")
        if member_chunk < 1:
            raise ValueError("member_chunk must be positive")
        if method not in ("euler", "heun"):
            raise ValueError("method must be 'euler' or 'heun'")
        was_training = self.training
        self.eval()
        try:
            encoded = self.encode_condition(condition)
            statistics = self.atom(encoded)
            if allocation is None:
                allocation = self.atom.allocate(
                    statistics, members=members, seed=seed + 1
                )
            expected_states = (
                len(condition),
                members,
                self.zones,
                self.hours,
            )
            if allocation.states.shape != expected_states:
                raise ValueError("provided atom allocation has the wrong shape")
            if allocation.states.device != condition.device:
                raise ValueError("provided atom allocation is on the wrong device")
            if not torch.equal(
                allocation.analytic_probabilities, statistics.probabilities
            ):
                raise ValueError(
                    "provided atom allocation was not created by the current E/A law"
                )
            states = allocation.states
            active = allocation.active_mask
            context = self.condition_context(encoded)
            if initial_noise is None:
                generator = torch.Generator(device=condition.device)
                generator.manual_seed(int(seed) + 2)
                latent = torch.randn(
                    expected_states,
                    dtype=condition.dtype,
                    device=condition.device,
                    generator=generator,
                )
            else:
                if initial_noise.shape != expected_states:
                    raise ValueError("initial noise has the wrong shape")
                if initial_noise.device != condition.device:
                    raise ValueError("initial noise is on the wrong device")
                if initial_noise.dtype != condition.dtype:
                    raise TypeError("initial noise dtype must match condition dtype")
                if not bool(torch.isfinite(initial_noise).all()):
                    raise ValueError("initial noise contains non-finite values")
                latent = initial_noise

            # Allocate all noise before chunking so common-random-number results
            # are independent of the inference memory setting.
            chunks: list[torch.Tensor] = []
            batched_forward_calls = 0
            per_path_nfe = steps if method == "euler" else 2 * steps - 1
            delta = 1.0 / steps
            for start in range(0, members, member_chunk):
                stop = min(start + member_chunk, members)
                width = stop - start
                chunk_active = active[:, start:stop].reshape(
                    -1, self.zones, self.hours
                )
                chunk_iid = torch.where(
                    active[:, start:stop],
                    latent[:, start:stop],
                    torch.zeros_like(latent[:, start:stop]),
                ).reshape(-1, self.zones, self.hours)
                chunk_encoded = encoded[:, None].expand(
                    -1, width, -1, -1, -1
                ).reshape(-1, self.zones, self.hours, self.encoder_dim)
                chunk_latent = self.prepare_source_noise(
                    chunk_iid, chunk_encoded, chunk_active
                )
                chunk_context = context[:, None].expand(
                    -1, width, -1, -1, -1
                ).reshape(-1, self.zones, self.hours, self.encoder_dim)
                for index in range(steps):
                    time0 = torch.full(
                        (len(chunk_latent),),
                        index / steps,
                        dtype=chunk_latent.dtype,
                        device=chunk_latent.device,
                    )
                    velocity0 = self.flow(
                        chunk_latent, time0, chunk_context, chunk_active
                    )
                    batched_forward_calls += 1
                    proposal = torch.where(
                        chunk_active,
                        chunk_latent + delta * velocity0,
                        torch.zeros_like(chunk_latent),
                    )
                    if method == "euler" or index == steps - 1:
                        chunk_latent = proposal
                    else:
                        time1 = torch.full(
                            (len(chunk_latent),),
                            (index + 1) / steps,
                            dtype=chunk_latent.dtype,
                            device=chunk_latent.device,
                        )
                        velocity1 = self.flow(
                            proposal, time1, chunk_context, chunk_active
                        )
                        batched_forward_calls += 1
                        chunk_latent = torch.where(
                            chunk_active,
                            chunk_latent
                            + 0.5 * delta * (velocity0 + velocity1),
                            torch.zeros_like(chunk_latent),
                        )
                chunks.append(
                    chunk_latent.reshape(
                        len(condition), width, self.zones, self.hours
                    )
                )
            latent = torch.cat(chunks, dim=1)
            expected_forward_calls = per_path_nfe * math.ceil(
                members / member_chunk
            )
            if batched_forward_calls != expected_forward_calls:
                raise RuntimeError("velocity call accounting invariant failed")
            values = self.atom.reconstruct(latent, states)
            return ScenarioBatch(
                values=values,
                states=states,
                active_mask=active,
                interior_latent=latent,
                atom_statistics=statistics,
                atom_allocation=allocation,
                per_path_nfe=per_path_nfe,
                batched_forward_calls=batched_forward_calls,
            )
        finally:
            self.train(was_training)

    def model_spec(self) -> dict[str, Any]:
        stable_source_recurrent = self.temporal_mode in (
            "stable_source_recurrent",
            "stable_source_recurrent_shuffled",
        )
        stable_source_modes = self.temporal_mode.startswith("stable_")
        shuffled = self.temporal_mode in (
            "feature_recurrent_shuffled",
            "source_recurrent_shuffled",
            "stable_feature_recurrent_shuffled",
            "stable_source_recurrent_shuffled",
        )
        return {
            "variant": self.variant,
            "condition_dim": self.condition_dim,
            "zones": self.zones,
            "hours": self.hours,
            "encoder_dim": self.encoder_dim,
            "temporal_mode": self.temporal_mode,
            "feature_recurrent": bool(
                self.feature_cell is not None and self.feature_cell.use_memory
            ),
            "source_recurrent": bool(
                self.source_cell is not None and self.source_cell.use_memory
            ) or stable_source_recurrent,
            "source_law": (
                "variance_preserving_conditional_ar1"
                if stable_source_recurrent
                else "iid_gaussian"
                if stable_source_modes
                else "unconstrained_residual_recurrent"
                if self.source_cell is not None and self.source_cell.use_memory
                else "residual_memoryless"
                if self.source_adapter is not None
                else "iid_gaussian"
            ),
            "stable_source_rho_max": (
                self.stable_source.rho_max if self.stable_source is not None else None
            ),
            "temporal_hour_order": self.temporal_hour_order.detach().cpu().tolist(),
            "shuffle_semantics": (
                "recurrent_order_only_outputs_restored_to_physical_hours"
                if shuffled
                else "chronological"
            ),
            "shuffle_target": (
                "feature"
                if self.temporal_mode in (
                    "feature_recurrent_shuffled",
                    "stable_feature_recurrent_shuffled",
                )
                else "source"
                if self.temporal_mode in (
                    "source_recurrent_shuffled",
                    "stable_source_recurrent_shuffled",
                )
                else None
            ),
            "logit_epsilon": self.logit_epsilon,
            "continuous_latent_domain": "interior_only",
            "joint_layout": "batch_member_zone_hour",
            "total_parameters": self.parameter_count(),
            "trainable_parameters": self.parameter_count(trainable_only=True),
            "effective_active_parameters": self.effective_active_parameter_count(),
            "shared_parameters": self.shared_parameter_count(),
            "atom_allocation_semantics": self.atom.allocation_semantics,
            "atom_fixed_one_probability": self.atom.head.fixed_one_probability,
            "atom_location_auxiliary_weight": self.atom_location_auxiliary_weight,
            "atom_location_normalizer": {
                "mean": self.atom_location_mean,
                "std": self.atom_location_std,
            },
        }


class R0JointRectifiedFlow(JointRectifiedFlowBase):
    """R0: clean joint time-domain common-shell rectified-flow reference."""

    variant = "R0"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs:
            raise ValueError("R0 fixes temporal_mode='none'")
        super().__init__(temporal_mode="none", **kwargs)


class T0MemorylessRectifiedFlow(JointRectifiedFlowBase):
    """T0: matched feature/source cells with both recurrent edges disabled.

    "Memoryless" applies to the two added cells only.  The shared axial
    attention in the velocity field can still inspect the full 24-hour field.
    """

    variant = "T0"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs:
            raise ValueError(
                "T0 fixes temporal_mode='matched_memoryless'"
            )
        super().__init__(temporal_mode="matched_memoryless", **kwargs)

    def recurrent_cell_reference(self) -> ParameterMatchedTemporalCell:
        """Return the state-compatible recurrent cell for a future T1 model."""

        assert self.temporal_cell is not None
        cell = ParameterMatchedTemporalCell(self.encoder_dim, use_memory=True)
        cell.load_state_dict(self.temporal_cell.state_dict(), strict=True)
        return cell


class T1FeatureRectifiedFlow(JointRectifiedFlowBase):
    """T1-feature: recurrent NWP features, memoryless conditional source."""

    variant = "T1_feature"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs or "temporal_hour_order" in kwargs:
            raise ValueError("T1-feature fixes its temporal mechanism and order")
        super().__init__(temporal_mode="feature_recurrent", **kwargs)


class T1SourceRectifiedFlow(JointRectifiedFlowBase):
    """T1-source: memoryless NWP features, recurrent conditional source."""

    variant = "T1_source"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs or "temporal_hour_order" in kwargs:
            raise ValueError("T1-source fixes its temporal mechanism and order")
        super().__init__(temporal_mode="source_recurrent", **kwargs)


class T1ShuffleRectifiedFlow(JointRectifiedFlowBase):
    """Negative control with non-chronological feature or source adjacency."""

    variant = "T1_shuffle"

    def __init__(
        self,
        *,
        temporal_hour_order: Sequence[int],
        shuffle_target: Literal["feature", "source"] = "source",
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs:
            raise ValueError("T1-shuffle derives temporal_mode from shuffle_target")
        if shuffle_target not in ("feature", "source"):
            raise ValueError("shuffle_target must be 'feature' or 'source'")
        super().__init__(
            temporal_mode=(
                "feature_recurrent_shuffled"
                if shuffle_target == "feature"
                else "source_recurrent_shuffled"
            ),
            temporal_hour_order=temporal_hour_order,
            **kwargs,
        )
        self.variant = f"T1_{shuffle_target}_shuffle"


class T0StableSourceRectifiedFlow(JointRectifiedFlowBase):
    """v3.1 matched control with an exactly IID standard-normal source."""

    variant = "T0_v3_1"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs or "temporal_hour_order" in kwargs:
            raise ValueError("T0-v3.1 fixes its temporal mechanism and order")
        super().__init__(temporal_mode="stable_memoryless", **kwargs)


class T1FeatureStableSourceRectifiedFlow(JointRectifiedFlowBase):
    """v3.1 recurrent NWP features with an unchanged IID source."""

    variant = "T1_feature_v3_1"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs or "temporal_hour_order" in kwargs:
            raise ValueError("T1-feature-v3.1 fixes its mechanism and order")
        super().__init__(temporal_mode="stable_feature_recurrent", **kwargs)


class T1SourceStableRectifiedFlow(JointRectifiedFlowBase):
    """v3.1 bounded conditional AR(1) source with unit marginal variance."""

    variant = "T1_source_v3_1"

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs or "temporal_hour_order" in kwargs:
            raise ValueError("T1-source-v3.1 fixes its mechanism and order")
        super().__init__(temporal_mode="stable_source_recurrent", **kwargs)


class T1StableShuffleRectifiedFlow(JointRectifiedFlowBase):
    """v3.1 negative control with shuffled feature or AR-source adjacency."""

    variant = "T1_stable_shuffle_v3_1"

    def __init__(
        self,
        *,
        temporal_hour_order: Sequence[int],
        shuffle_target: Literal["feature", "source"] = "source",
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        if "temporal_mode" in kwargs:
            raise ValueError("T1-stable-shuffle derives mode from shuffle_target")
        if shuffle_target not in ("feature", "source"):
            raise ValueError("shuffle_target must be 'feature' or 'source'")
        super().__init__(
            temporal_mode=(
                "stable_feature_recurrent_shuffled"
                if shuffle_target == "feature"
                else "stable_source_recurrent_shuffled"
            ),
            temporal_hour_order=temporal_hour_order,
            **kwargs,
        )
        self.variant = f"T1_{shuffle_target}_shuffle_v3_1"


# Short aliases are convenient in experiment registries while the long class
# names remain explicit in checkpoints and reports.
R0 = R0JointRectifiedFlow
T0 = T0MemorylessRectifiedFlow
T1Feature = T1FeatureRectifiedFlow
T1Source = T1SourceRectifiedFlow
T1Shuffle = T1ShuffleRectifiedFlow
T0V31 = T0StableSourceRectifiedFlow
T1FeatureV31 = T1FeatureStableSourceRectifiedFlow
T1SourceV31 = T1SourceStableRectifiedFlow
T1ShuffleV31 = T1StableShuffleRectifiedFlow
NWPEncoder = JointNWPEncoder


__all__ = [
    "AxialJointBlock",
    "FourierTimeEmbedding",
    "JointNWPEncoder",
    "JointRectifiedFlowBase",
    "JointTimeDomainVelocity",
    "NWPEncoder",
    "ParameterMatchedTemporalCell",
    "R0",
    "R0JointRectifiedFlow",
    "ScenarioBatch",
    "T0",
    "T0MemorylessRectifiedFlow",
    "T0StableSourceRectifiedFlow",
    "T0V31",
    "T1Feature",
    "T1FeatureRectifiedFlow",
    "T1FeatureStableSourceRectifiedFlow",
    "T1FeatureV31",
    "T1Shuffle",
    "T1ShuffleRectifiedFlow",
    "T1ShuffleV31",
    "T1StableShuffleRectifiedFlow",
    "T1Source",
    "T1SourceRectifiedFlow",
    "T1SourceStableRectifiedFlow",
    "T1SourceV31",
    "TemporalSourceAdapter",
    "VariancePreservingARSource",
]
