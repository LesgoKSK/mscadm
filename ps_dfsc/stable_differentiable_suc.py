from __future__ import annotations

from dataclasses import replace

from repro.suc import RTS24, rts24

from .differentiable_suc import DifferentiableSUC


class StableDifferentiableSUC(DifferentiableSUC):
    """Numerically scaled version of the differentiable SUC.

    The optimization layer divides all monetary coefficients by ``cost_scale``
    before canonicalization while retaining physical MW units in constraints.
    Returned decisions therefore solve the same regularized problem, but diffcp
    sees coefficients of order one rather than order 1e3.
    """

    def __init__(
        self,
        *,
        scenarios: int = 20,
        system: RTS24 | None = None,
        cost_scale: float = 100_000.0,
        strong_convexity: float = 1e-4,
        shedding_penalty: float = 1000.0,
        curtailment_penalty: float = 80.0,
        reserve_shortage_penalty: float = 500.0,
        reserve_up_fraction: float = 0.10,
        reserve_down_fraction: float = 0.05,
        reserve_up_cost_fraction: float = 0.10,
        reserve_down_cost_fraction: float = 0.05,
    ) -> None:
        if cost_scale <= 0.0:
            raise ValueError("cost_scale must be positive")
        physical = rts24() if system is None else system
        scaled = replace(
            physical,
            generators=tuple(
                replace(generator, energy_cost=generator.energy_cost / cost_scale)
                for generator in physical.generators
            ),
        )
        super().__init__(
            scenarios=scenarios,
            system=scaled,
            strong_convexity=strong_convexity,
            shedding_penalty=shedding_penalty / cost_scale,
            curtailment_penalty=curtailment_penalty / cost_scale,
            reserve_shortage_penalty=reserve_shortage_penalty / cost_scale,
            reserve_up_fraction=reserve_up_fraction,
            reserve_down_fraction=reserve_down_fraction,
            reserve_up_cost_fraction=reserve_up_cost_fraction,
            reserve_down_cost_fraction=reserve_down_cost_fraction,
        )
        self.system = physical
        self.cost_scale = float(cost_scale)
        self.shedding_penalty = float(shedding_penalty)
        self.curtailment_penalty = float(curtailment_penalty)
        self.reserve_shortage_penalty = float(reserve_shortage_penalty)


def activate_stable_training_layer() -> None:
    """Make the public trainer use the scaled layer without API changes."""

    from . import training

    training.DifferentiableSUC = StableDifferentiableSUC


__all__ = ["StableDifferentiableSUC", "activate_stable_training_layer"]
