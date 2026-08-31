"""Strictly-convex copper-plate QP used by publication training."""

from __future__ import annotations

import numpy as np

from .differentiable_suc import PlanningOutput, RealizedOutput, _dependencies
from .fast_differentiable_suc import (
    FastDifferentiableSUC,
    stable_soft_cvar,
)


class PublicationDifferentiableSUC(FastDifferentiableSUC):
    """Add the registered quadratic term to every continuous decision."""

    @staticmethod
    def _normalized_square(cp, *variables):
        return sum(cp.sum_squares(value / 1200.0) for value in variables)

    def _build_planning_layer(self):
        cp, CvxpyLayer = _dependencies()
        grid = self.system
        scenarios = self.scenarios
        units, farms, periods = (
            len(grid.generators),
            len(grid.wind_buses),
            24,
        )
        wind = cp.Parameter((scenarios, farms, periods), nonneg=True)
        probability = cp.Parameter(scenarios, nonneg=True)
        commitment = cp.Parameter((units, periods), nonneg=True)
        startup = cp.Parameter((units, periods), nonneg=True)
        shutdown = cp.Parameter((units, periods), nonneg=True)
        pda = cp.Variable((units, periods), nonneg=True)
        reserve_up = cp.Variable((units, periods), nonneg=True)
        reserve_down = cp.Variable((units, periods), nonneg=True)
        short_up = cp.Variable(periods, nonneg=True)
        short_down = cp.Variable(periods, nonneg=True)
        generation = cp.Variable((scenarios, units, periods), nonneg=True)
        wind_used = cp.Variable((scenarios, farms, periods), nonneg=True)
        shedding = cp.Variable((scenarios, periods), nonneg=True)
        constraints = self._common_constraints(
            cp,
            pda,
            reserve_up,
            reserve_down,
            generation,
            wind_used,
            shedding,
            commitment,
            startup,
            shutdown,
            wind,
        )
        constraints += [
            cp.sum(reserve_up, axis=0) + short_up
            >= self.reserve_up_fraction * grid.load,
            cp.sum(reserve_down, axis=0) + short_down
            >= self.reserve_down_fraction * grid.load,
            shedding <= grid.load[None],
        ]
        energy = np.asarray([item.energy_cost for item in grid.generators])
        scenario_cost = (
            cp.sum(
                cp.multiply(generation, energy[None, :, None]), axis=(1, 2)
            )
            - self.curtailment_penalty * cp.sum(wind_used, axis=(1, 2))
            + self.shedding_penalty * cp.sum(shedding, axis=1)
        ) / self.cost_scale
        reserve_cost = (
            cp.sum(
                cp.multiply(
                    reserve_up,
                    self.reserve_up_cost_fraction * energy[:, None],
                )
            )
            + cp.sum(
                cp.multiply(
                    reserve_down,
                    self.reserve_down_cost_fraction * energy[:, None],
                )
            )
            + self.reserve_shortage_penalty * cp.sum(short_up + short_down)
        ) / self.cost_scale
        regularizer = self.strong_convexity * self._normalized_square(
            cp,
            pda,
            reserve_up,
            reserve_down,
            short_up,
            short_down,
            generation,
            wind_used,
            shedding,
        )
        problem = cp.Problem(
            cp.Minimize(
                probability @ scenario_cost + reserve_cost + regularizer
            ),
            constraints,
        )
        if not problem.is_dpp():
            raise RuntimeError("publication planning relaxation is not DPP")
        return CvxpyLayer(
            problem,
            parameters=[wind, probability, commitment, startup, shutdown],
            variables=[
                pda,
                reserve_up,
                reserve_down,
                short_up,
                short_down,
                generation,
                wind_used,
                shedding,
            ],
        )

    def _build_realized_layer(self):
        cp, CvxpyLayer = _dependencies()
        grid = self.system
        units, farms, periods = (
            len(grid.generators),
            len(grid.wind_buses),
            24,
        )
        wind = cp.Parameter((1, farms, periods), nonneg=True)
        commitment = cp.Parameter((units, periods), nonneg=True)
        startup = cp.Parameter((units, periods), nonneg=True)
        shutdown = cp.Parameter((units, periods), nonneg=True)
        pda = cp.Parameter((units, periods), nonneg=True)
        reserve_up = cp.Parameter((units, periods), nonneg=True)
        reserve_down = cp.Parameter((units, periods), nonneg=True)
        generation = cp.Variable((1, units, periods), nonneg=True)
        wind_used = cp.Variable((1, farms, periods), nonneg=True)
        shedding = cp.Variable((1, periods), nonneg=True)
        constraints = self._common_constraints(
            cp,
            pda,
            reserve_up,
            reserve_down,
            generation,
            wind_used,
            shedding,
            commitment,
            startup,
            shutdown,
            wind,
        )
        constraints.append(shedding <= grid.load[None])
        energy = np.asarray([item.energy_cost for item in grid.generators])
        objective = (
            cp.sum(cp.multiply(generation, energy[None, :, None]))
            - self.curtailment_penalty * cp.sum(wind_used)
            + self.shedding_penalty * cp.sum(shedding)
        ) / self.cost_scale + self.strong_convexity * self._normalized_square(
            cp, generation, wind_used, shedding
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("publication realized relaxation is not DPP")
        return CvxpyLayer(
            problem,
            parameters=[
                wind,
                commitment,
                startup,
                shutdown,
                pda,
                reserve_up,
                reserve_down,
            ],
            variables=[generation, wind_used, shedding],
        )


def activate_publication_training_layer() -> None:
    from . import training_v2

    training_v2.DifferentiableSUC = PublicationDifferentiableSUC
    training_v2.soft_cvar = stable_soft_cvar


__all__ = [
    "PublicationDifferentiableSUC",
    "activate_publication_training_layer",
]
