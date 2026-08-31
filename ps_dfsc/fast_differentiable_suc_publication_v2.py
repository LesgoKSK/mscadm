"""Numerically scaled strictly-convex QP for publication training.

The previous publication layer expressed the optimization variables in MW and
only divided them by 1200 inside the quadratic regularizer.  At the registered
1e-4 coefficient this leaves cone-program curvature near 1e-10 and can make
SCS report an inaccurate unbounded status.  This layer formulates every power
variable and constraint in per-unit and converts results back to MW at the
public interface.
"""

from __future__ import annotations

import numpy as np

from .differentiable_suc import PlanningOutput, RealizedOutput, _dependencies
from .fast_differentiable_suc import (
    FastDifferentiableSUC,
    stable_soft_cvar,
)


class ScaledPublicationDifferentiableSUC(FastDifferentiableSUC):
    """Registered QP with all continuous power decisions in per-unit."""

    power_scale: float = 1200.0

    @staticmethod
    def _square(cp, *variables):
        return sum(cp.sum_squares(value) for value in variables)

    def _constraints(
        self,
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
        available_wind,
    ):
        grid = self.system
        scale = self.power_scale
        maximum = (
            np.asarray([item.maximum for item in grid.generators])[:, None]
            / scale
        )
        minimum = (
            np.asarray([item.minimum for item in grid.generators])[:, None]
            / scale
        )
        ramp_up = (
            np.asarray([item.ramp_up for item in grid.generators])[:, None]
            / scale
        )
        ramp_down = (
            np.asarray([item.ramp_down for item in grid.generators])[:, None]
            / scale
        )
        initial = (
            np.asarray([item.initial_power for item in grid.generators])
            / scale
        )
        load = np.asarray(grid.load) / scale
        constraints = [
            pda >= cp.multiply(minimum, commitment),
            pda <= cp.multiply(maximum, commitment),
            pda + reserve_up <= cp.multiply(maximum, commitment),
            pda - reserve_down >= cp.multiply(minimum, commitment),
            wind_used <= available_wind,
            pda[:, 0] - initial
            <= ramp_up[:, 0] + cp.multiply(maximum[:, 0], startup[:, 0]),
            initial - pda[:, 0]
            <= ramp_down[:, 0] + cp.multiply(maximum[:, 0], shutdown[:, 0]),
            pda[:, 1:] - pda[:, :-1]
            <= ramp_up + cp.multiply(maximum, startup[:, 1:]),
            pda[:, :-1] - pda[:, 1:]
            <= ramp_down + cp.multiply(maximum, shutdown[:, 1:]),
        ]
        for scenario in range(generation.shape[0]):
            output = generation[scenario]
            constraints += [
                output >= cp.multiply(minimum, commitment),
                output <= cp.multiply(maximum, commitment),
                output - pda <= reserve_up,
                pda - output <= reserve_down,
                output[:, 0] - initial
                <= ramp_up[:, 0]
                + cp.multiply(maximum[:, 0], startup[:, 0]),
                initial - output[:, 0]
                <= ramp_down[:, 0]
                + cp.multiply(maximum[:, 0], shutdown[:, 0]),
                output[:, 1:] - output[:, :-1]
                <= ramp_up + cp.multiply(maximum, startup[:, 1:]),
                output[:, :-1] - output[:, 1:]
                <= ramp_down + cp.multiply(maximum, shutdown[:, 1:]),
                cp.sum(output, axis=0)
                + cp.sum(wind_used[scenario], axis=0)
                + shedding[scenario]
                == load,
            ]
        return constraints

    def _build_planning_layer(self):
        cp, CvxpyLayer = _dependencies()
        grid = self.system
        scenarios = self.scenarios
        units = len(grid.generators)
        farms = len(grid.wind_buses)
        periods = 24
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
        constraints = self._constraints(
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
        load = np.asarray(grid.load) / self.power_scale
        constraints += [
            cp.sum(reserve_up, axis=0) + short_up
            >= self.reserve_up_fraction * load,
            cp.sum(reserve_down, axis=0) + short_down
            >= self.reserve_down_fraction * load,
            shedding <= load[None],
        ]
        energy = np.asarray([item.energy_cost for item in grid.generators])
        scale = self.power_scale
        scenario_cost = (
            cp.sum(
                cp.multiply(
                    generation * scale,
                    energy[None, :, None],
                ),
                axis=(1, 2),
            )
            - self.curtailment_penalty
            * scale
            * cp.sum(wind_used, axis=(1, 2))
            + self.shedding_penalty
            * scale
            * cp.sum(shedding, axis=1)
        ) / self.cost_scale
        reserve_cost = (
            cp.sum(
                cp.multiply(
                    reserve_up * scale,
                    self.reserve_up_cost_fraction * energy[:, None],
                )
            )
            + cp.sum(
                cp.multiply(
                    reserve_down * scale,
                    self.reserve_down_cost_fraction * energy[:, None],
                )
            )
            + self.reserve_shortage_penalty
            * scale
            * cp.sum(short_up + short_down)
        ) / self.cost_scale
        regularizer = self.strong_convexity * self._square(
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
            raise RuntimeError("scaled publication planning relaxation is not DPP")
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
        units = len(grid.generators)
        farms = len(grid.wind_buses)
        periods = 24
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
        constraints = self._constraints(
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
        constraints.append(
            shedding <= np.asarray(grid.load)[None] / self.power_scale
        )
        energy = np.asarray([item.energy_cost for item in grid.generators])
        scale = self.power_scale
        objective = (
            cp.sum(
                cp.multiply(
                    generation * scale,
                    energy[None, :, None],
                )
            )
            - self.curtailment_penalty * scale * cp.sum(wind_used)
            + self.shedding_penalty * scale * cp.sum(shedding)
        ) / self.cost_scale + self.strong_convexity * self._square(
            cp, generation, wind_used, shedding
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("scaled publication realized relaxation is not DPP")
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

    @staticmethod
    def _solver_arguments(solver_args):
        return {
            "eps": 1e-4,
            "max_iters": 50_000,
            "normalize": True,
            "acceleration_lookback": 10,
            **({} if solver_args is None else solver_args),
        }

    def plan(
        self,
        wind,
        probability,
        commitment,
        startup,
        shutdown,
        *,
        solver_args: dict | None = None,
    ) -> PlanningOutput:
        values = self._planning_layer(
            wind / self.power_scale,
            probability,
            commitment,
            startup,
            shutdown,
            solver_args=self._solver_arguments(solver_args),
        )
        return PlanningOutput(
            *(value * self.power_scale for value in values)
        )

    def realize(
        self,
        observed_wind,
        commitment,
        startup,
        shutdown,
        day_ahead_dispatch,
        reserve_up,
        reserve_down,
        *,
        solver_args: dict | None = None,
    ) -> RealizedOutput:
        wind = (
            observed_wind[None]
            if observed_wind.ndim == 2
            else observed_wind
        )
        values = self._realized_layer(
            wind / self.power_scale,
            commitment,
            startup,
            shutdown,
            day_ahead_dispatch / self.power_scale,
            reserve_up / self.power_scale,
            reserve_down / self.power_scale,
            solver_args=self._solver_arguments(solver_args),
        )
        return RealizedOutput(
            *(value * self.power_scale for value in values)
        )


def activate_scaled_publication_training_layer() -> None:
    from . import training_v2

    training_v2.DifferentiableSUC = ScaledPublicationDifferentiableSUC
    training_v2.soft_cvar = stable_soft_cvar


__all__ = [
    "ScaledPublicationDifferentiableSUC",
    "activate_scaled_publication_training_layer",
]
