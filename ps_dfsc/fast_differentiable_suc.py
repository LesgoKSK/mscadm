from __future__ import annotations

import numpy as np

from .differentiable_suc import (
    DifferentiableSUC,
    PlanningOutput,
    RealizedOutput,
    _dependencies,
)


class FastDifferentiableSUC(DifferentiableSUC):
    """Copper-plate QP used only for training gradients.

    Exact planning, validation and confirmation continue to use the DC-network
    MILP in :mod:`ps_dfsc.exact_suc`. This layer deliberately trades network
    fidelity for tractable backpropagation and must be covered by the
    relaxed-vs-exact mismatch report.
    """

    cost_scale: float = 100_000.0

    def _common_constraints(
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
        maximum = np.asarray([item.maximum for item in grid.generators])[:, None]
        minimum = np.asarray([item.minimum for item in grid.generators])[:, None]
        ramp_up = np.asarray([item.ramp_up for item in grid.generators])[:, None]
        ramp_down = np.asarray([item.ramp_down for item in grid.generators])[:, None]
        initial = np.asarray([item.initial_power for item in grid.generators])
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
                == grid.load,
            ]
        return constraints

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
            cp.sum(cp.multiply(generation, energy[None, :, None]), axis=(1, 2))
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
        regularizer = self.strong_convexity * (
            cp.sum_squares(pda / 1200.0)
            + cp.sum_squares(reserve_up / 1200.0)
            + cp.sum_squares(reserve_down / 1200.0)
            + cp.sum_squares(generation / 1200.0)
        )
        problem = cp.Problem(
            cp.Minimize(probability @ scenario_cost + reserve_cost + regularizer),
            constraints,
        )
        if not problem.is_dpp():
            raise RuntimeError("fast planning relaxation is not DPP")
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
        ) / self.cost_scale + self.strong_convexity * cp.sum_squares(
            generation / 1200.0
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("fast realized relaxation is not DPP")
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
        arguments = {
            "eps": 1e-5,
            "max_iters": 10_000,
            **({} if solver_args is None else solver_args),
        }
        values = self._planning_layer(
            wind,
            probability,
            commitment,
            startup,
            shutdown,
            solver_args=arguments,
        )
        return PlanningOutput(*values)

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
        arguments = {
            "eps": 1e-5,
            "max_iters": 10_000,
            **({} if solver_args is None else solver_args),
        }
        values = self._realized_layer(
            observed_wind[None] if observed_wind.ndim == 2 else observed_wind,
            commitment,
            startup,
            shutdown,
            day_ahead_dispatch,
            reserve_up,
            reserve_down,
            solver_args=arguments,
        )
        return RealizedOutput(*values)


def stable_soft_cvar(cost, alpha: float = 0.90):
    import torch

    if cost.ndim != 1 or len(cost) < 1:
        raise ValueError("cost must be a non-empty vector")
    eta = torch.quantile(cost.detach(), alpha)
    scale = cost.detach().std(unbiased=False).clamp_min(1.0) * 0.02
    excess = torch.nn.functional.softplus((cost - eta) / scale) * scale
    return eta + excess.mean() / (1.0 - alpha)


def activate_fast_training_layer() -> None:
    from . import training_v2

    training_v2.DifferentiableSUC = FastDifferentiableSUC
    training_v2.soft_cvar = stable_soft_cvar


__all__ = [
    "FastDifferentiableSUC",
    "activate_fast_training_layer",
    "stable_soft_cvar",
]
