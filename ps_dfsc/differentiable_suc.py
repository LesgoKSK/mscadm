from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repro.suc import RTS24, rts24


class DifferentiableSUCUnavailable(RuntimeError):
    pass


def _dependencies():
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except ImportError as error:
        raise DifferentiableSUCUnavailable(
            "Differentiable SUC requires requirements-ps-dfsc.txt "
            "(cvxpy, cvxpylayers and diffcp)."
        ) from error
    return cp, CvxpyLayer


def commitment_transitions(commitment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(commitment, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 24:
        raise ValueError("commitment must have shape [unit,24]")
    grid = rts24()
    if len(value) != len(grid.generators):
        raise ValueError("commitment has the wrong unit count")
    previous = np.asarray([generator.initial_on for generator in grid.generators])
    previous = np.concatenate((previous[:, None], value[:, :-1]), axis=1)
    return np.maximum(value - previous, 0.0), np.maximum(previous - value, 0.0)


@dataclass(frozen=True)
class PlanningOutput:
    day_ahead_dispatch: object
    reserve_up: object
    reserve_down: object
    reserve_shortage_up: object
    reserve_shortage_down: object
    scenario_dispatch: object
    used_wind: object
    load_shedding: object


@dataclass(frozen=True)
class RealizedOutput:
    dispatch: object
    used_wind: object
    load_shedding: object


class DifferentiableSUC:
    """CVXPYLayer relaxation with exact commitment supplied as a parameter."""

    def __init__(
        self,
        *,
        scenarios: int = 20,
        system: RTS24 | None = None,
        strong_convexity: float = 1e-4,
        shedding_penalty: float = 1000.0,
        curtailment_penalty: float = 80.0,
        reserve_shortage_penalty: float = 500.0,
        reserve_up_fraction: float = 0.10,
        reserve_down_fraction: float = 0.05,
        reserve_up_cost_fraction: float = 0.10,
        reserve_down_cost_fraction: float = 0.05,
    ) -> None:
        if scenarios < 1:
            raise ValueError("scenarios must be positive")
        if strong_convexity <= 0.0:
            raise ValueError("strong_convexity must be positive")
        self.system = rts24() if system is None else system
        self.scenarios = scenarios
        self.strong_convexity = float(strong_convexity)
        self.shedding_penalty = float(shedding_penalty)
        self.curtailment_penalty = float(curtailment_penalty)
        self.reserve_shortage_penalty = float(reserve_shortage_penalty)
        self.reserve_up_fraction = float(reserve_up_fraction)
        self.reserve_down_fraction = float(reserve_down_fraction)
        self.reserve_up_cost_fraction = float(reserve_up_cost_fraction)
        self.reserve_down_cost_fraction = float(reserve_down_cost_fraction)
        self._planning_layer = self._build_planning_layer()
        self._realized_layer = self._build_realized_layer()

    def _network_constraints(self, cp, generation, wind_used, shedding, theta):
        grid = self.system
        scenarios = generation.shape[0]
        constraints = []
        generators_at = {
            node: [i for i, item in enumerate(grid.generators) if item.bus == node]
            for node in range(1, 25)
        }
        winds_at = {
            node: [j for j, bus in enumerate(grid.wind_buses) if bus == node]
            for node in range(1, 25)
        }
        for s in range(scenarios):
            constraints.append(theta[s, 12, :] == 0)
            for node in range(1, 25):
                balance = shedding[s, node - 1]
                if generators_at[node]:
                    balance = balance + cp.sum(
                        generation[s, generators_at[node]], axis=0
                    )
                if winds_at[node]:
                    balance = balance + cp.sum(
                        wind_used[s, winds_at[node]], axis=0
                    )
                flow = 0
                for branch in grid.branches:
                    coefficient = grid.base_mva / branch.reactance
                    if branch.source == node:
                        flow += coefficient * (
                            theta[s, branch.target - 1]
                            - theta[s, branch.source - 1]
                        )
                    elif branch.target == node:
                        flow += coefficient * (
                            theta[s, branch.source - 1]
                            - theta[s, branch.target - 1]
                        )
                constraints.append(
                    balance + flow == grid.load * grid.load_share[node - 1]
                )
            for branch in grid.branches:
                coefficient = grid.base_mva / branch.reactance
                line_flow = coefficient * (
                    theta[s, branch.source - 1]
                    - theta[s, branch.target - 1]
                )
                constraints += [
                    line_flow <= branch.capacity,
                    line_flow >= -branch.capacity,
                ]
        return constraints

    def _operating_constraints(
        self,
        cp,
        pda,
        reserve_up,
        reserve_down,
        scenario_dispatch,
        wind_used,
        shedding,
        commitment,
        startup,
        shutdown,
        wind,
        short_up,
        short_down,
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
            cp.sum(reserve_up, axis=0) + short_up
            >= self.reserve_up_fraction * grid.load,
            cp.sum(reserve_down, axis=0) + short_down
            >= self.reserve_down_fraction * grid.load,
            wind_used <= wind,
        ]
        constraints += [
            pda[:, 0] - initial
            <= ramp_up[:, 0] + cp.multiply(maximum[:, 0], startup[:, 0]),
            initial - pda[:, 0]
            <= ramp_down[:, 0] + cp.multiply(maximum[:, 0], shutdown[:, 0]),
            pda[:, 1:] - pda[:, :-1]
            <= ramp_up + cp.multiply(maximum, startup[:, 1:]),
            pda[:, :-1] - pda[:, 1:]
            <= ramp_down + cp.multiply(maximum, shutdown[:, 1:]),
        ]
        for scenario in range(scenario_dispatch.shape[0]):
            generation = scenario_dispatch[scenario]
            constraints += [
                generation >= cp.multiply(minimum, commitment),
                generation <= cp.multiply(maximum, commitment),
                generation - pda <= reserve_up,
                pda - generation <= reserve_down,
                generation[:, 0] - initial
                <= ramp_up[:, 0]
                + cp.multiply(maximum[:, 0], startup[:, 0]),
                initial - generation[:, 0]
                <= ramp_down[:, 0]
                + cp.multiply(maximum[:, 0], shutdown[:, 0]),
                generation[:, 1:] - generation[:, :-1]
                <= ramp_up + cp.multiply(maximum, startup[:, 1:]),
                generation[:, :-1] - generation[:, 1:]
                <= ramp_down + cp.multiply(maximum, shutdown[:, 1:]),
            ]
        load_limit = grid.load[None, :] * grid.load_share[:, None]
        constraints += [shedding <= load_limit[None]]
        return constraints

    def _build_planning_layer(self):
        cp, CvxpyLayer = _dependencies()
        grid = self.system
        scenarios = self.scenarios
        units, farms, nodes, periods = (
            len(grid.generators),
            len(grid.wind_buses),
            24,
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
        shedding = cp.Variable((scenarios, nodes, periods), nonneg=True)
        theta = cp.Variable((scenarios, nodes, periods))
        constraints = self._operating_constraints(
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
            short_up,
            short_down,
        )
        constraints += self._network_constraints(
            cp, generation, wind_used, shedding, theta
        )
        energy_cost = np.asarray([item.energy_cost for item in grid.generators])
        scenario_cost = (
            cp.sum(cp.multiply(generation, energy_cost[None, :, None]), axis=(1, 2))
            - self.curtailment_penalty * cp.sum(wind_used, axis=(1, 2))
            + self.shedding_penalty * cp.sum(shedding, axis=(1, 2))
        )
        reserve_cost = cp.sum(
            cp.multiply(
                reserve_up,
                self.reserve_up_cost_fraction * energy_cost[:, None],
            )
        ) + cp.sum(
            cp.multiply(
                reserve_down,
                self.reserve_down_cost_fraction * energy_cost[:, None],
            )
        )
        shortage_cost = self.reserve_shortage_penalty * cp.sum(
            short_up + short_down
        )
        regularizer = self.strong_convexity * (
            cp.sum_squares(pda / 1200.0)
            + cp.sum_squares(reserve_up / 1200.0)
            + cp.sum_squares(reserve_down / 1200.0)
            + cp.sum_squares(generation / 1200.0)
        )
        objective = cp.Minimize(
            probability @ scenario_cost + reserve_cost + shortage_cost + regularizer
        )
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("planning relaxation is not DPP")
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
        units, farms, nodes, periods = (
            len(grid.generators),
            len(grid.wind_buses),
            24,
            24,
        )
        observed_wind = cp.Parameter((1, farms, periods), nonneg=True)
        commitment = cp.Parameter((units, periods), nonneg=True)
        startup = cp.Parameter((units, periods), nonneg=True)
        shutdown = cp.Parameter((units, periods), nonneg=True)
        pda_parameter = cp.Parameter((units, periods), nonneg=True)
        reserve_up_parameter = cp.Parameter((units, periods), nonneg=True)
        reserve_down_parameter = cp.Parameter((units, periods), nonneg=True)
        generation = cp.Variable((1, units, periods), nonneg=True)
        wind_used = cp.Variable((1, farms, periods), nonneg=True)
        shedding = cp.Variable((1, nodes, periods), nonneg=True)
        theta = cp.Variable((1, nodes, periods))
        zero_short = cp.Constant(np.zeros(periods))
        constraints = self._operating_constraints(
            cp,
            pda_parameter,
            reserve_up_parameter,
            reserve_down_parameter,
            generation,
            wind_used,
            shedding,
            commitment,
            startup,
            shutdown,
            observed_wind,
            zero_short,
            zero_short,
        )
        constraints += self._network_constraints(
            cp, generation, wind_used, shedding, theta
        )
        energy_cost = np.asarray([item.energy_cost for item in grid.generators])
        objective = cp.Minimize(
            cp.sum(cp.multiply(generation, energy_cost[None, :, None]))
            - self.curtailment_penalty * cp.sum(wind_used)
            + self.shedding_penalty * cp.sum(shedding)
            + self.strong_convexity * cp.sum_squares(generation / 1200.0)
        )
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("realized relaxation is not DPP")
        return CvxpyLayer(
            problem,
            parameters=[
                observed_wind,
                commitment,
                startup,
                shutdown,
                pda_parameter,
                reserve_up_parameter,
                reserve_down_parameter,
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
        values = self._planning_layer(
            wind,
            probability,
            commitment,
            startup,
            shutdown,
            solver_args={} if solver_args is None else solver_args,
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
        values = self._realized_layer(
            observed_wind[None] if observed_wind.ndim == 2 else observed_wind,
            commitment,
            startup,
            shutdown,
            day_ahead_dispatch,
            reserve_up,
            reserve_down,
            solver_args={} if solver_args is None else solver_args,
        )
        return RealizedOutput(*values)


__all__ = [
    "DifferentiableSUC",
    "DifferentiableSUCUnavailable",
    "PlanningOutput",
    "RealizedOutput",
    "commitment_transitions",
]
