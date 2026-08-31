from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, milp

from repro.suc import RTS24, _Rows, rts24


@dataclass(frozen=True)
class FirstStagePlan:
    commitment: np.ndarray
    day_ahead_dispatch: np.ndarray
    reserve_up: np.ndarray
    reserve_down: np.ndarray
    reserve_shortage_up: np.ndarray
    reserve_shortage_down: np.ndarray

    def __post_init__(self) -> None:
        commitment = np.asarray(self.commitment)
        if commitment.ndim != 2 or commitment.shape[1] != 24:
            raise ValueError("commitment must have shape [unit,24]")
        units = commitment.shape[0]
        for name in ("day_ahead_dispatch", "reserve_up", "reserve_down"):
            if np.asarray(getattr(self, name)).shape != (units, 24):
                raise ValueError(f"{name} must have shape [unit,24]")
        for name in ("reserve_shortage_up", "reserve_shortage_down"):
            if np.asarray(getattr(self, name)).shape != (24,):
                raise ValueError(f"{name} must have shape [24]")


@dataclass(frozen=True)
class SUCResult:
    status: str
    success: bool
    first_stage: FirstStagePlan
    scenario_dispatch: np.ndarray
    used_wind: np.ndarray
    load_shedding: float
    wind_curtailment: float
    reserve_shortage: float
    startup_cost: float
    energy_cost: float
    reserve_capacity_cost: float
    reserve_shortage_cost: float
    penalty_cost: float
    total_cost: float
    solve_time_seconds: float
    mip_gap: float
    mip_dual_bound: float
    mip_node_count: int


def _add_ramp_rows(
    rows: _Rows,
    power_indexer,
    start_indexer,
    stop_indexer,
    unit: int,
    hour: int,
    generator,
) -> None:
    if hour == 0:
        rows.add(
            [
                (power_indexer(unit, hour), 1),
                (start_indexer(unit, hour), -generator.maximum),
            ],
            -np.inf,
            generator.initial_power + generator.ramp_up,
        )
        rows.add(
            [
                (power_indexer(unit, hour), -1),
                (stop_indexer(unit, hour), -generator.maximum),
            ],
            -np.inf,
            generator.ramp_down - generator.initial_power,
        )
    else:
        rows.add(
            [
                (power_indexer(unit, hour), 1),
                (power_indexer(unit, hour - 1), -1),
                (start_indexer(unit, hour), -generator.maximum),
            ],
            -np.inf,
            generator.ramp_up,
        )
        rows.add(
            [
                (power_indexer(unit, hour - 1), 1),
                (power_indexer(unit, hour), -1),
                (stop_indexer(unit, hour), -generator.maximum),
            ],
            -np.inf,
            generator.ramp_down,
        )


def solve_two_stage_suc(
    wind_by_farm: np.ndarray,
    probabilities: np.ndarray,
    *,
    system: RTS24 | None = None,
    fixed_commitment: np.ndarray | None = None,
    fixed_first_stage: FirstStagePlan | None = None,
    shedding_penalty: float = 1000.0,
    curtailment_penalty: float = 80.0,
    reserve_shortage_penalty: float | None = None,
    reserve_up_fraction: float = 0.10,
    reserve_down_fraction: float = 0.05,
    reserve_up_cost_fraction: float = 0.10,
    reserve_down_cost_fraction: float = 0.05,
    mip_gap: float = 0.001,
    time_limit: float = 600.0,
) -> SUCResult:
    """Exact binary two-stage SUC with day-ahead dispatch and reserves."""

    grid = rts24() if system is None else system
    wind = np.asarray(wind_by_farm, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if wind.ndim != 3 or wind.shape[1:] != (len(grid.wind_buses), 24):
        raise ValueError("wind_by_farm must have shape [scenario,farm,24]")
    if not np.isfinite(wind).all() or np.any(wind < 0.0):
        raise ValueError("wind_by_farm must be finite and non-negative")
    if probability.shape != (len(wind),) or np.any(probability < 0.0):
        raise ValueError("probabilities must align with scenarios and be non-negative")
    if not np.isfinite(probability).all() or probability.sum() <= 0.0:
        raise ValueError("probabilities must have positive finite mass")
    fractions = (
        reserve_up_fraction,
        reserve_down_fraction,
        reserve_up_cost_fraction,
        reserve_down_cost_fraction,
    )
    if any(value < 0.0 for value in fractions):
        raise ValueError("reserve fractions must be non-negative")
    if fixed_commitment is not None and fixed_first_stage is not None:
        raise ValueError("provide fixed_commitment or fixed_first_stage, not both")
    probability = probability / probability.sum()
    reserve_shortage_penalty = (
        0.5 * shedding_penalty
        if reserve_shortage_penalty is None
        else float(reserve_shortage_penalty)
    )
    if reserve_shortage_penalty < 0.0:
        raise ValueError("reserve_shortage_penalty must be non-negative")

    scenarios, farms, periods = wind.shape
    units, nodes = len(grid.generators), 24
    offset_u = 0
    offset_start = offset_u + units * periods
    offset_stop = offset_start + units * periods
    offset_pda = offset_stop + units * periods
    offset_rup = offset_pda + units * periods
    offset_rdn = offset_rup + units * periods
    offset_rshort_up = offset_rdn + units * periods
    offset_rshort_down = offset_rshort_up + periods
    offset_p = offset_rshort_down + periods
    offset_wind = offset_p + scenarios * units * periods
    offset_shed = offset_wind + scenarios * farms * periods
    offset_theta = offset_shed + scenarios * nodes * periods
    variables = offset_theta + scenarios * nodes * periods

    def u(i: int, t: int) -> int:
        return offset_u + i * periods + t

    def start(i: int, t: int) -> int:
        return offset_start + i * periods + t

    def stop(i: int, t: int) -> int:
        return offset_stop + i * periods + t

    def pda(i: int, t: int) -> int:
        return offset_pda + i * periods + t

    def rup(i: int, t: int) -> int:
        return offset_rup + i * periods + t

    def rdn(i: int, t: int) -> int:
        return offset_rdn + i * periods + t

    def rshort_up(t: int) -> int:
        return offset_rshort_up + t

    def rshort_down(t: int) -> int:
        return offset_rshort_down + t

    def dispatch(s: int, i: int, t: int) -> int:
        return offset_p + (s * units + i) * periods + t

    def used_wind(s: int, j: int, t: int) -> int:
        return offset_wind + (s * farms + j) * periods + t

    def shed(s: int, n: int, t: int) -> int:
        return offset_shed + (s * nodes + n) * periods + t

    def theta(s: int, n: int, t: int) -> int:
        return offset_theta + (s * nodes + n) * periods + t

    lower = np.zeros(variables)
    upper = np.full(variables, np.inf)
    upper[offset_u:offset_pda] = 1.0
    lower[offset_theta:] = -np.pi
    upper[offset_theta:] = np.pi
    for s in range(scenarios):
        for t in range(periods):
            lower[theta(s, 12, t)] = upper[theta(s, 12, t)] = 0.0
            for j in range(farms):
                upper[used_wind(s, j, t)] = wind[s, j, t]
            for n in range(nodes):
                upper[shed(s, n, t)] = grid.load[t] * grid.load_share[n]

    if fixed_commitment is not None:
        fixed = np.asarray(fixed_commitment, dtype=np.float64)
        if fixed.shape != (units, periods):
            raise ValueError(f"fixed_commitment must have shape [{units},24]")
        for i in range(units):
            for t in range(periods):
                lower[u(i, t)] = upper[u(i, t)] = fixed[i, t]
    if fixed_first_stage is not None:
        if fixed_first_stage.commitment.shape != (units, periods):
            raise ValueError("fixed_first_stage has the wrong unit count")
        for array, indexer in (
            (fixed_first_stage.commitment, u),
            (fixed_first_stage.day_ahead_dispatch, pda),
            (fixed_first_stage.reserve_up, rup),
            (fixed_first_stage.reserve_down, rdn),
        ):
            values = np.asarray(array, dtype=np.float64)
            if values.shape != (units, periods) or not np.isfinite(values).all():
                raise ValueError("fixed_first_stage unit arrays do not align")
            for i in range(units):
                for t in range(periods):
                    lower[indexer(i, t)] = upper[indexer(i, t)] = values[i, t]
        for values, indexer in (
            (fixed_first_stage.reserve_shortage_up, rshort_up),
            (fixed_first_stage.reserve_shortage_down, rshort_down),
        ):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (periods,) or not np.isfinite(values).all():
                raise ValueError("fixed_first_stage shortage arrays do not align")
            for t in range(periods):
                lower[indexer(t)] = upper[indexer(t)] = values[t]

    objective = np.zeros(variables)
    for i, generator in enumerate(grid.generators):
        for t in range(periods):
            objective[start(i, t)] = generator.startup_cost
            objective[rup(i, t)] = reserve_up_cost_fraction * generator.energy_cost
            objective[rdn(i, t)] = reserve_down_cost_fraction * generator.energy_cost
    objective[offset_rshort_up:offset_p] = reserve_shortage_penalty
    for s, scenario_probability in enumerate(probability):
        for t in range(periods):
            for i, generator in enumerate(grid.generators):
                objective[dispatch(s, i, t)] = (
                    scenario_probability * generator.energy_cost
                )
            for j in range(farms):
                objective[used_wind(s, j, t)] = (
                    -scenario_probability * curtailment_penalty
                )
            for n in range(nodes):
                objective[shed(s, n, t)] = (
                    scenario_probability * shedding_penalty
                )

    rows = _Rows(variables)
    for i, generator in enumerate(grid.generators):
        for t in range(periods):
            transition = [(u(i, t), 1), (start(i, t), -1), (stop(i, t), 1)]
            if t:
                transition.append((u(i, t - 1), -1))
                rhs = 0
            else:
                rhs = generator.initial_on
            rows.add(transition, rhs, rhs)
            begin_up = max(0, t - generator.minimum_up + 1)
            rows.add(
                [(start(i, k), 1) for k in range(begin_up, t + 1)]
                + [(u(i, t), -1)],
                -np.inf,
                0,
            )
            begin_down = max(0, t - generator.minimum_down + 1)
            rows.add(
                [(stop(i, k), 1) for k in range(begin_down, t + 1)]
                + [(u(i, t), 1)],
                -np.inf,
                1,
            )
            rows.add(
                [(pda(i, t), 1), (u(i, t), -generator.maximum)],
                -np.inf,
                0,
            )
            rows.add(
                [(pda(i, t), -1), (u(i, t), generator.minimum)],
                -np.inf,
                0,
            )
            rows.add(
                [
                    (pda(i, t), 1),
                    (rup(i, t), 1),
                    (u(i, t), -generator.maximum),
                ],
                -np.inf,
                0,
            )
            rows.add(
                [
                    (pda(i, t), 1),
                    (rdn(i, t), -1),
                    (u(i, t), -generator.minimum),
                ],
                0,
                np.inf,
            )
            _add_ramp_rows(rows, pda, start, stop, i, t, generator)
        for s in range(scenarios):
            scenario_power = lambda unit, hour, scenario=s: dispatch(
                scenario, unit, hour
            )
            for t in range(periods):
                rows.add(
                    [(dispatch(s, i, t), 1), (u(i, t), -generator.maximum)],
                    -np.inf,
                    0,
                )
                rows.add(
                    [(dispatch(s, i, t), -1), (u(i, t), generator.minimum)],
                    -np.inf,
                    0,
                )
                rows.add(
                    [
                        (dispatch(s, i, t), 1),
                        (pda(i, t), -1),
                        (rup(i, t), -1),
                    ],
                    -np.inf,
                    0,
                )
                rows.add(
                    [
                        (pda(i, t), 1),
                        (dispatch(s, i, t), -1),
                        (rdn(i, t), -1),
                    ],
                    -np.inf,
                    0,
                )
                _add_ramp_rows(
                    rows, scenario_power, start, stop, i, t, generator
                )
    for t in range(periods):
        rows.add(
            [(rup(i, t), 1) for i in range(units)] + [(rshort_up(t), 1)],
            reserve_up_fraction * grid.load[t],
            np.inf,
        )
        rows.add(
            [(rdn(i, t), 1) for i in range(units)] + [(rshort_down(t), 1)],
            reserve_down_fraction * grid.load[t],
            np.inf,
        )

    generators_at = {node: [] for node in range(1, nodes + 1)}
    winds_at = {node: [] for node in range(1, nodes + 1)}
    for i, generator in enumerate(grid.generators):
        generators_at[generator.bus].append(i)
    for j, bus in enumerate(grid.wind_buses):
        winds_at[bus].append(j)
    for s in range(scenarios):
        for t in range(periods):
            for node in range(1, nodes + 1):
                terms = [
                    (dispatch(s, i, t), 1) for i in generators_at[node]
                ]
                terms += [
                    (used_wind(s, j, t), 1) for j in winds_at[node]
                ]
                terms.append((shed(s, node - 1, t), 1))
                for branch in grid.branches:
                    coefficient = grid.base_mva / branch.reactance
                    if branch.source == node:
                        terms += [
                            (theta(s, branch.source - 1, t), -coefficient),
                            (theta(s, branch.target - 1, t), coefficient),
                        ]
                    elif branch.target == node:
                        terms += [
                            (theta(s, branch.source - 1, t), coefficient),
                            (theta(s, branch.target - 1, t), -coefficient),
                        ]
                demand = grid.load[t] * grid.load_share[node - 1]
                rows.add(terms, demand, demand)
            for branch in grid.branches:
                coefficient = grid.base_mva / branch.reactance
                rows.add(
                    [
                        (theta(s, branch.source - 1, t), coefficient),
                        (theta(s, branch.target - 1, t), -coefficient),
                    ],
                    -branch.capacity,
                    branch.capacity,
                )

    integrality = np.zeros(variables, dtype=np.uint8)
    integrality[offset_u:offset_pda] = 1
    started = perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=rows.constraint(),
        options={"mip_rel_gap": mip_gap, "time_limit": time_limit},
    )
    elapsed = perf_counter() - started
    if result.x is None:
        raise RuntimeError(f"two-stage SUC failed: {result.message}")

    solution = result.x
    commitment = solution[offset_u:offset_start].reshape(units, periods).round()
    day_ahead = solution[offset_pda:offset_rup].reshape(units, periods)
    reserve_up = solution[offset_rup:offset_rdn].reshape(units, periods)
    reserve_down = solution[offset_rdn:offset_rshort_up].reshape(units, periods)
    short_up = solution[offset_rshort_up:offset_rshort_down]
    short_down = solution[offset_rshort_down:offset_p]
    dispatch_value = solution[offset_p:offset_wind].reshape(
        scenarios, units, periods
    )
    used_value = solution[offset_wind:offset_shed].reshape(
        scenarios, farms, periods
    )
    shedding = solution[offset_shed:offset_theta].reshape(
        scenarios, nodes, periods
    ).sum(axis=1)
    startup_cost = float(
        sum(
            grid.generators[i].startup_cost * solution[start(i, t)]
            for i in range(units)
            for t in range(periods)
        )
    )
    energy_cost = float(
        sum(
            probability[s]
            * grid.generators[i].energy_cost
            * dispatch_value[s, i].sum()
            for s in range(scenarios)
            for i in range(units)
        )
    )
    reserve_capacity_cost = float(
        sum(
            reserve_up_cost_fraction
            * grid.generators[i].energy_cost
            * reserve_up[i, t]
            + reserve_down_cost_fraction
            * grid.generators[i].energy_cost
            * reserve_down[i, t]
            for i in range(units)
            for t in range(periods)
        )
    )
    reserve_shortage = float(short_up.sum() + short_down.sum())
    reserve_shortage_cost = reserve_shortage_penalty * reserve_shortage
    expected_shedding = float(np.sum(probability[:, None] * shedding))
    curtailment = wind.sum(axis=1) - used_value.sum(axis=1)
    expected_curtailment = float(np.sum(probability[:, None] * curtailment))
    penalty_cost = (
        shedding_penalty * expected_shedding
        + curtailment_penalty * expected_curtailment
    )
    total_cost = (
        startup_cost
        + energy_cost
        + reserve_capacity_cost
        + reserve_shortage_cost
        + penalty_cost
    )
    constant = float(
        curtailment_penalty
        * np.sum(probability[:, None] * wind.sum(axis=1))
    )
    raw_bound = float(getattr(result, "mip_dual_bound", np.nan))
    return SUCResult(
        status=str(result.message),
        success=bool(result.success),
        first_stage=FirstStagePlan(
            commitment=commitment,
            day_ahead_dispatch=day_ahead,
            reserve_up=reserve_up,
            reserve_down=reserve_down,
            reserve_shortage_up=short_up,
            reserve_shortage_down=short_down,
        ),
        scenario_dispatch=dispatch_value,
        used_wind=used_value,
        load_shedding=expected_shedding,
        wind_curtailment=expected_curtailment,
        reserve_shortage=reserve_shortage,
        startup_cost=startup_cost,
        energy_cost=energy_cost,
        reserve_capacity_cost=reserve_capacity_cost,
        reserve_shortage_cost=reserve_shortage_cost,
        penalty_cost=float(penalty_cost),
        total_cost=float(total_cost),
        solve_time_seconds=float(elapsed),
        mip_gap=float(getattr(result, "mip_gap", np.nan)),
        mip_dual_bound=raw_bound + constant if np.isfinite(raw_bound) else np.nan,
        mip_node_count=int(getattr(result, "mip_node_count", -1)),
    )


def evaluate_realized(
    first_stage: FirstStagePlan,
    observed_wind_by_farm: np.ndarray,
    *,
    system: RTS24 | None = None,
    **solver_options: float,
) -> SUCResult:
    """Evaluate one observed day while fixing every day-ahead decision."""

    observed = np.asarray(observed_wind_by_farm, dtype=np.float64)
    grid = rts24() if system is None else system
    if observed.shape != (len(grid.wind_buses), 24):
        raise ValueError("observed_wind_by_farm must have shape [farm,24]")
    return solve_two_stage_suc(
        observed[None],
        np.ones(1),
        system=grid,
        fixed_first_stage=first_stage,
        **solver_options,
    )


__all__ = [
    "FirstStagePlan",
    "SUCResult",
    "evaluate_realized",
    "solve_two_stage_suc",
]
