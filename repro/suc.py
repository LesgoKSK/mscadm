from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class Generator:
    name: str
    bus: int
    minimum: float
    maximum: float
    ramp_up: float
    ramp_down: float
    minimum_up: int
    minimum_down: int
    energy_cost: float
    startup_cost: float
    initial_power: float
    initial_on: int


@dataclass(frozen=True)
class Branch:
    source: int
    target: int
    reactance: float
    capacity: float


@dataclass(frozen=True)
class RTS24:
    generators: tuple[Generator, ...]
    branches: tuple[Branch, ...]
    load: np.ndarray
    load_share: np.ndarray
    wind_buses: tuple[int, ...]
    base_mva: float = 100.0


def rts24() -> RTS24:
    buses = [1, 2, 7, 13, 15, 15, 16, 18, 21, 22, 23, 23]
    names = ["Unit1", "Unit2", "Unit7", "Unit13", "Unit15a", "Unit15b", "Unit16", "Unit18", "Unit21", "Unit22", "Unit23a", "Unit23b"]
    maximum = [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350]
    minimum = [30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140]
    ramp = [120, 120, 350, 240, 60, 155, 155, 280, 280, 300, 180, 240]
    up = [8, 8, 8, 12, 4, 8, 8, 1, 1, 0, 8, 8]
    down = [4, 4, 8, 10, 2, 8, 8, 1, 1, 0, 8, 8]
    energy = [13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0, 10.52, 10.89]
    startup = [1430.4, 1430.4, 1725, 3056.7, 437, 312, 312, 0, 0, 0, 624, 2298]
    initial_power = [76, 76, 0, 0, 0, 0, 124, 240, 240, 240, 248, 280]
    initial_on = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    generators = tuple(
        Generator(n, b, pmin, pmax, ru, ru, mu, md, c, su, p0, on)
        for n, b, pmin, pmax, ru, mu, md, c, su, p0, on in zip(
            names, buses, minimum, maximum, ramp, up, down, energy, startup, initial_power, initial_on
        )
    )
    raw_branches = [
        (1,2,.0146,175),(1,3,.2253,175),(1,5,.0907,350),(2,4,.1356,175),(2,6,.205,175),
        (3,9,.1271,175),(3,24,.084,400),(4,9,.111,175),(5,10,.094,350),(6,10,.0642,175),
        (7,8,.0652,350),(8,9,.1762,175),(8,10,.1762,175),(9,11,.084,400),(9,12,.084,400),
        (10,11,.084,400),(10,12,.084,400),(11,13,.0488,500),(11,14,.0426,500),
        (12,13,.0488,500),(12,23,.0985,500),(13,23,.0884,250),(14,16,.0594,250),
        (15,16,.0172,500),(15,21,.0249,400),(15,24,.0529,500),(16,17,.0263,500),
        (16,19,.0234,500),(17,18,.0143,500),(17,22,.1069,500),(18,21,.0132,1000),
        (19,20,.0203,1000),(20,23,.0112,1000),(21,22,.0692,500),
    ]
    branches = tuple(Branch(*value) for value in raw_branches)
    load = np.asarray([
        1775.835,1669.815,1590.3,1563.795,1563.795,1590.3,1961.37,2279.43,
        2517.975,2544.48,2544.48,2517.975,2517.975,2517.975,2464.965,2464.965,
        2623.995,2650.5,2650.5,2544.48,2411.955,2199.915,1934.865,1669.815,
    ])
    shares = np.zeros(24)
    for bus, percentage in zip(
        [1,2,3,4,5,6,7,8,9,10,13,14,15,16,18,19,20],
        [3.8,3.4,6.3,2.6,2.5,4.8,4.4,6.0,6.1,6.8,9.3,6.8,11.1,3.5,11.7,6.4,4.5],
    ):
        shares[bus - 1] = percentage / 100
    return RTS24(generators, branches, load, shares, (3, 5, 7, 16, 21, 23))


def reduce_scenarios(
    scenarios: np.ndarray, clusters: int, *, capacity: float = 1200.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if scenarios.ndim != 2 or scenarios.shape[1] != 24:
        raise ValueError("scenarios must have shape [scenario, 24]")
    if not 1 <= clusters <= len(scenarios):
        raise ValueError("clusters must be between one and the scenario count")
    fitted = KMeans(n_clusters=clusters, random_state=seed, n_init=20).fit(scenarios)
    probabilities = np.bincount(fitted.labels_, minlength=clusters) / len(scenarios)
    return np.clip(fitted.cluster_centers_, 0, 1) * capacity, probabilities


class _Rows:
    def __init__(self, variables: int) -> None:
        self.variables = variables
        self.row: list[int] = []
        self.column: list[int] = []
        self.value: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, terms: list[tuple[int, float]], lower: float, upper: float) -> None:
        index = len(self.lower)
        for column, value in terms:
            if value:
                self.row.append(index); self.column.append(column); self.value.append(value)
        self.lower.append(lower); self.upper.append(upper)

    def constraint(self) -> LinearConstraint:
        matrix = coo_matrix((self.value, (self.row, self.column)), shape=(len(self.lower), self.variables)).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def solve_suc(
    wind_total: np.ndarray,
    probabilities: np.ndarray,
    *,
    system: RTS24 | None = None,
    fixed_commitment: np.ndarray | None = None,
    shedding_penalty: float = 1000.0,
    curtailment_penalty: float = 80.0,
    mip_gap: float = 0.01,
    time_limit: float = 300.0,
) -> dict[str, np.ndarray | float | str]:
    grid = rts24() if system is None else system
    wind_total = np.asarray(wind_total, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if wind_total.ndim != 2 or wind_total.shape[1] != 24:
        raise ValueError("wind_total must be [scenarios, 24]")
    probabilities = probabilities / probabilities.sum()
    scenarios, periods = wind_total.shape
    units, nodes, farms = len(grid.generators), 24, len(grid.wind_buses)
    offset_u = 0
    offset_start = offset_u + units * periods
    offset_stop = offset_start + units * periods
    offset_p = offset_stop + units * periods
    offset_wind = offset_p + scenarios * units * periods
    offset_shed = offset_wind + scenarios * farms * periods
    offset_theta = offset_shed + scenarios * nodes * periods
    variables = offset_theta + scenarios * nodes * periods
    u = lambda i,t: offset_u + i*periods+t
    start = lambda i,t: offset_start + i*periods+t
    stop = lambda i,t: offset_stop + i*periods+t
    p = lambda s,i,t: offset_p + (s*units+i)*periods+t
    w = lambda s,j,t: offset_wind + (s*farms+j)*periods+t
    shed = lambda s,n,t: offset_shed + (s*nodes+n)*periods+t
    theta = lambda s,n,t: offset_theta + (s*nodes+n)*periods+t
    lower = np.zeros(variables); upper = np.full(variables, np.inf)
    upper[offset_u:offset_p] = 1
    lower[offset_theta:] = -np.pi; upper[offset_theta:] = np.pi
    for s in range(scenarios):
        for t in range(periods):
            lower[theta(s, 12, t)] = upper[theta(s, 12, t)] = 0
            for j in range(farms):
                upper[w(s,j,t)] = wind_total[s,t] / farms
            for n in range(nodes):
                upper[shed(s,n,t)] = grid.load[t] * grid.load_share[n]
    if fixed_commitment is not None:
        fixed = np.asarray(fixed_commitment, dtype=float)
        if fixed.shape != (units, periods):
            raise ValueError("fixed_commitment must have shape [12, 24]")
        for i in range(units):
            for t in range(periods):
                lower[u(i,t)] = upper[u(i,t)] = fixed[i,t]
    objective = np.zeros(variables)
    for i, generator in enumerate(grid.generators):
        for t in range(periods):
            objective[start(i,t)] = generator.startup_cost
    for s, probability in enumerate(probabilities):
        for t in range(periods):
            for i, generator in enumerate(grid.generators):
                objective[p(s,i,t)] = probability * generator.energy_cost
            for j in range(farms):
                objective[w(s,j,t)] = -probability * curtailment_penalty
            for n in range(nodes):
                objective[shed(s,n,t)] = probability * shedding_penalty
    rows = _Rows(variables)
    for i, generator in enumerate(grid.generators):
        for t in range(periods):
            previous_on = generator.initial_on if t == 0 else None
            transition = [(u(i,t),1),(start(i,t),-1),(stop(i,t),1)]
            if t:
                transition.append((u(i,t-1),-1)); rhs = 0
            else:
                rhs = previous_on
            rows.add(transition, rhs, rhs)
            begin_up = max(0, t-generator.minimum_up+1)
            rows.add([(start(i,k),1) for k in range(begin_up,t+1)] + [(u(i,t),-1)], -np.inf, 0)
            begin_down = max(0, t-generator.minimum_down+1)
            rows.add([(stop(i,k),1) for k in range(begin_down,t+1)] + [(u(i,t),1)], -np.inf, 1)
        for s in range(scenarios):
            for t in range(periods):
                rows.add([(p(s,i,t),1),(u(i,t),-generator.maximum)], -np.inf, 0)
                rows.add([(p(s,i,t),-1),(u(i,t),generator.minimum)], -np.inf, 0)
                if t == 0:
                    rows.add([(p(s,i,t),1),(start(i,t),-generator.maximum)], -np.inf, generator.initial_power + generator.ramp_up)
                    rows.add([(p(s,i,t),-1),(stop(i,t),-generator.maximum)], -np.inf, generator.ramp_down-generator.initial_power)
                else:
                    rows.add([(p(s,i,t),1),(p(s,i,t-1),-1),(start(i,t),-generator.maximum)], -np.inf, generator.ramp_up)
                    rows.add([(p(s,i,t-1),1),(p(s,i,t),-1),(stop(i,t),-generator.maximum)], -np.inf, generator.ramp_down)
    generators_at = {node: [] for node in range(1, nodes+1)}
    winds_at = {node: [] for node in range(1, nodes+1)}
    for i, generator in enumerate(grid.generators): generators_at[generator.bus].append(i)
    for j, bus in enumerate(grid.wind_buses): winds_at[bus].append(j)
    for s in range(scenarios):
        for t in range(periods):
            for node in range(1, nodes+1):
                terms = [(p(s,i,t),1) for i in generators_at[node]]
                terms += [(w(s,j,t),1) for j in winds_at[node]]
                terms.append((shed(s,node-1,t),1))
                for branch in grid.branches:
                    coefficient = grid.base_mva / branch.reactance
                    if branch.source == node:
                        terms += [(theta(s,branch.source-1,t),-coefficient),(theta(s,branch.target-1,t),coefficient)]
                    elif branch.target == node:
                        terms += [(theta(s,branch.source-1,t),coefficient),(theta(s,branch.target-1,t),-coefficient)]
                demand = grid.load[t] * grid.load_share[node-1]
                rows.add(terms, demand, demand)
            for branch in grid.branches:
                coefficient = grid.base_mva / branch.reactance
                rows.add([(theta(s,branch.source-1,t),coefficient),(theta(s,branch.target-1,t),-coefficient)], -branch.capacity, branch.capacity)
    integrality = np.zeros(variables, dtype=np.uint8); integrality[offset_u:offset_p] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=rows.constraint(),
        options={"mip_rel_gap": mip_gap, "time_limit": time_limit},
    )
    if result.x is None:
        raise RuntimeError(f"SUC failed: {result.message}")
    commitment = result.x[offset_u:offset_start].reshape(units, periods).round()
    dispatch = result.x[offset_p:offset_wind].reshape(scenarios, units, periods)
    used_wind = result.x[offset_wind:offset_shed].reshape(scenarios, farms, periods).sum(axis=1)
    shedding = result.x[offset_shed:offset_theta].reshape(scenarios, nodes, periods).sum(axis=1)
    startup_cost = sum(grid.generators[i].startup_cost * result.x[start(i,t)] for i in range(units) for t in range(periods))
    energy_cost = sum(probabilities[s] * grid.generators[i].energy_cost * dispatch[s,i].sum() for s in range(scenarios) for i in range(units))
    expected_shedding = float(np.sum(probabilities[:,None] * shedding))
    curtailment = wind_total - used_wind
    expected_curtailment = float(np.sum(probabilities[:,None] * curtailment))
    penalty = shedding_penalty * expected_shedding + curtailment_penalty * expected_curtailment
    return {
        "status": str(result.message),
        "commitment": commitment,
        "dispatch": dispatch,
        "load_shedding": expected_shedding,
        "wind_curtailment": expected_curtailment,
        "startup_cost": float(startup_cost),
        "energy_cost": float(energy_cost),
        "penalty_cost": float(penalty),
        "total_cost": float(startup_cost + energy_cost + penalty),
    }


__all__ = ["Generator", "Branch", "RTS24", "rts24", "reduce_scenarios", "solve_suc"]
