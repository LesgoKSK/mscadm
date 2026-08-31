from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .exact_suc import evaluate_realized, solve_two_stage_suc
from .selection import empirical_cvar
from .types import CalibratedDistribution


def run_exact_cases(
    distributions: list[CalibratedDistribution],
    observed_wind_by_farm: np.ndarray,
    *,
    outer: int,
    method: str,
    dates: np.ndarray | None = None,
    mip_gap: float = 0.001,
    time_limit: float = 600.0,
    solver_options: dict[str, float] | None = None,
) -> pd.DataFrame:
    truth = np.asarray(observed_wind_by_farm, dtype=np.float64)
    if truth.shape != (len(distributions), 6, 24):
        raise ValueError("observed_wind_by_farm must have shape [day,6,24]")
    if dates is None:
        date_labels = np.arange(len(distributions)).astype(str)
    else:
        if len(dates) != len(distributions):
            raise ValueError("dates do not align with distributions")
        date_labels = np.asarray(dates).astype("datetime64[D]").astype(str)
    options = {} if solver_options is None else dict(solver_options)
    options.update({"mip_gap": mip_gap, "time_limit": time_limit})
    rows: list[dict[str, object]] = []
    for day, distribution in enumerate(distributions):
        base = {
            "outer": int(outer),
            "date": date_labels[day],
            "method": method,
            "used_fallback": distribution.used_fallback,
            "fallback_reason": distribution.fallback_reason,
            "ESS": distribution.ess,
            "entropy": distribution.entropy,
            "transport_cost": distribution.transport_cost,
        }
        try:
            planned = solve_two_stage_suc(
                distribution.suc_scenarios,
                distribution.suc_probabilities,
                **options,
            )
            realized = evaluate_realized(
                planned.first_stage, truth[day], **options
            )
            oracle = solve_two_stage_suc(
                truth[day][None], np.ones(1), **options
            )
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "planned_total_cost": planned.total_cost,
                    "realized_total_cost": realized.total_cost,
                    "oracle_total_cost": oracle.total_cost,
                    "cost_regret": realized.total_cost - oracle.total_cost,
                    "load_shedding": realized.load_shedding,
                    "wind_curtailment": realized.wind_curtailment,
                    "reserve_shortage": realized.reserve_shortage,
                    "planned_solve_time": planned.solve_time_seconds,
                    "realized_solve_time": realized.solve_time_seconds,
                    "oracle_solve_time": oracle.solve_time_seconds,
                    "planned_mip_gap": planned.mip_gap,
                    "realized_mip_gap": realized.mip_gap,
                    "oracle_mip_gap": oracle.mip_gap,
                    "planned_dual_bound": planned.mip_dual_bound,
                    "realized_dual_bound": realized.mip_dual_bound,
                    "oracle_dual_bound": oracle.mip_dual_bound,
                }
            )
        except Exception as error:
            rows.append(
                {
                    **base,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return pd.DataFrame(rows)


def summarize_exact_cases(frame: pd.DataFrame) -> dict[str, float | int]:
    successful = frame.loc[frame["status"] == "ok"]
    result: dict[str, float | int] = {
        "cases": int(len(frame)),
        "successful_cases": int(len(successful)),
        "paired_success_rate": float(len(successful) / len(frame)) if len(frame) else 0.0,
    }
    if successful.empty:
        return result
    realized = successful["realized_total_cost"].to_numpy(np.float64)
    result.update(
        {
            "realized_total_cost_mean": float(realized.mean()),
            "cost_regret_mean": float(successful["cost_regret"].mean()),
            "CVaR90": empirical_cvar(realized, 0.90),
            "CVaR95": empirical_cvar(realized, 0.95),
            "load_shedding_mean": float(successful["load_shedding"].mean()),
            "wind_curtailment_mean": float(
                successful["wind_curtailment"].mean()
            ),
            "reserve_shortage_mean": float(
                successful["reserve_shortage"].mean()
            ),
            "planned_solve_time_mean": float(
                successful["planned_solve_time"].mean()
            ),
            "planned_mip_gap_max": float(successful["planned_mip_gap"].max()),
        }
    )
    return result


def relaxed_exact_mismatch(
    relaxed_commitment: np.ndarray,
    relaxed_dispatch: np.ndarray,
    relaxed_reserve_up: np.ndarray,
    relaxed_reserve_down: np.ndarray,
    exact_result,
    *,
    relaxed_objective: float,
) -> dict[str, float]:
    exact = exact_result.first_stage
    return {
        "commitment_hamming": float(
            np.mean(np.rint(relaxed_commitment) != exact.commitment)
        ),
        "day_ahead_dispatch_MAE": float(
            np.mean(np.abs(relaxed_dispatch - exact.day_ahead_dispatch))
        ),
        "reserve_up_MAE": float(
            np.mean(np.abs(relaxed_reserve_up - exact.reserve_up))
        ),
        "reserve_down_MAE": float(
            np.mean(np.abs(relaxed_reserve_down - exact.reserve_down))
        ),
        "planned_objective_relative_gap": float(
            (relaxed_objective - exact_result.total_cost)
            / max(abs(exact_result.total_cost), 1e-12)
        ),
    }


def stratified_paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    outer: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    block = np.asarray(outer)
    if left.shape != right.shape or left.shape != block.shape:
        raise ValueError("candidate, baseline and outer must align")
    rng = np.random.default_rng(seed)
    differences = []
    unique = np.unique(block)
    for _ in range(samples):
        pieces = []
        for value in unique:
            indices = np.flatnonzero(block == value)
            pieces.append(
                rng.choice(indices, size=len(indices), replace=True)
            )
        selected = np.concatenate(pieces)
        differences.append(float(np.mean(left[selected] - right[selected])))
    difference = np.asarray(differences)
    return {
        "mean_difference": float(np.mean(left - right)),
        "one_sided_95_upper": float(np.quantile(difference, 0.95)),
        "two_sided_95_lower": float(np.quantile(difference, 0.025)),
        "two_sided_95_upper": float(np.quantile(difference, 0.975)),
    }


__all__ = [
    "relaxed_exact_mismatch",
    "run_exact_cases",
    "stratified_paired_bootstrap",
    "summarize_exact_cases",
]
