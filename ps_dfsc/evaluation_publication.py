"""Publication-grade exact evaluation with solver termination audit fields."""

from __future__ import annotations

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
    tolerance = mip_gap * (1.0 + 1e-6) + 1e-12
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
            "requested_mip_gap": mip_gap,
            "time_limit_seconds": time_limit,
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
                    "status": "solution_available",
                    "all_solver_success": bool(
                        planned.success and realized.success and oracle.success
                    ),
                    "all_target_gaps_met": bool(
                        planned.mip_gap <= tolerance
                        and realized.mip_gap <= tolerance
                        and oracle.mip_gap <= tolerance
                    ),
                    "planned_solver_success": planned.success,
                    "planned_termination": planned.status,
                    "realized_solver_success": realized.success,
                    "realized_termination": realized.status,
                    "oracle_solver_success": oracle.success,
                    "oracle_termination": oracle.status,
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
                    "planned_node_count": planned.mip_node_count,
                    "realized_node_count": realized.mip_node_count,
                    "oracle_node_count": oracle.mip_node_count,
                }
            )
        except Exception as error:
            rows.append(
                {
                    **base,
                    "status": "failed_no_solution",
                    "all_solver_success": False,
                    "all_target_gaps_met": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return pd.DataFrame(rows)


def summarize_exact_cases(frame: pd.DataFrame) -> dict[str, float | int]:
    successful = frame.loc[frame["status"] == "solution_available"]
    result: dict[str, float | int] = {
        "cases": int(len(frame)),
        "successful_cases": int(len(successful)),
        "paired_success_rate": (
            float(len(successful) / len(frame)) if len(frame) else 0.0
        ),
        "solver_success_rate": (
            float(successful["all_solver_success"].mean())
            if len(successful)
            else 0.0
        ),
        "target_gap_pass_rate": (
            float(successful["all_target_gaps_met"].mean())
            if len(successful)
            else 0.0
        ),
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


__all__ = ["run_exact_cases", "summarize_exact_cases"]
