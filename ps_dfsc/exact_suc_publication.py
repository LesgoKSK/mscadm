"""Audited wrappers around the exact SUC solver.

The legacy formulation removes a constant curtailment term from the objective
seen by HiGHS and adds it back to both the reported incumbent and dual bound.
Consequently, HiGHS' raw relative gap is not the relative gap of the reported
total operating cost.  This module recomputes that gap on the reported scale
while retaining the raw value in the termination text.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .exact_suc import (
    FirstStagePlan,
    SUCResult,
    solve_two_stage_suc as _solve_two_stage_suc,
)


def _reported_cost_gap(result: SUCResult) -> float:
    if not np.isfinite(result.mip_dual_bound):
        return np.nan
    denominator = max(abs(result.total_cost), 1e-12)
    return float(
        max(0.0, result.total_cost - result.mip_dual_bound) / denominator
    )


def solve_two_stage_suc(*args, **kwargs) -> SUCResult:
    result = _solve_two_stage_suc(*args, **kwargs)
    raw_gap = result.mip_gap
    actual_gap = _reported_cost_gap(result)
    status = (
        f"{result.status}; raw_highs_mip_gap={raw_gap:.12g}; "
        f"reported_cost_mip_gap={actual_gap:.12g}"
    )
    return replace(result, status=status, mip_gap=actual_gap)


def evaluate_realized(
    first_stage: FirstStagePlan,
    observed_wind_by_farm: np.ndarray,
    *,
    system=None,
    **solver_options,
) -> SUCResult:
    observed = np.asarray(observed_wind_by_farm, dtype=np.float64)
    return solve_two_stage_suc(
        observed[None],
        np.ones(1),
        system=system,
        fixed_first_stage=first_stage,
        **solver_options,
    )


__all__ = [
    "FirstStagePlan",
    "SUCResult",
    "evaluate_realized",
    "solve_two_stage_suc",
]
