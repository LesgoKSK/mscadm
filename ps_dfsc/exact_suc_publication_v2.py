"""Adaptive exact wrapper targeting the reported-cost relative MIP gap."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .exact_suc import (
    FirstStagePlan,
    SUCResult,
    solve_two_stage_suc as _legacy_solve,
)


def _reported_gap(result: SUCResult) -> float:
    if not np.isfinite(result.mip_dual_bound):
        return np.nan
    return float(
        max(0.0, result.total_cost - result.mip_dual_bound)
        / max(abs(result.total_cost), 1e-12)
    )


def _annotated(
    result: SUCResult,
    *,
    raw_gap: float,
    requested_gap: float,
    elapsed: float,
    passes: int,
) -> SUCResult:
    actual = _reported_gap(result)
    status = (
        f"{result.status}; raw_highs_mip_gap={raw_gap:.12g}; "
        f"reported_cost_mip_gap={actual:.12g}; "
        f"requested_reported_gap={requested_gap:.12g}; "
        f"adaptive_passes={passes}"
    )
    return replace(
        result,
        status=status,
        mip_gap=actual,
        solve_time_seconds=float(elapsed),
    )


def solve_two_stage_suc(*args, **kwargs) -> SUCResult:
    requested_gap = float(kwargs.get("mip_gap", 0.001))
    total_limit = float(kwargs.get("time_limit", 600.0))
    started = perf_counter()
    first = _legacy_solve(*args, **kwargs)
    first_elapsed = perf_counter() - started
    first_raw_gap = float(first.mip_gap)
    first_actual_gap = _reported_gap(first)
    if (
        first.success
        and np.isfinite(first_actual_gap)
        and first_actual_gap > requested_gap + 1e-12
        and np.isfinite(first_raw_gap)
        and first_raw_gap > 0.0
    ):
        remaining = total_limit - first_elapsed
        ratio = first_actual_gap / first_raw_gap
        tightened = min(
            requested_gap * 0.8 / max(ratio, 1e-12),
            requested_gap * 0.5,
        )
        if remaining >= 1.0 and tightened < requested_gap:
            second_kwargs = dict(kwargs)
            second_kwargs["mip_gap"] = max(tightened, 1e-8)
            second_kwargs["time_limit"] = remaining
            second = _legacy_solve(*args, **second_kwargs)
            elapsed = perf_counter() - started
            second_actual_gap = _reported_gap(second)
            chosen = (
                second
                if np.isfinite(second_actual_gap)
                and (
                    not np.isfinite(first_actual_gap)
                    or second_actual_gap <= first_actual_gap
                )
                else first
            )
            raw = (
                float(second.mip_gap)
                if chosen is second
                else first_raw_gap
            )
            return _annotated(
                chosen,
                raw_gap=raw,
                requested_gap=requested_gap,
                elapsed=elapsed,
                passes=2,
            )
    return _annotated(
        first,
        raw_gap=first_raw_gap,
        requested_gap=requested_gap,
        elapsed=first_elapsed,
        passes=1,
    )


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
