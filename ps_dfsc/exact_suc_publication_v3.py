"""Final exact wrapper with reported-gap success semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .exact_suc import FirstStagePlan, SUCResult
from .exact_suc_publication_v2 import solve_two_stage_suc as _adaptive_solve


def solve_two_stage_suc(*args, **kwargs) -> SUCResult:
    requested_gap = float(kwargs.get("mip_gap", 0.001))
    result = _adaptive_solve(*args, **kwargs)
    raw_adaptive_success = bool(result.success)
    publication_success = bool(
        raw_adaptive_success
        and np.isfinite(result.mip_gap)
        and result.mip_gap <= requested_gap + 1e-12
    )
    return replace(
        result,
        success=publication_success,
        status=(
            f"{result.status}; raw_adaptive_success="
            f"{str(raw_adaptive_success).lower()}; "
            f"publication_gap_success={str(publication_success).lower()}"
        ),
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
