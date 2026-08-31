"""Explicit feasibility recovery for training-only commitment refreshes."""

from __future__ import annotations

from dataclasses import replace

from .exact_suc_publication_v3 import solve_two_stage_suc as strict_solve


def solve_midpoint_suc_with_recovery(*args, **kwargs):
    """Run the registered solve, then one exact-binary feasibility recovery.

    This wrapper is used only for the stopped-gradient midpoint commitment
    cache.  Validation, selection and confirmation keep the strict 0.1% solve.
    A recovered incumbent is never labelled successful or target-gap passing.
    """
    try:
        return strict_solve(*args, **kwargs)
    except RuntimeError as error:
        recovery_kwargs = dict(kwargs)
        recovery_kwargs["mip_gap"] = 1.0
        recovery_kwargs["time_limit"] = float(
            kwargs.get("time_limit", 600.0)
        )
        recovered = strict_solve(*args, **recovery_kwargs)
        return replace(
            recovered,
            success=False,
            status=(
                "midpoint_feasibility_recovery_after_no_incumbent; "
                f"registered_failure={error}; "
                f"recovery_status={recovered.status}"
            ),
        )


__all__ = ["solve_midpoint_suc_with_recovery"]
