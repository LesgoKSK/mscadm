"""Strict publication preparer that requires finite solver bounds."""

from __future__ import annotations

import numpy as np

import repro_scripts.prepare_ps_dfsc_training_final as preparation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)


_base_load_cache = preparation._load_cache


def _strict_load_cache(*args, **kwargs):
    record = _base_load_cache(*args, **kwargs)
    if record is None:
        return None
    origin = str(np.asarray(record.get("cache_origin", "")).item())
    planned_bound = float(
        np.asarray(record.get("planned_dual_bound", np.nan)).item()
    )
    realized_bound = float(
        np.asarray(record.get("realized_dual_bound", np.nan)).item()
    )
    if origin == "legacy_gap_certified":
        return None
    if not np.isfinite(planned_bound) or not np.isfinite(realized_bound):
        return None
    return record


preparation._load_cache = _strict_load_cache
preparation.solve_two_stage_suc = solve_two_stage_suc
preparation.evaluate_realized = evaluate_realized


if __name__ == "__main__":
    preparation.main()
