from __future__ import annotations

import numpy as np
import pandas as pd

from repro_scripts.ps_dfsc_final_report_publication import (
    _method_summary,
    _paired,
    _worst_case_penalty,
)


def _frame(method, costs, statuses):
    rows = []
    for index, (cost, status) in enumerate(zip(costs, statuses)):
        rows.append(
            {
                "outer": 1,
                "date": f"2020-01-0{index + 1}",
                "method": method,
                "status": status,
                "realized_total_cost": cost,
                "cost_regret": 1.0,
                "load_shedding": 0.0,
                "wind_curtailment": 0.0,
                "reserve_shortage": 0.0,
                "all_solver_success": status == "solution_available",
                "all_target_gaps_met": status == "solution_available",
                "planned_mip_gap": 0.001,
                "planned_solve_time": 2.0,
            }
        )
    return pd.DataFrame(rows)


def test_report_pairs_solution_available_and_penalizes_failure():
    candidate = _frame(
        "candidate",
        [90.0, np.nan],
        ["solution_available", "failed_no_solution"],
    )
    baseline = _frame(
        "baseline",
        [100.0, 110.0],
        ["solution_available", "solution_available"],
    )
    paired = _paired(candidate, baseline)
    assert len(paired) == 1
    summary = _method_summary(candidate)
    assert summary["solution_available"] == 1
    sensitivity = _worst_case_penalty(
        [candidate], [baseline], multipliers=(2.0,)
    )
    assert sensitivity["2.0"]["penalty_cost"] == 220.0
    assert sensitivity["2.0"]["candidate_mean_cost"] == 155.0
    assert sensitivity["2.0"]["baseline_mean_cost"] == 105.0
