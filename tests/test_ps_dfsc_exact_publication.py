from __future__ import annotations

import numpy as np

import ps_dfsc.exact_suc_publication as publication
from ps_dfsc.exact_suc import FirstStagePlan, SUCResult


def _result() -> SUCResult:
    first = FirstStagePlan(
        commitment=np.ones((1, 24)),
        day_ahead_dispatch=np.zeros((1, 24)),
        reserve_up=np.zeros((1, 24)),
        reserve_down=np.zeros((1, 24)),
        reserve_shortage_up=np.zeros(24),
        reserve_shortage_down=np.zeros(24),
    )
    return SUCResult(
        status="Time limit reached",
        success=False,
        first_stage=first,
        scenario_dispatch=np.zeros((1, 1, 24)),
        used_wind=np.zeros((1, 1, 24)),
        load_shedding=0.0,
        wind_curtailment=0.0,
        reserve_shortage=0.0,
        startup_cost=0.0,
        energy_cost=0.0,
        reserve_capacity_cost=0.0,
        reserve_shortage_cost=0.0,
        penalty_cost=0.0,
        total_cost=100.0,
        solve_time_seconds=600.0,
        mip_gap=0.5,
        mip_dual_bound=99.0,
        mip_node_count=10,
    )


def test_reported_gap_uses_adjusted_cost_and_dual_bound(monkeypatch):
    monkeypatch.setattr(
        publication, "_solve_two_stage_suc", lambda *args, **kwargs: _result()
    )
    solved = publication.solve_two_stage_suc(
        np.zeros((1, 6, 24)), np.ones(1)
    )
    assert solved.mip_gap == 0.01
    assert "raw_highs_mip_gap=0.5" in solved.status
    assert "reported_cost_mip_gap=0.01" in solved.status
    assert not solved.success
