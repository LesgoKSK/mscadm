from __future__ import annotations

import numpy as np

import ps_dfsc.exact_suc_publication_v2 as publication
from ps_dfsc.exact_suc import FirstStagePlan, SUCResult


def _result(raw_gap, bound, *, success=True):
    first = FirstStagePlan(
        commitment=np.ones((1, 24)),
        day_ahead_dispatch=np.zeros((1, 24)),
        reserve_up=np.zeros((1, 24)),
        reserve_down=np.zeros((1, 24)),
        reserve_shortage_up=np.zeros(24),
        reserve_shortage_down=np.zeros(24),
    )
    return SUCResult(
        status="Optimal",
        success=success,
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
        solve_time_seconds=1.0,
        mip_gap=raw_gap,
        mip_dual_bound=bound,
        mip_node_count=1,
    )


def test_adaptive_second_pass_targets_reported_gap(monkeypatch):
    calls = []
    results = iter([_result(0.001, 99.5), _result(0.0001, 99.95)])

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(publication, "_legacy_solve", solve)
    result = publication.solve_two_stage_suc(
        np.zeros((1, 6, 24)),
        np.ones(1),
        mip_gap=0.001,
        time_limit=600.0,
    )
    assert len(calls) == 2
    assert calls[1]["mip_gap"] < 0.001
    assert calls[1]["time_limit"] <= 600.0
    assert np.isclose(result.mip_gap, 0.0005)
    assert "adaptive_passes=2" in result.status


def test_time_limit_incumbent_is_not_retried(monkeypatch):
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return _result(0.1, 90.0, success=False)

    monkeypatch.setattr(publication, "_legacy_solve", solve)
    result = publication.solve_two_stage_suc(
        np.zeros((1, 6, 24)),
        np.ones(1),
        mip_gap=0.001,
        time_limit=600.0,
    )
    assert len(calls) == 1
    assert not result.success
    assert result.mip_gap == 0.1
