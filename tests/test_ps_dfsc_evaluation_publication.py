from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.types import CalibratedDistribution


def _result(*, success: bool, gap: float, cost: float, status: str):
    return SimpleNamespace(
        success=success,
        status=status,
        total_cost=cost,
        first_stage=object(),
        load_shedding=1.0,
        wind_curtailment=2.0,
        reserve_shortage=3.0,
        solve_time_seconds=4.0,
        mip_gap=gap,
        mip_dual_bound=cost - 1.0,
        mip_node_count=5,
    )


def test_exact_evaluation_preserves_time_limit_incumbent_audit(monkeypatch):
    calls = iter(
        [
            _result(
                success=False,
                gap=0.01,
                cost=100.0,
                status="Time limit reached",
            ),
            _result(success=True, gap=0.0, cost=80.0, status="Optimal"),
        ]
    )
    monkeypatch.setattr(
        evaluation, "solve_two_stage_suc", lambda *args, **kwargs: next(calls)
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_realized",
        lambda *args, **kwargs: _result(
            success=True, gap=0.0, cost=90.0, status="Optimal"
        ),
    )
    distribution = CalibratedDistribution(
        full_scenarios=np.zeros((2, 10, 24)),
        probabilities=np.full(2, 0.5),
        suc_scenarios=np.zeros((2, 6, 24)),
        suc_probabilities=np.full(2, 0.5),
        ess=2.0,
        entropy=np.log(2.0),
        transport_cost=0.0,
    )
    frame = evaluation.run_exact_cases(
        [distribution],
        np.zeros((1, 6, 24)),
        outer=1,
        method="candidate",
        mip_gap=0.001,
    )
    row = frame.iloc[0]
    assert row["status"] == "solution_available"
    assert not row["all_solver_success"]
    assert not row["all_target_gaps_met"]
    assert row["planned_termination"] == "Time limit reached"
    assert row["planned_dual_bound"] == 99.0
    assert row["planned_node_count"] == 5
    summary = evaluation.summarize_exact_cases(frame)
    assert summary["paired_success_rate"] == 1.0
    assert summary["solver_success_rate"] == 0.0
    assert summary["target_gap_pass_rate"] == 0.0
