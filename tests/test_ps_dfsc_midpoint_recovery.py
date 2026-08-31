from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import ps_dfsc.midpoint_suc_recovery as recovery


def test_midpoint_recovery_is_explicitly_marked_unsuccessful(monkeypatch):
    calls = []
    template = SimpleNamespace(
        success=True,
        status="recovered exact incumbent",
    )

    def fake_solve(*args, **kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            raise RuntimeError("no incumbent")
        return template

    monkeypatch.setattr(recovery, "strict_solve", fake_solve)
    monkeypatch.setattr(
        recovery,
        "replace",
        lambda value, **changes: SimpleNamespace(
            **{**value.__dict__, **changes}
        ),
    )
    result = recovery.solve_midpoint_suc_with_recovery(
        "wind", "probability", mip_gap=0.001, time_limit=600
    )
    assert calls[0]["mip_gap"] == 0.001
    assert calls[1]["mip_gap"] == 1.0
    assert calls[1]["time_limit"] == 600
    assert result.success is False
    assert "midpoint_feasibility_recovery" in result.status
