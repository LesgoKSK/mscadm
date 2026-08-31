from __future__ import annotations

from dataclasses import replace

import ps_dfsc.exact_suc_publication_v3 as publication
from tests.test_ps_dfsc_exact_publication_v2 import _result


def test_publication_success_requires_reported_gap(monkeypatch):
    base = replace(_result(0.001, 99.5), mip_gap=0.005, success=True)
    monkeypatch.setattr(
        publication, "_adaptive_solve", lambda *args, **kwargs: base
    )
    solved = publication.solve_two_stage_suc(
        None, None, mip_gap=0.001, time_limit=600.0
    )
    assert not solved.success
    assert "raw_adaptive_success=true" in solved.status
    assert "publication_gap_success=false" in solved.status
