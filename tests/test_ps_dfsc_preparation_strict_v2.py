from __future__ import annotations

import numpy as np

import repro_scripts.prepare_ps_dfsc_publication_strict_v2 as strict


def test_successful_cache_above_reported_gap_is_rejected(monkeypatch):
    monkeypatch.setattr(
        strict,
        "_base_load_cache",
        lambda *args, **kwargs: {
            "cache_origin": np.asarray("solved"),
            "planned_dual_bound": np.asarray(10.0),
            "realized_dual_bound": np.asarray(10.0),
            "planned_mip_gap": np.asarray(0.005),
            "planned_success": np.asarray(True),
        },
    )
    assert strict._strict_load_cache(None, requested_gap=0.001) is None


def test_time_limit_incumbent_is_retained_for_audit(monkeypatch):
    expected = {
        "cache_origin": np.asarray("solved"),
        "planned_dual_bound": np.asarray(10.0),
        "realized_dual_bound": np.asarray(10.0),
        "planned_mip_gap": np.asarray(0.005),
        "planned_success": np.asarray(False),
    }
    monkeypatch.setattr(
        strict, "_base_load_cache", lambda *args, **kwargs: expected
    )
    assert (
        strict._strict_load_cache(None, requested_gap=0.001) is expected
    )
