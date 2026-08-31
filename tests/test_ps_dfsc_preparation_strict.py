from __future__ import annotations

import numpy as np

import repro_scripts.prepare_ps_dfsc_publication_strict as strict


def test_strict_cache_rejects_missing_dual_bounds(monkeypatch):
    monkeypatch.setattr(
        strict,
        "_base_load_cache",
        lambda *args, **kwargs: {
            "cache_origin": np.asarray("legacy_gap_certified"),
            "planned_dual_bound": np.asarray(np.nan),
            "realized_dual_bound": np.asarray(np.nan),
        },
    )
    assert strict._strict_load_cache(None) is None


def test_strict_cache_accepts_solved_finite_bounds(monkeypatch):
    expected = {
        "cache_origin": np.asarray("solved"),
        "planned_dual_bound": np.asarray(10.0),
        "realized_dual_bound": np.asarray(11.0),
    }
    monkeypatch.setattr(
        strict, "_base_load_cache", lambda *args, **kwargs: expected
    )
    assert strict._strict_load_cache(None) is expected
