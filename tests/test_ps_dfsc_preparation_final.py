from __future__ import annotations

import numpy as np

from repro_scripts.prepare_ps_dfsc_training_final import (
    _array_sha256,
    _load_cache,
)


def test_array_hash_tracks_shape_dtype_and_values():
    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _array_sha256(left) == _array_sha256(left.copy())
    assert _array_sha256(left) != _array_sha256(left.astype(np.float64))
    changed = left.copy()
    changed[0, 0] = 99
    assert _array_sha256(left) != _array_sha256(changed)


def test_legacy_cache_reuse_requires_certified_gap(tmp_path):
    path = tmp_path / "day_000.npz"
    np.savez_compressed(
        path,
        commitment=np.ones((12, 24)),
        realized_total_cost=np.asarray(10.0),
        planned_total_cost=np.asarray(9.0),
        planned_mip_gap=np.asarray(0.001),
        planned_solve_time=np.asarray(3.0),
        realized_mip_gap=np.asarray(0.0),
        realized_solve_time=np.asarray(0.1),
    )
    loaded = _load_cache(
        path,
        input_sha256="input",
        day_sha256="day",
        mapping_sha256="mapping",
        requested_gap=0.001,
    )
    assert loaded is not None
    assert str(loaded["cache_origin"]) == "legacy_gap_certified"
    assert bool(loaded["planned_success"])

    np.savez_compressed(
        path,
        commitment=np.ones((12, 24)),
        realized_total_cost=np.asarray(10.0),
        planned_mip_gap=np.asarray(0.01),
    )
    assert (
        _load_cache(
            path,
            input_sha256="input",
            day_sha256="day",
            mapping_sha256="mapping",
            requested_gap=0.001,
        )
        is None
    )


def test_v2_cache_reuse_requires_matching_provenance(tmp_path):
    path = tmp_path / "day_000.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("ps_dfsc_identity_day_cache_v2"),
        input_sha256=np.asarray("input"),
        day_sha256=np.asarray("day"),
        mapping_sha256=np.asarray("mapping"),
        commitment=np.ones((12, 24)),
        realized_total_cost=np.asarray(10.0),
        planned_mip_gap=np.asarray(0.001),
    )
    assert (
        _load_cache(
            path,
            input_sha256="input",
            day_sha256="day",
            mapping_sha256="mapping",
            requested_gap=0.001,
        )
        is not None
    )
    assert (
        _load_cache(
            path,
            input_sha256="changed",
            day_sha256="day",
            mapping_sha256="mapping",
            requested_gap=0.001,
        )
        is None
    )
