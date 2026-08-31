from __future__ import annotations

import numpy as np

from repro_scripts.status_ps_dfsc_formal import midpoint_cache_counts


def test_midpoint_status_distinguishes_registered_600_second_cache(tmp_path):
    root = tmp_path / "beta000"
    cache = root.with_suffix(".refresh_cache")
    cache.mkdir()
    np.savez_compressed(
        cache / "day_000.npz",
        time_limit_seconds=np.asarray(60.0),
    )
    np.savez_compressed(
        cache / "day_001.npz",
        time_limit_seconds=np.asarray(600.0),
    )
    assert midpoint_cache_counts(root) == (2, 1)
