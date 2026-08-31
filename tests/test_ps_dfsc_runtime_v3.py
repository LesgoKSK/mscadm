from __future__ import annotations

import numpy as np

from repro_scripts.ps_dfsc_publication_runtime_v3 import (
    _diversity_per_date,
)


def test_weighted_pairwise_diversity():
    scenarios = np.zeros((1, 2, 10, 24), dtype=np.float64)
    scenarios[0, 1] = 1.0
    probabilities = np.array([[0.25, 0.75]])
    observed = _diversity_per_date(scenarios, probabilities)[0]
    distance = np.sqrt(240.0)
    expected = 2.0 * 0.25 * 0.75 * distance
    assert np.isclose(observed, expected)
