from __future__ import annotations

import numpy as np

from ps_dfsc.safety_publication import evaluate_safety_gate
from repro_scripts.ps_dfsc_publication_runtime import _dates


def test_dates_accepts_registered_singular_day_field(tmp_path):
    path = tmp_path / "archive.npz"
    expected = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    np.savez_compressed(path, day=expected)
    archive = np.load(path, allow_pickle=False)
    assert np.array_equal(_dates(archive, 2), expected)


def test_coverage_bootstrap_uses_two_sided_95_percent_interval():
    covered = np.ones((2, 10, 24), dtype=bool)
    covered[0].reshape(-1)[:48] = False
    daily_coverage = covered.mean(axis=(1, 2))
    metrics = {
        "CRPS": np.ones(2),
        "ES": np.ones(2),
        "VS": np.ones(2),
        "ramp_CRPS": np.ones(2),
        "zero_Brier": np.ones(2),
        "coverage_90": daily_coverage,
    }
    assignments = {
        "season": np.array([0, 1]),
        "hour_block": np.broadcast_to(np.arange(24) // 6, (10, 24)),
        "wind_level": np.array([0, 1]),
        "ramp_level": np.array([0, 1]),
    }
    samples = 200
    seed = 17
    decision = evaluate_safety_gate(
        metrics,
        metrics,
        candidate_covered=covered,
        baseline_covered=covered,
        assignments=assignments,
        bootstrap_samples=samples,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 2, size=(samples, 2))
    bootstrap = daily_coverage[indices].mean(axis=1)
    assert np.isclose(
        decision.diagnostics["coverage_lower_bound"],
        np.quantile(bootstrap, 0.025),
    )
    assert np.isclose(
        decision.diagnostics["coverage_upper_bound"],
        np.quantile(bootstrap, 0.975),
    )
    assert decision.diagnostics["conditional_ACE_difference"] == 0.0
