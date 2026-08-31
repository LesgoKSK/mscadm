import numpy as np

from repro.metrics import all_scores, interval_scores


def test_perfect_ensemble_scores_are_zero() -> None:
    observations = np.asarray([[0.1, 0.4, 0.9], [0.2, 0.3, 0.8]], dtype=np.float32)
    scenarios = np.repeat(observations[:, None], 100, axis=1)
    scores = all_scores(scenarios, observations)
    assert all(abs(value) < 1e-7 for value in scores.values())


def test_interval_width_is_nonnegative_and_coverage_is_bounded() -> None:
    rng = np.random.default_rng(0)
    observations = rng.random((5, 24))
    scenarios = rng.random((5, 100, 24))
    intervals = interval_scores(scenarios, observations)
    assert all(width >= 0 for width in intervals["piaw"])
    assert all(0 <= coverage <= 1 for coverage in intervals["coverage"])

