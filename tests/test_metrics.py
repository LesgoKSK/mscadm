import numpy as np

from mscadm.metrics import evaluate_all


def test_perfect_deterministic_scenarios_have_zero_scores() -> None:
    observations = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]], dtype=np.float32)
    scenarios = np.repeat(observations[:, None, :], 4, axis=1)
    mask = np.ones_like(observations, dtype=bool)
    metrics = evaluate_all(scenarios, observations, mask, [0.1, 0.5, 0.9])
    for name in ("mae", "rmse", "crps", "qs", "es", "vs"):
        assert abs(float(metrics[name])) < 1e-7

