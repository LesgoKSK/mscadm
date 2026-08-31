import numpy as np

from caa_rahc.outer_evaluation import evaluate_outer_tests


def test_outer_evaluation_pools_disjoint_dates():
    rng = np.random.default_rng(1)
    methods = {"A0": {}, "A4": {}}
    truth = {}
    day = {}
    assignments = {}
    for outer in (1, 2, 3):
        n, m, h = 4, 9, 4
        truth[outer] = rng.uniform(0.2, 0.8, size=(n, h))
        start = np.datetime64("2012-01-01") + (outer - 1) * 10
        day[outer] = start + np.arange(n).astype("timedelta64[D]")
        assignments[outer] = {
            "zone": np.broadcast_to(np.arange(n)[:, None] + 1, (n, h)),
            "hour": np.broadcast_to(np.arange(h)[None, :] + 1, (n, h)),
        }
        a0 = []
        a4 = []
        for seed in range(3):
            poor = np.clip(truth[outer][:, None] + rng.normal(0, 0.15, size=(n, m, h)), 0, 1)
            good = np.clip(truth[outer][:, None] + rng.normal(0, 0.10, size=(n, m, h)), 0, 1)
            a0.append(poor)
            a4.append(good)
        methods["A0"][outer] = a0
        methods["A4"][outer] = a4
    result = evaluate_outer_tests(
        methods,
        truth,
        day,
        assignments,
        bootstrap_replicates=20,
        strict_reversals=0,
    )
    assert result["protocol"]["unique_test_dates"] == 12
    assert len(result["rows"]) == 2 * 4 * 3
    assert "A4" in result["noninferiority"]
