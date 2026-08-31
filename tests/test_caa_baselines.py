import numpy as np

from caa_rahc.baselines import crossfit_c0, fit_c0, transform_c0


def test_c0_date_crossfit_and_full_transform():
    rng = np.random.default_rng(8)
    days = np.repeat(np.arange(10).astype("datetime64[D]"), 2)
    truth = rng.uniform(0, 1, size=(20, 24))
    scenarios = np.clip(truth[:, None] + rng.normal(0, 0.1, size=(20, 21, 24)), 0, 1)
    result = crossfit_c0([scenarios, scenarios], truth, days, folds=5)
    assert len(result.scenarios_by_seed) == 2
    assert len(np.unique(result.fold_assignments)) == 5
    assert np.isfinite(result.scenarios_by_seed[0]).all()
    fitted = fit_c0([scenarios, scenarios], truth)
    transformed = transform_c0(fitted, [scenarios, scenarios])
    assert transformed[0].shape == scenarios.shape
