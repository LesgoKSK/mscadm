import numpy as np

from caa_rahc.metrics import analytic_atom_scores, per_date_metrics, stack_per_date_metrics


def test_date_metrics_shapes_and_seed_stack():
    rng = np.random.default_rng(5)
    cases, members, hours = 20, 11, 24
    truth = rng.uniform(0, 1, size=(cases, hours))
    values = np.clip(truth[:, None, :] + rng.normal(0, 0.1, size=(cases, members, hours)), 0, 1)
    day = np.repeat(np.arange(2).astype("datetime64[D]"), 10)
    one = per_date_metrics(values, truth, day)
    assert one["CRPS"].shape == (2,)
    stacked = stack_per_date_metrics([values, values], truth, day)
    assert stacked["VS"].shape == (2, 2)
    np.testing.assert_allclose(stacked["CRPS"][0], stacked["CRPS"][1])


def test_analytic_atom_scores_are_member_count_independent():
    truth = np.array([[0.0, 0.2, 1.0]])
    p0 = np.array([[0.8, 0.1, 0.0]])
    result = analytic_atom_scores(p0, 0.01, truth)
    assert result["zero_Brier"] >= 0.0
    assert result["zero_observed_rate"] == 1 / 3
    assert np.isfinite(list(result.values())).all()
