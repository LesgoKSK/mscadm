import numpy as np

from repro.suc import reduce_scenarios, rts24


def test_rts24_paper_capacities_and_wind_layout() -> None:
    system = rts24()
    assert len(system.generators) == 12
    assert sum(unit.maximum for unit in system.generators) == 3375
    assert len(system.wind_buses) == 6
    assert len(system.branches) == 34
    assert np.isclose(system.load_share.sum(), 1.0)


def test_kmeans_reduction_returns_probabilities() -> None:
    scenarios = np.linspace(0, 1, 20 * 24).reshape(20, 24)
    reduced, probability = reduce_scenarios(scenarios, 4, seed=2)
    assert reduced.shape == (4, 24)
    assert probability.shape == (4,)
    assert np.isclose(probability.sum(), 1)
    assert np.all((reduced >= 0) & (reduced <= 1200))
