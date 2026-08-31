import numpy as np

from repro.classical import QRGBMConfig, QuantileRegressionGBM, random_scenarios
from repro.data import SplitData


def split(rows: int = 20) -> SplitData:
    rng = np.random.default_rng(2)
    zone = np.repeat(np.asarray([1, 2]), rows // 2)
    target = rng.uniform(0, 1, (rows, 24)).astype(np.float32)
    return SplitData(
        condition=rng.normal(size=(rows, 24, 20)).astype(np.float32),
        flat_condition=rng.normal(size=(rows, 250)).astype(np.float32),
        target=target,
        target_standard=target,
        zone=zone,
        day=np.arange(rows).astype("datetime64[D]"),
    )


def test_rand_has_expected_shape_and_zone_source() -> None:
    data = split()
    scenarios = random_scenarios(data, scenarios=7, seed=3)
    assert scenarios.shape == (20, 7, 24)
    zone_one_candidates = {tuple(row) for row in data.target[data.zone == 1]}
    assert all(tuple(row) in zone_one_candidates for row in scenarios[0])


def test_qrgbm_quantiles_and_ecc_sampling() -> None:
    data = split(24)
    model = QuantileRegressionGBM(
        QRGBMConfig(quantiles=3, estimators=2, max_depth=2, seed=4)
    ).fit(data)
    quantiles = model.predict_quantiles(data.flat_condition[:3])
    assert quantiles.shape == (3, 24, 3)
    assert np.all(np.diff(quantiles, axis=-1) >= 0)
    generated = model.sample(data, scenarios=5, seed=5)
    assert generated.shape == (24, 5, 24)
    assert np.all((generated >= 0) & (generated <= 1))
