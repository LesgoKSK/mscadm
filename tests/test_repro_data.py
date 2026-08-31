from pathlib import Path

import numpy as np

from repro.data import build_gefcom2014, load_complete_hours


ROOT = Path(__file__).resolve().parents[1]


def test_reference_forward_fill_and_feature_formulas() -> None:
    hours = load_complete_hours(ROOT / "Data", zones=(1,))
    missing_timestamp = np.datetime64("2013-04-22T09:00")
    previous_timestamp = np.datetime64("2013-04-22T08:00")
    missing = hours.loc[hours["timestamp"] == missing_timestamp].iloc[0]
    previous = hours.loc[hours["timestamp"] == previous_timestamp].iloc[0]
    assert missing["TARGETVAR"] == previous["TARGETVAR"]
    assert np.isclose(missing["WE10"], 0.5 * missing["WS10"] ** 3)
    assert np.isclose(
        missing["WD10"], np.arctan2(missing["U10"], missing["V10"]) * 180 / np.pi
    )


def test_dumas_protocol_counts_and_shapes() -> None:
    bundle = build_gefcom2014(ROOT / "Data")
    assert len(bundle.train) == 6_310
    assert len(bundle.validation) == 500
    assert len(bundle.test) == 500
    assert bundle.train.condition.shape == (6_310, 24, 20)
    assert bundle.train.flat_condition.shape == (6_310, 250)
    assert bundle.train.target.shape == (6_310, 24)
    assert not np.intersect1d(bundle.train.day, bundle.test.day).size
    assert not np.intersect1d(bundle.validation.day, bundle.test.day).size
    assert np.isfinite(bundle.train.condition).all()

