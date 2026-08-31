from pathlib import Path

import numpy as np

from mscadm.data import build_datasets, load_test_hours, load_training_hours


DATA = Path(__file__).resolve().parents[1] / "Data"


def test_raw_dataset_shape_and_alignment() -> None:
    train = load_training_hours(DATA)
    test = load_test_hours(DATA)
    assert len(train) == 168_000
    assert len(test) == 7_440
    assert train["ZONEID"].nunique() == 10
    assert test["ZONEID"].nunique() == 10
    assert test[["U10", "V10", "U100", "V100"]].isna().sum().sum() == 0


def test_daily_dataset_and_scaling() -> None:
    train, validation, test, stats = build_datasets(DATA, validation_days=70)
    assert len(train) > 6_000
    assert len(validation) > 600
    assert len(test) == 310
    sample = train[0]
    assert sample["condition"].shape == (24, 20)
    assert sample["target"].shape == (24, 1)
    assert sample["target_mask"].all()
    assert np.isfinite(stats.mean).all()
    assert np.isfinite(stats.std).all()

