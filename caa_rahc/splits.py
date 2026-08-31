"""Explicit, auditable outer date splits for the confirmatory CAA-RAHC study."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from repro.data import (
    ArrayStandardizer,
    DataBundle,
    SplitData,
    _daily_arrays,
    _flat_feature_major,
    build_gefcom2014,
    load_complete_hours,
)


def _dates(values: Iterable[object]) -> np.ndarray:
    result = np.asarray(list(values)).astype("datetime64[D]")
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("date collection must be a non-empty vector")
    if len(np.unique(result)) != len(result):
        raise ValueError("date collection contains duplicates")
    return np.sort(result)


def make_confirmatory_split_protocol(
    data_dir: str | Path = "Data",
    *,
    development_split_seed: int = 0,
    allocation_seed: int = 20260719,
    outer_splits: int = 3,
    validation_days_per_split: int = 50,
    test_days_per_split: int = 50,
) -> dict[str, object]:
    """Allocate disjoint, previously unexamined confirmatory test dates.

    Test dates are drawn only from the *training dates* of the historical
    seed-0 development protocol, so they are disjoint from the old validation
    and test dates that motivated CAA-RAHC.  The union of all confirmatory test
    dates is excluded from every new base-model training set, even when a date
    belongs to another outer split.  This makes the three test blocks genuine
    holdouts for every newly trained checkpoint.
    """

    if outer_splits < 2:
        raise ValueError("the registered confirmatory study requires at least two outer splits")
    historical = build_gefcom2014(data_dir, seed=development_split_seed)
    historical_train = np.sort(np.unique(historical.train.day.astype("datetime64[D]")))
    historical_validation = np.sort(
        np.unique(historical.validation.day.astype("datetime64[D]"))
    )
    historical_test = np.sort(np.unique(historical.test.day.astype("datetime64[D]")))
    required = outer_splits * (validation_days_per_split + test_days_per_split)
    if len(historical_train) < required:
        raise ValueError("not enough historical training dates for disjoint outer blocks")
    shuffled = historical_train.copy()
    np.random.default_rng(allocation_seed).shuffle(shuffled)
    test_count = outer_splits * test_days_per_split
    validation_count = outer_splits * validation_days_per_split
    test_pool = shuffled[:test_count]
    validation_pool = shuffled[test_count : test_count + validation_count]
    all_days = np.sort(
        np.unique(
            np.concatenate((historical_train, historical_validation, historical_test))
        )
    )
    splits: list[dict[str, object]] = []
    for index in range(outer_splits):
        test = np.sort(
            test_pool[index * test_days_per_split : (index + 1) * test_days_per_split]
        )
        validation = np.sort(
            validation_pool[
                index * validation_days_per_split : (index + 1) * validation_days_per_split
            ]
        )
        train = np.setdiff1d(all_days, np.union1d(test_pool, validation), assume_unique=True)
        splits.append(
            {
                "index": index,
                "model_seed": 7300 + index,
                "train_days": [str(value) for value in train],
                "validation_days": [str(value) for value in validation],
                "test_days": [str(value) for value in test],
                "counts": {
                    "train_days": int(len(train)),
                    "validation_days": int(len(validation)),
                    "test_days": int(len(test)),
                },
            }
        )
    return {
        "name": "caa_rahc_confirmatory_outer_dates_v1",
        "frozen_allocation_seed": allocation_seed,
        "development_split_seed": development_split_seed,
        "outer_splits": outer_splits,
        "validation_days_per_split": validation_days_per_split,
        "test_days_per_split": test_days_per_split,
        "historical_development_validation_days": [str(value) for value in historical_validation],
        "historical_development_test_days": [str(value) for value in historical_test],
        "all_confirmatory_test_days": [str(value) for value in np.sort(test_pool)],
        "all_confirmatory_validation_days": [str(value) for value in np.sort(validation_pool)],
        "test_allocation_source": "historical seed-0 training dates only",
        "cross_split_test_exclusion": (
            "the union of all confirmatory test dates is excluded from every new base-model train set"
        ),
        "splits": splits,
    }


def build_explicit_gefcom2014(
    data_dir: str | Path,
    *,
    train_days: Sequence[object],
    validation_days: Sequence[object],
    test_days: Sequence[object],
    protocol_metadata: Mapping[str, object] | None = None,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> DataBundle:
    """Build a GEFCom bundle from explicit, disjoint calendar-date lists."""

    train_date = _dates(train_days)
    validation_date = _dates(validation_days)
    test_date = _dates(test_days)
    if np.intersect1d(train_date, validation_date).size:
        raise ValueError("train and validation dates overlap")
    if np.intersect1d(train_date, test_date).size:
        raise ValueError("train and test dates overlap")
    if np.intersect1d(validation_date, test_date).size:
        raise ValueError("validation and test dates overlap")

    hours = load_complete_hours(data_dir, zones)
    condition, target, zone, day = _daily_arrays(hours)
    available = np.sort(np.unique(day.astype("datetime64[D]")))
    requested = np.sort(np.concatenate((train_date, validation_date, test_date)))
    missing = np.setdiff1d(requested, available)
    if missing.size:
        raise ValueError(f"requested dates are not in GEFCom data: {missing.astype(str).tolist()}")
    if len(np.unique(requested)) != len(available) or not np.array_equal(
        np.unique(requested), available
    ):
        raise ValueError("explicit split must partition every available calendar date exactly once")

    masks = {
        "train": np.isin(day, train_date),
        "validation": np.isin(day, validation_date),
        "test": np.isin(day, test_date),
    }
    condition_scaler = ArrayStandardizer.fit(condition[masks["train"]], axes=0)
    raw_flat = _flat_feature_major(condition, zone)
    flat_scaler = ArrayStandardizer.fit(raw_flat[masks["train"]], axes=0)
    target_scaler = ArrayStandardizer.fit(target[masks["train"]], axes=0)
    scaled_condition = condition_scaler.transform(condition)
    zone_one_hot = np.eye(10, dtype=np.float32)[zone - 1]
    condition_with_zone = np.concatenate(
        (scaled_condition, np.repeat(zone_one_hot[:, None, :], 24, axis=1)), axis=-1
    )
    scaled_flat = flat_scaler.transform(raw_flat)
    scaled_target = target_scaler.transform(target)

    def select(name: str) -> SplitData:
        mask = masks[name]
        return SplitData(
            condition=condition_with_zone[mask],
            flat_condition=scaled_flat[mask],
            target=target[mask],
            target_standard=scaled_target[mask],
            zone=zone[mask],
            day=day[mask],
        )

    protocol: dict[str, object] = {
        "name": "explicit_calendar_date_partition",
        "train_days": int(len(train_date)),
        "validation_days": int(len(validation_date)),
        "test_days": int(len(test_date)),
        "standardizers_fit_on": "train only",
    }
    if protocol_metadata:
        protocol.update(dict(protocol_metadata))
    return DataBundle(
        train=select("train"),
        validation=select("validation"),
        test=select("test"),
        condition_standardizer=condition_scaler,
        flat_condition_standardizer=flat_scaler,
        target_standardizer=target_scaler,
        protocol=protocol,
    )


__all__ = ["build_explicit_gefcom2014", "make_confirmatory_split_protocol"]
