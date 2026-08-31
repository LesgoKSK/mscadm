"""Authoritative explicit data builder for CAA confirmatory outer splits.

Unlike the historical builder, an outer split may deliberately leave dates
unused.  In CAA, the test dates belonging to the other two outer splits are
excluded from the current model's training data as an additional safeguard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from repro.data import (
    ArrayStandardizer,
    DataBundle,
    SplitData,
    _daily_arrays,
    _flat_feature_major,
    load_complete_hours,
)


def _dates(values: Sequence[object]) -> np.ndarray:
    result = np.asarray(values).astype("datetime64[D]")
    if result.ndim != 1 or not len(result) or len(np.unique(result)) != len(result):
        raise ValueError("each date list must be a non-empty unique vector")
    return np.sort(result)


def build_outer_gefcom2014(
    data_dir: str | Path,
    *,
    train_days: Sequence[object],
    validation_days: Sequence[object],
    test_days: Sequence[object],
    protocol_metadata: Mapping[str, object] | None = None,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> DataBundle:
    """Build one train/validation/test bundle while allowing unused dates."""

    train_date = _dates(train_days)
    validation_date = _dates(validation_days)
    test_date = _dates(test_days)
    if (
        np.intersect1d(train_date, validation_date).size
        or np.intersect1d(train_date, test_date).size
        or np.intersect1d(validation_date, test_date).size
    ):
        raise ValueError("train, validation, and test dates must be pairwise disjoint")
    hours = load_complete_hours(data_dir, zones)
    condition, target, zone, day = _daily_arrays(hours)
    available = np.sort(np.unique(day.astype("datetime64[D]")))
    requested = np.unique(np.concatenate((train_date, validation_date, test_date)))
    missing = np.setdiff1d(requested, available)
    if missing.size:
        raise ValueError(f"requested dates are unavailable: {missing.astype(str).tolist()}")
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
        "name": "caa_explicit_outer_date_subset",
        "train_days": int(len(train_date)),
        "validation_days": int(len(validation_date)),
        "test_days": int(len(test_date)),
        "unused_days": int(len(available) - len(requested)),
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


__all__ = ["build_outer_gefcom2014"]
