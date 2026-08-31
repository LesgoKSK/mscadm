from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from caa_rahc.splits import build_explicit_gefcom2014
from mm_jdwind.data import JointDataBundle, JointSplitData, jointify_split
from repro.data import SplitData, build_gefcom2014

from .splits import load_ps_dfsc_splits, registry_sha256


@dataclass(frozen=True)
class PSDFSCOuterData:
    base_generator: JointDataBundle
    development_train: JointSplitData
    development_validation: JointSplitData
    confirmation_test: JointSplitData
    protocol: dict[str, Any]


def _select_dates(split: SplitData, dates: np.ndarray) -> SplitData:
    mask = np.isin(
        split.day.astype("datetime64[D]"), dates.astype("datetime64[D]")
    )
    return SplitData(
        condition=split.condition[mask],
        flat_condition=split.flat_condition[mask],
        target=split.target[mask],
        target_standard=split.target_standard[mask],
        zone=split.zone[mask],
        day=split.day[mask],
    )


def build_ps_dfsc_outer_data(
    data_dir: str | Path,
    *,
    outer: int,
    split_registry: str | Path,
) -> PSDFSCOuterData:
    """Build one leakage-audited outer pipeline.

    The MM-JDWind base generator excludes all 150 PS development dates and the
    current 50-day confirmation block. Other outer confirmation dates may
    remain in its training data, as pre-registered.
    """

    registry = load_ps_dfsc_splits(split_registry)
    if outer not in (1, 2, 3):
        raise ValueError("outer must be 1, 2, or 3")
    legacy = build_gefcom2014(data_dir, seed=0)
    legacy_train = np.sort(
        np.unique(legacy.train.day.astype("datetime64[D]"))
    )
    head_dates = np.sort(
        np.unique(legacy.validation.day.astype("datetime64[D]"))
    )
    calibration_dates = np.sort(
        np.unique(legacy.test.day.astype("datetime64[D]"))
    )
    development_train_dates = registry["_development_train"]
    development_validation_dates = registry["_development_validation"]
    development_dates = np.union1d(
        development_train_dates, development_validation_dates
    )
    test_dates = registry["_outer"][outer]
    if np.intersect1d(development_dates, test_dates).size:
        raise ValueError("development and confirmation dates overlap")
    excluded = np.union1d(development_dates, test_dates)
    base_train_dates = np.setdiff1d(
        legacy_train, excluded, assume_unique=True
    )
    combined_validation = np.unique(
        np.concatenate(
            (
                head_dates,
                calibration_dates,
                development_train_dates,
                development_validation_dates,
            )
        )
    )
    explicit = build_explicit_gefcom2014(
        data_dir,
        train_days=base_train_dates,
        validation_days=combined_validation,
        test_days=test_dates,
        protocol_metadata={
            "name": "ps_dfsc_outer_v1",
            "outer": outer,
            "split_registry": str(Path(split_registry).resolve()),
            "split_registry_sha256": registry_sha256(registry),
            "test_date_sha256": registry["outer_test_sha256"][str(outer)],
        },
    )
    head = _select_dates(explicit.validation, head_dates)
    calibration = _select_dates(explicit.validation, calibration_dates)
    development_train = _select_dates(
        explicit.validation, development_train_dates
    )
    development_validation = _select_dates(
        explicit.validation, development_validation_dates
    )
    protocol: dict[str, Any] = {
        **explicit.protocol,
        "roles": {
            "base_train": "MM-JDWind fitting; excludes all PS development and current outer test",
            "base_validation": "MM-JDWind checkpoint selection only",
            "base_calibration": "MM-JDWind base-method calibration only",
            "ps_development_train": "PS-DFSC optimization only",
            "ps_development_validation": "PS-DFSC gate and candidate selection",
            "confirmation_test": "locked confirmation only",
        },
        "calendar_day_counts": {
            "base_train": int(len(base_train_dates)),
            "base_validation": int(len(head_dates)),
            "base_calibration": int(len(calibration_dates)),
            "ps_development_train": int(len(development_train_dates)),
            "ps_development_validation": int(len(development_validation_dates)),
            "confirmation_test": int(len(test_dates)),
        },
        "other_outer_test_dates_may_remain_in_base_train": True,
    }
    base_bundle = JointDataBundle(
        train=jointify_split(explicit.train),
        validation=jointify_split(head),
        calibration=jointify_split(calibration),
        test=jointify_split(explicit.test),
        protocol=protocol,
        source=explicit,
    )
    result = PSDFSCOuterData(
        base_generator=base_bundle,
        development_train=jointify_split(development_train),
        development_validation=jointify_split(development_validation),
        confirmation_test=jointify_split(explicit.test),
        protocol=protocol,
    )
    audit_ps_dfsc_outer_data(result)
    return result


def audit_ps_dfsc_outer_data(data: PSDFSCOuterData) -> None:
    base_train = data.base_generator.train.day.astype("datetime64[D]")
    development = np.union1d(
        data.development_train.day.astype("datetime64[D]"),
        data.development_validation.day.astype("datetime64[D]"),
    )
    test = data.confirmation_test.day.astype("datetime64[D]")
    if len(np.unique(data.development_train.day)) != 100:
        raise ValueError("PS development train is not 100 days")
    if len(np.unique(data.development_validation.day)) != 50:
        raise ValueError("PS development validation is not 50 days")
    if len(np.unique(test)) != 50:
        raise ValueError("PS confirmation test is not 50 days")
    if np.intersect1d(base_train, development).size:
        raise ValueError("base generator train leaks PS development dates")
    if np.intersect1d(base_train, test).size:
        raise ValueError("base generator train leaks current confirmation dates")
    if np.intersect1d(development, test).size:
        raise ValueError("PS development leaks confirmation dates")


__all__ = [
    "PSDFSCOuterData",
    "audit_ps_dfsc_outer_data",
    "build_ps_dfsc_outer_data",
]
