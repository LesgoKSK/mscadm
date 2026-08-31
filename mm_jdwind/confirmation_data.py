from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from caa_rahc.nested_data import (
    NestedDataBundle,
    OUTER_TEST_DATES,
    date_list_sha256,
)
from caa_rahc.splits import build_explicit_gefcom2014
from repro.data import SplitData, build_gefcom2014

from .data import JointDataBundle, jointify_split


def _select_dates(split: SplitData, dates: np.ndarray) -> SplitData:
    mask = np.isin(split.day.astype("datetime64[D]"), dates.astype("datetime64[D]"))
    return SplitData(
        condition=split.condition[mask],
        flat_condition=split.flat_condition[mask],
        target=split.target[mask],
        target_standard=split.target_standard[mask],
        zone=split.zone[mask],
        day=split.day[mask],
    )


def _protocol_sha256(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_confirmation_splits(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "mm_jdwind_confirmation_splits_v1":
        raise ValueError("unexpected MM-JDWind split schema")
    blocks = {
        int(key): np.sort(np.asarray(dates, dtype="datetime64[D]"))
        for key, dates in value["outer_test_dates"].items()
    }
    if set(blocks) != {1, 2, 3}:
        raise ValueError("confirmation protocol must contain outers 1..3")
    for outer, dates in blocks.items():
        if len(dates) != 50 or len(np.unique(dates)) != 50:
            raise ValueError(f"outer {outer} is not a unique 50-day block")
        if date_list_sha256(dates) != value["outer_test_sha256"][str(outer)]:
            raise ValueError(f"outer {outer} date hash mismatch")
    union = np.sort(np.concatenate(tuple(blocks.values())))
    if len(np.unique(union)) != 150:
        raise ValueError("confirmation outer blocks overlap")
    if date_list_sha256(union) != value["all_outer_test_sha256"]:
        raise ValueError("confirmation union date hash mismatch")
    old = np.asarray(
        sorted({date for values in OUTER_TEST_DATES.values() for date in values}),
        dtype="datetime64[D]",
    )
    if np.intersect1d(union, old).size:
        raise ValueError("confirmation dates overlap prior CAA outer tests")
    value["_blocks"] = blocks
    return value


def build_confirmation_gefcom2014(
    data_dir: str | Path,
    *,
    outer: int,
    split_path: str | Path,
) -> JointDataBundle:
    split_protocol = load_confirmation_splits(split_path)
    blocks: dict[int, np.ndarray] = split_protocol["_blocks"]
    if outer not in blocks:
        raise ValueError("outer must be 1, 2, or 3")
    legacy = build_gefcom2014(data_dir, seed=0)
    train_dates = np.sort(np.unique(legacy.train.day.astype("datetime64[D]")))
    head_dates = np.sort(np.unique(legacy.validation.day.astype("datetime64[D]")))
    calibration_dates = np.sort(np.unique(legacy.test.day.astype("datetime64[D]")))
    test_dates = blocks[outer]
    if not np.array_equal(np.intersect1d(test_dates, train_dates), test_dates):
        raise ValueError("confirmation test dates are not from legacy training")
    current_train = np.setdiff1d(train_dates, test_dates, assume_unique=True)
    combined_validation = np.union1d(head_dates, calibration_dates)
    bundle = build_explicit_gefcom2014(
        data_dir,
        train_days=current_train,
        validation_days=combined_validation,
        test_days=test_dates,
        protocol_metadata={
            "name": "mm_jdwind_postfreeze_internal_confirmation_v1",
            "outer": outer,
            "split_registry": str(Path(split_path).resolve()),
            "test_date_sha256": split_protocol["outer_test_sha256"][str(outer)],
        },
    )
    head = _select_dates(bundle.validation, head_dates)
    calibration = _select_dates(bundle.validation, calibration_dates)
    protocol: dict[str, Any] = {
        **bundle.protocol,
        "roles": {
            "train": "legacy train minus current newly frozen outer test",
            "validation": "legacy validation; training checkpoint selection only",
            "calibration": "legacy test; method selection and calibration only",
            "test": "new frozen outer test; evaluation only",
        },
        "calendar_day_counts": {
            "train": 581,
            "head_validation": 50,
            "calibration": 50,
            "test": 50,
        },
        "confirmation_registry": {
            key: value
            for key, value in split_protocol.items()
            if key != "_blocks"
        },
    }
    protocol["protocol_sha256"] = _protocol_sha256(protocol)
    nested = NestedDataBundle(
        train=bundle.train,
        validation=head,
        calibration=calibration,
        test=bundle.test,
        condition_standardizer=bundle.condition_standardizer,
        flat_condition_standardizer=bundle.flat_condition_standardizer,
        target_standardizer=bundle.target_standardizer,
        protocol=protocol,
    )
    return JointDataBundle(
        train=jointify_split(nested.train),
        validation=jointify_split(nested.validation),
        calibration=jointify_split(nested.calibration),
        test=jointify_split(nested.test),
        protocol={
            **protocol,
            "joint_layout": {
                "sample": "calendar day",
                "shape": ["day", "zone", "hour", "feature"],
                "zones": list(range(1, 11)),
                "hours": 24,
            },
        },
        source=nested,
    )


__all__ = [
    "build_confirmation_gefcom2014",
    "load_confirmation_splits",
]
