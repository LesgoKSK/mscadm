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
from mm_jdwind.confirmation_data import _select_dates
from mm_jdwind.data import (
    JointDataBundle,
    build_joint_nested_gefcom2014,
    jointify_split,
)
from repro.data import build_gefcom2014


def _protocol_sha256(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_stgf_confirmation_splits(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "stgf_confirmation_splits_v1":
        raise ValueError("unexpected STGF split schema")
    blocks = {
        int(key): np.sort(np.asarray(dates, dtype="datetime64[D]"))
        for key, dates in value["outer_test_dates"].items()
    }
    if set(blocks) != {1, 2, 3}:
        raise ValueError("STGF registry must contain outers 1..3")
    for outer, dates in blocks.items():
        if len(dates) != 50 or len(np.unique(dates)) != 50:
            raise ValueError(f"outer {outer} is not a unique 50-day block")
        if date_list_sha256(dates) != value["outer_test_sha256"][str(outer)]:
            raise ValueError(f"outer {outer} date hash mismatch")
    union = np.sort(np.concatenate(tuple(blocks.values())))
    if len(np.unique(union)) != 150:
        raise ValueError("STGF outer blocks overlap")
    if date_list_sha256(union) != value["all_outer_test_sha256"]:
        raise ValueError("STGF union hash mismatch")
    caa = np.asarray(
        sorted({date for dates in OUTER_TEST_DATES.values() for date in dates}),
        dtype="datetime64[D]",
    )
    mm_path = Path(value["prior_catalogs"]["MM_registry_path"])
    mm = json.loads(mm_path.read_text(encoding="utf-8"))
    mm_dates = np.asarray(
        sorted(
            {
                date
                for dates in mm["outer_test_dates"].values()
                for date in dates
            }
        ),
        dtype="datetime64[D]",
    )
    if np.intersect1d(union, np.union1d(caa, mm_dates)).size:
        raise ValueError("STGF test dates overlap prior CAA/MM catalogs")
    value["_blocks"] = blocks
    return value


def build_stgf_development_gefcom2014(
    data_dir: str | Path, *, outer: int
) -> JointDataBundle:
    bundle = build_joint_nested_gefcom2014(str(data_dir), outer=outer)
    protocol = dict(bundle.protocol)
    protocol["stgf_evidence_role"] = (
        "exploratory development; calibration-only architecture selection"
    )
    return JointDataBundle(
        train=bundle.train,
        validation=bundle.validation,
        calibration=bundle.calibration,
        test=bundle.test,
        protocol=protocol,
        source=bundle.source,
    )


def build_stgf_confirmation_gefcom2014(
    data_dir: str | Path,
    *,
    outer: int,
    split_path: str | Path,
) -> JointDataBundle:
    registry = load_stgf_confirmation_splits(split_path)
    blocks: dict[int, np.ndarray] = registry["_blocks"]
    if outer not in blocks:
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
    all_stgf_tests = np.sort(np.concatenate(tuple(blocks.values())))
    train_dates = np.setdiff1d(
        legacy_train, all_stgf_tests, assume_unique=True
    )
    test_dates = blocks[outer]
    other_test_dates = np.setdiff1d(
        all_stgf_tests, test_dates, assume_unique=True
    )
    combined_validation = np.sort(
        np.concatenate((head_dates, calibration_dates, other_test_dates))
    )
    explicit = build_explicit_gefcom2014(
        data_dir,
        train_days=train_dates,
        validation_days=combined_validation,
        test_days=test_dates,
        protocol_metadata={
            "name": "stgf_postfreeze_internal_confirmation_v1",
            "outer": outer,
            "split_registry": str(Path(split_path).resolve()),
            "test_date_sha256": registry["outer_test_sha256"][str(outer)],
            "all_stgf_test_sha256": registry["all_outer_test_sha256"],
            "cross_outer_exclusion": registry["cross_outer_exclusion"],
        },
    )
    head = _select_dates(explicit.validation, head_dates)
    calibration = _select_dates(explicit.validation, calibration_dates)
    protocol: dict[str, Any] = {
        **explicit.protocol,
        "roles": registry["roles"],
        "calendar_day_counts": registry["calendar_day_counts"],
        "unused_other_outer_test_dates": other_test_dates.astype(str).tolist(),
        "confirmation_registry": {
            key: value for key, value in registry.items() if key != "_blocks"
        },
    }
    protocol["protocol_sha256"] = _protocol_sha256(protocol)
    nested = NestedDataBundle(
        train=explicit.train,
        validation=head,
        calibration=calibration,
        test=explicit.test,
        condition_standardizer=explicit.condition_standardizer,
        flat_condition_standardizer=explicit.flat_condition_standardizer,
        target_standardizer=explicit.target_standardizer,
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
    "build_stgf_confirmation_gefcom2014",
    "build_stgf_development_gefcom2014",
    "load_stgf_confirmation_splits",
]
