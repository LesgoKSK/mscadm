"""Frozen four-way GEFCom2014 data protocol for CAA-RAHC confirmation.

This module is the authoritative builder for the three registered CAA-RAHC
outer splits.  It deliberately reclassifies the historical seed-0 validation
dates as head-validation data and the historical seed-0 test dates as
calibration/development data.  Each new outer test contains 50 dates selected
from the historical training pool, and the corresponding base-model training
set is exactly ``historical_train - current_outer_test``.

The other two outer-test blocks remain available to the current outer model's
training set.  This is intentional repeated outer holdout: no model sees its
own outer test, all three outer tests are mutually disjoint, and all results
must be generated under one method lock before any outer score is inspected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from repro.data import (
    ArrayStandardizer,
    DataBundle,
    SplitData,
    _daily_arrays,
    _flat_feature_major,
    _split_days,
    load_complete_hours,
)


OUTER_TEST_DATES: dict[int, tuple[str, ...]] = {
    1: (
        "2012-01-08", "2012-01-31", "2012-02-17", "2012-02-29", "2012-03-07",
        "2012-03-23", "2012-04-18", "2012-04-24", "2012-05-06", "2012-05-20",
        "2012-06-18", "2012-06-21", "2012-07-16", "2012-07-20", "2012-08-15",
        "2012-08-20", "2012-09-08", "2012-09-11", "2012-10-04", "2012-10-05",
        "2012-10-29", "2012-11-12", "2012-11-23", "2012-12-19", "2012-12-27",
        "2013-01-18", "2013-01-28", "2013-02-25", "2013-02-27", "2013-03-16",
        "2013-03-18", "2013-04-28", "2013-04-30", "2013-05-01", "2013-05-05",
        "2013-05-10", "2013-06-19", "2013-06-26", "2013-07-08", "2013-07-21",
        "2013-08-10", "2013-08-17", "2013-09-13", "2013-09-30", "2013-10-14",
        "2013-10-24", "2013-11-12", "2013-11-16", "2013-12-04", "2013-12-31",
    ),
    2: (
        "2012-01-01", "2012-01-26", "2012-02-08", "2012-02-24", "2012-03-06",
        "2012-03-18", "2012-04-06", "2012-04-19", "2012-05-02", "2012-05-19",
        "2012-06-10", "2012-06-23", "2012-07-17", "2012-07-29", "2012-08-26",
        "2012-08-28", "2012-09-10", "2012-09-12", "2012-10-25", "2012-10-30",
        "2012-11-10", "2012-11-20", "2012-12-17", "2012-12-28", "2012-12-31",
        "2013-01-05", "2013-01-09", "2013-02-07", "2013-02-09", "2013-03-02",
        "2013-03-28", "2013-04-06", "2013-04-24", "2013-05-03", "2013-05-19",
        "2013-05-20", "2013-06-15", "2013-06-17", "2013-07-01", "2013-07-05",
        "2013-08-04", "2013-08-28", "2013-09-08", "2013-09-29", "2013-10-02",
        "2013-10-28", "2013-11-02", "2013-11-30", "2013-12-15", "2013-12-19",
    ),
    3: (
        "2012-01-05", "2012-01-18", "2012-02-02", "2012-02-27", "2012-03-02",
        "2012-03-29", "2012-04-26", "2012-04-29", "2012-05-01", "2012-05-10",
        "2012-06-04", "2012-06-17", "2012-07-22", "2012-07-26", "2012-08-03",
        "2012-08-06", "2012-09-02", "2012-09-05", "2012-10-03", "2012-10-11",
        "2012-10-23", "2012-11-02", "2012-11-30", "2012-12-08", "2012-12-13",
        "2012-12-25", "2013-01-08", "2013-01-20", "2013-02-04", "2013-02-13",
        "2013-03-03", "2013-03-29", "2013-04-15", "2013-04-17", "2013-05-07",
        "2013-05-24", "2013-06-23", "2013-06-25", "2013-07-10", "2013-07-28",
        "2013-08-09", "2013-08-13", "2013-09-06", "2013-09-21", "2013-10-18",
        "2013-10-29", "2013-11-18", "2013-11-22", "2013-12-02", "2013-12-26",
    ),
}

OUTER_TEST_DATE_HASHES = {
    1: "adc254f8283b0b3f4699c51d75f59b55b4695e9e2b7465626c1edd5cd1729431",
    2: "0c5dbec382d12d931ddd577a216b9ade7bd1fe9c7371e8ec29ac236f21723225",
    3: "78c371c86c51337993f861faedd3f0fcc5494e196e93b3973460c8b915bd0923",
}

LEGACY_DATE_HASHES = {
    "train": "420c87109bb239dd07312c5dd602dc4ee4af973704b6cfc1da393794e193e673",
    "validation": "ccb58fc472247170c3a88bbae94081b9c26a6156abb5fd1ecb40a0d25a860dcd",
    "test": "cecc2c1ab04779ecbc8ca762238e7359e8291b5585b0c37bd875b371be5291dd",
}


@dataclass(frozen=True)
class NestedDataBundle(DataBundle):
    """A :class:`DataBundle` with an independent calibration split.

    Inheriting from ``DataBundle`` preserves ``train`` and ``validation``
    compatibility with ``CRTrainer``.  ``validation`` is exclusively the
    statistics-head validation split; downstream CAA code must use
    ``calibration`` for fitting and model selection.
    """

    calibration: SplitData

    @property
    def head_validation(self) -> SplitData:
        return self.validation


def _as_dates(values: Iterable[object]) -> np.ndarray:
    return np.sort(np.asarray(list(values), dtype="datetime64[D]"))


def date_list_sha256(values: Iterable[object]) -> str:
    """Hash sorted ISO dates joined by newlines, with no trailing newline."""

    dates = _as_dates(values).astype(str).tolist()
    return hashlib.sha256("\n".join(dates).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _input_fingerprint(data_dir: Path) -> dict[str, Any]:
    names = ["TestTar_W.csv"]
    names.extend(f"TestPred_W_Zone{zone}.csv" for zone in range(1, 11))
    names.extend(f"Train_W_Zone{zone}.csv" for zone in range(1, 11))
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"path": name, "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }


def _array_fingerprint(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if np.issubdtype(value.dtype, np.datetime64):
            value = value.astype("datetime64[D]").astype(np.int64)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _assert_registered_outer_catalog(legacy_train: np.ndarray) -> None:
    outer_arrays = {index: _as_dates(values) for index, values in OUTER_TEST_DATES.items()}
    for index, values in outer_arrays.items():
        if len(values) != 50 or len(np.unique(values)) != 50:
            raise RuntimeError(f"registered outer {index} is not a unique 50-date block")
        if date_list_sha256(values) != OUTER_TEST_DATE_HASHES[index]:
            raise RuntimeError(f"registered outer {index} date hash mismatch")
        if not np.array_equal(np.intersect1d(values, legacy_train), values):
            raise RuntimeError(f"registered outer {index} contains a non-legacy-train date")
    for left in OUTER_TEST_DATES:
        for right in OUTER_TEST_DATES:
            if left < right and np.intersect1d(outer_arrays[left], outer_arrays[right]).size:
                raise RuntimeError(f"registered outer blocks {left} and {right} overlap")


def _split_protocol_sha256(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_nested_gefcom2014(
    data_dir: str | Path = "Data",
    *,
    outer: int,
) -> NestedDataBundle:
    """Build one frozen 581/50/50/50 CAA-RAHC outer split.

    Parameters
    ----------
    data_dir:
        Directory containing the 21 official GEFCom2014 wind CSV files.
    outer:
        One-based registered outer index in ``{1, 2, 3}``.
    """

    if outer not in OUTER_TEST_DATES:
        raise ValueError(f"outer must be one of {sorted(OUTER_TEST_DATES)}, got {outer!r}")
    root = Path(data_dir)
    input_fingerprint = _input_fingerprint(root)
    hours = load_complete_hours(root, tuple(range(1, 11)))
    condition, target, zone, day = _daily_arrays(hours)
    day = day.astype("datetime64[D]")
    available = np.sort(np.unique(day))
    if len(available) != 731:
        raise RuntimeError(f"expected 731 complete calendar dates, found {len(available)}")

    legacy_train, legacy_validation, legacy_test = _split_days(available, 50, 0)
    legacy = {
        "train": np.sort(legacy_train.astype("datetime64[D]")),
        "validation": np.sort(legacy_validation.astype("datetime64[D]")),
        "test": np.sort(legacy_test.astype("datetime64[D]")),
    }
    for name, values in legacy.items():
        actual_hash = date_list_sha256(values)
        if actual_hash != LEGACY_DATE_HASHES[name]:
            raise RuntimeError(
                f"legacy seed-0 {name} dates drifted: {actual_hash} != {LEGACY_DATE_HASHES[name]}"
            )
    _assert_registered_outer_catalog(legacy["train"])

    outer_test = _as_dates(OUTER_TEST_DATES[outer])
    train_days = np.setdiff1d(legacy["train"], outer_test, assume_unique=True)
    head_validation_days = legacy["validation"]
    calibration_days = legacy["test"]
    split_dates = {
        "train": train_days,
        "head_validation": head_validation_days,
        "calibration": calibration_days,
        "test": outer_test,
    }
    expected_counts = {"train": 581, "head_validation": 50, "calibration": 50, "test": 50}
    for name, values in split_dates.items():
        if len(values) != expected_counts[name]:
            raise RuntimeError(f"{name} has {len(values)} dates, expected {expected_counts[name]}")
    names = tuple(split_dates)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if np.intersect1d(split_dates[left], split_dates[right]).size:
                raise RuntimeError(f"{left} and {right} date sets overlap")
    union = np.unique(np.concatenate(tuple(split_dates.values())))
    if not np.array_equal(union, available):
        raise RuntimeError("four-way split does not partition all 731 available dates")
    for other, values in OUTER_TEST_DATES.items():
        if other != outer and not np.array_equal(
            np.intersect1d(_as_dates(values), train_days), _as_dates(values)
        ):
            raise RuntimeError(f"outer {other} test block must remain in outer {outer} train")

    masks = {
        "train": np.isin(day, train_days),
        "validation": np.isin(day, head_validation_days),
        "calibration": np.isin(day, calibration_days),
        "test": np.isin(day, outer_test),
    }
    expected_zone_days = {"train": 5_810, "validation": 500, "calibration": 500, "test": 500}
    for name, mask in masks.items():
        if int(mask.sum()) != expected_zone_days[name]:
            raise RuntimeError(
                f"{name} has {int(mask.sum())} zone-days, expected {expected_zone_days[name]}"
            )

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

    date_strings = {
        name: values.astype(str).tolist() for name, values in split_dates.items()
    }
    date_hashes = {name: date_list_sha256(values) for name, values in split_dates.items()}
    legacy_protocol = {
        name: {
            "dates": values.astype(str).tolist(),
            "sha256": LEGACY_DATE_HASHES[name],
        }
        for name, values in legacy.items()
    }
    outer_catalog = {
        str(index): {
            "dates": list(values),
            "sha256": OUTER_TEST_DATE_HASHES[index],
        }
        for index, values in OUTER_TEST_DATES.items()
    }
    data_fingerprint = _array_fingerprint(
        condition=condition,
        target=target,
        zone=zone,
        day=day,
    )
    protocol: dict[str, Any] = {
        "name": "caa_rahc_fixed_nested_outer_v1",
        "outer": int(outer),
        "legacy_split": {"seed": 0, "split_days": 50, "dates": legacy_protocol},
        "roles": {
            "train": "legacy seed-0 train minus current outer test",
            "validation": "legacy seed-0 validation; CR statistics-head selection only",
            "calibration": "legacy seed-0 test; CAA fitting and selection only",
            "test": "current registered outer test; evaluation only",
        },
        "dates": date_strings,
        "date_sha256": date_hashes,
        "outer_test_catalog": outer_catalog,
        "calendar_day_counts": expected_counts,
        "zone_day_counts": expected_zone_days,
        "zones": list(range(1, 11)),
        "hours_per_zone_day": 24,
        "other_outer_test_dates_remain_in_train": True,
        "standardizers_fit_on": "train only",
        "input_fingerprint": input_fingerprint,
        "data_fingerprint": {"algorithm": "sha256", "sha256": data_fingerprint},
    }
    protocol["protocol_sha256"] = _split_protocol_sha256(protocol)

    return NestedDataBundle(
        train=select("train"),
        validation=select("validation"),
        calibration=select("calibration"),
        test=select("test"),
        condition_standardizer=condition_scaler,
        flat_condition_standardizer=flat_scaler,
        target_standardizer=target_scaler,
        protocol=protocol,
    )


__all__ = [
    "LEGACY_DATE_HASHES",
    "NestedDataBundle",
    "OUTER_TEST_DATES",
    "OUTER_TEST_DATE_HASHES",
    "build_nested_gefcom2014",
    "date_list_sha256",
]

