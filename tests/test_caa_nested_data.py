from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from caa_rahc.nested_data import (
    LEGACY_DATE_HASHES,
    OUTER_TEST_DATES,
    OUTER_TEST_DATE_HASHES,
    NestedDataBundle,
    build_nested_gefcom2014,
    date_list_sha256,
)
from repro.data import ArrayStandardizer, DataBundle, SplitData


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outer_one() -> NestedDataBundle:
    return build_nested_gefcom2014(ROOT / "Data", outer=1)


def test_registered_outer_lists_are_fixed_balanced_and_disjoint() -> None:
    assert set(OUTER_TEST_DATES) == {1, 2, 3}
    arrays = {
        index: np.asarray(values, dtype="datetime64[D]")
        for index, values in OUTER_TEST_DATES.items()
    }
    for index, values in arrays.items():
        assert len(values) == 50
        assert len(np.unique(values)) == 50
        assert date_list_sha256(values) == OUTER_TEST_DATE_HASHES[index]
        month_counts = [np.sum(np.char.startswith(values.astype(str), f"2012-{m:02d}"))
                        + np.sum(np.char.startswith(values.astype(str), f"2013-{m:02d}"))
                        for m in range(1, 13)]
        assert min(month_counts) == 4
        assert max(month_counts) == 5
    assert not np.intersect1d(arrays[1], arrays[2]).size
    assert not np.intersect1d(arrays[1], arrays[3]).size
    assert not np.intersect1d(arrays[2], arrays[3]).size


def test_nested_bundle_is_crtrainer_compatible_and_has_exact_counts(
    outer_one: NestedDataBundle,
) -> None:
    bundle = outer_one
    assert isinstance(bundle, DataBundle)
    assert isinstance(bundle.train, SplitData)
    assert isinstance(bundle.validation, SplitData)
    assert bundle.head_validation is bundle.validation
    assert isinstance(bundle.calibration, SplitData)
    assert isinstance(bundle.test, SplitData)

    assert len(bundle.train) == 5_810
    assert len(bundle.validation) == 500
    assert len(bundle.calibration) == 500
    assert len(bundle.test) == 500
    assert bundle.train.condition.shape == (5_810, 24, 20)
    assert bundle.validation.condition.shape == (500, 24, 20)
    assert bundle.calibration.condition.shape == (500, 24, 20)
    assert bundle.test.condition.shape == (500, 24, 20)

    split_days = {
        "train": np.unique(bundle.train.day),
        "head_validation": np.unique(bundle.validation.day),
        "calibration": np.unique(bundle.calibration.day),
        "test": np.unique(bundle.test.day),
    }
    assert {name: len(values) for name, values in split_days.items()} == {
        "train": 581,
        "head_validation": 50,
        "calibration": 50,
        "test": 50,
    }
    names = tuple(split_days)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            assert not np.intersect1d(split_days[left], split_days[right]).size
    assert len(np.unique(np.concatenate(tuple(split_days.values())))) == 731
    for split in (bundle.train, bundle.validation, bundle.calibration, bundle.test):
        _, counts = np.unique(split.day, return_counts=True)
        assert np.array_equal(np.unique(counts), np.asarray([10]))


def test_protocol_freezes_legacy_roles_and_other_outer_dates_remain_in_train(
    outer_one: NestedDataBundle,
) -> None:
    protocol = outer_one.protocol
    assert protocol["name"] == "caa_rahc_fixed_nested_outer_v1"
    assert protocol["outer"] == 1
    assert protocol["calendar_day_counts"] == {
        "train": 581,
        "head_validation": 50,
        "calibration": 50,
        "test": 50,
    }
    assert protocol["other_outer_test_dates_remain_in_train"] is True

    legacy = protocol["legacy_split"]["dates"]
    for name in ("train", "validation", "test"):
        assert date_list_sha256(legacy[name]["dates"]) == LEGACY_DATE_HASHES[name]
        assert legacy[name]["sha256"] == LEGACY_DATE_HASHES[name]
    assert protocol["dates"]["head_validation"] == legacy["validation"]["dates"]
    assert protocol["dates"]["calibration"] == legacy["test"]["dates"]
    expected_train = np.setdiff1d(
        np.asarray(legacy["train"]["dates"], dtype="datetime64[D]"),
        np.asarray(OUTER_TEST_DATES[1], dtype="datetime64[D]"),
    )
    assert np.array_equal(np.unique(outer_one.train.day), expected_train)
    train_days = set(np.unique(outer_one.train.day).astype(str).tolist())
    assert set(OUTER_TEST_DATES[2]).issubset(train_days)
    assert set(OUTER_TEST_DATES[3]).issubset(train_days)

    for name, dates in protocol["dates"].items():
        assert date_list_sha256(dates) == protocol["date_sha256"][name]
    for index, registered in protocol["outer_test_catalog"].items():
        assert registered["dates"] == list(OUTER_TEST_DATES[int(index)])
        assert registered["sha256"] == OUTER_TEST_DATE_HASHES[int(index)]


def test_all_standardizers_are_fit_on_train_only(outer_one: NestedDataBundle) -> None:
    expected_target = ArrayStandardizer.fit(outer_one.train.target, axes=0)
    np.testing.assert_allclose(outer_one.target_standardizer.mean, expected_target.mean)
    np.testing.assert_allclose(outer_one.target_standardizer.std, expected_target.std)
    np.testing.assert_allclose(
        outer_one.train.target_standard.mean(axis=0), 0.0, atol=2e-6
    )
    np.testing.assert_allclose(
        outer_one.train.condition[:, :, :10].mean(axis=0), 0.0, atol=3e-5
    )
    np.testing.assert_allclose(outer_one.train.flat_condition.mean(axis=0), 0.0, atol=3e-5)

    all_targets = np.concatenate(
        (
            outer_one.train.target,
            outer_one.validation.target,
            outer_one.calibration.target,
            outer_one.test.target,
        )
    )
    all_target_mean = all_targets.mean(axis=0, keepdims=True)
    assert np.max(np.abs(all_target_mean - outer_one.target_standardizer.mean)) > 1e-4
    assert outer_one.protocol["standardizers_fit_on"] == "train only"


def test_protocol_contains_verifiable_input_data_and_protocol_fingerprints(
    outer_one: NestedDataBundle,
) -> None:
    protocol = outer_one.protocol
    input_fingerprint = protocol["input_fingerprint"]
    assert input_fingerprint["algorithm"] == "sha256"
    assert len(input_fingerprint["files"]) == 21
    assert len(input_fingerprint["combined_sha256"]) == 64
    assert all(record["bytes"] > 0 and len(record["sha256"]) == 64
               for record in input_fingerprint["files"])
    assert protocol["data_fingerprint"]["algorithm"] == "sha256"
    assert len(protocol["data_fingerprint"]["sha256"]) == 64

    recorded = protocol["protocol_sha256"]
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == recorded


def test_invalid_outer_is_rejected_before_reading_data() -> None:
    with pytest.raises(ValueError, match="outer must be one of"):
        build_nested_gefcom2014(ROOT / "Data", outer=0)
