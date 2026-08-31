from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd

from architecture_v1.data import (
    INTERIOR_STATE,
    build_architecture_v1_data,
    fill_targets_split_first,
    load_nwp_hours,
)
from architecture_v1.protocol import (
    ALL_ROLES,
    build_architecture_protocol,
    date_list_sha256,
    file_sha256,
    write_protocol_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1.json"
DATA = ROOT / "Data"


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert re.search(pattern, str(error)), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def test_frozen_protocol_quarantines_all_300_diagnostic_dates() -> None:
    available = np.arange(
        np.datetime64("2012-01-01"),
        np.datetime64("2014-01-01"),
        dtype="datetime64[D]",
    )
    protocol = build_architecture_protocol(
        CONFIG, available_days=available, smoke=False
    )
    expected_counts = {
        "train": 281,
        "validation": 50,
        "calibration": 50,
        "selection": 50,
        "r_seen": 300,
        "final": 0,
    }
    assert {role: len(protocol.role_dates(role)) for role in ALL_ROLES} == expected_counts
    assert date_list_sha256(protocol.role_dates("r_seen")) == (
        "bcb8fc9429642f9a4a0a7cde97ecec60a88ad35579991e820293e7cecba1c5f0"
    )
    assert np.intersect1d(
        protocol.role_dates("r_seen"), protocol.role_dates("selection")
    ).size == 0
    assert np.intersect1d(
        protocol.role_dates("r_seen"), protocol.role_dates("final")
    ).size == 0
    local_union = np.concatenate(
        [protocol.role_dates(role) for role in ALL_ROLES if role != "final"]
    )
    assert len(local_union) == 731
    assert len(np.unique(local_union)) == 731
    assert protocol.manifest["safety_invariants"] == {
        "split_before_target_imputation": True,
        "cross_role_target_fill": False,
        "r_seen_intersection_selection": 0,
        "r_seen_intersection_final": 0,
        "local_final_date_count": 0,
    }
    with _raises(RuntimeError, "external-only"):
        protocol.require_final_available()


def test_predictor_only_loader_never_reads_target_columns_or_test_targets() -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    original = pd.read_csv

    def recording_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        usecols = tuple(kwargs.get("usecols", ()))
        calls.append((Path(path).name, usecols))
        return original(path, *args, **kwargs)

    with patch.object(pd, "read_csv", recording_read_csv):
        values = load_nwp_hours(DATA, zones=(1,))
    assert len(values) == 731 * 24
    assert {name for name, _ in calls} == {
        "Train_W_Zone1.csv",
        "TestPred_W_Zone1.csv",
    }
    assert all("TARGETVAR" not in usecols for _, usecols in calls)
    assert all(name != "TestTar_W.csv" for name, _ in calls)


def test_target_fill_is_split_first_and_never_crosses_role_or_day() -> None:
    train_day = pd.Timestamp("2012-01-01")
    selection_day = pd.Timestamp("2012-01-02")
    hours = pd.DataFrame(
        {
            "role": ["train", "train", "selection", "selection"],
            "ZONEID": [1, 1, 1, 1],
            "day": [train_day, train_day, selection_day, selection_day],
            "timestamp": pd.to_datetime(
                [
                    "2012-01-01 01:00",
                    "2012-01-01 02:00",
                    "2012-01-02 01:00",
                    "2012-01-02 02:00",
                ]
            ),
            "TARGETVAR": [0.7, 0.9, np.nan, 0.2],
        }
    )
    filled, audit = fill_targets_split_first(hours)
    # A pre-split global ffill would leak 0.9 into selection.  The safe loader
    # instead backfills from the same role/zone/day value 0.2.
    assert np.isclose(filled.loc[2, "TARGETVAR"], 0.2)
    assert np.isnan(filled.loc[2, "TARGETVAR_RAW"])
    assert bool(filled.loc[2, "raw_missing_mask"])
    assert audit["cross_role_fill"] is False
    assert audit["cross_day_fill"] is False
    assert audit["raw_missing_cells_by_role"] == {"selection": 1, "train": 0}


def test_full_bundle_preserves_all_raw_missing_cells_and_clean_masks() -> None:
    bundle = build_architecture_v1_data(DATA, config_path=CONFIG, smoke=False)
    expected_counts = {
        "train": 281,
        "validation": 50,
        "calibration": 50,
        "selection": 50,
        "r_seen": 300,
    }
    assert {role: len(bundle.role(role)) for role in expected_counts} == expected_counts
    missing_by_role = {
        role: int(bundle.role(role).raw_missing_mask.sum()) for role in expected_counts
    }
    assert missing_by_role == {
        "train": 119,
        "validation": 9,
        "calibration": 2,
        "selection": 7,
        "r_seen": 38,
    }
    assert sum(missing_by_role.values()) == 175
    assert bundle.manifest["data_audit"]["split_first_missing"][
        "total_raw_missing_cells"
    ] == 175
    for role in expected_counts:
        split = bundle.role(role)
        assert split.condition.shape == (len(split), 10, 24, 20)
        assert np.isfinite(split.target).all()
        assert np.array_equal(np.isnan(split.target_raw), split.raw_missing_mask)
        assert np.all(split.state[split.raw_missing_mask] == INTERIOR_STATE)
        expected_ramp_mask = (
            split.observed_mask[..., :-1] & split.observed_mask[..., 1:]
        )
        assert np.array_equal(split.ramp_observed_mask, expected_ramp_mask)

    train = bundle.train
    expected_mean = np.nanmean(train.target_raw.astype(np.float64), axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(bundle.target_standardizer.mean, expected_mean, atol=1e-7)
    with _raises(RuntimeError, "external-only"):
        bundle.role("final")


def test_smoke_view_and_manifest_sidecar_are_deterministic() -> None:
    first = build_architecture_protocol(CONFIG, data_dir=DATA, smoke=True)
    second = build_architecture_protocol(CONFIG, data_dir=DATA, smoke=True)
    assert first.manifest["protocol_sha256"] == second.manifest["protocol_sha256"]
    assert first.manifest["smoke"]["date_counts"] == {
        "train": 8,
        "validation": 2,
        "calibration": 2,
        "selection": 2,
        "r_seen": 2,
        "final": 0,
    }
    assert first.manifest["config_path"] == "repro_configs/architecture_v1.json"
    assert set(first.manifest["calendar_month_blocks"]) == {
        "train",
        "validation",
        "calibration",
        "selection",
        "r_seen",
    }

    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "architecture_v1.protocol.manifest.json"
        hashes = write_protocol_manifest(first, destination)
        sidecar = Path(hashes["sidecar"])
        assert destination.is_file() and sidecar.is_file()
        assert hashes["manifest_file_sha256"] == file_sha256(destination)
        expected_sidecar = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert sidecar.read_text(encoding="ascii") == (
            f"{expected_sidecar}  {destination.name}\n"
        )


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
