from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from caa_rahc.nested_data import OUTER_TEST_DATES, date_list_sha256
from repro.data import build_gefcom2014, load_complete_hours


def _load_outer_dates(path: str | Path) -> np.ndarray:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = value.get("outer_test_dates")
    if not isinstance(blocks, dict):
        raise ValueError(f"{path} has no outer_test_dates registry")
    return np.asarray(
        sorted({date for dates in blocks.values() for date in dates}),
        dtype="datetime64[D]",
    )


def _nwp_level_by_day(data_dir: str | Path) -> dict[np.datetime64, float]:
    frame = load_complete_hours(data_dir)
    grouped = frame.groupby("day", sort=True)["WS100"].mean()
    return {
        np.datetime64(day, "D"): float(value)
        for day, value in grouped.items()
    }


def _strata(
    dates: np.ndarray, levels: dict[np.datetime64, float]
) -> dict[tuple[int, int], list[np.datetime64]]:
    values = np.asarray([levels[date] for date in dates], dtype=np.float64)
    thresholds = np.quantile(values, [1 / 3, 2 / 3])
    result: dict[tuple[int, int], list[np.datetime64]] = {}
    for date, value in zip(dates, values):
        month = int(str(date)[5:7])
        level = int(np.digitize(value, thresholds))
        result.setdefault((month, level), []).append(date)
    return result


def _allocate(
    dates: np.ndarray,
    levels: dict[np.datetime64, float],
    capacities: tuple[int, ...],
    *,
    seed: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    blocks: list[list[np.datetime64]] = [[] for _ in capacities]
    per_stratum: list[dict[tuple[int, int], int]] = [{} for _ in capacities]
    strata = _strata(dates, levels)
    for key in sorted(strata):
        values = np.asarray(sorted(strata[key]), dtype="datetime64[D]")
        rng.shuffle(values)
        for date in values:
            candidates = [
                index
                for index, capacity in enumerate(capacities)
                if len(blocks[index]) < capacity
            ]
            if not candidates:
                break
            selected = min(
                candidates,
                key=lambda index: (
                    per_stratum[index].get(key, 0),
                    len(blocks[index]) / capacities[index],
                    index,
                ),
            )
            blocks[selected].append(date)
            per_stratum[selected][key] = per_stratum[selected].get(key, 0) + 1
    if tuple(map(len, blocks)) != capacities:
        raise RuntimeError(
            f"stratified allocation produced {tuple(map(len, blocks))}, "
            f"expected {capacities}"
        )
    return tuple(np.sort(np.asarray(block, dtype="datetime64[D]")) for block in blocks)


def build_ps_dfsc_split_registry(
    data_dir: str | Path,
    *,
    mm_registry: str | Path,
    stgf_registry: str | Path,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Build the frozen development and fresh outer split registry.

    Allocation uses dates, calendar month and NWP WS100 only; no target is read
    for stratification.
    """

    legacy = build_gefcom2014(data_dir, seed=0)
    legacy_train = np.sort(
        np.unique(legacy.train.day.astype("datetime64[D]"))
    )
    caa = np.asarray(
        sorted({date for dates in OUTER_TEST_DATES.values() for date in dates}),
        dtype="datetime64[D]",
    )
    mm = _load_outer_dates(mm_registry)
    stgf = _load_outer_dates(stgf_registry)
    prior = np.union1d(np.union1d(caa, mm), stgf)
    eligible = np.setdiff1d(legacy_train, prior, assume_unique=True)
    if len(mm) != 150:
        raise ValueError("MM development source must contain 150 unique dates")
    if len(eligible) < 150:
        raise ValueError("fewer than 150 fresh eligible dates remain")
    levels = _nwp_level_by_day(data_dir)
    development_train, development_validation = _allocate(
        mm, levels, (100, 50), seed=seed
    )
    outer_one, outer_two, outer_three = _allocate(
        eligible, levels, (50, 50, 50), seed=seed + 1
    )
    outer = {1: outer_one, 2: outer_two, 3: outer_three}
    union = np.sort(np.concatenate(tuple(outer.values())))
    if len(np.unique(union)) != 150 or np.intersect1d(union, prior).size:
        raise AssertionError("PS-DFSC confirmation allocation is not fresh/disjoint")
    return {
        "schema": "ps_dfsc_splits_v1",
        "allocation_seed": seed,
        "allocation_features": ["calendar_month", "NWP_WS100_tercile"],
        "allocation_uses_target": False,
        "development_source": str(Path(mm_registry)),
        "development_train_dates": [str(value) for value in development_train],
        "development_validation_dates": [
            str(value) for value in development_validation
        ],
        "development_train_sha256": date_list_sha256(development_train),
        "development_validation_sha256": date_list_sha256(
            development_validation
        ),
        "eligible_unseen_pool_count": int(len(eligible)),
        "eligible_unseen_pool_sha256": date_list_sha256(eligible),
        "outer_test_dates": {
            str(key): [str(value) for value in dates]
            for key, dates in outer.items()
        },
        "outer_test_sha256": {
            str(key): date_list_sha256(dates) for key, dates in outer.items()
        },
        "all_outer_test_sha256": date_list_sha256(union),
        "unused_fresh_dates": [
            str(value) for value in np.setdiff1d(eligible, union)
        ],
        "roles": {
            "development_train": "PS-DFSC optimization only",
            "development_validation": "safety gate, beta and candidate selection",
            "outer_test": "locked confirmation only",
        },
    }


def load_ps_dfsc_splits(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "ps_dfsc_splits_v1":
        raise ValueError("unexpected PS-DFSC split schema")
    development_train = np.asarray(
        value["development_train_dates"], dtype="datetime64[D]"
    )
    development_validation = np.asarray(
        value["development_validation_dates"], dtype="datetime64[D]"
    )
    outer = {
        int(key): np.asarray(dates, dtype="datetime64[D]")
        for key, dates in value["outer_test_dates"].items()
    }
    if len(development_train) != 100 or len(development_validation) != 50:
        raise ValueError("development split must be 100/50 days")
    if np.intersect1d(development_train, development_validation).size:
        raise ValueError("development train and validation overlap")
    if set(outer) != {1, 2, 3} or any(len(value) != 50 for value in outer.values()):
        raise ValueError("outer confirmation blocks must be 3x50")
    union = np.concatenate(tuple(outer.values()))
    if len(np.unique(union)) != 150:
        raise ValueError("outer confirmation blocks overlap")
    if date_list_sha256(development_train) != value["development_train_sha256"]:
        raise ValueError("development train hash mismatch")
    if (
        date_list_sha256(development_validation)
        != value["development_validation_sha256"]
    ):
        raise ValueError("development validation hash mismatch")
    for key, dates in outer.items():
        if date_list_sha256(dates) != value["outer_test_sha256"][str(key)]:
            raise ValueError(f"outer {key} hash mismatch")
    if date_list_sha256(np.sort(union)) != value["all_outer_test_sha256"]:
        raise ValueError("outer union hash mismatch")
    value["_development_train"] = np.sort(development_train)
    value["_development_validation"] = np.sort(development_validation)
    value["_outer"] = {key: np.sort(dates) for key, dates in outer.items()}
    return value


def registry_sha256(value: dict[str, Any]) -> str:
    public = {key: item for key, item in value.items() if not key.startswith("_")}
    payload = json.dumps(
        public, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "build_ps_dfsc_split_registry",
    "load_ps_dfsc_splits",
    "registry_sha256",
]
