from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from caa_rahc.nested_data import OUTER_TEST_DATES, date_list_sha256
from repro.data import build_gefcom2014


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "repro_configs" / "stgf_confirmation_splits.json"
MM_REGISTRY = ROOT / "repro_configs" / "mm_jdwind_confirmation_splits.json"
ALLOCATION_SEED = 20260730


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if DESTINATION.exists():
        raise RuntimeError(f"split registry is already frozen: {DESTINATION}")
    legacy = build_gefcom2014(ROOT / "Data", seed=0)
    legacy_train = np.sort(
        np.unique(legacy.train.day.astype("datetime64[D]"))
    )
    legacy_validation = np.sort(
        np.unique(legacy.validation.day.astype("datetime64[D]"))
    )
    legacy_test = np.sort(
        np.unique(legacy.test.day.astype("datetime64[D]"))
    )
    caa_dates = np.asarray(
        sorted({date for values in OUTER_TEST_DATES.values() for date in values}),
        dtype="datetime64[D]",
    )
    mm_value = json.loads(MM_REGISTRY.read_text(encoding="utf-8"))
    mm_dates = np.asarray(
        sorted(
            {
                date
                for values in mm_value["outer_test_dates"].values()
                for date in values
            }
        ),
        dtype="datetime64[D]",
    )
    prior_union = np.union1d(caa_dates, mm_dates)
    if len(caa_dates) != 150 or len(mm_dates) != 150 or len(prior_union) != 300:
        raise RuntimeError("prior outer-test catalogs are not disjoint 150-day blocks")
    available = np.setdiff1d(legacy_train, prior_union, assume_unique=True)
    if len(available) != 331:
        raise RuntimeError(f"expected 331 unseen legacy-train dates, found {len(available)}")
    shuffled = available.copy()
    np.random.default_rng(ALLOCATION_SEED).shuffle(shuffled)
    blocks = {
        outer: np.sort(shuffled[(outer - 1) * 50 : outer * 50])
        for outer in (1, 2, 3)
    }
    union = np.sort(np.concatenate(tuple(blocks.values())))
    if len(np.unique(union)) != 150:
        raise RuntimeError("new STGF outer blocks overlap")
    if np.intersect1d(union, prior_union).size:
        raise RuntimeError("new STGF tests overlap prior test catalogs")
    universal_train = np.setdiff1d(legacy_train, union, assume_unique=True)
    if len(universal_train) != 481:
        raise RuntimeError("unexpected universal training-day count")
    registry = {
        "schema": "stgf_confirmation_splits_v1",
        "created_before_stgf_development_test_generation": True,
        "allocation_seed": ALLOCATION_SEED,
        "allocation_source": (
            "legacy seed-0 training dates excluding all CAA and MM-JDWind "
            "outer-test dates"
        ),
        "cross_outer_exclusion": (
            "the union of all 150 STGF test dates is excluded from every "
            "STGF confirmation training set"
        ),
        "roles": {
            "train": "481 legacy-train dates excluding all STGF outer tests",
            "head_validation": "legacy seed-0 validation dates",
            "calibration": "legacy seed-0 test dates",
            "test": "current 50-day STGF outer block",
        },
        "calendar_day_counts": {
            "train": 481,
            "head_validation": 50,
            "calibration": 50,
            "test": 50,
            "unused_other_outer_tests": 100,
        },
        "prior_catalogs": {
            "CAA_outer_test_union_sha256": date_list_sha256(caa_dates),
            "MM_JDWind_outer_test_union_sha256": date_list_sha256(mm_dates),
            "prior_union_sha256": date_list_sha256(prior_union),
            "MM_registry_path": str(MM_REGISTRY.resolve()),
            "MM_registry_file_sha256": sha256_file(MM_REGISTRY),
        },
        "legacy_role_sha256": {
            "train": date_list_sha256(legacy_train),
            "validation": date_list_sha256(legacy_validation),
            "test": date_list_sha256(legacy_test),
        },
        "eligible_unseen_pool_count": len(available),
        "eligible_unseen_pool_sha256": date_list_sha256(available),
        "all_outer_test_sha256": date_list_sha256(union),
        "universal_train_sha256": date_list_sha256(universal_train),
        "outer_test_sha256": {
            str(outer): date_list_sha256(dates)
            for outer, dates in blocks.items()
        },
        "outer_test_dates": {
            str(outer): dates.astype(str).tolist()
            for outer, dates in blocks.items()
        },
        "evidence_scope": (
            "post-freeze internal confirmation on dates not used as test by "
            "CAA or MM-JDWind; not external validation"
        ),
    }
    DESTINATION.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frozen": str(DESTINATION.resolve()),
                "all_outer_test_sha256": registry["all_outer_test_sha256"],
                "outer_test_sha256": registry["outer_test_sha256"],
                "eligible_unseen_pool_count": len(available),
                "universal_train_count": len(universal_train),
            }
        )
    )


if __name__ == "__main__":
    main()
