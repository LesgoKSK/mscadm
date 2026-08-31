from __future__ import annotations

from pathlib import Path

import numpy as np

from caa_rahc.nested_data import OUTER_TEST_DATES
from mm_jdwind.confirmation_data import load_confirmation_splits


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_registry_is_disjoint_from_caa_outer_tests() -> None:
    registry = load_confirmation_splits(
        ROOT / "repro_configs" / "mm_jdwind_confirmation_splits.json"
    )
    blocks = registry["_blocks"]
    assert set(blocks) == {1, 2, 3}
    union = np.concatenate(tuple(blocks.values()))
    assert len(np.unique(union)) == 150
    old = np.asarray(
        sorted({date for values in OUTER_TEST_DATES.values() for date in values}),
        dtype="datetime64[D]",
    )
    assert np.intersect1d(union, old).size == 0
    assert registry["created_before_mm_development_test_generation"] is True
