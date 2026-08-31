"""Fast contract tests for the retained family-v1 training runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro_scripts.run_architecture_v1_family_v1_formal import (
    DEFAULT_CONFIG,
    _epoch_order,
    _gradient_audit,
    _validate_p0,
    _validation_bank_manifest,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_p0_artifact_authorizes_retained_training_without_weights() -> None:
    result = _validate_p0(_config())
    assert result["payload"]["status"] == "P0_GO"
    root = Path(result["path"]).parent
    assert not list(root.rglob("*.pt"))
    assert not list(root.rglob("*.pth"))


def test_epoch_order_is_common_deterministic_and_complete() -> None:
    first = _epoch_order(267, seed=13000, epoch=4)
    second = _epoch_order(267, seed=13000, epoch=4)
    different = _epoch_order(267, seed=13000, epoch=5)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.array_equal(np.sort(first), np.arange(267))


def test_validation_bank_uses_registered_four_seed_pairs() -> None:
    class Split:
        day = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-02-20"),
            dtype="datetime64[D]",
        )

    bank = _validation_bank_manifest(Split(), _config())
    assert bank["role"] == "validation"
    assert bank["replicates"] == 4
    assert len(bank["combined_deterministic_seeds"]) == 4
    assert len(set(bank["combined_deterministic_seeds"])) == 4


def test_gradient_audit_applies_frozen_late_epoch_thresholds() -> None:
    records = [
        {
            "epoch_1_based": epoch,
            "preclip_gradient_norm": 2.0 + (epoch % 2),
        }
        for epoch in range(1, 41)
    ]
    report = _gradient_audit(records, _config())
    assert report["audit_epoch_start_one_based"] == 31
    assert report["update_count"] == 10
    assert report["passed"] is True


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
