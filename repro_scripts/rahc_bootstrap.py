"""Run the registered calendar-day cluster bootstrap comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rahc.bootstrap import paired_calendar_day_bootstrap
from rahc.group_metrics import GroupingProtocol


WORKSPACE = Path(__file__).resolve().parents[1]
RAHC_ROOT = WORKSPACE / "outputs" / "rahc_cr_mscadm"
CR_ROOT = WORKSPACE / "outputs" / "cr_mscadm" / "scenarios"


def _load(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive["scenarios"]


def main() -> None:
    validation = [
        np.load(CR_ROOT / f"full_seed{seed}_validation_raw.npz", allow_pickle=False)
        for seed in range(3)
    ]
    test = [
        np.load(CR_ROOT / f"full_seed{seed}_test_raw.npz", allow_pickle=False)
        for seed in range(3)
    ]
    protocol = GroupingProtocol.fit(
        np.stack([archive["scenarios"] for archive in validation]), test[0]["zone"]
    )
    assignments = protocol.assign(
        np.stack([archive["scenarios"] for archive in test]), test[0]["zone"]
    )
    methods: dict[str, list[np.ndarray]] = {
        "legacy_G0": [
            _load(CR_ROOT / f"full_seed{seed}_test_calibrated.npz") for seed in range(3)
        ],
    }
    for name in (
        "C0_empirical",
        "C3_hour_zone",
        "C5_additive",
        "C6_RAHC",
        "C6_RAHC_linear",
    ):
        methods[name] = [
            _load(RAHC_ROOT / "scenarios" / "test" / f"{name}_seed{seed}.npz")
            for seed in range(3)
        ]
    comparisons = (
        ("legacy_G0", "C6_RAHC"),
        ("C0_empirical", "C6_RAHC"),
        ("legacy_G0", "C6_RAHC_linear"),
        ("legacy_G0", "C3_hour_zone"),
        ("legacy_G0", "C5_additive"),
    )
    results: dict[str, object] = {}
    for baseline, method in comparisons:
        result = paired_calendar_day_bootstrap(
            methods[baseline],
            methods[method],
            test[0]["observations"],
            test[0]["day"],
            assignments,
            replicates=10_000,
            random_seed=20260718,
            expected_unique_days=50,
        )
        key = f"{baseline}_vs_{method}"
        results[key] = result
        print(json.dumps({"comparison": key, "metrics": result["metrics"]}), flush=True)
    output = RAHC_ROOT / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "paired_calendar_day_bootstrap.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary.replace(destination)
    for archive in (*validation, *test):
        archive.close()


if __name__ == "__main__":
    main()
