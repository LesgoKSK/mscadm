"""Audit the nine frozen MM-JDWind retrains without touching test truth.

This audit deliberately inspects only:
  * the pre-registered split registry,
  * training metadata/checkpoint file hashes, and
  * development-train/development-validation scenario archives.

It does not open or generate confirmation-test scenario archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ps_dfsc.splits import load_ps_dfsc_splits, registry_sha256


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def date_strings(values: np.ndarray) -> list[str]:
    return [str(value.astype("datetime64[D]")) for value in values]


def inspect_archive(
    path: Path,
    *,
    expected_days: list[str],
    expected_members: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "scenarios",
            "observations",
            "zero_probability",
            "one_probability",
            "states",
            "day",
            "zones",
            "metadata",
        }
        missing = sorted(required.difference(archive.files))
        scenarios_shape = list(archive["scenarios"].shape)
        observations_shape = list(archive["observations"].shape)
        states_shape = list(archive["states"].shape)
        days = date_strings(archive["day"])
        finite = bool(
            np.isfinite(archive["scenarios"]).all()
            and np.isfinite(archive["observations"]).all()
            and np.isfinite(archive["zero_probability"]).all()
            and np.isfinite(archive["one_probability"]).all()
        )
        bounds_ok = bool(
            (archive["scenarios"] >= 0.0).all()
            and (archive["scenarios"] <= 1.0).all()
        )

    expected_shape = [len(expected_days), expected_members, 10, 24]
    checks = {
        "required_arrays_present": not missing,
        "scenario_shape": scenarios_shape == expected_shape,
        "observation_shape": observations_shape == [len(expected_days), 10, 24],
        "state_shape": states_shape == expected_shape,
        "dates_exact": days == expected_days,
        "finite": finite,
        "unit_interval": bounds_ok,
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "arrays": sorted(required),
        "missing_arrays": missing,
        "scenario_shape": scenarios_shape,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ps_dfsc" / "base" / "retrain_audit.json",
    )
    args = parser.parse_args()

    split_path = ROOT / "repro_configs" / "ps_dfsc_splits.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_semantic_sha256 = registry_sha256(load_ps_dfsc_splits(split_path))
    development = {
        "development_train": split["development_train_dates"],
        "development_validation": split["development_validation_dates"],
    }
    outer_dates = {
        int(key.replace("outer", "")): value
        for key, value in split["outer_test_dates"].items()
    }

    split_checks = {
        "development_train_count_100": len(development["development_train"]) == 100,
        "development_validation_count_50": len(development["development_validation"]) == 50,
        "outer_count_3": sorted(outer_dates) == [1, 2, 3],
        "outer_each_50": all(len(values) == 50 for values in outer_dates.values()),
        "development_roles_disjoint": not (
            set(development["development_train"])
            & set(development["development_validation"])
        ),
        "outer_tests_pairwise_disjoint": all(
            not (set(outer_dates[left]) & set(outer_dates[right]))
            for left in outer_dates
            for right in outer_dates
            if left < right
        ),
        "development_disjoint_from_all_outer_tests": not (
            (
                set(development["development_train"])
                | set(development["development_validation"])
            )
            & set().union(*(set(values) for values in outer_dates.values()))
        ),
        "allocation_does_not_use_target": split["allocation_uses_target"] is False,
    }

    outputs = ROOT / "outputs" / "ps_dfsc"
    base = outputs / "base"
    retrains: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    for outer in (1, 2, 3):
        outer_root = base / f"outer{outer}"
        for seed in (0, 1, 2):
            run = outer_root / "runs" / f"seed{seed}"
            required = [
                run / "final.pt",
                run / "history.json",
                run / "protocol.json",
                run / "run_config.json",
            ]
            present = all(path.is_file() for path in required)
            protocol = (
                json.loads((run / "protocol.json").read_text(encoding="utf-8"))
                if present
                else {}
            )
            config = (
                json.loads((run / "run_config.json").read_text(encoding="utf-8"))
                if present
                else {}
            )
            checks = {
                "required_files_present": present,
                "protocol_outer_matches": protocol.get("outer") == outer,
                "config_seed_matches": config.get("seed") == seed,
                "base_train_count_431": (
                    protocol.get("calendar_day_counts", {}).get("base_train") == 431
                ),
                "development_excluded": (
                    protocol.get("roles", {}).get("base_train")
                    == "MM-JDWind fitting; excludes all PS development and current outer test"
                ),
                "current_outer_test_excluded": protocol.get("test_days") == 50,
                "split_registry_hash_matches": (
                    protocol.get("split_registry_sha256") == split_semantic_sha256
                ),
                "test_date_hash_matches_registry": (
                    protocol.get("test_date_sha256")
                    == split["outer_test_sha256"][str(outer)]
                ),
                "standardizers_train_only": (
                    protocol.get("standardizers_fit_on") == "train only"
                ),
            }
            retrains.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "paths": [str(path.resolve()) for path in required],
                    "sha256": {
                        path.name: sha256(path) for path in required if path.is_file()
                    },
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )

            for role, expected_dates in development.items():
                archive_path = outer_root / "scenarios" / f"seed{seed}_{role}.npz"
                if archive_path.is_file():
                    detail = inspect_archive(
                        archive_path,
                        expected_days=expected_dates,
                        expected_members=100,
                    )
                else:
                    detail = {
                        "path": str(archive_path.resolve()),
                        "passed": False,
                        "checks": {"file_present": False},
                    }
                detail.update({"outer": outer, "seed": seed, "role": role})
                archives.append(detail)

        for role, expected_dates in development.items():
            pooled_path = outer_root / "pooled" / f"{role}_M100.npz"
            if pooled_path.is_file():
                detail = inspect_archive(
                    pooled_path,
                    expected_days=expected_dates,
                    expected_members=100,
                )
            else:
                detail = {
                    "path": str(pooled_path.resolve()),
                    "passed": False,
                    "checks": {"file_present": False},
                }
            detail.update({"outer": outer, "seed": "pooled", "role": role})
            archives.append(detail)

    confirmation_artifacts = sorted(
        str(path.resolve())
        for path in base.rglob("*confirmation_test*")
        if path.is_file()
    )
    overall_checks = {
        "nine_retrains": len(retrains) == 9,
        "all_retrains_pass": all(item["passed"] for item in retrains),
        "eighteen_seed_archives_and_six_pooled_archives": len(archives) == 24,
        "all_development_archives_pass": all(item["passed"] for item in archives),
        "split_checks_pass": all(split_checks.values()),
        "confirmation_artifact_count_zero": len(confirmation_artifacts) == 0,
    }
    audit = {
        "schema": "ps_dfsc_base_retrain_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training metadata and development scenarios only; test truth not opened",
        "split_registry": {
            "path": str(split_path.resolve()),
            "sha256": split_semantic_sha256,
            "checks": split_checks,
        },
        "retrain_count": len(retrains),
        "retrains": retrains,
        "archive_count": len(archives),
        "development_archives": archives,
        "confirmation_artifacts": confirmation_artifacts,
        "checks": overall_checks,
        "passed": all(overall_checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passed": audit["passed"],
                "checks": overall_checks,
            },
            ensure_ascii=False,
        )
    )
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
