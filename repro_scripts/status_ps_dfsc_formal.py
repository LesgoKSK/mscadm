"""Concise authoritative status for the formal PS-DFSC experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "ps_dfsc"
NAMES = ("beta000", "beta025", "beta050", "beta100")

def midpoint_cache_counts(root: Path) -> tuple[int, int]:
    files = list(root.with_suffix(".refresh_cache").glob("day_*.npz"))
    compatible = 0
    for path in files:
        with np.load(path, allow_pickle=False) as value:
            compatible += int(
                "time_limit_seconds" in value
                and float(value["time_limit_seconds"]) == 600.0
            )
    return len(files), compatible


def main() -> None:
    outers = {}
    for outer in (1, 2, 3):
        outer_root = OUTPUT / f"outer{outer}"
        candidates = {}
        for name in NAMES:
            root = outer_root / "candidates" / name
            resume_path = root.with_suffix(".epoch_resume.pt")
            resume = None
            if resume_path.is_file():
                payload = torch.load(
                    resume_path,
                    map_location="cpu",
                    weights_only=False,
                )
                state = payload["state"]
                resume = {
                    "schema": payload["metadata"].get("schema"),
                    "next_epoch": int(state["next_epoch"]),
                    "history_epochs": len(state["history_epochs"]),
                    "midpoint_refresh_completed": bool(
                        payload["midpoint_refresh_completed"]
                    ),
                }
            candidates[name] = {
                "final_checkpoint": root.with_suffix(".pt").is_file(),
                "validation_record": root.with_suffix(
                    ".validation.json"
                ).is_file(),
                "resume": resume,
                "midpoint_cache_days_total": midpoint_cache_counts(root)[0],
                "midpoint_cache_days_registered_600s": midpoint_cache_counts(root)[1],
            }
        baseline_cache = outer_root / "baseline_cache"
        outers[str(outer)] = {
            "baseline_train_days": len(
                list((baseline_cache / "train").glob("day_*.npz"))
            ),
            "baseline_validation_days": len(
                list((baseline_cache / "validation").glob("day_*.npz"))
            ),
            "training_prepared": (
                outer_root / "development_train_prepared.npz"
            ).is_file(),
            "validation_prepared": (
                outer_root / "development_validation_prepared.npz"
            ).is_file(),
            "candidates": candidates,
            "lock_manifest": (
                outer_root / "lock_manifest.json"
            ).is_file(),
        }
    confirmation = OUTPUT / "confirmation"
    result = {
        "schema": "ps_dfsc_formal_status_v1",
        "outers": outers,
        "candidate_final_count": sum(
            item["final_checkpoint"]
            for outer in outers.values()
            for item in outer["candidates"].values()
        ),
        "candidate_validation_count": sum(
            item["validation_record"]
            for outer in outers.values()
            for item in outer["candidates"].values()
        ),
        "lock_count": sum(
            outer["lock_manifest"] for outer in outers.values()
        ),
        "confirmation_test_archives": len(
            list(
                (OUTPUT / "base").glob(
                    "outer*/**/*confirmation_test*"
                )
            )
        ),
        "confirmation_exact_csv_count": len(
            list(confirmation.glob("outer*/exact_*.csv"))
        ),
        "confirmation_complete": (
            confirmation / "completion_audit.json"
        ).is_file(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
