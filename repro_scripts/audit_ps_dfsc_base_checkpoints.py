"""Deep finite-value audit for all nine formal MM-JDWind checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "ps_dfsc"
            / "base"
            / "checkpoint_finite_audit.json"
        ),
    )
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for outer in (1, 2, 3):
        for seed in (0, 1, 2):
            run = (
                ROOT
                / "outputs"
                / "ps_dfsc"
                / "base"
                / f"outer{outer}"
                / "runs"
                / f"seed{seed}"
            )
            checkpoint = torch.load(
                run / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
            history_file = json.loads(
                (run / "history.json").read_text(encoding="utf-8")
            )
            checks = {
                "checkpoint_dict": isinstance(checkpoint, dict),
                "name_mm_jdwind": checkpoint.get("name") == "mm_jdwind",
                "seed_matches": checkpoint.get("seed") == seed,
                "protocol_outer_matches": (
                    checkpoint.get("protocol", {}).get("outer") == outer
                ),
                "stage_final": checkpoint.get("stage") == "flow_validation_selected",
                "positive_step": int(checkpoint.get("step", -1)) >= 0,
                "model_nonempty": bool(checkpoint.get("model")),
                "model_finite": finite_tree(checkpoint.get("model", {})),
                "embedded_history_finite": finite_tree(
                    checkpoint.get("history", {})
                ),
                "validation_finite": finite_tree(
                    checkpoint.get("validation", {})
                ),
                "history_file_finite": finite_tree(history_file),
            }
            records.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "checkpoint": str((run / "final.pt").resolve()),
                    "step": int(checkpoint.get("step", -1)),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            del checkpoint
            gc.collect()
    result = {
        "schema": "ps_dfsc_base_checkpoint_finite_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "count": len(records),
        "passed": len(records) == 9 and all(item["passed"] for item in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": result["count"],
                "passed": result["passed"],
            },
            ensure_ascii=False,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
