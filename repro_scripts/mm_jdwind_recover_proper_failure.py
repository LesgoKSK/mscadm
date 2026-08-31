from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def nonfinite_parameters(payload: dict[str, Any]) -> list[str]:
    return [
        name
        for name, value in payload["model"].items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reject a non-finite proper-score checkpoint and create a transparent "
            "flow-best fallback checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing flow_best.pt and proper_latest.pt",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    flow_path = run_dir / "flow_best.pt"
    proper_path = run_dir / "proper_latest.pt"
    final_path = run_dir / "final.pt"
    failure_path = run_dir / "proper_failure.json"
    if final_path.exists() or failure_path.exists():
        raise RuntimeError("recovery artifacts already exist; refusing to overwrite")
    flow = torch.load(flow_path, map_location="cpu", weights_only=False)
    proper = torch.load(proper_path, map_location="cpu", weights_only=False)
    flow_bad = nonfinite_parameters(flow)
    proper_bad = nonfinite_parameters(proper)
    validation = proper.get("validation", {})
    nonfinite_metrics = [
        name
        for name, value in validation.items()
        if not torch.isfinite(torch.tensor(float(value)))
    ]
    if flow_bad:
        raise RuntimeError(f"flow_best is not finite: {flow_bad[:5]}")
    if not proper_bad and not nonfinite_metrics:
        raise RuntimeError("proper checkpoint is finite; recovery is not justified")
    failure = {
        "schema": "mm_jdwind_proper_failure_v1",
        "decision": "reject_proper_and_fallback_to_flow_best",
        "reason": "non-finite proper-score validation and/or parameters",
        "proper_checkpoint": str(proper_path),
        "proper_checkpoint_sha256": sha256_file(proper_path),
        "proper_nonfinite_parameter_count": len(proper_bad),
        "proper_nonfinite_parameters": proper_bad,
        "proper_nonfinite_validation_metrics": nonfinite_metrics,
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": sha256_file(flow_path),
        "flow_nonfinite_parameter_count": 0,
        "scientific_status": (
            "The proper-score fine-tuning branch is a failed ablation. The final "
            "checkpoint is exactly the validation-selected flow checkpoint and "
            "must not be described as proper-score fine-tuned."
        ),
    }
    fallback = dict(flow)
    fallback["stage"] = "proper_fallback_flow"
    fallback["proper_status"] = "failed_nonfinite_rejected"
    fallback["fallback_source"] = str(flow_path)
    fallback["fallback_source_sha256"] = failure["flow_checkpoint_sha256"]
    fallback["recovery_record"] = str(failure_path)
    temporary = final_path.with_suffix(".pt.tmp")
    torch.save(fallback, temporary)
    temporary.replace(final_path)
    failure_path.write_text(
        json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
    )
    check = torch.load(final_path, map_location="cpu", weights_only=False)
    if nonfinite_parameters(check):
        raise RuntimeError("recovered final checkpoint is not finite")
    print(
        json.dumps(
            {
                "final": str(final_path),
                "failure_record": str(failure_path),
                "proper_nonfinite_parameter_count": len(proper_bad),
                "fallback_stage": fallback["stage"],
            }
        )
    )


if __name__ == "__main__":
    main()
