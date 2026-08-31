#!/usr/bin/env python3
"""Freeze family-v1 as a D0 sampler-contract No-Go without modifying old evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.training import file_sha256


def verified(relative: str, expected: str) -> dict[str, str]:
    path = ROOT / relative
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"evidence hash drifted: {relative}")
    return {"path": relative, "sha256": actual}


def main() -> int:
    root = ROOT / "outputs" / "architecture_v1_family_v1"
    output = root / "FAMILY_V1_NO_GO.freeze.json"
    if output.exists():
        raise FileExistsError(f"freeze already exists: {output}")
    evidence = {
        "protocol": verified("repro_configs/architecture_v1_family_v1.json", "d084852e74839b6aba0703cd6bc27388f73dad7ca52bf731a49ec89ac91c59e4"),
        "P0": verified("outputs/architecture_v1_family_v1/P0_preflight/P0_RESULT.json", "847f1337d41257609077a7396b41d95edcb8e76750d8e1f005ae9ceae636fc1b"),
        "training": verified("outputs/architecture_v1_family_v1/formal_training/TRAINING_RESULT.json", "b3c1342de12945e5af8e3223a472ff72b0d9ad5135750facd15a66d5c38026f7"),
        "evaluation_failure": verified("outputs/architecture_v1_family_v1/formal_evaluation/EVALUATION_FAILURE.json", "1a2de82aadd38dbad3b577d6fbb117a7783520f959b82f6e045c26ad4b1ad6de"),
        "boundary_diagnostic": verified("outputs/architecture_v1_family_v1/formal_evaluation/diagnostics/D0_seed3_sampling21000_boundary_diagnostic.json", "d3033230892afb13f893418690c94e053485cd99ef4ccf927e63af5b3d7f5cb7"),
    }
    f0_archives = []
    for seed in (21000, 21001, 21002):
        path = root / "formal_evaluation" / "archives" / f"F0_seed3_sampling{seed}.npz"
        f0_archives.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)})
    payload = {
        "schema": "architecture_v1_family_v1_no_go_freeze_v1",
        "status": "D0_SAMPLER_CONTRACT_NO_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "epsilon-prediction DDIM violated strict interior output semantics during first formal D0 validation archive",
        "diagnostic_summary": {
            "active_interior_coordinates": 1094220,
            "decoded_exact_zero": 469518,
            "decoded_exact_one": 507402,
            "decoded_boundary_total": 976920,
            "decoded_boundary_fraction": 0.89280035,
            "affected_validation_days": "50/50",
            "latent_min": -11196.73,
            "latent_max": 12674.73,
            "all_latents_finite": True
        },
        "decision": {
            "selected_family": None,
            "family_comparison_permitted": False,
            "old_D0_weights_must_not_be_reinterpreted_as_valid": True,
            "next_revision": "family-v1.1 prospectively frozen before implementation and training"
        },
        "evidence": evidence,
        "valid_partial_F0_archives": f0_archives,
        "target_access": {
            "selection": "sealed_and_unaccessed",
            "calibration": "sealed_and_unaccessed",
            "final": "unaccessed"
        }
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    digest = file_sha256(output)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"path": str(output), "sha256": digest, "status": payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
