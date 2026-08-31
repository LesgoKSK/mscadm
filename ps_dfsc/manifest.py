from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock_manifest(
    *,
    split_registry: str | Path,
    config: dict[str, Any],
    model_paths: list[str | Path],
    outer: int,
    beta: float,
    candidate_id: str,
    safety_decision: dict[str, Any],
) -> dict[str, Any]:
    if outer not in (1, 2, 3):
        raise ValueError("outer must be 1, 2, or 3")
    models = {
        str(Path(path).resolve()): file_sha256(path) for path in model_paths
    }
    manifest: dict[str, Any] = {
        "schema": "ps_dfsc_lock_manifest_v1",
        "outer": outer,
        "split_registry": str(Path(split_registry).resolve()),
        "split_registry_sha256": file_sha256(split_registry),
        "config": config,
        "model_sha256": models,
        "beta": float(beta),
        "candidate_id": str(candidate_id),
        "safety_decision": safety_decision,
        "test_truth_accessed": False,
    }
    payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return manifest


def verify_lock_manifest(manifest: dict[str, Any]) -> None:
    value = dict(manifest)
    expected = value.pop("manifest_sha256", None)
    if value.get("schema") != "ps_dfsc_lock_manifest_v1":
        raise ValueError("unexpected lock manifest schema")
    if value.get("test_truth_accessed") is not False:
        raise ValueError("manifest was not created before test truth access")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed != expected:
        raise ValueError("lock manifest hash mismatch")
    if file_sha256(value["split_registry"]) != value["split_registry_sha256"]:
        raise ValueError("split registry changed after locking")
    for path, digest in value["model_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"locked model changed: {path}")


__all__ = ["build_lock_manifest", "file_sha256", "verify_lock_manifest"]
