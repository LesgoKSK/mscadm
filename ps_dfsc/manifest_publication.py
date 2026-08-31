"""Publication lock manifest with data, model and selection provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import file_sha256


def _hashed_paths(paths) -> dict[str, str]:
    result = {}
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = file_sha256(path)
    return result


def _payload_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_lock_manifest(
    *,
    split_registry: str | Path,
    config_path: str | Path,
    model_paths,
    data_paths,
    selection_path: str | Path,
    outer: int,
) -> dict[str, Any]:
    if outer not in (1, 2, 3):
        raise ValueError("outer must be 1, 2, or 3")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    fallback = bool(selection["used_identity_fallback"])
    candidate = selection.get("candidate")
    if fallback and candidate is not None:
        raise ValueError("identity fallback selection cannot contain a candidate")
    if not fallback and not isinstance(candidate, dict):
        raise ValueError("non-fallback selection must contain a candidate")
    candidate_id = "identity" if fallback else str(candidate["candidate_id"])
    beta = None if fallback else float(candidate["beta"])
    models = _hashed_paths(model_paths)
    if not fallback:
        checkpoint = str(Path(candidate["checkpoint"]).resolve())
        if checkpoint not in models:
            raise ValueError("selected candidate checkpoint is not locked")
    config_file = Path(config_path).resolve()
    manifest: dict[str, Any] = {
        "schema": "ps_dfsc_lock_manifest_v2",
        "outer": int(outer),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_registry": str(Path(split_registry).resolve()),
        "split_registry_sha256": file_sha256(split_registry),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "config": json.loads(config_file.read_text(encoding="utf-8")),
        "model_sha256": models,
        "data_sha256": _hashed_paths(data_paths),
        "selection_path": str(Path(selection_path).resolve()),
        "selection_sha256": file_sha256(selection_path),
        "selection": selection,
        "beta": beta,
        "candidate_id": candidate_id,
        "used_identity_fallback": fallback,
        "safety_decision": selection,
        "test_truth_accessed": False,
    }
    manifest["manifest_sha256"] = _payload_hash(manifest)
    return manifest


def verify_lock_manifest(manifest: dict[str, Any]) -> None:
    value = dict(manifest)
    expected = value.pop("manifest_sha256", None)
    if value.get("schema") != "ps_dfsc_lock_manifest_v2":
        raise ValueError("unexpected publication lock schema")
    if value.get("test_truth_accessed") is not False:
        raise ValueError("manifest was not created before test truth access")
    if _payload_hash(value) != expected:
        raise ValueError("lock manifest hash mismatch")
    scalar_files = {
        value["split_registry"]: value["split_registry_sha256"],
        value["config_path"]: value["config_sha256"],
        value["selection_path"]: value["selection_sha256"],
    }
    for path, digest in {
        **scalar_files,
        **value["model_sha256"],
        **value["data_sha256"],
    }.items():
        if file_sha256(path) != digest:
            raise ValueError(f"locked artifact changed: {path}")
    selection = json.loads(
        Path(value["selection_path"]).read_text(encoding="utf-8")
    )
    if selection != value["selection"]:
        raise ValueError("selection payload changed after locking")
    if bool(selection["used_identity_fallback"]) != bool(
        value["used_identity_fallback"]
    ):
        raise ValueError("selection/fallback mismatch")


__all__ = ["build_lock_manifest", "verify_lock_manifest"]
