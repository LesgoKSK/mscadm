"""Authoritative entry point for the CAA-RAHC completion audit.

``caa_completion_audit`` is retained as the frozen audit engine.  This thin
wrapper installs datetime-aware JSON normalization without changing any audit
contract, then re-exports the registered API.  Formal completion commands
must invoke this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from . import caa_completion_audit as _engine


_ENGINE_JSONABLE = _engine._jsonable


def _datetime_aware_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return _ENGINE_JSONABLE(value)


_engine._jsonable = _datetime_aware_jsonable

AUTHORITATIVE_MODULE = _engine.AUTHORITATIVE_MODULE
EXPECTED_DAYS_PER_OUTER = _engine.EXPECTED_DAYS_PER_OUTER
EXPERIMENT_MANIFEST = _engine.EXPERIMENT_MANIFEST
FAMILIES = _engine.FAMILIES
FROZEN_MANIFEST = _engine.FROZEN_MANIFEST
OUTERS = _engine.OUTERS
SEEDS = _engine.SEEDS
SELECTION_ANALYSIS = _engine.SELECTION_ANALYSIS
SELECTION_LOCK = _engine.SELECTION_LOCK

audit_completion = _engine.audit_completion
run = audit_completion
canonical_json_sha256 = _engine.canonical_json_sha256
checkpoint_path = _engine.checkpoint_path
final_archive_path = _engine.final_archive_path
raw_archive_path = _engine.raw_archive_path
sha256_file = _engine.sha256_file
sidecar_path = _engine.sidecar_path


def main(argv: Sequence[str] | None = None) -> int:
    return _engine.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_MODULE",
    "EXPECTED_DAYS_PER_OUTER",
    "FAMILIES",
    "OUTERS",
    "SEEDS",
    "audit_completion",
    "canonical_json_sha256",
    "checkpoint_path",
    "final_archive_path",
    "main",
    "raw_archive_path",
    "run",
    "sha256_file",
    "sidecar_path",
]
