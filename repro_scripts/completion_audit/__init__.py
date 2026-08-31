"""BOM-aware command wrapper for the completion audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_source = Path(__file__).resolve().parent.parent / "completion_audit.py"
_spec = importlib.util.spec_from_file_location("repro_scripts._completion_audit_impl", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load completion audit from {_source}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)
_original_read_text = Path.read_text


def _bom_aware_read_text(self: Path, encoding=None, errors=None):
    raw = self.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors=errors or "strict")
    return raw.decode(encoding or "utf-8", errors=errors or "strict")


def main() -> None:
    Path.read_text = _bom_aware_read_text
    try:
        _implementation.main()
    finally:
        Path.read_text = _original_read_text


__all__ = ["main"]
