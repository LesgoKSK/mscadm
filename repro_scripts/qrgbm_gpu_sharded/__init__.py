"""Compatibility entry point for the memory-bounded QRGBM implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from repro.data import build_gefcom2014 as _build_gefcom2014


_source = Path(__file__).resolve().parent.parent / "qrgbm_gpu_sharded.py"
_spec = importlib.util.spec_from_file_location("repro_scripts._qrgbm_gpu_sharded_impl", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load QRGBM implementation from {_source}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)


def _build_with_all_zone_default(data_dir, zones=None, seed: int = 0):
    selected = tuple(range(1, 11)) if zones is None else zones
    return _build_gefcom2014(data_dir, zones=selected, seed=seed)


_implementation.build_gefcom2014 = _build_with_all_zone_default
main = _implementation.main

__all__ = ["main"]
