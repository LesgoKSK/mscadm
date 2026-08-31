"""Corrected entry point for parallel midpoint cache filling."""

from __future__ import annotations

import repro_scripts.parallel_midpoint_refresh as _impl


def main() -> None:
    original = _impl._model_sha256

    def remember(model):
        value = original(model)
        _impl._MODEL_SHA = value
        return value

    _impl._model_sha256 = remember
    _impl.main()


if __name__ == "__main__":
    main()
