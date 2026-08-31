"""Fair-comparison SUC command package.

This package shadows the early command module and guarantees that every model
is evaluated on the same test-day indices.
"""

from repro_scripts.suc_cli import main

__all__ = ["main"]
