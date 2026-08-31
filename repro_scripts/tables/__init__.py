"""Authoritative table builder package.

This package intentionally shadows the early ``repro_scripts/tables.py``
prototype.  The package version distinguishes independently trained
single-zone models from slices of all-zone models.
"""

from repro_scripts.tables_cli import main

__all__ = ["main"]
