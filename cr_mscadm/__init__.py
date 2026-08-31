"""Calibrated Residual MS-CADM research extension.

The package is intentionally separate from :mod:`repro` so the completed
baseline reproduction remains byte-for-byte untouched.
"""

from .calibration import CopulaPITCalibrator
from .model import ConditionalLocationScale, ResidualMSCADM

__all__ = ["ConditionalLocationScale", "ResidualMSCADM", "CopulaPITCalibrator"]
