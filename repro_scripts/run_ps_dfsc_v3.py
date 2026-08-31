"""Corrected PS-DFSC CLI entry point.

Use this module for training. It replaces the v1 trainer with the corrected
four-dimensional archive implementation before dispatching CLI commands.
"""

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.training_v2 import train_calibrator

pipeline.train_calibrator = train_calibrator


if __name__ == "__main__":
    pipeline.main()
