"""Recommended PS-DFSC CLI.

Training uses the tractable copper-plate fixed-commitment QP. All validation
costs and final results still use the exact DC-network MILP.
"""

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.fast_differentiable_suc import activate_fast_training_layer
from ps_dfsc.training_v2 import train_calibrator

activate_fast_training_layer()
pipeline.train_calibrator = train_calibrator


if __name__ == "__main__":
    pipeline.main()
