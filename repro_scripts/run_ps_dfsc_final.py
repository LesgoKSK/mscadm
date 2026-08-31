"""Canonical PS-DFSC pipeline entry point."""

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.fast_differentiable_suc import activate_fast_training_layer
from ps_dfsc.pipeline_runtime import train_command

activate_fast_training_layer()
pipeline.train = train_command


if __name__ == "__main__":
    pipeline.main()
