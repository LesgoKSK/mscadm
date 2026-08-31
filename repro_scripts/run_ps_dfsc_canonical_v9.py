"""Final canonical PS-DFSC CLI with per-unit differentiable QP."""

import ps_dfsc.pipeline_runtime_publication as training_runtime
import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.exact_suc_publication_v3 import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.fast_differentiable_suc_publication_v2 import (
    activate_scaled_publication_training_layer,
)
from ps_dfsc.pipeline_runtime_v2 import calibrate_command
from repro_scripts.ps_dfsc_publication_runtime_v3 import safety_gate_command


activate_scaled_publication_training_layer()
pipeline.train = training_runtime.train_command
pipeline.calibrate_command = calibrate_command
pipeline.safety_gate_command = safety_gate_command
training_runtime.solve_two_stage_suc = solve_two_stage_suc
pipeline.solve_two_stage_suc = solve_two_stage_suc
pipeline.evaluate_realized = evaluate_realized


if __name__ == "__main__":
    pipeline.main()
