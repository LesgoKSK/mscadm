"""Final canonical PS-DFSC training and validation CLI."""

import repro_scripts.run_ps_dfsc as pipeline
import ps_dfsc.evaluation_publication as evaluation
import ps_dfsc.pipeline_runtime_publication as training_runtime
from ps_dfsc.exact_suc_publication_v3 import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.fast_differentiable_suc_publication import (
    activate_publication_training_layer,
)
from repro_scripts.ps_dfsc_publication_runtime_v3 import (
    calibrate_archive,
    safety_gate,
)


activate_publication_training_layer()
evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
training_runtime.solve_two_stage_suc = solve_two_stage_suc
pipeline.solve_two_stage_suc = solve_two_stage_suc
pipeline.train = training_runtime.train_command
pipeline.calibrate_archive = calibrate_archive
pipeline.safety_gate = safety_gate
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases


if __name__ == "__main__":
    pipeline.main()
