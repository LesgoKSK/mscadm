"""Canonical formal PS-DFSC training/validation CLI."""

import repro_scripts.run_ps_dfsc as pipeline
import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.fast_differentiable_suc_publication import (
    activate_publication_training_layer,
)
from ps_dfsc.pipeline_runtime_publication import train_command
from repro_scripts.ps_dfsc_publication_runtime_v2 import (
    calibrate_archive,
    safety_gate,
)


activate_publication_training_layer()
evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
pipeline.solve_two_stage_suc = solve_two_stage_suc
pipeline.train = train_command
pipeline.calibrate_archive = calibrate_archive
pipeline.safety_gate = safety_gate
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases


if __name__ == "__main__":
    pipeline.main()
