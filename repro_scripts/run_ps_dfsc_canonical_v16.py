"""Canonical finite-gradient PS-DFSC CLI with fail-closed checkpoints."""

import ps_dfsc.evaluation_publication as evaluation
import ps_dfsc.pipeline_runtime_publication as training_runtime
import ps_dfsc.resumable_training_runtime_v2 as resumable
import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.exact_suc_publication_v3 import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.midpoint_suc_recovery import (
    solve_midpoint_suc_with_recovery,
)
from ps_dfsc.training_v6 import train_calibrator
from repro_scripts.ps_dfsc_publication_runtime_v3 import (
    calibrate_archive,
    safety_gate,
)


resumable.train_calibrator = train_calibrator
evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
training_runtime.solve_two_stage_suc = solve_midpoint_suc_with_recovery
pipeline.solve_two_stage_suc = solve_two_stage_suc
pipeline.train = resumable.train_command
pipeline.calibrate_archive = calibrate_archive
pipeline.safety_gate = safety_gate
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases


if __name__ == "__main__":
    pipeline.main()
