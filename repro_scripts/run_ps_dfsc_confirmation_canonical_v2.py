"""Locked confirmation using corrected calibration, safety and exact audit."""

import repro_scripts.run_ps_dfsc as pipeline
import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)
from repro_scripts.ps_dfsc_publication_runtime import (
    calibrate_archive,
    safety_gate,
)
from repro_scripts.run_ps_dfsc_confirmation_final import main


evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
pipeline.calibrate_archive = calibrate_archive
pipeline.safety_gate = safety_gate
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases


if __name__ == "__main__":
    main()
