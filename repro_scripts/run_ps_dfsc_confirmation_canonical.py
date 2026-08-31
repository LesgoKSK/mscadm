"""Canonical locked confirmation with audited exact-gap semantics."""

import repro_scripts.run_ps_dfsc as pipeline
import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)
from repro_scripts.run_ps_dfsc_confirmation_final import main


evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases


if __name__ == "__main__":
    main()
