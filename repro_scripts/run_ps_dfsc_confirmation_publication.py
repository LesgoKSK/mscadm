"""Canonical locked confirmation entry with publication-grade solver audit."""

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.evaluation_publication import (
    run_exact_cases,
    summarize_exact_cases,
)
from repro_scripts.run_ps_dfsc_confirmation_final import main


pipeline.run_exact_cases = run_exact_cases
pipeline.summarize_exact_cases = summarize_exact_cases


if __name__ == "__main__":
    main()
