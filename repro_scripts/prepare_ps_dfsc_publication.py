"""Canonical audited identity-baseline preparation entry point."""

import repro_scripts.prepare_ps_dfsc_training_final as preparation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)


preparation.solve_two_stage_suc = solve_two_stage_suc
preparation.evaluate_realized = evaluate_realized


if __name__ == "__main__":
    preparation.main()
