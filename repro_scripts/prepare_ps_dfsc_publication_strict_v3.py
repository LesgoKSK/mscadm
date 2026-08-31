"""Final strict preparer with publication reported-gap success semantics."""

import repro_scripts.prepare_ps_dfsc_publication_strict_v2 as strict
from ps_dfsc.exact_suc_publication_v3 import (
    evaluate_realized,
    solve_two_stage_suc,
)


strict.preparation.solve_two_stage_suc = solve_two_stage_suc
strict.preparation.evaluate_realized = evaluate_realized


if __name__ == "__main__":
    strict.preparation.main()
