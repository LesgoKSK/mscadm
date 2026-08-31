"""Canonical locked confirmation with complete safety diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import repro_scripts.run_ps_dfsc as pipeline
import repro_scripts.run_ps_dfsc_confirmation_final as confirmation
import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.manifest_publication import verify_lock_manifest
from repro_scripts.ps_dfsc_publication_runtime_v3 import (
    calibrate_archive,
    safety_gate,
)


def _verified_lock(path, outer):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_lock_manifest(value)
    if int(value["outer"]) != int(outer):
        raise ValueError("lock manifest outer does not match requested outer")
    return value


evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized
pipeline.calibrate_archive = calibrate_archive
pipeline.safety_gate = safety_gate
pipeline.run_exact_cases = evaluation.run_exact_cases
pipeline.summarize_exact_cases = evaluation.summarize_exact_cases
confirmation._verified_lock = _verified_lock


if __name__ == "__main__":
    confirmation.main()
