from __future__ import annotations

import json

import numpy as np
import pandas as pd

import repro_scripts.run_ps_dfsc_locked_confirmation_v3 as reuse


def distribution(path, scenario_delta=0.0):
    np.savez_compressed(
        path,
        suc_scenarios=np.asarray([[[1.0 + scenario_delta]]]),
        suc_probabilities=np.asarray([1.0]),
    )


def test_exact_reuse_requires_array_identical_suc_inputs(tmp_path):
    candidate = tmp_path / "ps_dfsc_test.npz"
    identity = tmp_path / "identity_test.npz"
    distribution(candidate)
    distribution(identity)
    assert reuse._exact_inputs_identical(candidate, identity)
    distribution(identity, scenario_delta=0.1)
    assert not reuse._exact_inputs_identical(candidate, identity)


def test_fallback_reuse_is_explicit_and_changes_method_metadata(tmp_path):
    lock = tmp_path / "lock_manifest.json"
    lock.write_text(
        json.dumps({"used_identity_fallback": True}),
        encoding="utf-8",
    )
    candidate_distribution = tmp_path / "ps_dfsc_test.npz"
    identity_distribution = tmp_path / "identity_test.npz"
    distribution(candidate_distribution)
    distribution(identity_distribution)
    candidate_csv = tmp_path / "exact_ps_dfsc.csv"
    pd.DataFrame(
        [
            {
                "outer": 1,
                "date": "2014-01-01",
                "method": "PS-DFSC identity fallback",
                "used_fallback": True,
                "fallback_reason": "outer fallback",
                "status": "solution_available",
                "realized_total_cost": 1.0,
            }
        ]
    ).to_csv(candidate_csv, index=False)
    candidate_csv.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "schema": "ps_dfsc_cached_exact_evaluation_v3",
                "cases": 1,
                "successful_cases": 1,
                "method": "PS-DFSC identity fallback",
            }
        ),
        encoding="utf-8",
    )
    identity_csv = tmp_path / "exact_identity.csv"
    identity_summary = identity_csv.with_suffix(".summary.json")
    (tmp_path / "truth.npz").write_bytes(b"truth")
    (tmp_path / "mapping.json").write_text("{}", encoding="utf-8")
    original_root = reuse.pipeline.ROOT
    reuse.pipeline.ROOT = tmp_path
    try:
        reuse._write_fallback_identity_reuse(
            label="outer1_identity_exact",
            command=["python", "exact"],
            inputs=[
                lock,
                identity_distribution,
                tmp_path / "truth.npz",
                tmp_path / "mapping.json",
            ],
            outputs=[identity_csv, identity_summary],
            attempts=1,
        )
    finally:
        reuse.pipeline.ROOT = original_root
    frame = pd.read_csv(identity_csv)
    summary = json.loads(identity_summary.read_text(encoding="utf-8"))
    assert frame.loc[0, "method"] == "MM-JDWind identity"
    assert not bool(frame.loc[0, "used_fallback"])
    assert summary["identity_fallback_exact_reuse"] is True
    assert summary["newly_solved_cases"] == 0
