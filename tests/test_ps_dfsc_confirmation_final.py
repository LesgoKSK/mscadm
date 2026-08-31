from __future__ import annotations

import json

import numpy as np
import pytest

from repro_scripts.run_ps_dfsc_confirmation_final import (
    _identity_archive,
    _lock_uses_identity_fallback,
    _verify_selected_checkpoint,
)


def test_identity_fallback_detection_and_locked_checkpoint(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    lock = {
        "candidate_id": "beta025",
        "safety_decision": {"used_identity_fallback": False},
        "model_sha256": {str(checkpoint.resolve()): "digest"},
    }
    assert not _lock_uses_identity_fallback(lock)
    _verify_selected_checkpoint(lock, checkpoint)
    with pytest.raises(ValueError, match="not included"):
        _verify_selected_checkpoint(lock, tmp_path / "other.pt")

    lock["candidate_id"] = "identity"
    assert _lock_uses_identity_fallback(lock)
    lock["candidate_id"] = "none"
    lock["safety_decision"]["used_identity_fallback"] = True
    assert _lock_uses_identity_fallback(lock)


def test_identity_archive_marks_whole_outer_fallback(tmp_path):
    rng = np.random.default_rng(9)
    scenarios = rng.uniform(size=(2, 4, 10, 24)).astype(np.float32)
    pooled = tmp_path / "pooled.npz"
    np.savez_compressed(
        pooled,
        scenarios=scenarios,
        day=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "groups": [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]],
                "wind_buses": [3, 5, 7, 16, 21, 23],
                "zone_capacity_mw": 120.0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "identity_fallback.npz"
    _identity_archive(
        pooled,
        mapping,
        output,
        clusters=2,
        used_fallback=True,
        fallback_reason="outer_identity_fallback",
    )
    result = np.load(output, allow_pickle=False)
    assert np.array_equal(result["full_scenarios"], scenarios)
    assert np.allclose(result["probabilities"], 0.25)
    assert np.all(result["used_fallback"])
    assert set(result["fallback_reason"]) == {"outer_identity_fallback"}
    assert result["suc_scenarios"].shape == (2, 2, 6, 24)
    assert np.allclose(result["suc_probabilities"].sum(axis=1), 1.0)
