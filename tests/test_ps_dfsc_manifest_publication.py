from __future__ import annotations

import json

import pytest

from ps_dfsc.manifest_publication import (
    build_lock_manifest,
    verify_lock_manifest,
)
from repro_scripts.lock_ps_dfsc_publication import _assert_confirmation_absent


def _file(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_publication_manifest_hashes_data_models_and_selection(tmp_path):
    splits = _file(tmp_path, "splits.json", "{}")
    config = _file(tmp_path, "config.json", '{"clusters": 20}')
    model = _file(tmp_path, "base.pt", "base")
    data = _file(tmp_path, "development.npz", "data")
    selection = _file(
        tmp_path,
        "selection.json",
        json.dumps(
            {
                "used_identity_fallback": True,
                "candidate": None,
                "reason": "no_safety_feasible_candidate",
            }
        ),
    )
    manifest = build_lock_manifest(
        split_registry=splits,
        config_path=config,
        model_paths=[model],
        data_paths=[data],
        selection_path=selection,
        outer=1,
    )
    verify_lock_manifest(manifest)
    assert manifest["candidate_id"] == "identity"
    assert manifest["beta"] is None
    data.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="locked artifact changed"):
        verify_lock_manifest(manifest)


def test_nonfallback_checkpoint_must_be_locked(tmp_path):
    splits = _file(tmp_path, "splits.json", "{}")
    config = _file(tmp_path, "config.json", "{}")
    base = _file(tmp_path, "base.pt", "base")
    candidate = _file(tmp_path, "candidate.pt", "candidate")
    data = _file(tmp_path, "development.npz", "data")
    selection = _file(
        tmp_path,
        "selection.json",
        json.dumps(
            {
                "used_identity_fallback": False,
                "candidate": {
                    "candidate_id": "beta025",
                    "beta": 0.25,
                    "checkpoint": str(candidate),
                },
            }
        ),
    )
    with pytest.raises(ValueError, match="checkpoint is not locked"):
        build_lock_manifest(
            split_registry=splits,
            config_path=config,
            model_paths=[base],
            data_paths=[data],
            selection_path=selection,
            outer=1,
        )


def test_lock_refuses_preexisting_confirmation_artifact(tmp_path):
    artifact = _file(tmp_path, "seed0_confirmation_test.npz", "forbidden")
    assert artifact.exists()
    with pytest.raises(RuntimeError, match="already exist"):
        _assert_confirmation_absent(tmp_path)
