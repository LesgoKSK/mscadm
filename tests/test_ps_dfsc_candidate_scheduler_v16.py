import json

import pytest
import torch

import repro_scripts.schedule_ps_dfsc_candidates_v16 as scheduler
from ps_dfsc.manifest import file_sha256
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.pipeline_runtime_publication import _model_sha256


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "ROOT", tmp_path)
    prepared = (
        tmp_path
        / "outputs"
        / "ps_dfsc"
        / "outer3"
        / "development_train_prepared.npz"
    )
    mapping = tmp_path / "repro_configs" / "ps_dfsc_mapping_outer3.json"
    prepared.parent.mkdir(parents=True)
    mapping.parent.mkdir(parents=True)
    prepared.write_bytes(b"registered training archive")
    mapping.write_text("{}", encoding="utf-8")
    checkpoint = prepared.parent / "candidates" / "beta025.pt"
    checkpoint.parent.mkdir()
    model = PSDFSCNetwork()
    payload = {
        "schema": "ps_dfsc_checkpoint_v3",
        "model_state": model.state_dict(),
        "model_state_sha256": _model_sha256(model),
        "training_config": {
            "beta": 0.25,
            "epochs": 50,
            "proper_warmup_epochs": 20,
            "batch_size": 4,
            "strong_convexity": 1.0e-4,
            "seed": 30025,
        },
        "commitment_refresh_calls": 1,
        "commitment_refresh_summary": {
            "cases": 100,
            "target_gap_pass_rate": 0.75,
        },
        "training_input_sha256": file_sha256(prepared),
        "mapping_sha256": file_sha256(mapping),
    }
    torch.save(payload, checkpoint)
    history = {"epochs": [{"loss": float(i)} for i in range(50)]}
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    return checkpoint


def test_scheduler_validates_complete_candidate(tmp_path, monkeypatch):
    checkpoint = _fixture(tmp_path, monkeypatch)
    evidence = scheduler._validate_candidate(
        checkpoint, outer=3, beta=0.25, seed=30025
    )
    assert evidence["refresh_cases"] == 100
    assert evidence["checkpoint_sha256"] == file_sha256(checkpoint)


def test_scheduler_rejects_nonfinite_history(tmp_path, monkeypatch):
    checkpoint = _fixture(tmp_path, monkeypatch)
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(
            {
                "epochs": [
                    {"loss": float("nan") if i == 49 else float(i)}
                    for i in range(50)
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FloatingPointError):
        scheduler._validate_candidate(
            checkpoint, outer=3, beta=0.25, seed=30025
        )
