import json
import sys

import torch

import repro_scripts.validate_ps_dfsc_candidate_v2 as validator
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.stable_training_metrics import (
    SMOOTHING_EPSILON,
    SURROGATE_SCHEMA,
)


def test_validator_requires_identical_final_and_epoch50_models(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "beta025.pt"
    output = tmp_path / "validation.json"
    model = PSDFSCNetwork()
    torch.save({"model_state": model.state_dict()}, checkpoint)
    torch.save(
        {
            "metadata": {
                "schema": "ps_dfsc_epoch_resume_metadata_v2",
                "proper_score_surrogate": SURROGATE_SCHEMA,
                "proper_score_smoothing_epsilon": SMOOTHING_EPSILON,
                "beta": 0.25,
                "seed": 30025,
                "epochs": 50,
                "warmup_epochs": 20,
                "batch_size": 4,
            },
            "midpoint_refresh_completed": True,
            "state": {
                "next_epoch": 50,
                "history_epochs": [{"loss": float(i)} for i in range(50)],
                "model_state": model.state_dict(),
            },
        },
        checkpoint.with_suffix(".epoch_resume.pt"),
    )
    monkeypatch.setattr(
        validator,
        "_validate_candidate",
        lambda *_args, **_kwargs: {"base_validation": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--checkpoint",
            str(checkpoint),
            "--outer",
            "3",
            "--beta",
            "0.25",
            "--seed",
            "30025",
            "--output",
            str(output),
        ],
    )
    validator.main()
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["final_resume_model_identical"] is True
    assert evidence["resume_next_epoch"] == 50
    assert evidence["proper_score_surrogate"] == SURROGATE_SCHEMA
