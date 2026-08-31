from __future__ import annotations

import numpy as np

import ps_dfsc.training_v7 as training
from ps_dfsc.training import TrainingConfig


def test_warmup_uses_all_days_and_decision_uses_registered_subset(monkeypatch):
    observed = []

    def fake_train(*args, config, resume_state, **kwargs):
        rng = np.random.default_rng(config.seed)
        if resume_state is not None:
            rng.bit_generator.state = resume_state["numpy_rng_state"]
            start = int(resume_state["next_epoch"])
        else:
            start = 0
        for _epoch in range(start, config.epochs):
            observed.append(len(rng.permutation(100)))
        return observed

    monkeypatch.setattr(training, "_finite_train", fake_train)
    config = TrainingConfig(
        epochs=4,
        proper_warmup_epochs=2,
        batch_size=4,
    )
    result = training.train_calibrator(
        config=config,
        decision_batches_per_epoch=2,
    )
    assert result == [100, 100, 8, 8]


def test_resumed_sampler_starts_at_resume_epoch(monkeypatch):
    observed = []
    state = np.random.default_rng(7).bit_generator.state

    def fake_train(*args, config, resume_state, **kwargs):
        rng = np.random.default_rng(config.seed)
        rng.bit_generator.state = resume_state["numpy_rng_state"]
        for _epoch in range(
            int(resume_state["next_epoch"]), config.epochs
        ):
            observed.append(len(rng.permutation(100)))
        return observed

    monkeypatch.setattr(training, "_finite_train", fake_train)
    config = TrainingConfig(
        epochs=4,
        proper_warmup_epochs=2,
        batch_size=4,
    )
    result = training.train_calibrator(
        config=config,
        resume_state={
            "next_epoch": 3,
            "numpy_rng_state": state,
        },
        decision_batches_per_epoch=2,
    )
    assert result == [8]
