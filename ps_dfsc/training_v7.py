"""Deterministic stochastic-day decision training.

Proper-score warm-up still uses all 100 development days.  Once the expensive
decision-focused QP path is enabled, each epoch uses a deterministic random
subset of complete days.  Validation and every safety gate continue to use all
50 validation days.  This is ordinary stochastic optimization and does not
change the model, objective, constraints, K=20 reduction, or exact midpoint
refresh.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import ps_dfsc.training_v4 as _base
from .training import TrainingConfig
from .training_v6 import train_calibrator as _finite_train


class _EpochDaySampler:
    def __init__(
        self,
        generator: np.random.Generator,
        *,
        start_epoch: int,
        warmup_epochs: int,
        decision_days: int,
    ) -> None:
        self._generator = generator
        self._epoch = int(start_epoch)
        self._warmup_epochs = int(warmup_epochs)
        self._decision_days = int(decision_days)

    @property
    def bit_generator(self):
        return self._generator.bit_generator

    def permutation(self, value):
        order = self._generator.permutation(value)
        epoch = self._epoch
        self._epoch += 1
        if epoch < self._warmup_epochs:
            return order
        return order[: min(len(order), self._decision_days)]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)


def train_calibrator(
    *args,
    config: TrainingConfig = TrainingConfig(),
    resume_state=None,
    decision_batches_per_epoch: int = 2,
    **kwargs,
):
    if decision_batches_per_epoch <= 0:
        raise ValueError("decision_batches_per_epoch must be positive")
    start_epoch = (
        0 if resume_state is None else int(resume_state["next_epoch"])
    )
    decision_days = int(config.batch_size) * int(
        decision_batches_per_epoch
    )
    original = _base.np.random.default_rng

    def sampled_generator(seed=None):
        return _EpochDaySampler(
            original(seed),
            start_epoch=start_epoch,
            warmup_epochs=config.proper_warmup_epochs,
            decision_days=decision_days,
        )

    _base.np.random.default_rng = sampled_generator
    try:
        return _finite_train(
            *args,
            config=config,
            resume_state=resume_state,
            **kwargs,
        )
    finally:
        _base.np.random.default_rng = original


__all__ = ["train_calibrator"]
