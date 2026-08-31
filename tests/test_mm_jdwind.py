from __future__ import annotations

import numpy as np
import torch

from mm_jdwind.data import INTERIOR_STATE, JointSplitData, flatten_joint_cases
from mm_jdwind.metrics import evaluate_joint
from mm_jdwind.model import MMJDWind, mixed_measure_head_loss
from mm_jdwind.sampling import mass_preserving_states, sample_joint


def _model() -> MMJDWind:
    common = {
        "condition_dim": 20,
        "model_dim": 16,
        "depth": 1,
        "heads": 4,
        "ff_multiplier": 2,
        "dropout": 0.0,
    }
    return MMJDWind(
        head={
            **common,
            "minimum_scale": 0.05,
            "maximum_scale": 2.0,
            "upper_logit_prior": -8.0,
            "upper_logit_deviation": 1.0,
        },
        jump=common,
        flow=common,
    )


def test_head_probabilities_and_losses_are_valid() -> None:
    model = _model()
    condition = torch.randn(2, 10, 24, 20)
    target = torch.rand(2, 10, 24)
    target[:, :, 0] = 0.0
    state = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    state[:, :, 0] = 0
    output = model.statistics(condition)
    probabilities = output.state_probabilities
    assert probabilities.shape == (2, 10, 24, 3)
    assert torch.allclose(probabilities.sum(-1), torch.ones_like(target), atol=1e-6)
    losses = mixed_measure_head_loss(target, state, output)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_jump_flow_and_all_sampling_modes_are_bounded() -> None:
    torch.manual_seed(3)
    model = _model()
    condition = torch.randn(1, 10, 24, 20)
    target = torch.rand(1, 10, 24)
    state = torch.full((1, 10, 24), INTERIOR_STATE, dtype=torch.long)
    statistics = model.statistics(condition)
    jump = model.jump.loss(condition, state, statistics.state_probabilities)
    assert torch.isfinite(jump["loss"])
    residual = model.residual(target, state, statistics)
    flow = model.flow.loss(residual, condition, state)
    assert torch.isfinite(flow["loss"])
    for mode in ("none", "independent", "correlated", "mass_preserving"):
        values, _, sampled = sample_joint(
            model,
            condition,
            members=3,
            jump_steps=2,
            flow_steps=2,
            seed=19,
            state_mode=mode,
            member_chunk=2,
        )
        assert values.shape == (1, 3, 10, 24)
        assert sampled.shape == (1, 3, 10, 24)
        assert torch.isfinite(values).all()
        assert float(values.min()) >= 0.0
        assert float(values.max()) <= 1.0


def test_mass_preserving_sampler_matches_rounded_atom_counts() -> None:
    sampled = torch.ones(1, 10, 2, 2, dtype=torch.long)
    member = torch.rand(1, 10, 2, 2, 3)
    member = member / member.sum(-1, keepdim=True)
    analytic = torch.zeros(1, 2, 2, 3)
    analytic[..., 0] = 0.2
    analytic[..., 1] = 0.7
    analytic[..., 2] = 0.1
    result = mass_preserving_states(sampled, member, analytic)
    assert torch.all((result == 0).sum(1) == 2)
    assert torch.all((result == 2).sum(1) == 1)


def test_joint_flattening_and_metrics() -> None:
    rng = np.random.default_rng(7)
    target = rng.uniform(0.05, 0.95, size=(2, 10, 24)).astype(np.float32)
    split = JointSplitData(
        condition=rng.normal(size=(2, 10, 24, 20)).astype(np.float32),
        target=target,
        day=np.asarray(["2013-01-01", "2013-01-02"], dtype="datetime64[D]"),
        zones=np.arange(1, 11, dtype=np.int64),
    )
    scenarios = np.clip(
        target[:, None] + rng.normal(0, 0.1, size=(2, 5, 10, 24)),
        0.0,
        1.0,
    ).astype(np.float32)
    flat, truth = flatten_joint_cases(scenarios, target)
    assert flat.shape == (20, 5, 24)
    assert truth.shape == (20, 24)
    metrics = evaluate_joint(
        scenarios,
        split,
        zero_probability=np.full_like(target, 0.05),
        one_probability=np.full_like(target, 0.01),
    )
    for value in metrics.values():
        assert np.isfinite(value)
