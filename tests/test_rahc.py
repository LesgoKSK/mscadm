from __future__ import annotations

import numpy as np
import pytest
import torch

from rahc.v2_calibration import (
    RAHCalibratorV2,
    apply_probability_map,
    rank_audit,
    rank_intervals,
)
from rahc.v2_features import FEATURE_NAMES, FeatureScaler
from rahc.v2_model import beta_binomial_log_pmf
from rahc.validation import grouped_day_folds


def test_beta_binomial_identity_shapes_give_uniform_rank_pmf() -> None:
    members = 17
    rank = torch.arange(members + 1, dtype=torch.float64)
    shape = torch.ones_like(rank)

    probability = torch.exp(beta_binomial_log_pmf(rank, members, shape, shape))

    expected = torch.full_like(probability, 1.0 / (members + 1))
    assert torch.allclose(probability, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(probability.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_rank_intervals_represent_ties_as_inclusive_pooled_ranks() -> None:
    scenarios = np.tile(
        np.array([0.1, 0.5, 0.5, 0.9], dtype=np.float32)[None, :, None],
        (1, 1, 24),
    )
    observations = np.full((1, 24), 0.7, dtype=np.float32)
    observations[0, 0] = 0.5
    observations[0, 1] = 0.1
    observations[0, 2] = 1.0

    lower, upper = rank_intervals(scenarios, observations)

    assert (lower[0, 0], upper[0, 0]) == (1, 3)
    assert (lower[0, 1], upper[0, 1]) == (0, 1)
    assert (lower[0, 2], upper[0, 2]) == (4, 4)
    assert np.all(lower[0, 3:] == 3)
    assert np.all(upper[0, 3:] == 3)


@pytest.mark.parametrize("tail_rule", ("bounded", "linear", "clamp"))
def test_zero_strength_transform_is_elementwise_identity(tail_rule: str) -> None:
    rng = np.random.default_rng(41)
    scenarios = rng.uniform(0.05, 0.95, size=(4, 11, 24)).astype(np.float32)
    zones = np.array([1, 3, 7, 10], dtype=np.int64)
    scaler = FeatureScaler(
        mean=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        std=np.ones(len(FEATURE_NAMES), dtype=np.float32),
    )
    calibrator = RAHCalibratorV2("full", scaler, device="cpu")

    transformed = calibrator.transform(
        scenarios, zones, strength=0.0, tail_rule=tail_rule
    )

    assert transformed.dtype == scenarios.dtype
    assert np.allclose(transformed, scenarios, atol=2e-6, rtol=2e-6)


def test_grouped_day_folds_never_split_a_calendar_date() -> None:
    days = np.datetime64("2020-01-01") + np.repeat(np.arange(10), 10)
    assignments = grouped_day_folds(days, folds=5, seed=19)

    assert set(assignments.tolist()) == set(range(5))
    for day in np.unique(days):
        assert np.unique(assignments[days == day]).size == 1
    for fold in range(5):
        train_days = set(days[assignments != fold].tolist())
        held_out_days = set(days[assignments == fold].tolist())
        assert train_days.isdisjoint(held_out_days)


def test_conditional_spread_fit_reduces_loss_and_learns_rank_shift() -> None:
    rng = np.random.default_rng(20260718)
    cases, members, hours = 30, 9, 24
    low_spread = np.arange(cases) % 2 == 0
    spread = np.where(low_spread, 0.012, 0.09)
    center = (
        0.45
        + 0.05 * np.sin(np.arange(hours)[None, :] * 2.0 * np.pi / hours)
        + rng.normal(0.0, 0.015, size=(cases, 1))
    )
    scenarios = np.clip(
        center[:, None, :]
        + spread[:, None, None] * rng.normal(size=(cases, members, hours)),
        0.001,
        0.999,
    ).astype(np.float32)
    observations = center.copy()
    observations[low_spread] += 0.045
    observations[~low_spread] += rng.normal(
        0.0, 0.012, size=(int((~low_spread).sum()), hours)
    )
    observations = np.clip(observations, 0.0, 1.0).astype(np.float32)
    zones = (np.arange(cases) % 10 + 1).astype(np.int64)

    calibrator = RAHCalibratorV2.fit(
        scenarios,
        observations,
        zones,
        variant="hour_regime",
        regularization=1e-3,
        epochs=30,
        learning_rate=0.05,
        seed=13,
        device="cpu",
    )

    summary = calibrator.fit_summary
    assert summary is not None
    assert summary.final_nll < summary.initial_nll - 0.2
    alpha, beta = calibrator.predict_shapes(scenarios, zones)
    expected_rank = members * alpha / (alpha + beta)
    assert expected_rank[low_spread].mean() > expected_rank[~low_spread].mean() + 1.0


def test_rank_audit_reports_zero_reversals_and_stable_ordinal_ranks() -> None:
    rng = np.random.default_rng(73)
    scenarios = rng.uniform(0.08, 0.92, size=(3, 13, 24)).astype(np.float32)
    probabilities = np.broadcast_to(
        np.linspace(0.02, 0.98, scenarios.shape[1])[None, :, None],
        scenarios.shape,
    ).copy()

    transformed = apply_probability_map(
        scenarios, probabilities, tail_rule="bounded"
    )
    audit = rank_audit(scenarios, transformed)

    assert audit["strict_reversals"] == 0
    assert audit["stable_ordinal_rank_matches"] == audit["stable_ordinal_rank_total"]
    assert audit["stable_ordinal_rank_fraction"] == 1.0
