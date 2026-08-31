from __future__ import annotations

import math

import numpy as np
import torch

from mm_jdwind.metrics import (
    adjacency_variogram_score,
    joint_energy_score,
    per_date_joint_metrics,
)
from ps_dfsc.exact_suc import evaluate_realized, solve_two_stage_suc
from ps_dfsc.inference import calibrate
from ps_dfsc.mapping import WindFarmMapping, fit_wind_farm_mapping
from ps_dfsc.metrics import weighted_per_date_metrics
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.reduction import fit_fixed_assignments, weighted_cluster_reduction
from ps_dfsc.safety import evaluate_safety_gate


def test_initial_model_is_uniform_and_identity() -> None:
    torch.manual_seed(0)
    values = torch.rand(2, 12, 10, 24)
    values[:, :, 0, 0] = 0.0
    values[:, :, 1, 0] = 1.0
    model = PSDFSCNetwork()
    result = model(values)
    assert torch.allclose(
        result.probabilities, torch.full((2, 12), 1 / 12), atol=1e-7
    )
    assert torch.allclose(result.scenarios, values, atol=2e-6)
    assert torch.equal(result.scenarios[:, :, 0, 0], values[:, :, 0, 0])
    assert torch.equal(result.scenarios[:, :, 1, 0], values[:, :, 1, 0])
    assert torch.allclose(result.ess, torch.full((2,), 12.0), atol=1e-5)


def test_model_is_permutation_equivariant() -> None:
    torch.manual_seed(2)
    values = torch.rand(1, 15, 10, 24)
    permutation = torch.randperm(15)
    model = PSDFSCNetwork()
    direct = model(values)
    permuted = model(values[:, permutation])
    assert torch.allclose(
        direct.scenarios[:, permutation], permuted.scenarios, atol=1e-6
    )
    assert torch.allclose(
        direct.probabilities[:, permutation], permuted.probabilities, atol=1e-7
    )


def test_transport_is_monotone_bounded_and_endpoint_preserving() -> None:
    values = torch.linspace(0, 1, 101).reshape(1, 101, 1, 1)
    scale = torch.full((1, 1, 1), 1.1)
    shift = torch.full((1, 1, 1), -0.15)
    moved = PSDFSCNetwork.monotone_transport(values, scale, shift)
    assert moved[0, 0, 0, 0] == 0.0
    assert moved[0, -1, 0, 0] == 1.0
    assert torch.all(torch.diff(moved.flatten()) >= 0.0)
    assert float(moved.min()) >= 0.0
    assert float(moved.max()) <= 1.0


def test_mapping_has_four_pairs_and_preserves_total_capacity() -> None:
    rng = np.random.default_rng(3)
    training = rng.random((40, 10, 24))
    training[:, 1] = training[:, 0] + rng.normal(0, 1e-3, (40, 24))
    training = np.clip(training, 0, 1)
    mapping = fit_wind_farm_mapping(training)
    assert sorted(map(len, mapping.groups)) == [1, 1, 2, 2, 2, 2]
    mapped = mapping.transform(training[:2])
    assert mapped.shape == (2, 6, 24)
    assert np.allclose(
        mapped.sum(axis=1), training[:2].sum(axis=1) * 120.0
    )


def test_weighted_reduction_preserves_probability_and_mean() -> None:
    rng = np.random.default_rng(4)
    values = rng.random((30, 6, 24)) * 300
    probability = rng.random(30)
    probability /= probability.sum()
    labels = fit_fixed_assignments(values, 5, seed=2)
    reduced, mass = weighted_cluster_reduction(values, probability, labels)
    assert reduced.shape == (5, 6, 24)
    assert np.isclose(mass.sum(), 1.0)
    assert np.allclose(
        np.tensordot(mass, reduced, axes=(0, 0)),
        np.tensordot(probability, values, axes=(0, 0)),
    )


def test_uniform_weighted_scores_match_existing_metrics() -> None:
    rng = np.random.default_rng(5)
    scenarios = rng.random((3, 8, 10, 24))
    truth = rng.random((3, 10, 24))
    probability = np.full((3, 8), 1 / 8)
    weighted = weighted_per_date_metrics(scenarios, probability, truth)
    existing = per_date_joint_metrics(scenarios, truth)
    assert np.allclose(weighted["CRPS"], existing["CRPS"])
    assert np.isclose(weighted["ES"].mean(), joint_energy_score(scenarios, truth))
    assert np.isclose(
        weighted["VS"].mean(), adjacency_variogram_score(scenarios, truth)
    )


def test_safety_gate_accepts_identical_safe_metrics() -> None:
    days = 20
    metrics = {
        "CRPS": np.full(days, 1.0),
        "ES": np.full(days, 2.0),
        "VS": np.full(days, 3.0),
        "ramp_CRPS": np.full(days, 1.5),
        "zero_Brier": np.full(days, 0.2),
        "coverage_90": np.full(days, 0.90),
        "conditional_ACE": np.full(days, 0.10),
    }
    decision = evaluate_safety_gate(
        metrics, metrics, bootstrap_samples=100, seed=0
    )
    assert decision.passed
    assert not decision.reasons


def test_truth_independent_structural_fallback_is_exact_identity() -> None:
    rng = np.random.default_rng(6)
    base = rng.random((20, 10, 24))
    mapping = WindFarmMapping(
        groups=((0, 1), (2, 3), (4, 5), (6, 7), (8,), (9,))
    )
    model = PSDFSCNetwork()
    with torch.no_grad():
        model.transport_head[-1].bias.fill_(100.0)
    result = calibrate(model, base, mapping, clusters=5, transport_budget=0.0)
    assert result.used_fallback
    assert np.array_equal(result.full_scenarios, base)
    assert np.array_equal(result.probabilities, np.full(20, 1 / 20))
    assert result.ess == 20
    assert result.entropy == math.log(20)


def test_exact_two_stage_suc_and_realized_recourse() -> None:
    wind = np.full((1, 6, 24), 100.0)
    planned = solve_two_stage_suc(
        wind, np.ones(1), mip_gap=0.02, time_limit=60
    )
    realized = evaluate_realized(
        planned.first_stage, wind[0], mip_gap=0.02, time_limit=60
    )
    assert planned.success
    assert realized.success
    assert planned.first_stage.day_ahead_dispatch.shape == (12, 24)
    assert planned.first_stage.reserve_up.shape == (12, 24)
    assert planned.reserve_shortage >= 0.0
    assert planned.solve_time_seconds > 0.0
    assert np.isclose(planned.total_cost, realized.total_cost, rtol=1e-7)
