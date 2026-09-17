"""Synthetic and frozen-input checks for G0-B formal evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from architecture_v1.g0_predictability import build_mode_groups
from architecture_v1.g0b_evaluation import (
    CONTRAST_IDS,
    adjudicate,
    balanced_reconstruction_risk,
    learning_curve_aulc,
    month_cluster_max_t_bands,
    relative_benefit,
    sample_reconstruction_metrics,
)
from architecture_v1.g0b_tiny_denoiser import ModeProjectorBank
from architecture_v1.family_diffusion import MaskedJointDDPM
from architecture_v1.g0b_tiny_denoiser import PATH_IDS, TinyDenoisingSystem
from repro_scripts import run_architecture_v1_g0_b_tiny_denoiser_evaluation as evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _groups():
    return build_mode_groups(
        {"low": (0, 4), "mid": (4, 12), "high": (12, 24)},
        (
            "low_common",
            "low_local",
            "mid_common",
            "mid_local",
            "high_common",
            "high_local",
        ),
    )


def test_sample_metrics_apply_active_and_increment_masks_and_group_ranks() -> None:
    groups = _groups()
    projectors = ModeProjectorBank(groups)
    target = torch.zeros(2, 10, 24)
    active = torch.ones_like(target, dtype=torch.bool)
    prediction = torch.ones_like(target)
    effective = torch.tensor(
        [[group.full_rank for group in groups]] * 2, dtype=torch.float32
    )
    metrics = sample_reconstruction_metrics(
        prediction,
        target,
        active,
        projectors,
        effective,
        outer_train_second_moment=2.0,
    )
    assert torch.allclose(metrics["cell_MSE"], torch.ones(2))
    assert torch.equal(metrics["increment_MSE"], torch.zeros(2))
    assert torch.allclose(
        metrics["joint_day_normalized_SSE"], torch.full((2,), 0.5)
    )
    assert metrics["six_group_MSE"].shape == (2, 6)
    assert torch.isfinite(metrics["six_group_MSE"]).all()


def test_balanced_risk_aulc_and_relative_benefit_have_registered_direction() -> None:
    values = np.ones((4, 2, 3, 3), dtype=np.float64)
    values[..., 0] *= 2.0
    values[..., 1] *= 4.0
    values[..., 2] *= 8.0
    denominators = np.asarray([[2.0, 4.0, 8.0]] * 3)
    balanced = balanced_reconstruction_risk(values, denominators)
    assert np.array_equal(balanced, np.ones((4, 2, 3)))
    aulc = learning_curve_aulc(balanced, (0.25, 0.5, 1.0))
    assert np.array_equal(aulc, np.ones((4, 2)))
    benefit = relative_benefit(np.asarray([2.0, 4.0]), np.asarray([1.0, 3.0]))
    assert np.isclose(benefit.mean(), 1.0 / 3.0)


def test_month_cluster_max_t_is_paired_reproducible_and_simultaneous() -> None:
    days = np.arange(
        np.datetime64("2022-01-01"), np.datetime64("2024-01-01")
    ).astype("datetime64[D]")
    rng = np.random.default_rng(19)
    contributions = rng.normal(0.03, 0.01, size=(len(days), 13))
    first = month_cluster_max_t_bands(
        contributions,
        days,
        repetitions=500,
        seed=34400,
        confidence=0.95,
    )
    second = month_cluster_max_t_bands(
        contributions,
        days,
        repetitions=500,
        seed=34400,
        confidence=0.95,
    )
    assert first["month_clusters"] == 24
    assert first["endpoint_count"] == 13
    assert np.array_equal(first["estimate"], second["estimate"])
    assert np.array_equal(first["simultaneous_low"], second["simultaneous_low"])
    assert np.all(first["simultaneous_low"] <= first["estimate"])
    assert np.all(first["simultaneous_high"] >= first["estimate"])
    assert first["critical_value"] > 1.0


def test_adjudication_requires_every_registered_gate_and_decision_order() -> None:
    bands = {
        name: {
            "estimate": 0.03,
            "simultaneous_low": 0.01,
            "simultaneous_high": 0.05,
        }
        for name in CONTRAST_IDS
    }
    go = adjudicate(
        bands,
        positive_outer_folds=6,
        positive_model_seeds=3,
        technical_eligibility=True,
        thresholds=_config()["go_no_go"],
    )
    assert go["status"] == "G0_B_PREDICTABILITY_ALLOCATION_UTILITY_GO"
    assert go["all_GO_gates_passed"] is True

    failed = {name: dict(value) for name, value in bands.items()}
    failed[CONTRAST_IDS[3]]["estimate"] = 0.005
    failed[CONTRAST_IDS[3]]["simultaneous_low"] = -0.002
    no_go = adjudicate(
        failed,
        positive_outer_folds=6,
        positive_model_seeds=3,
        technical_eligibility=True,
        thresholds=_config()["go_no_go"],
    )
    assert no_go["status"] == "G0_B_NWP_ALIGNMENT_NO_GO"


def test_noise_bank_formula_is_fold_specific_reproducible_and_target_free() -> None:
    folds = [
        SimpleNamespace(fold=0, outer_test=np.arange(2)),
        SimpleNamespace(fold=1, outer_test=np.arange(3)),
    ]
    first, first_identity = evaluation._noise_banks(_config(), folds)
    second, second_identity = evaluation._noise_banks(_config(), folds)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0][0], first[1][0])
    assert first_identity == second_identity
    assert first_identity["per_fold"]["0"]["seed"] == 34300
    assert first_identity["per_fold"]["1"]["seed"] == 34301


def test_all_six_paths_execute_the_frozen_synthetic_evaluation_bank() -> None:
    groups = _groups()
    rng = np.random.default_rng(27)
    days = 2
    residual_test = rng.normal(size=(days, 10, 24)).astype(np.float64)
    active_test = np.ones((days, 10, 24), dtype=bool)
    ranks = np.asarray([group.full_rank for group in groups], dtype=np.float64)
    effective_test = np.broadcast_to(ranks, (days, 6)).copy()
    weights_test = effective_test / effective_test.sum(axis=1, keepdims=True)
    fold = SimpleNamespace(
        outer_test=np.arange(days),
        residual_test=residual_test,
        active_test=active_test,
        condition_test=rng.normal(size=(days, 47)),
        variance_test=np.exp(rng.normal(scale=0.2, size=(days, 6))),
        weights_test=weights_test,
        effective_test=effective_test,
        fixed_variance=np.ones(6),
        residual_train=rng.normal(size=(5, 10, 24)),
        active_train=np.ones((5, 10, 24), dtype=bool),
    )
    timesteps = tuple(_config()["evaluation_bank"]["timesteps"])
    noise = torch.randn(days, len(timesteps), 4, 10, 24)
    baseline = MaskedJointDDPM().alpha_bar.detach().cpu().numpy()
    swap = np.asarray([1, 0], dtype=np.int64)
    for path in PATH_IDS:
        values, budget_error = evaluation._evaluate_one(
            system=TinyDenoisingSystem(path, model_seed=3),
            fold=fold,
            groups=groups,
            noise_bank=noise,
            timesteps=timesteps,
            baseline_alpha_numpy=baseline,
            config=_config(),
            device=torch.device("cpu"),
            shuffle_permutations=(swap, swap, swap, swap),
            batch_size=64,
        )
        assert set(values) == {
            "cell_MSE",
            "increment_MSE",
            "joint_day_normalized_SSE",
            "six_group_MSE",
            "gaussian_reference_risk",
            "oracle_efficiency_ratio",
        }
        assert values["six_group_MSE"].shape == (days, 6)
        assert all(np.isfinite(value).all() for value in values.values())
        assert budget_error < 1e-4


def test_target_free_dry_run_verifies_complete_training_freeze() -> None:
    result = evaluation.dry_run(CONFIG)
    assert result["mode"] == "target_free_no_evaluation_bank_constructed"
    assert result["target_roles_materialized"] == []
    assert result["verified_runs"] == 324
    assert result["verified_final_EMA_checkpoints"] == 324
    assert tuple(result["confirmatory_contrasts"]) == CONTRAST_IDS
    assert result["bootstrap_replicates"] == 10000
