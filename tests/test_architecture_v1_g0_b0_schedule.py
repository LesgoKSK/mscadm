"""Fail-closed tests for the schedule-only G0-B0 allocation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from architecture_v1.family_diffusion import MaskedJointDDPM
from architecture_v1.g0b_allocation import (
    gaussian_information,
    gaussian_posterior_variance,
    reserve_then_water_fill,
    reverse_water_fill_information,
    snr_from_information,
    weighted_information_budget,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b0_schedule.json"
RUNNER = ROOT / "repro_scripts" / "run_architecture_v1_g0_b0_schedule.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_config_and_no_training_boundary() -> None:
    value = _config()
    assert value["schema"] == "architecture_v1_g0_b0_schedule_audit_v1"
    assert value["status"] == (
        "frozen_after_documented_schedule_exploration_before_executable_G0_B0_audit"
    )
    assert value["allocation"]["primary_eta"] == 0.5
    assert value["allocation"]["primary_eta_must_not_be_tuned"] is True
    assert value["input_access"]["raw_dataset_loader_imported"] is False
    assert value["input_access"]["raw_target_arrays_materialized"] is False
    assert "a G0-B denoiser implementation or training run" in value["not_authorized"]
    assert CONFIG.with_name(CONFIG.name + ".sha256").read_text().split() == [
        _sha(CONFIG),
        CONFIG.name,
    ]

    source = RUNNER.read_text(encoding="utf-8")
    assert "architecture_v1.data" not in source
    assert "build_architecture_v1_train_data" not in source


def test_registered_cosine_schedule_identity() -> None:
    value = _config()["baseline_schedule"]
    diffusion = MaskedJointDDPM(
        timesteps=value["timesteps"],
        cosine_offset=value["cosine_offset"],
        beta_min=value["beta_min"],
        beta_max=value["beta_max"],
    )
    alpha_bar = diffusion.alpha_bar.detach().cpu().numpy()
    assert alpha_bar.dtype == np.float32
    digest = hashlib.sha256(
        alpha_bar.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    assert digest == value["alpha_bar_raw_little_endian_float32_sha256"]
    assert float(alpha_bar[0]) == value["expected_clean_endpoint_alpha_bar"]
    assert float(alpha_bar[-1]) == value["expected_noisy_endpoint_alpha_bar"]


def test_gaussian_information_inverse_and_posterior_identity() -> None:
    variance = np.asarray([[0.2, 1.0, 8.0]], dtype=np.float64)
    snr = np.asarray([[0.01, 1.0, 100.0]], dtype=np.float64)
    information = gaussian_information(variance, snr)
    recovered = snr_from_information(variance, information)
    assert np.allclose(recovered, snr, atol=1e-12, rtol=1e-12)
    posterior = gaussian_posterior_variance(variance, snr)
    assert np.allclose(
        posterior,
        variance * np.exp(-2.0 * information),
        atol=1e-14,
        rtol=1e-12,
    )


def test_reverse_water_filling_conserves_budget_and_equalizes_active_risk() -> None:
    variance = np.asarray([[0.2, 1.0, 8.0]], dtype=np.float64)
    weights = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
    budget = np.asarray([0.4], dtype=np.float64)
    information, log_level = reverse_water_fill_information(
        variance,
        weights,
        budget,
    )
    assert np.allclose(
        weighted_information_budget(information, weights),
        budget,
        atol=1e-12,
        rtol=0.0,
    )
    risk = variance * np.exp(-2.0 * information)
    active = information[0] > 0.0
    assert np.allclose(
        risk[0, active],
        np.exp(log_level[0]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.all(risk[0, ~active] <= np.exp(log_level[0]) + 1e-12)


def test_reserve_contract_endpoints_retention_and_equal_variance_identity() -> None:
    variance = np.asarray([[0.2, 1.0, 8.0]], dtype=np.float64)
    weights = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
    baseline_snr = 1.7
    primary = reserve_then_water_fill(
        variance,
        weights,
        baseline_snr,
        eta=0.5,
    )
    assert np.all(primary.information + 1e-14 >= 0.5 * primary.baseline_information)
    assert np.max(np.abs(primary.budget_error)) <= 1e-12
    assert np.all(primary.snr > 0.0)

    iid = reserve_then_water_fill(
        variance,
        weights,
        baseline_snr,
        eta=1.0,
    )
    baseline_alpha = baseline_snr / (1.0 + baseline_snr)
    assert np.allclose(iid.alpha_bar, baseline_alpha, atol=1e-12, rtol=0.0)

    oracle = reserve_then_water_fill(
        variance,
        weights,
        baseline_snr,
        eta=0.0,
    )
    assert np.array_equal(oracle.reserved_information, np.zeros_like(variance))

    equal = np.full((1, 3), 2.5, dtype=np.float64)
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        allocation = reserve_then_water_fill(
            equal,
            weights,
            baseline_snr,
            eta=eta,
        )
        assert np.allclose(
            allocation.alpha_bar,
            baseline_alpha,
            atol=1e-12,
            rtol=0.0,
        )


def test_primary_allocation_is_monotone_on_a_synthetic_forward_grid() -> None:
    variance = np.asarray(
        [[0.2, 0.5, 1.0, 2.0, 4.0, 8.0]], dtype=np.float64
    )
    weights = np.asarray([[4, 36, 8, 72, 12, 108]], dtype=np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    baseline_snr = np.exp(np.linspace(8.0, -16.0, 250))
    allocated = np.stack(
        [
            reserve_then_water_fill(
                variance,
                weights,
                float(ratio),
                eta=0.5,
            ).snr[0]
            for ratio in baseline_snr
        ],
        axis=0,
    )
    assert np.isfinite(allocated).all()
    assert np.all(allocated > 0.0)
    assert not np.any(np.diff(allocated, axis=0) > 1e-10)


def test_invalid_variance_weights_budget_and_eta_fail_closed() -> None:
    variance = np.asarray([[1.0, 2.0]], dtype=np.float64)
    weights = np.asarray([[0.5, 0.5]], dtype=np.float64)
    for bad_variance in (
        np.asarray([[0.0, 2.0]]),
        np.asarray([[np.nan, 2.0]]),
    ):
        try:
            reserve_then_water_fill(bad_variance, weights, 1.0, eta=0.5)
        except (ValueError, FloatingPointError):
            pass
        else:
            raise AssertionError("invalid variance did not fail closed")
    try:
        reserve_then_water_fill(variance, np.asarray([[0.4, 0.4]]), 1.0, eta=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("non-normalized weights did not fail closed")
    try:
        reserve_then_water_fill(variance, weights, 1.0, eta=1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid eta did not fail closed")
