"""Fail-closed unit tests for the frozen train-only G0-A audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from architecture_v1.g0_predictability import (
    build_mode_groups,
    continuous_logit_target,
    derangement,
    fit_log_variance_glm,
    fit_static_variance,
    gaussian_variance_score,
    masked_band_energies,
    orthonormal_dct_ii,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_a_predictability.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _groups():
    value = _config()
    temporal = value["mode_registry"]["temporal"]
    ranges = {
        label: temporal[label]["indices_half_open"]
        for label in ("low", "mid", "high")
    }
    return build_mode_groups(ranges, value["mode_registry"]["ordered_groups"])


def test_frozen_config_and_train_only_role_contract() -> None:
    value = _config()
    assert value["schema"] == "architecture_v1_g0_a_predictability_audit_v1"
    assert value["status"] == "frozen_before_any_G0_A_target_aware_execution"
    assert value["role_access"]["allowed_target_roles"] == ["train"]
    assert set(value["role_access"]["forbidden_target_roles"]) == {
        "validation",
        "calibration",
        "selection",
        "r_seen",
        "final",
    }
    assert CONFIG.with_name(CONFIG.name + ".sha256").read_text().split() == [
        _sha(CONFIG),
        CONFIG.name,
    ]
    assert "G0-B diffusion-path implementation or training" in value["not_authorized"]


def test_dct_and_six_projectors_form_the_frozen_partition() -> None:
    dct = orthonormal_dct_ii(24)
    assert np.allclose(dct.T @ dct, np.eye(24), atol=1e-12, rtol=0.0)
    groups = _groups()
    assert [group.name for group in groups] == [
        "low_common",
        "low_local",
        "mid_common",
        "mid_local",
        "high_common",
        "high_local",
    ]
    assert np.allclose(
        [group.full_rank for group in groups],
        [4.0, 36.0, 8.0, 72.0, 12.0, 108.0],
        atol=1e-12,
        rtol=0.0,
    )
    for group in groups:
        assert np.allclose(
            group.spatial_projector @ group.spatial_projector,
            group.spatial_projector,
            atol=1e-12,
        )
        assert np.allclose(
            group.temporal_projector @ group.temporal_projector,
            group.temporal_projector,
            atol=1e-12,
        )


def test_atom_values_never_enter_continuous_band_energy() -> None:
    groups = _groups()
    target = np.full((2, 10, 24), 0.5, dtype=np.float64)
    state = np.ones_like(target, dtype=np.int64)
    observed = np.ones_like(target, dtype=bool)
    state[:, 0, 0] = 0
    state[:, 1, 1] = 2
    observed[:, 2, 2] = False
    latent, active = continuous_logit_target(
        target, state, observed, epsilon=1e-4
    )
    assert np.all(latent[~active] == 0.0)
    residual = np.arange(target.size, dtype=np.float64).reshape(target.shape) / 100.0
    first, first_rank = masked_band_energies(
        residual,
        active,
        groups,
        minimum_effective_rank=1.5,
        energy_floor=1e-8,
    )
    changed = residual.copy()
    changed[~active] = 1e12
    second, second_rank = masked_band_energies(
        changed,
        active,
        groups,
        minimum_effective_rank=1.5,
        energy_floor=1e-8,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_rank, second_rank)


def test_variance_glm_detects_synthetic_conditional_uncertainty() -> None:
    generator = np.random.default_rng(123)
    feature = generator.normal(size=(400, 1))
    true_variance = np.exp(0.9 * feature[:, 0])
    energy = true_variance * generator.chisquare(df=8, size=400) / 8.0
    rank = np.full(400, 8.0)
    fit = np.arange(300)
    test = np.arange(300, 400)
    model = fit_log_variance_glm(
        feature[fit],
        energy[fit],
        rank[fit],
        alpha=0.01,
        variance_clip=(1e-6, 1e6),
        maximum_iterations=100,
        tolerance=1e-8,
    )
    predicted = model.predict(feature[test])
    static = fit_static_variance(
        energy[fit], rank[fit], variance_clip=(1e-6, 1e6)
    )
    conditional_score = gaussian_variance_score(energy[test], predicted).mean()
    static_score = gaussian_variance_score(
        energy[test], np.full(len(test), static)
    ).mean()
    assert conditional_score < static_score


def test_registered_shuffle_primitive_has_no_fixed_points() -> None:
    permutation = derangement(43, 28100)
    assert np.array_equal(np.sort(permutation), np.arange(43))
    assert not np.any(permutation == np.arange(43))
