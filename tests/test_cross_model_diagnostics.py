from __future__ import annotations

import numpy as np

from cross_model_diagnostics.core import (
    atom_duration_summary,
    classify_states,
    cross_lag_summary,
    empirical_crps,
    generated_state_attribution,
    masked_per_day_scores,
    per_day_metrics,
    stratified_paired_bootstrap,
)
from cross_model_diagnostics.data import assign_regimes, fit_regime_thresholds


def _toy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    truth = rng.uniform(0.05, 0.95, size=(4, 10, 24))
    truth[:, :, :3] = 0.0
    scenarios = np.clip(
        truth[:, None] + rng.normal(0.0, 0.08, size=(4, 12, 10, 24)), 0.0, 1.0
    )
    return scenarios, truth


def test_fast_empirical_crps_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(3)
    sample = rng.normal(size=(5, 7, 3))
    truth = rng.normal(size=(5, 3))
    fast = empirical_crps(sample, truth, member_axis=1)
    first = np.abs(sample - truth[:, None]).mean(axis=1)
    second = np.abs(sample[:, :, None] - sample[:, None, :]).mean(axis=(1, 2))
    assert np.allclose(fast, first - 0.5 * second)


def test_per_day_and_atom_diagnostics_are_finite() -> None:
    scenarios, truth = _toy()
    metrics = per_day_metrics(scenarios, truth, zero_epsilon=0.01)
    assert metrics
    assert all(value.shape == (4,) for value in metrics.values())
    assert all(np.isfinite(value).all() for value in metrics.values())
    duration = atom_duration_summary(scenarios, truth, epsilon=0.01)
    assert 0.0 <= duration["duration_total_variation"] <= 1.0
    assert np.isclose(sum(duration["forecast_duration_distribution"]), 1.0)
    no_atom = atom_duration_summary(np.full_like(scenarios, 0.5), truth, epsilon=0.0)
    assert no_atom["forecast_run_count"] == 0
    assert no_atom["duration_total_variation"] is None

    valid = np.ones_like(truth, dtype=bool)
    valid[0, 0, 3] = False
    clean = masked_per_day_scores(scenarios, truth, valid)
    assert clean["valid_level_cells"][0] == 239
    assert clean["valid_ramp_cells"][0] == 228
    assert np.isfinite(clean["level_CRPS_clean"]).all()
    assert np.isfinite(clean["ramp_CRPS_clean"]).all()


def test_state_attribution_labels_truth_and_generated_transitions() -> None:
    scenarios, truth = _toy()
    states = classify_states(scenarios, 0.01)
    records = generated_state_attribution(
        scenarios, truth, scenario_states=states, epsilon=0.01
    )
    assert len(records) == 12
    names = {record["truth_transition"] for record in records}
    assert names == {"interior_to_interior", "atom_to_interior", "atom_to_atom"}
    assert any(record["generated_transition"] == "oracle_same_category" for record in records)


def test_cross_lag_detects_independent_zone_forecast() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=(30, 1, 24))
    truth = np.repeat(base, 10, axis=1)
    independent = rng.normal(size=(30, 20, 10, 24))
    summary = cross_lag_summary(independent, truth, lags=(0, 1))
    level0 = next(
        row for row in summary["records"] if row["domain"] == "level" and row["lag"] == 0
    )
    assert level0["cross_zone_RMSE"] > 0.5


def test_bootstrap_and_train_only_regimes() -> None:
    differences = np.asarray([-1.0, -0.5, 0.2, -0.2, -0.1, -0.4])
    strata = np.asarray([1, 1, 2, 2, 3, 3])
    result = stratified_paired_bootstrap(
        differences, strata, repetitions=200, seed=5
    )
    assert result["n_days"] == 6
    assert result["mean_difference"] < 0.0

    features = {
        "mean_ws100": np.linspace(0.0, 1.0, 120),
        "rms_dws100": np.linspace(0.5, 1.5, 120),
        "weighted_turn100": np.linspace(0.1, 0.4, 120),
        "nwp_ramp": np.linspace(1.0, 2.0, 120),
        "direction_shift": np.linspace(2.0, 3.0, 120),
        "spatial_dispersion": np.linspace(3.0, 4.0, 120),
        "mean_shear100_10": np.linspace(1.0, 3.0, 120),
    }
    train = np.zeros(120, dtype=bool)
    train[:100] = True
    thresholds = fit_regime_thresholds(features, train)
    labels = assign_regimes(features, thresholds)
    assert {
        "calm",
        "strong",
        "speed_volatile",
        "turning",
        "spatial_heterogeneous",
        "wind_level",
        "nwp_dynamics",
        "direction_regime",
        "spatial_regime",
    }.issubset(labels)
