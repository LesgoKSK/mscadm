"""Tests for the three-point lag-1 state attribution diagnostic.

These tests are assertion based so they can run with either pytest or plain
Python in the lightweight diagnostics environment.
"""

from __future__ import annotations

import numpy as np

from cross_model_diagnostics.advanced import lag1_increment_state_decomposition
from cross_model_diagnostics.core import classify_states


def _example() -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(
        [
            [
                [0.00, 0.14, 0.31, 0.24, 0.48, 0.67],
                [0.22, 0.36, 0.28, 0.51, 0.46, 0.73],
            ],
            [
                [0.18, 0.09, 0.27, 0.43, 0.35, 0.61],
                [0.00, 0.00, 0.17, 0.11, 0.38, 0.29],
            ],
            [
                [0.71, 0.56, 0.62, 0.41, 0.30, 0.12],
                [0.35, 0.49, 0.39, 0.58, 0.52, 0.69],
            ],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(314)
    scenarios = np.clip(
        truth[:, None]
        + rng.normal(0.0, 0.09, size=(truth.shape[0], 7, *truth.shape[1:])),
        0.0,
        1.0,
    )
    return scenarios, truth


def test_three_point_contributions_reconstruct_daily_gap() -> None:
    scenarios, truth = _example()
    result = lag1_increment_state_decomposition(scenarios, truth, epsilon=0.0)

    member_increment = np.diff(scenarios, axis=-1)
    truth_increment = np.diff(truth, axis=-1)
    direct_forecast = np.asarray(
        [
            np.corrcoef(
                member_increment[day, ..., :-1].reshape(-1),
                member_increment[day, ..., 1:].reshape(-1),
            )[0, 1]
            for day in range(len(truth))
        ]
    )
    direct_truth = np.asarray(
        [
            np.corrcoef(
                truth_increment[day, ..., :-1].reshape(-1),
                truth_increment[day, ..., 1:].reshape(-1),
            )[0, 1]
            for day in range(len(truth))
        ]
    )

    assert result["unit_of_analysis"] == "calendar_day"
    assert result["forecast_case_count"].shape == (3, 2)
    assert np.array_equal(
        result["forecast_case_count"].sum(axis=1),
        np.full(3, 7 * 2 * (6 - 2)),
    )
    assert np.array_equal(
        result["truth_case_count"].sum(axis=1), np.full(3, 2 * (6 - 2))
    )
    assert np.allclose(result["forecast_case_share"].sum(axis=1), 1.0)
    assert np.allclose(result["truth_case_share"].sum(axis=1), 1.0)
    assert np.allclose(
        result["forecast_correlation_contribution"].sum(axis=1),
        result["forecast_total_correlation"],
    )
    assert np.allclose(result["forecast_total_correlation"], direct_forecast)
    assert np.allclose(
        result["truth_correlation_contribution"].sum(axis=1),
        result["truth_total_correlation"],
    )
    assert np.allclose(result["truth_total_correlation"], direct_truth)
    assert np.allclose(
        result["signed_correlation_gap_contribution"].sum(axis=1),
        result["signed_total_correlation_gap"],
    )
    assert np.allclose(
        result["reconstructed_signed_total_correlation_gap"],
        result["signed_total_correlation_gap"],
    )
    assert np.allclose(
        result["absolute_total_correlation_error"],
        np.abs(result["signed_total_correlation_gap"]),
    )


def test_perfect_predictive_paths_have_zero_gap_by_category() -> None:
    _, truth = _example()
    scenarios = np.repeat(truth[:, None], 5, axis=1)
    states = classify_states(scenarios, epsilon=0.0)
    result = lag1_increment_state_decomposition(
        scenarios, truth, scenario_states=states, epsilon=0.0
    )

    assert np.all(result["valid_day"])
    assert np.allclose(result["signed_total_correlation_gap"], 0.0)
    assert np.allclose(result["signed_correlation_gap_contribution"], 0.0)
    assert np.allclose(
        result["forecast_case_share"], result["truth_case_share"]
    )
    assert result["forecast_state_source"] == "archive_explicit_state"


def test_threshold_labels_are_not_claimed_as_ddpm_latent_states() -> None:
    scenarios, truth = _example()
    # Remove all numerical boundary values from the generated sample.  The
    # threshold-derived version is then entirely interior at epsilon=0.
    interior = np.clip(scenarios, 0.05, 0.95)
    threshold = lag1_increment_state_decomposition(interior, truth, epsilon=0.0)
    assert np.all(threshold["forecast_case_count"][:, 1] == 0)
    assert "not a latent generative state" in threshold["forecast_state_semantics"]
    assert threshold["forecast_state_source"].startswith(
        "threshold_derived_output_boundary"
    )

    # An explicit state tensor is authoritative even when its label is not
    # recoverable by thresholding the continuous values.
    explicit_states = np.ones_like(interior, dtype=np.int8)
    explicit_states[..., 0] = 0
    explicit = lag1_increment_state_decomposition(
        interior, truth, scenario_states=explicit_states, epsilon=0.0
    )
    assert np.all(explicit["forecast_case_count"][:, 1] > 0)
    assert explicit["forecast_state_source"] == "archive_explicit_state"


def test_whole_member_permutation_is_invariant() -> None:
    scenarios, truth = _example()
    states = classify_states(scenarios, epsilon=0.01)
    permutation = np.asarray([6, 2, 0, 5, 1, 4, 3])
    left = lag1_increment_state_decomposition(
        scenarios, truth, scenario_states=states, epsilon=0.01
    )
    right = lag1_increment_state_decomposition(
        scenarios[:, permutation],
        truth,
        scenario_states=states[:, permutation],
        epsilon=0.01,
    )
    for key in (
        "forecast_case_count",
        "forecast_case_share",
        "forecast_correlation_contribution",
        "forecast_conditional_global_standardized_moment",
        "forecast_conditional_correlation",
        "forecast_total_correlation",
        "signed_total_correlation_gap",
    ):
        assert np.allclose(left[key], right[key], equal_nan=True)


def run_all_tests() -> None:
    tests = (
        test_three_point_contributions_reconstruct_daily_gap,
        test_perfect_predictive_paths_have_zero_gap_by_category,
        test_threshold_labels_are_not_claimed_as_ddpm_latent_states,
        test_whole_member_permutation_is_invariant,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
