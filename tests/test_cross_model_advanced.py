"""Assertion-based tests; runnable directly without pytest."""

from __future__ import annotations

import numpy as np

from cross_model_diagnostics.advanced import (
    atom_event_metrics,
    generated_transition_crps_decomposition,
    lagged_variogram_score,
    lagged_variogram_suite,
    ramp_crps_metrics,
    shuffle_member_trajectories,
    zero_run_statistics,
)


def _bounded_truth() -> np.ndarray:
    rng = np.random.default_rng(7)
    truth = rng.uniform(0.1, 0.9, size=(5, 4, 12))
    truth[0, :, :3] = 0.0
    truth[1, 0, ::2] = 0.0
    return truth


def test_perfect_forecast_scores_zero() -> None:
    truth = _bounded_truth()
    scenarios = np.repeat(truth[:, None], 8, axis=1)
    for value in ramp_crps_metrics(scenarios, truth).values():
        assert np.allclose(value, 0.0)
    for value in atom_event_metrics(scenarios, truth, epsilon=0.0).values():
        assert np.allclose(value, 0.0)
    suite = lagged_variogram_suite(scenarios, truth, lags=(0, 1, 3))
    assert all(np.allclose(value, 0.0) for value in suite["per_day"].values())


def test_member_permutation_invariance() -> None:
    rng = np.random.default_rng(19)
    truth = _bounded_truth()
    scenarios = np.clip(
        truth[:, None] + rng.normal(0.0, 0.13, size=(5, 11, 4, 12)), 0.0, 1.0
    )
    permutation = rng.permutation(scenarios.shape[1])
    permuted = scenarios[:, permutation]
    left = ramp_crps_metrics(scenarios, truth)
    right = ramp_crps_metrics(permuted, truth)
    assert left.keys() == right.keys()
    assert all(np.allclose(left[key], right[key]) for key in left)
    for domain in ("level", "ramp"):
        for pair_type in ("cross_zone", "same_zone_temporal"):
            score_left = lagged_variogram_score(
                scenarios,
                truth,
                lag=2,
                domain=domain,
                pair_type=pair_type,
            )
            score_right = lagged_variogram_score(
                permuted,
                truth,
                lag=2,
                domain=domain,
                pair_type=pair_type,
            )
            assert np.allclose(score_left, score_right)

    decomposition_left = generated_transition_crps_decomposition(scenarios, truth)
    decomposition_right = generated_transition_crps_decomposition(permuted, truth)
    for key in ("first_term", "pair_term", "net_contribution", "total_local_ramp_CRPS"):
        assert np.allclose(decomposition_left[key], decomposition_right[key])


def test_generated_transition_contributions_are_exact() -> None:
    rng = np.random.default_rng(23)
    truth = _bounded_truth()
    scenarios = np.clip(
        truth[:, None] + rng.normal(0.0, 0.2, size=(5, 13, 4, 12)), 0.0, 1.0
    )
    result = generated_transition_crps_decomposition(scenarios, truth, epsilon=0.02)
    assert np.allclose(
        result["first_term"] - result["pair_term"], result["net_contribution"]
    )
    assert np.allclose(
        result["net_contribution"].sum(axis=1), result["total_local_ramp_CRPS"]
    )
    assert np.allclose(
        result["reconstructed_local_ramp_CRPS"], result["total_local_ramp_CRPS"]
    )
    assert int(result["member_cases"].sum()) == int(np.prod(np.diff(scenarios, axis=-1).shape))

    # Verify the O(M log M) pair allocation against the explicit MxM identity.
    ramp = np.diff(scenarios, axis=-1)
    observed = np.diff(truth, axis=-1)
    first = np.abs(ramp - observed[:, None]).mean(axis=1)
    pair = np.abs(ramp[:, :, None] - ramp[:, None, :]).mean(axis=(1, 2)) / 2.0
    explicit = (first - pair).mean(axis=(1, 2))
    assert np.allclose(result["total_local_ramp_CRPS"], explicit)


def test_synthetic_lag_and_shuffle_control() -> None:
    days, members, zones, hours = 8, 17, 4, 14
    time = np.arange(hours, dtype=np.float64)
    truth = np.empty((days, zones, hours), dtype=np.float64)
    for day in range(days):
        for zone in range(zones):
            truth[day, zone] = 0.35 * np.sin((time + day) / 3.0) + 0.08 * zone
    offsets = np.linspace(-0.25, 0.25, members)
    scenarios = truth[:, None] + offsets[None, :, None, None]

    original = lagged_variogram_score(
        scenarios,
        truth,
        lag=2,
        domain="level",
        pair_type="cross_zone",
    )
    shuffled = shuffle_member_trajectories(scenarios, seed=41)
    negative = lagged_variogram_score(
        shuffled,
        truth,
        lag=2,
        domain="level",
        pair_type="cross_zone",
    )
    assert np.allclose(original, 0.0)
    assert float(negative.mean()) > float(original.mean()) + 1e-5

    # Whole local trajectories (and hence local ramps/durations) are intact.
    for day in range(days):
        for zone in range(zones):
            assert np.allclose(
                np.sort(scenarios[day, :, zone], axis=0),
                np.sort(shuffled[day, :, zone], axis=0),
            )
            original_paths = sorted(map(tuple, scenarios[day, :, zone]))
            shuffled_paths = sorted(map(tuple, shuffled[day, :, zone]))
            assert original_paths == shuffled_paths
    same_original = lagged_variogram_score(
        scenarios,
        truth,
        lag=3,
        domain="level",
        pair_type="same_zone_temporal",
    )
    same_shuffled = lagged_variogram_score(
        shuffled,
        truth,
        lag=3,
        domain="level",
        pair_type="same_zone_temporal",
    )
    assert np.allclose(same_original, same_shuffled)


def test_all_zero_interior_and_alternating_runs() -> None:
    all_zero = np.zeros((2, 3, 24), dtype=np.float64)
    zero_stats = zero_run_statistics(all_zero)
    assert np.array_equal(zero_stats["H_total_zero_hours"], np.asarray([72, 72]))
    assert np.array_equal(zero_stats["K_zero_run_count"], np.asarray([3, 3]))
    assert np.array_equal(zero_stats["L_longest_zero_run"], np.asarray([24, 24]))

    interior = np.full((2, 3, 24), 0.4)
    interior_stats = zero_run_statistics(interior)
    assert all(np.all(value == 0) for value in interior_stats.values())

    alternating = np.ones((2, 3, 24), dtype=np.float64)
    alternating[..., ::2] = 0.0
    alternating_stats = zero_run_statistics(alternating)
    assert np.array_equal(
        alternating_stats["H_total_zero_hours"], np.asarray([36, 36])
    )
    assert np.array_equal(alternating_stats["K_zero_run_count"], np.asarray([36, 36]))
    assert np.array_equal(alternating_stats["L_longest_zero_run"], np.asarray([1, 1]))

    # Epsilon is active and the three limiting patterns remain perfectly scored.
    for truth in (all_zero, interior, alternating):
        scenarios = np.repeat(truth[:, None], 6, axis=1)
        metrics = atom_event_metrics(scenarios, truth, epsilon=0.01)
        assert all(np.allclose(value, 0.0) for value in metrics.values())


def run_all_tests() -> None:
    tests = [
        test_perfect_forecast_scores_zero,
        test_member_permutation_invariance,
        test_generated_transition_contributions_are_exact,
        test_synthetic_lag_and_shuffle_control,
        test_all_zero_interior_and_alternating_runs,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
