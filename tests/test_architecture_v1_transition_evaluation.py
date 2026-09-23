from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from architecture_v1.transition_evaluation import (
    ARCHIVE_SCHEMA,
    METRIC_NAMES,
    aggregate_metric_replicates,
    fit_train_only_ramp_thresholds,
    load_transition_archive,
    transition_per_day_metrics,
    validate_transition_archive_arrays,
    write_transition_archive,
)


def _perfect_archive_arrays() -> dict[str, np.ndarray]:
    days = 3
    members = 10
    hour = np.arange(24, dtype=np.float64)
    base = 0.5 + 0.25 * np.sin(2.0 * np.pi * hour / 24.0)
    truth = np.broadcast_to(base, (days, 10, 24)).copy()
    truth[0, 0, 0] = 0.0
    truth[1, 1, 1] = 1.0
    scenarios = np.repeat(truth[:, None], members, axis=1)
    states = np.ones_like(scenarios, dtype=np.int8)
    states[scenarios == 0.0] = 0
    states[scenarios == 1.0] = 2
    zero = (truth == 0.0).astype(np.float64)
    one = (truth == 1.0).astype(np.float64)
    observed = np.ones_like(truth, dtype=bool)
    return {
        "scenarios": scenarios,
        "observations": truth,
        "observed_mask": observed,
        "raw_missing_mask": ~observed,
        "states": states,
        "zero_probability": zero,
        "one_probability": one,
        "day": np.asarray(["2012-01-01", "2012-01-02", "2012-01-03"], dtype="datetime64[D]"),
        "zones": np.arange(1, 11, dtype=np.int64),
    }


def test_train_only_ramp_thresholds_use_only_observed_pairs() -> None:
    generator = np.random.default_rng(7501)
    truth = generator.uniform(0.05, 0.95, size=(8, 10, 24))
    observed = np.ones_like(truth, dtype=bool)
    observed[0, 0, :3] = False
    first = fit_train_only_ramp_thresholds(truth, observed)
    changed = truth.copy()
    changed[~observed] = 1e6
    second = fit_train_only_ramp_thresholds(changed, observed)
    assert first == second
    assert float(first["down_threshold"]) < 0.0
    assert float(first["up_threshold"]) > 0.0


def test_all_registered_metrics_are_per_day_and_perfect_sample_scores_zero() -> None:
    arrays = _perfect_archive_arrays()
    thresholds = {
        "down_threshold": -0.1,
        "up_threshold": 0.1,
    }
    metrics = transition_per_day_metrics(
        arrays["scenarios"],
        arrays["observations"],
        arrays["observed_mask"],
        zero_probability=arrays["zero_probability"],
        one_probability=arrays["one_probability"],
        ramp_thresholds=thresholds,
    )
    assert tuple(metrics) == METRIC_NAMES
    assert all(value.shape == (3,) for value in metrics.values())
    for name, value in metrics.items():
        if name == "coverage90":
            assert np.array_equal(value, np.ones(3))
        else:
            assert np.allclose(value, 0.0, atol=1e-12), name


def test_transition_archive_round_trip_is_hashed_and_exact() -> None:
    arrays = _perfect_archive_arrays()
    near_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    assert np.float32(near_one) == 1.0
    arrays["scenarios"][0, 0, 0, 1] = near_one
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "config_sha256": "1" * 64,
        "training_freeze_sha256": "2" * 64,
        "atom_checkpoint_sha256": "3" * 64,
        "denoiser_checkpoint_sha256": "4" * 64,
        "outer_fold": 0,
        "model_seed": 3,
        "path_id": "LEVEL_IID",
        "sampling_seed": 61000,
        "members": 10,
        "DDIM_steps": 31,
        "member_chunk": 5,
        "atom_allocation_sha256": "5" * 64,
        "native_epsilon_sha256": "6" * 64,
        "target_state_argument_used_for_sampling": False,
    }
    with tempfile.TemporaryDirectory(prefix="tgo_archive_", dir="/tmp") as raw:
        path = Path(raw) / "scenario.npz"
        digest, manifest = write_transition_archive(
            path, arrays=arrays, metadata=metadata
        )
        assert len(digest) == 64
        assert manifest.is_file()
        restored, restored_metadata, restored_digest = load_transition_archive(path)
        assert restored_digest == digest
        assert restored["scenarios"].dtype == np.float64
        assert restored["scenarios"][0, 0, 0, 1] == near_one
        assert restored_metadata == metadata
        for name, value in arrays.items():
            if name == "day":
                assert np.array_equal(restored[name], value.astype("datetime64[D]"))
            elif np.issubdtype(np.asarray(value).dtype, np.floating):
                assert np.allclose(restored[name], value, rtol=0.0, atol=1e-7)
            else:
                assert np.array_equal(restored[name], value)


def test_metric_replicates_average_within_calendar_day() -> None:
    first = {name: np.asarray([1.0, 2.0]) for name in METRIC_NAMES}
    second = {name: np.asarray([3.0, 4.0]) for name in METRIC_NAMES}
    averaged = aggregate_metric_replicates([first, second])
    assert tuple(averaged) == METRIC_NAMES
    assert all(np.array_equal(value, np.asarray([2.0, 3.0])) for value in averaged.values())

def test_transition_archive_rejects_fp32_scenario_payload() -> None:
    arrays = _perfect_archive_arrays()
    arrays["scenarios"] = arrays["scenarios"].astype(np.float32)
    try:
        validate_transition_archive_arrays(**arrays)
    except TypeError as error:
        assert "FP64" in str(error)
    else:
        raise AssertionError("FP32 scenario archive was accepted")


def test_late_horizon_score_is_explicitly_unavailable_without_observed_cells() -> None:
    arrays = _perfect_archive_arrays()
    observed = arrays["observed_mask"].copy()
    observed[2, :, -6:] = False
    scores = transition_per_day_metrics(
        arrays["scenarios"],
        arrays["observations"],
        observed,
        zero_probability=arrays["zero_probability"],
        one_probability=arrays["one_probability"],
        ramp_thresholds={"down_threshold": -0.1, "up_threshold": 0.1},
    )
    assert np.isfinite(scores["late_horizon_level_CRPS"][:2]).all()
    assert np.isnan(scores["late_horizon_level_CRPS"][2])
    assert all(
        np.isfinite(value).all()
        for name, value in scores.items()
        if name != "late_horizon_level_CRPS"
    )
