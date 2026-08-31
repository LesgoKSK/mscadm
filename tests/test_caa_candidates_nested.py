import json

import numpy as np
import pytest

import caa_rahc.candidates as prototype_module
import caa_rahc.candidates_nested as nested_module
from caa_rahc.candidates_nested import CandidateGrid, generate_oof_candidates


class _FakeGate:
    def __init__(self, regularization: float, training_cases: int) -> None:
        self.regularization = float(regularization)
        self.fit_summary = {
            "regularization": self.regularization,
            "training_cases": int(training_cases),
            "mean_lower_gate": 0.10,
            "mean_upper_gate": 0.12,
        }

    def predict(
        self, features: np.ndarray, zone: np.ndarray, *, strength: float = 1.0
    ) -> np.ndarray:
        assert features.shape == (len(zone), 24, 7)
        result = np.empty((len(zone), 24, 2), dtype=np.float64)
        result[..., 0] = 0.10 * float(strength)
        result[..., 1] = 0.12 * float(strength)
        return result


def _inputs() -> dict[str, object]:
    rng = np.random.default_rng(321)
    cases, members, hours = 15, 11, 24
    hour = np.arange(hours)
    observations = np.clip(
        0.45
        + 0.15 * np.sin(2.0 * np.pi * hour / hours)[None, :]
        + rng.normal(0.0, 0.04, (cases, hours)),
        0.08,
        0.92,
    )
    raw = [
        np.clip(
            observations[:, None, :]
            + rng.normal(0.0, 0.065 + 0.004 * seed, (cases, members, hours)),
            0.02,
            0.98,
        ).astype(np.float32)
        for seed in range(3)
    ]
    zone = np.arange(cases) % 10 + 1
    day = np.arange("2020-01-01", "2020-01-16", dtype="datetime64[D]")

    calibration_condition = rng.normal(size=(cases, hours, 20)).astype(np.float32)
    calibration_condition[..., 10:] = 0.0
    for index, zone_id in enumerate(zone):
        calibration_condition[index, :, 10 + zone_id - 1] = 1.0

    train_cases = 20
    train_condition = rng.normal(size=(train_cases, hours, 20)).astype(np.float32)
    train_condition[..., 10:] = 0.0
    for index in range(train_cases):
        train_condition[index, :, 10 + index % 10] = 1.0
    train_target = np.clip(
        0.30 + rng.normal(0.0, 0.14, (train_cases, hours)), 0.01, 0.95
    )
    train_target[::2, :3] = 0.0
    train_target[1, -1] = 1.0

    assignments = {
        "zone": np.broadcast_to(zone[:, None], (cases, hours)).copy(),
        "hour": np.broadcast_to(np.arange(hours)[None, :], (cases, hours)).copy(),
        "fixed_regime": np.broadcast_to(
            (np.arange(hours) % 3)[None, :], (cases, hours)
        ).copy(),
    }
    return {
        "raw_scenarios_by_seed": raw,
        "observations": observations,
        "zone": zone,
        "day": day,
        "train_condition": train_condition,
        "train_target": train_target,
        "calibration_condition": calibration_condition,
        "train_day": np.arange(
            "2019-01-01", "2019-01-21", dtype="datetime64[D]"
        ),
        "assignments": assignments,
        "a1_scenarios_by_seed": raw,
    }


def _grid() -> CandidateGrid:
    return CandidateGrid(
        atom_strengths=(0.5,),
        tail_strengths=(0.25,),
        maximum_logit_shifts=(0.10,),
        maximum_width_increases=(1.0,),
        atom_regularization_c=0.1,
        zero_model_maximum_iterations=100,
        maximum_lambda=0.2,
        gate_regularization=0.03,
        gate_no_shrink_regularization=0.0,
        gate_steps=1,
        gate_batch_cells=64,
        gate_learning_rate=0.01,
        fold_seed=19,
    )


def test_outer_gate_held_dates_never_enter_any_inner_c0_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    all_dates = np.asarray(inputs["day"]).astype("datetime64[D]")
    inner_crossfit_inputs: list[np.ndarray] = []
    gate_fit_calls: list[dict[str, object]] = []

    real_inner_crossfit = nested_module.crossfit_c0

    def recording_inner_crossfit(
        scenarios_by_seed: list[np.ndarray],
        observations: np.ndarray,
        day: np.ndarray,
        **kwargs: object,
    ) -> object:
        inner_crossfit_inputs.append(np.asarray(day).astype("datetime64[D]").copy())
        return real_inner_crossfit(
            scenarios_by_seed, observations, day, **kwargs
        )

    def fake_gate_fit(
        scenarios_by_seed: list[np.ndarray],
        observations: np.ndarray,
        features_by_seed: list[np.ndarray],
        zone: np.ndarray,
        **kwargs: object,
    ) -> _FakeGate:
        gate_fit_calls.append(
            {
                "seed_count": len(scenarios_by_seed),
                "scenario_cases": [len(value) for value in scenarios_by_seed],
                "feature_cases": [len(value) for value in features_by_seed],
                "observation_cases": len(observations),
                "zone_cases": len(zone),
                "regularization": float(kwargs["regularization"]),
            }
        )
        return _FakeGate(float(kwargs["regularization"]), len(observations))

    monkeypatch.setattr(nested_module, "crossfit_c0", recording_inner_crossfit)
    monkeypatch.setattr(nested_module, "fit_tail_gate", fake_gate_fit)
    prototype_gate_before = prototype_module.fit_tail_gate

    result = generate_oof_candidates(**inputs, grid=_grid())

    nested_audit = result.audit["strict_nested_C0_gate_training"]
    assert len(inner_crossfit_inputs) == len(nested_audit) == 5
    for passed_dates, outer_fold_audit in zip(inner_crossfit_inputs, nested_audit):
        outer_held = np.asarray(
            outer_fold_audit["outer_held_dates"], dtype="datetime64[D]"
        )
        outer_training = np.asarray(
            outer_fold_audit["outer_training_dates"], dtype="datetime64[D]"
        )

        # This is the core leakage guard: the entire input universe available
        # to every inner C0 fold excludes the current outer gate-held dates.
        assert np.intersect1d(passed_dates, outer_held).size == 0
        assert np.array_equal(np.unique(passed_dates), np.unique(outer_training))
        assert np.array_equal(
            np.unique(np.concatenate([outer_training, outer_held])), all_dates
        )
        assert outer_fold_audit[
            "outer_held_dates_used_anywhere_in_inner_C0"
        ] == 0

        for inner_fold_audit in outer_fold_audit["inner_C0_folds"]:
            inner_train = np.asarray(
                inner_fold_audit["training_dates"], dtype="datetime64[D]"
            )
            inner_held = np.asarray(
                inner_fold_audit["held_dates"], dtype="datetime64[D]"
            )
            assert np.intersect1d(inner_train, outer_held).size == 0
            assert np.intersect1d(inner_held, outer_held).size == 0
            assert np.intersect1d(inner_train, inner_held).size == 0
            assert inner_fold_audit["outer_gate_held_overlap_count"] == 0

    # Five outer folds, each with regularized and no-shrink fits.  Both fits
    # receive only nested OOF curves on the 12 outer-training dates.
    assert len(gate_fit_calls) == 10
    assert all(call["seed_count"] == 3 for call in gate_fit_calls)
    assert all(call["scenario_cases"] == [12, 12, 12] for call in gate_fit_calls)
    assert all(call["feature_cases"] == [12, 12, 12] for call in gate_fit_calls)
    assert all(call["observation_cases"] == 12 for call in gate_fit_calls)

    assert prototype_module.fit_tail_gate is prototype_gate_before
    assert result.audit["strict_nested_invariant"] == {
        "outer_gate_held_dates_used_anywhere_in_inner_C0": 0,
        "gate_fit_calls": 10,
        "adapter_restored_after_call": True,
    }
    assert result.audit["protocol"]["authoritative_module"] == (
        "caa_rahc.candidates_nested"
    )
    assert [
        record.metadata["family"] for record in result.primary_constrained_records
    ] == ["A0", "A4"]
    assert [
        record.metadata["family"] for record in result.primary_unconstrained_records
    ] == ["A0", "A5"]
    for family in ("A3", "A4", "A5", "A6"):
        assert all(
            record.metadata["outer_gate_held_dates_used_by_inner_C0"] == 0
            for record in result.records_for_family(family)
        )
    json.dumps(result.audit, sort_keys=True, allow_nan=False)
