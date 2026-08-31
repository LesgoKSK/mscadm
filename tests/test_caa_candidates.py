import json

import numpy as np
import pytest

import caa_rahc.candidates as candidate_module
from caa_rahc.candidates import CandidateGrid, generate_oof_candidates
from caa_rahc.hinge_tail import WidthCapInfeasibleError
from caa_rahc.selection import METRIC_KEYS


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


def _synthetic_inputs() -> dict[str, object]:
    rng = np.random.default_rng(123)
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
    train_day = np.arange("2019-01-01", "2019-01-21", dtype="datetime64[D]")

    assignments = {
        "zone": np.broadcast_to(zone[:, None], (cases, hours)).copy(),
        "hour": np.broadcast_to(np.arange(hours)[None, :], (cases, hours)).copy(),
        "fixed_regime": np.broadcast_to((np.arange(hours) % 3)[None, :], (cases, hours)).copy(),
    }
    return {
        "raw_scenarios_by_seed": raw,
        "observations": observations,
        "zone": zone,
        "day": day,
        "train_condition": train_condition,
        "train_target": train_target,
        "calibration_condition": calibration_condition,
        "train_day": train_day,
        "assignments": assignments,
        "a1_scenarios_by_seed": raw,
    }


def _small_grid() -> CandidateGrid:
    return CandidateGrid(
        atom_strengths=(0.5,),
        tail_strengths=(0.25,),
        maximum_logit_shifts=(0.10,),
        # A large local cap keeps this synthetic test focused on orchestration;
        # the selector still applies its registered global 0.05 guard.
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


def _install_fake_gate(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_fit(
        scenarios_by_seed: list[np.ndarray],
        observations: np.ndarray,
        features_by_seed: list[np.ndarray],
        zone: np.ndarray,
        **kwargs: object,
    ) -> _FakeGate:
        calls.append(
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

    monkeypatch.setattr(candidate_module, "fit_tail_gate", fake_fit)
    return calls


def test_small_grid_streams_all_families_without_date_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_gate(monkeypatch)
    inputs = _synthetic_inputs()
    original_assignments = {
        key: value.copy() for key, value in inputs["assignments"].items()
    }
    result = generate_oof_candidates(**inputs, grid=_small_grid())

    assert {family: len(values) for family, values in result.records_by_family.items()} == {
        "A0": 1,
        "A1": 1,
        "A2": 1,
        "A3": 1,
        "A4": 1,
        "A5": 1,
        "A6": 1,
    }
    assert [record.metadata["family"] for record in result.primary_constrained_records] == [
        "A0",
        "A4",
    ]
    assert [record.metadata["family"] for record in result.primary_unconstrained_records] == [
        "A0",
        "A5",
    ]
    assert all(
        record.metadata["family"] not in {"A2", "A3", "A6"}
        for record in result.primary_constrained_records
    )

    a4 = result.records_for_family("A4")[0]
    a5 = result.records_for_family("A5")[0]
    assert a4.selection_policy == "constrained"
    assert a5.selection_policy == "unconstrained"
    assert a5.metadata["mirrors_A4_record"] == a4.name
    for metric in METRIC_KEYS:
        assert np.array_equal(a4.metrics[metric], a5.metrics[metric])
        assert np.asarray(a4.metrics[metric]).shape == (3, 15)
    assert np.array_equal(a4.conditional_ace90, a5.conditional_ace90)

    # Five folds times regularized/no-shrink; every fit sees only 12 of 15
    # dates and pools all three model seeds.
    assert len(calls) == 10
    assert all(call["seed_count"] == 3 for call in calls)
    assert all(call["scenario_cases"] == [12, 12, 12] for call in calls)
    assert all(call["feature_cases"] == [12, 12, 12] for call in calls)
    for fold in result.audit["folds"]:
        assert fold["overlap_count"] == 0
        training = set(fold["training_dates"])
        held = set(fold["held_dates"])
        assert training.isdisjoint(held)
        assert len(training | held) == 15

    assert type(result.zero_model).__module__ == "caa_rahc.structural_atom"
    assert result.zero_model.train_cells == np.asarray(inputs["train_target"]).size
    assert result.audit["zero_model"]["train_date_audit"]["calibration_overlap_count"] == 0
    assert result.audit["protocol"]["width_cap_reference"] == "atom_only for local transformation"
    assert result.audit["invalid_candidates"] == []
    assert all(
        "streamed per seed" in record.metadata.get("scenario_storage", "")
        for family in ("A2", "A3", "A4", "A6")
        for record in result.records_for_family(family)
    )
    for key, value in original_assignments.items():
        assert np.array_equal(inputs["assignments"][key], value)
    json.dumps(result.audit, sort_keys=True, allow_nan=False)


def test_width_cap_infeasibility_is_explicitly_invalid_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gate(monkeypatch)

    def fail_width_cap(*args: object, **kwargs: object) -> object:
        assert kwargs["width_cap_reference"] == "atom_only"
        raise WidthCapInfeasibleError("synthetic atom/tail cap conflict")

    monkeypatch.setattr(
        candidate_module, "atom_aware_logit_hinge_tail", fail_width_cap
    )
    inputs = _synthetic_inputs()
    # A1 is irrelevant to this failure-path test.
    inputs["a1_scenarios_by_seed"] = None
    result = generate_oof_candidates(**inputs, grid=_small_grid())

    assert len(result.records_for_family("A0")) == 1
    assert all(
        len(result.records_for_family(family)) == 0
        for family in ("A2", "A3", "A4", "A5", "A6")
    )
    assert result.primary_constrained_records[0].name == "A0"
    assert len(result.primary_constrained_records) == 1
    assert len(result.audit["invalid_candidates"]) == 4
    for failure in result.audit["invalid_candidates"]:
        assert failure["reason"] == "WidthCapInfeasibleError"
        assert failure["silently_replaced"] is False
        assert "synthetic atom/tail cap conflict" in failure["message"]
        assert failure["config"]["width_cap_reference"] == "atom_only"
    assert result.audit["invalid_handling"].startswith(
        "WidthCapInfeasibleError is recorded as invalid"
    )


def test_overlapping_structural_train_dates_are_rejected_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gate(monkeypatch)
    inputs = _synthetic_inputs()
    inputs["train_day"] = np.asarray(inputs["train_day"]).copy()
    inputs["train_day"][0] = np.asarray(inputs["day"])[0]
    with pytest.raises(ValueError, match="train dates overlap calibration"):
        generate_oof_candidates(**inputs, grid=_small_grid())

