"""Protocol tests for the frozen TGO-v1 physical evaluation runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import torch

from architecture_v1.transition_evaluation import METRIC_NAMES
from architecture_v1.transition_probe import PATH_IDS
from repro_scripts import run_architecture_v1_transition_object_evaluation as runner


def _margins() -> dict[str, float | int]:
    return {
        "ramp_CRPS_absolute_improvement_min": 0.001,
        "lagged_variogram_relative_improvement_min": 0.05,
        "attribution_ramp_CRPS_absolute_noninferiority_margin": 0.001,
        "attribution_lagged_variogram_relative_noninferiority_margin": 0.05,
        "level_CRPS_absolute_noninferiority_margin": 0.0015,
        "normalized_joint_ES_relative_noninferiority_margin": 0.02,
        "coverage90_absolute_difference_max": 0.02,
        "width90_relative_increase_max": 0.1,
        "daily_mean_power_CRPS_relative_noninferiority_margin": 0.02,
        "late_horizon_level_CRPS_absolute_noninferiority_margin": 0.0015,
        "minimum_positive_outer_folds": 5,
        "minimum_positive_model_seeds": 2,
    }


def _synthetic_metrics(days: int = 24) -> dict[str, dict[str, np.ndarray]]:
    base = {name: np.full(days, 1.0, dtype=np.float64) for name in METRIC_NAMES}
    values: dict[str, dict[str, np.ndarray]] = {}
    for index, path_id in enumerate(PATH_IDS):
        values[path_id] = {name: value.copy() for name, value in base.items()}
        if path_id != "LEVEL_IID":
            values[path_id]["ramp_CRPS"] -= 0.002 + index * 0.0001
            values[path_id]["lagged_increment_variogram_score"] -= 0.06 + index * 0.001
    # The true path must also beat each attribution control in the synthetic GO case.
    values["TRANSITION_TRUE"]["ramp_CRPS"][:] = 0.990
    values["TRANSITION_TRUE"]["lagged_increment_variogram_score"][:] = 0.88
    return values


def test_endpoint_registry_is_exact_unique_and_has_28_entries() -> None:
    endpoints = runner._endpoint_registry()
    assert len(endpoints) == len(set(endpoints)) == 28
    assert endpoints[0] == (
        "TRANSITION_TRUE_vs_LEVEL_IID__ramp_CRPS__absolute_benefit"
    )
    assert endpoints[-1] == (
        "TRANSITION_TRUE_vs_LEVEL_IID__late_horizon_level_CRPS__absolute_harm"
    )


def test_contributions_preserve_pairing_direction_and_relative_scale() -> None:
    metrics = _synthetic_metrics()
    values = runner._inference_contributions(metrics)
    assert tuple(values) == runner._endpoint_registry()
    assert np.allclose(
        values["TRANSITION_TRUE_vs_LEVEL_IID__ramp_CRPS__absolute_benefit"],
        0.01,
    )
    assert np.allclose(
        values[
            "TRANSITION_TRUE_vs_LEVEL_IID__lagged_increment_variogram_score__relative_benefit"
        ],
        0.12,
    )
    assert np.allclose(
        values["TRANSITION_TRUE_vs_LEVEL_IID__level_CRPS__absolute_harm"],
        0.0,
    )


def test_adjudication_go_requires_all_attribution_safety_and_stability_gates() -> None:
    bands = {
        name: {
            "estimate": 0.10,
            "simultaneous_low": 0.08,
            "simultaneous_high": 0.12,
        }
        for name in runner._endpoint_registry()
    }
    for name in runner._endpoint_registry()[-6:]:
        bands[name] = {
            "estimate": 0.0,
            "simultaneous_low": -0.0001,
            "simultaneous_high": 0.0001,
        }
    go = runner._adjudicate(
        bands,
        positive_outer_folds=6,
        positive_model_seeds=2,
        technical_eligibility=True,
        margins=_margins(),
    )
    assert go["status"] == "TGO_V1_ATTRIBUTED_GO"
    assert go["all_GO_gates_passed"] is True

    failed = {name: dict(value) for name, value in bands.items()}
    wrong = (
        "TRANSITION_TRUE_vs_TRANSITION_WRONG__ramp_CRPS__absolute_benefit"
    )
    failed[wrong]["simultaneous_low"] = -0.001
    no_go = runner._adjudicate(
        failed,
        positive_outer_folds=6,
        positive_model_seeds=2,
        technical_eligibility=True,
        margins=_margins(),
    )
    assert no_go["status"] == "ADJACENCY_NOT_IDENTIFIED"
    assert no_go["all_GO_gates_passed"] is False


def test_candidate_failure_cannot_be_rescued_by_unregistered_sampling_label() -> None:
    bands = {
        name: {
            "estimate": 0.0,
            "simultaneous_low": -0.01,
            "simultaneous_high": 0.01,
        }
        for name in runner._endpoint_registry()
    }
    decision = runner._adjudicate(
        bands,
        positive_outer_folds=6,
        positive_model_seeds=2,
        technical_eligibility=True,
        margins=_margins(),
    )
    assert decision["status"] == "TGO_V1_NO_GO"
    assert decision["status"] != "SAMPLING_NO_GO"



def test_strict_interior_encoding_changes_only_fp64_saturated_values() -> None:
    logits = torch.tensor([0.0, 37.614, -750.0, 0.0, 0.0], dtype=torch.float64)
    states = torch.tensor([0, 1, 1, 2, 1], dtype=torch.long)
    decoded, counts = runner._decode_strict_interior_sigmoid(logits, states)
    assert counts == {"zero_count": 1, "one_count": 1}
    assert decoded.dtype == torch.float64
    assert decoded[0] == 0.0
    assert decoded[3] == 1.0
    assert decoded[4] == 0.5
    assert decoded[1] == torch.nextafter(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert decoded[2] == torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_metric_artifact_records_the_single_unavailable_late_horizon_day() -> None:
    days = np.asarray(["2013-12-30", "2013-12-31"], dtype="datetime64[D]")
    metrics = {name: np.zeros(2, dtype=np.float64) for name in METRIC_NAMES}
    metrics["late_horizon_level_CRPS"] = np.asarray([0.0, np.nan])
    with tempfile.TemporaryDirectory(prefix="tgo_v13_metric_", dir="/tmp") as raw:
        path = Path(raw) / "metrics.npz"
        digest = runner._write_metrics(path, day=days, metrics=metrics)
        loaded_days, loaded, loaded_digest = runner._load_metrics(path)
        assert loaded_digest == digest
        assert np.array_equal(loaded_days, days)
        assert np.isnan(loaded["late_horizon_level_CRPS"][1])
        with np.load(path, allow_pickle=False) as stored:
            assert np.array_equal(
                stored["late_horizon_valid"], np.asarray([True, False])
            )


def test_complete_case_mask_is_shared_by_every_path() -> None:
    days = np.asarray(
        ["2013-12-30", "2013-12-31", "2014-01-01"], dtype="datetime64[D]"
    )
    metrics = {
        path_id: {
            "late_horizon_level_CRPS": np.asarray([0.1, np.nan, 0.2])
        }
        for path_id in PATH_IDS
    }
    assert np.array_equal(
        runner._complete_case_mask(days, metrics),
        np.asarray([True, False, True]),
    )
    metrics["TRANSITION_WRONG"]["late_horizon_level_CRPS"][1] = 0.0
    try:
        runner._complete_case_mask(days, metrics)
    except RuntimeError as error:
        assert "valid days drifted" in str(error)
    else:
        raise AssertionError("path-specific complete-case days were accepted")
