"""Array-level and frozen-input tests for family-v1.2 formal evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from architecture_v1.family_v1_2_evaluation import (
    adjudicate,
    block_by_nwp_interaction_omnibus,
    daily_contribution,
    daily_harm,
    max_t_simultaneous_bands,
    nwp_regime_homogeneity_omnibus,
    stage_homogeneity_omnibus,
)
from repro_scripts.run_architecture_v1_family_v1_2_evaluation import (
    AMENDMENT,
    _dry_run,
    _load_amendment,
    _shuffle_bank_indices,
    _variant_orders,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amendment() -> dict:
    return json.loads(AMENDMENT.read_text(encoding="utf-8"))


def test_evaluation_amendment_and_all_frozen_inputs_verify() -> None:
    amendment = _load_amendment()
    assert AMENDMENT.with_name(AMENDMENT.name + ".sha256").read_text().split() == [
        _sha(AMENDMENT),
        AMENDMENT.name,
    ]
    assert amendment["freeze_boundary"][
        "family_v1_2_validation_scenarios_generated_before_this_amendment"
    ] is False
    dry = _dry_run()
    assert dry["verified_completions"] == 48
    assert dry["verified_reused_D0_v_archives"] == 9
    assert dry["balanced_shuffle_bank_counts"] == {
        str(index): 25 for index in range(6)
    }
    assert dry["selection_state"] == dry["calibration_state"] == "sealed"


def test_shuffle_inference_rule_is_balanced_target_free_and_shared() -> None:
    amendment = _amendment()
    combined = []
    for sampling_seed in (21000, 21001, 21002):
        values = _shuffle_bank_indices(
            sampling_seed=sampling_seed, days=50, amendment=amendment
        )
        assert values.shape == (50,)
        assert set(values.tolist()).issubset(set(range(6)))
        combined.extend(values.tolist())
    assert [combined.count(index) for index in range(6)] == [25] * 6
    assert np.array_equal(
        _variant_orders(
            "chronological_heldout6",
            sampling_seed=21000,
            days=50,
            amendment=amendment,
        ),
        np.full(50, 6),
    )


def test_daily_absolute_relative_benefit_and_harm_directions() -> None:
    reference = np.asarray([2.0, 4.0, 6.0])
    candidate = np.asarray([1.0, 3.0, 5.0])
    absolute = daily_contribution(reference, candidate, relative=False)
    relative = daily_contribution(reference, candidate, relative=True)
    harm = daily_harm(candidate, reference, relative=True)
    assert np.array_equal(absolute, np.ones(3))
    assert np.isclose(relative.mean(), 0.25)
    assert np.isclose(harm.mean(), -0.25)


def test_joint_max_t_bands_keep_endpoint_shape_and_pairing() -> None:
    rng = np.random.default_rng(71)
    values = rng.normal(0.02, 0.005, size=(50, 2, 8))
    result = max_t_simultaneous_bands(
        values, repetitions=500, seed=291001, confidence=0.95
    )
    assert result["endpoint_shape"] == [2, 8]
    assert result["endpoint_count"] == 16
    assert result["estimate"].shape == (2, 8)
    assert np.all(result["simultaneous_low"] <= result["estimate"])
    assert np.all(result["simultaneous_high"] >= result["estimate"])
    assert result["critical_value"] > 1.0


def test_stage_and_nwp_omnibus_detect_clear_registered_structure() -> None:
    rng = np.random.default_rng(72)
    labels = np.asarray(["stable"] * 17 + ["moderate"] * 16 + ["dynamic"] * 17)
    values = rng.normal(0.0, 0.01, size=(50, 2, 8))
    values += np.linspace(-0.04, 0.04, 8)[None, None, :]
    values += np.asarray([0.0, 0.04, 0.08])[
        np.asarray([{"stable": 0, "moderate": 1, "dynamic": 2}[x] for x in labels])
    ][:, None, None]
    stage = stage_homogeneity_omnibus(values, repetitions=500, seed=291004)
    nwp = nwp_regime_homogeneity_omnibus(
        values, labels, repetitions=500, seed=291005
    )
    assert stage["p_value"] < 0.05
    assert nwp["p_value"] < 0.05


def test_interaction_omnibus_detects_nonadditive_dynamic_final_block() -> None:
    rng = np.random.default_rng(73)
    labels = np.asarray(["stable"] * 17 + ["moderate"] * 16 + ["dynamic"] * 17)
    values = rng.normal(0.0, 0.008, size=(50, 2, 8))
    dynamic = labels == "dynamic"
    values[dynamic, :, 6:] += 0.12
    result = block_by_nwp_interaction_omnibus(
        values, labels, repetitions=500, seed=291006
    )
    assert result["p_value"] < 0.05
    assert result["cell_interaction"].shape == (3, 2, 8)


def _bands(rows: int, blocks: int, *, estimate: float, low: float, high: float) -> dict:
    return {
        "estimate": np.full((rows, blocks), estimate),
        "simultaneous_low": np.full((rows, blocks), low),
        "simultaneous_high": np.full((rows, blocks), high),
    }


def test_decision_tree_requires_adjacent_primary_and_same_checkpoint_support() -> None:
    primary = _bands(2, 8, estimate=0.06, low=0.01, high=0.08)
    primary["estimate"][0] = 0.002
    same = _bands(2, 8, estimate=0.01, low=0.001, high=0.02)
    safety = _bands(4, 8, estimate=0.0, low=-0.001, high=0.001)
    margins = {
        "ramp_CRPS_improvement_min": 0.001,
        "lagged_variogram_relative_improvement_min": 0.05,
        "level_CRPS_noninferiority_margin": 0.0015,
        "normalized_joint_ES_relative_noninferiority_margin": 0.02,
        "coverage90_absolute_difference_max": 0.02,
        "width90_relative_increase_max": 0.10,
    }
    decision = adjudicate(
        primary_bands=primary,
        same_checkpoint_bands=same,
        safety_bands=safety,
        stage_omnibus={"p_value": 0.01},
        nwp_omnibus={"p_value": 0.20},
        interaction_omnibus={"p_value": 0.20},
        margins=margins,
    )
    assert decision["status"] == "FAMILY_V1_3_SNR_STAGE_GATE"
    assert len(decision["full_adjacent_pairs"]) == 7

    same["simultaneous_low"][:] = -0.001
    no_go = adjudicate(
        primary_bands=primary,
        same_checkpoint_bands=same,
        safety_bands=safety,
        stage_omnibus={"p_value": 0.01},
        nwp_omnibus={"p_value": 0.01},
        interaction_omnibus={"p_value": 0.01},
        margins=margins,
    )
    assert no_go["status"] == "FAMILY_V1_2_ORDERED_RECURRENCE_NO_GO"
