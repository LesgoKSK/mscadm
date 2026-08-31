from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from caa_rahc.hinge_tail import (
    HINGE_ANCHOR,
    _shifted_width_90,
    WidthCapInfeasibleError,
    atom_aware_logit_hinge_tail,
)
from rahc.v2_calibration import rank_audit


def _interior_baseline(
    cases: int = 2, members: int = 100, hours: int = 3
) -> np.ndarray:
    q = (np.arange(members, dtype=np.float64) + 0.5) / members
    curve = expit(-2.0 + 4.0 * q)
    values = np.broadcast_to(curve[None, :, None], (cases, members, hours)).copy()
    for case in range(cases):
        for hour in range(hours):
            values[case, :, hour] += 0.002 * case + 0.001 * hour
    return np.clip(values, 1e-5, 1.0 - 1e-5)


def test_gate_zero_and_atoms_disabled_is_exact_float64_a0() -> None:
    baseline = _interior_baseline(cases=2, members=37, hours=4).astype(np.float32)
    baseline[0, :3, 0] = 0.0
    baseline[1, -2:, 3] = 1.0
    shape = (baseline.shape[0], baseline.shape[2])

    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=np.full(shape, 0.2),
        rho1=np.full(shape, 0.1),
        lower_gate=np.zeros(shape),
        upper_gate=np.zeros(shape),
        dmax=1.0,
        width_delta_cap=0.0,
        atom_enabled=False,
    )

    expected = baseline.astype(np.float64)
    assert result.atom_only.dtype == np.float64
    assert result.calibrated.dtype == np.float64
    assert np.array_equal(result.atom_only, expected)
    assert np.array_equal(result.calibrated, expected)
    assert result.diagnostics.strict_reversals == 0
    assert result.diagnostics.central_values_changed_by_tail == 0
    assert result.diagnostics.nonfinite_input_values == 0
    assert result.diagnostics.nonfinite_intermediate_values == 0


def test_hundred_member_atoms_are_exact_and_midpoint_quantized() -> None:
    baseline = _interior_baseline(cases=2, members=100, hours=3)
    pi0 = np.array([[0.123, 0.207, 0.004], [0.055, 0.301, 0.099]])
    rho1 = np.array([[0.021, 0.008, 0.113], [0.030, 0.002, 0.070]])
    zeros = np.zeros_like(pi0)

    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0,
        rho1,
        zeros,
        zeros,
        dmax=0.8,
        width_delta_cap=1.0,
    )
    diagnostics = result.diagnostics
    expected_pi1 = (1.0 - pi0) * rho1

    assert np.array_equal(result.atom_only, result.calibrated)
    assert np.all(np.isin(result.atom_only, [0.0, 1.0]) | ((result.atom_only > 0) & (result.atom_only < 1)))
    assert np.array_equal(
        np.mean(result.atom_only == 0.0, axis=1), diagnostics.realized_pi0
    )
    assert np.array_equal(
        np.mean(result.atom_only == 1.0, axis=1), diagnostics.realized_pi1
    )
    assert np.allclose(diagnostics.requested_pi1, expected_pi1)
    assert diagnostics.is_hundred_member_ensemble
    assert diagnostics.member_probability_resolution == pytest.approx(0.01)
    assert diagnostics.midpoint_atom_quantization_bound == pytest.approx(0.005)
    assert diagnostics.zero_atom_quantization_max_abs <= 0.005 + 1e-14
    assert diagnostics.one_atom_quantization_max_abs <= 0.005 + 1e-14
    assert np.any(result.atom_only == 0.0)
    assert np.any(result.atom_only == 1.0)


def test_asymmetric_hinge_changes_only_requested_tail_and_preserves_central_curve() -> None:
    baseline = _interior_baseline(cases=1, members=100, hours=1)
    pi0 = np.array([[0.03]])
    rho1 = np.array([[0.02]])
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0,
        rho1,
        lower_gate=np.ones((1, 1)),
        upper_gate=np.zeros((1, 1)),
        dmax=0.9,
        width_delta_cap=1.0,
    )

    atom = np.sort(result.atom_only[0, :, 0])
    calibrated = np.sort(result.calibrated[0, :, 0])
    q = (np.arange(100, dtype=np.float64) + 0.5) / 100.0
    p1 = float((1.0 - pi0[0, 0]) * rho1[0, 0])
    continuous = (q > pi0[0, 0]) & (q <= 1.0 - p1)
    qc = (q - pi0[0, 0]) / (1.0 - pi0[0, 0] - p1)
    central = continuous & (qc >= HINGE_ANCHOR) & (qc <= 1.0 - HINGE_ANCHOR)
    lower = continuous & (qc < HINGE_ANCHOR)
    upper = continuous & (qc > 1.0 - HINGE_ANCHOR)

    assert np.array_equal(calibrated[central], atom[central])
    assert np.all(calibrated[lower] <= atom[lower])
    assert np.any(calibrated[lower] < atom[lower])
    assert np.array_equal(calibrated[upper], atom[upper])
    assert result.diagnostics.central_values_changed_by_tail == 0
    assert result.diagnostics.requested_max_abs_logit_shift <= 0.9 + 1e-14
    assert result.diagnostics.applied_max_abs_logit_shift <= 0.9 + 1e-14


def test_upper_hinge_is_outward_and_lower_tail_is_unchanged() -> None:
    baseline = _interior_baseline(cases=1, members=100, hours=1)
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.0,
        rho1=0.0,
        lower_gate=0.0,
        upper_gate=0.75,
        dmax=1.2,
        width_delta_cap=1.0,
    )
    atom = np.sort(result.atom_only[0, :, 0])
    calibrated = np.sort(result.calibrated[0, :, 0])
    q = (np.arange(100, dtype=np.float64) + 0.5) / 100.0

    assert np.array_equal(calibrated[q <= 0.9], atom[q <= 0.9])
    assert np.all(calibrated[q > 0.9] >= atom[q > 0.9])
    assert np.any(calibrated[q > 0.9] > atom[q > 0.9])


def test_width_cap_is_enforced_per_cell_by_scaling_only_tail_shift() -> None:
    baseline = _interior_baseline(cases=2, members=100, hours=2)
    shape = (2, 2)
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=np.zeros(shape),
        rho1=np.zeros(shape),
        lower_gate=np.ones(shape),
        upper_gate=np.ones(shape),
        dmax=4.0,
        width_delta_cap=0.01,
    )
    diagnostics = result.diagnostics

    # Regression oracle for the pre-vectorization arithmetic: solve each cell
    # independently with the same 56-step four-order-statistic bisection.
    q = (np.arange(100, dtype=np.float64) + 0.5) / 100.0
    expected_scales = np.empty(shape, dtype=np.float64)
    expected_sorted = np.empty_like(baseline)
    for case in range(shape[0]):
        for hour in range(shape[1]):
            atom = baseline[case, :, hour]
            z0 = np.log(atom) - np.log1p(-atom)
            lower_anchor = np.interp(0.1, q, z0)
            upper_anchor = np.interp(0.9, q, z0)
            shift = np.zeros_like(atom)
            lower = q < 0.1
            upper = q > 0.9
            shift[lower] = -np.maximum(lower_anchor - z0[lower], 0.0)
            shift[upper] = np.maximum(z0[upper] - upper_anchor, 0.0)
            quantiles = np.quantile(atom, (0.05, 0.95))
            limit = float(quantiles[1] - quantiles[0] + 0.01)
            low = 0.0
            high = 1.0
            for _ in range(56):
                middle = 0.5 * (low + high)
                if _shifted_width_90(atom, shift, middle) <= limit:
                    low = middle
                else:
                    high = middle
            expected_scales[case, hour] = low
            expected_sorted[case, :, hour] = expit(z0 + low * shift)

    assert np.array_equal(diagnostics.tail_scale, expected_scales)
    np.testing.assert_allclose(
        np.sort(result.calibrated, axis=1), expected_sorted, rtol=0.0, atol=np.finfo(float).eps
    )
    assert diagnostics.width_cap_activated_cells == 4
    assert np.all(diagnostics.tail_scale < 1.0)
    assert np.all(diagnostics.tail_scale > 0.0)
    assert np.all(
        diagnostics.calibrated_width_90
        <= diagnostics.baseline_width_90 + 0.01 + 5e-11
    )
    assert diagnostics.maximum_final_width_cap_excess <= 5e-11
    q = (np.arange(100, dtype=np.float64) + 0.5) / 100.0
    central = (q >= 0.1) & (q <= 0.9)
    assert np.array_equal(
        np.sort(result.calibrated, axis=1)[:, central],
        np.sort(result.atom_only, axis=1)[:, central],
    )


def test_atom_only_width_cap_infeasibility_is_never_silently_repaired() -> None:
    members = 100
    baseline = np.linspace(0.42, 0.58, members, dtype=np.float64)[None, :, None]

    with pytest.raises(WidthCapInfeasibleError, match="requested atoms violate"):
        atom_aware_logit_hinge_tail(
            baseline,
            pi0=0.20,
            rho1=0.0,
            lower_gate=0.0,
            upper_gate=0.0,
            dmax=1.0,
            width_delta_cap=0.0,
        )


def test_stable_rank_restoration_has_no_strict_reversals_with_ties_and_atoms() -> None:
    rng = np.random.default_rng(20260719)
    cases, members, hours = 3, 41, 4
    baseline = rng.uniform(0.05, 0.95, size=(cases, members, hours))
    baseline[:, 4:8, :] = baseline[:, 3:4, :]
    for case in range(cases):
        for hour in range(hours):
            baseline[case, :, hour] = baseline[case, rng.permutation(members), hour]
    shape = (cases, hours)
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=np.full(shape, 0.08),
        rho1=np.full(shape, 0.03),
        lower_gate=np.full(shape, 0.7),
        upper_gate=np.full(shape, 0.4),
        dmax=0.7,
        width_delta_cap=1.0,
    )
    audit = rank_audit(baseline, result.calibrated)

    assert audit["strict_reversals"] == 0
    assert result.diagnostics.strict_reversals == 0
    assert np.all(np.diff(np.sort(result.calibrated, axis=1), axis=1) >= 0.0)
    assert result.diagnostics.atom_preprojection_adjacent_violations == 0
    assert result.diagnostics.tail_preprojection_adjacent_violations == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("baseline", np.nan, FloatingPointError),
        ("pi0", np.nan, FloatingPointError),
        ("rho1", 1.1, ValueError),
        ("lower_gate", -0.1, ValueError),
        ("upper_gate", np.inf, FloatingPointError),
    ),
)
def test_invalid_or_nonfinite_inputs_raise_explicitly(
    field: str, value: float, error: type[Exception]
) -> None:
    baseline = _interior_baseline(cases=1, members=20, hours=1)
    kwargs: dict[str, object] = {
        "baseline_scenarios": baseline,
        "pi0": 0.05,
        "rho1": 0.01,
        "lower_gate": 0.5,
        "upper_gate": 0.5,
        "dmax": 0.8,
        "width_delta_cap": 1.0,
    }
    if field == "baseline":
        changed = baseline.copy()
        changed[0, 0, 0] = value
        kwargs["baseline_scenarios"] = changed
    else:
        kwargs[field] = value

    with pytest.raises(error):
        atom_aware_logit_hinge_tail(**kwargs)


def test_logit_hinge_shift_matches_requested_cap_in_logit_space() -> None:
    baseline = _interior_baseline(cases=1, members=100, hours=1)
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.0,
        rho1=0.0,
        lower_gate=1.0,
        upper_gate=1.0,
        dmax=0.6,
        width_delta_cap=1.0,
    )
    atom = np.sort(result.atom_only[0, :, 0])
    calibrated = np.sort(result.calibrated[0, :, 0])
    applied = logit(calibrated) - logit(atom)

    assert np.max(np.abs(applied)) <= 0.6 + 1e-12
    assert np.all(applied[:10] <= 0.0)
    assert np.all(applied[10:90] == 0.0)
    assert np.all(applied[90:] >= 0.0)



def test_frozen_hinge_uses_logit_distance_from_point_one_anchors() -> None:
    baseline = _interior_baseline(cases=1, members=100, hours=1)
    lower_gate = 0.4
    upper_gate = 0.7
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.0,
        rho1=0.0,
        lower_gate=lower_gate,
        upper_gate=upper_gate,
        dmax=10.0,
        width_delta_cap=1.0,
    )
    atom = np.sort(result.atom_only[0, :, 0])
    calibrated = np.sort(result.calibrated[0, :, 0])
    q = (np.arange(100, dtype=np.float64) + 0.5) / 100.0
    z0 = logit(atom)
    lower_anchor = np.interp(0.1, q, z0)
    upper_anchor = np.interp(0.9, q, z0)
    expected = np.zeros_like(q)
    lower = q < 0.1
    upper = q > 0.9
    expected[lower] = -lower_gate * np.maximum(lower_anchor - z0[lower], 0.0)
    expected[upper] = upper_gate * np.maximum(z0[upper] - upper_anchor, 0.0)
    actual = logit(calibrated) - z0

    assert np.allclose(actual, expected, atol=2e-15, rtol=2e-15)
    assert np.array_equal(calibrated[~(lower | upper)], atom[~(lower | upper)])
    assert result.diagnostics.requested_max_abs_logit_shift == pytest.approx(
        np.max(np.abs(expected))
    )
    assert result.diagnostics.logit_cap_hit_values == 0


def test_frozen_hinge_reports_pre_cap_shift_and_final_dmax() -> None:
    baseline = _interior_baseline(cases=1, members=100, hours=1)
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.0,
        rho1=0.0,
        lower_gate=1.0,
        upper_gate=1.0,
        dmax=0.05,
        width_delta_cap=1.0,
    )

    assert result.diagnostics.requested_max_abs_logit_shift > 0.05
    assert result.diagnostics.applied_max_abs_logit_shift <= 0.05 + 1e-15
    assert result.diagnostics.logit_cap_hit_values > 0

def test_atom_only_width_reference_isolates_structural_atom_jump() -> None:
    baseline = np.linspace(0.42, 0.58, 100, dtype=np.float64)[None, :, None]
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.20,
        rho1=0.0,
        lower_gate=0.0,
        upper_gate=0.0,
        dmax=1.0,
        width_delta_cap=0.0,
        width_cap_reference="atom_only",
    )
    diagnostics = result.diagnostics
    summary = diagnostics.summary()

    assert np.array_equal(result.calibrated, result.atom_only)
    assert diagnostics.width_cap_reference == "atom_only"
    assert np.array_equal(diagnostics.width_cap_reference_90, diagnostics.atom_only_width_90)
    assert diagnostics.maximum_final_width_cap_excess == pytest.approx(0.0, abs=1e-14)
    assert summary["maximum_final_width_change_from_atom_only"] == pytest.approx(0.0)
    assert summary["maximum_final_width_change_from_baseline"] > 0.0


def test_atom_only_reference_caps_only_hinge_widening_per_cell() -> None:
    baseline = np.linspace(0.42, 0.58, 100, dtype=np.float64)[None, :, None]
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.20,
        rho1=0.0,
        lower_gate=1.0,
        upper_gate=1.0,
        dmax=3.0,
        width_delta_cap=0.001,
        width_cap_reference="atom_only",
    )
    diagnostics = result.diagnostics
    summary = diagnostics.summary()

    assert diagnostics.width_cap_activated_cells == 1
    assert diagnostics.tail_scale[0, 0] < 1.0
    assert diagnostics.calibrated_width_90[0, 0] <= (
        diagnostics.atom_only_width_90[0, 0] + 0.001 + 5e-11
    )
    assert summary["maximum_final_width_change_from_atom_only"] <= 0.001 + 5e-11
    assert summary["maximum_final_width_change_from_baseline"] > 0.001


def test_width_cap_reference_rejects_unknown_protocol() -> None:
    baseline = _interior_baseline(cases=1, members=20, hours=1)
    with pytest.raises(ValueError, match="width_cap_reference"):
        atom_aware_logit_hinge_tail(
            baseline,
            pi0=0.0,
            rho1=0.0,
            lower_gate=0.0,
            upper_gate=0.0,
            dmax=1.0,
            width_delta_cap=0.0,
            width_cap_reference="unknown",  # type: ignore[arg-type]
        )

@pytest.mark.parametrize(
    "baseline",
    (
        np.r_[np.zeros(70), np.ones(30)][None, :, None],
        np.zeros((1, 100, 1), dtype=np.float64),
        np.ones((1, 100, 1), dtype=np.float64),
    ),
)
def test_empty_empirical_continuous_support_uses_exact_local_a0_fallback(
    baseline: np.ndarray,
) -> None:
    result = atom_aware_logit_hinge_tail(
        baseline,
        pi0=0.20,
        rho1=0.10,
        lower_gate=1.0,
        upper_gate=1.0,
        dmax=0.8,
        width_delta_cap=0.05,
        width_cap_reference="atom_only",
    )

    assert np.array_equal(result.atom_only, baseline)
    assert np.array_equal(result.calibrated, baseline)
    assert result.diagnostics.central_values_changed_by_tail == 0
    assert result.diagnostics.strict_reversals == 0
    assert result.diagnostics.nonfinite_intermediate_values == 0
