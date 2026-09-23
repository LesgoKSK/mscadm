from __future__ import annotations

import numpy as np
import torch

from architecture_v1.transition_object import (
    MaskConditionedOperator,
    ddim_direct_x0_step,
    fit_level_rms,
    fit_operator_scale,
    matched_noise_level_forward,
    transition_native_forward,
    wrong_adjacency_audit,
    wrong_adjacency_order,
)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        float(torch.max(torch.abs(actual - expected))),
        actual,
        expected,
    )


def test_true_transition_uses_contiguous_active_runs_only() -> None:
    value = torch.tensor(
        [[[10.0, 11.0, 99.0, 20.0, 23.0, 50.0, 7.0, 9.0]]],
        dtype=torch.float64,
    )
    active = torch.tensor(
        [[[True, True, False, True, True, False, True, True]]]
    )
    operator = MaskConditionedOperator("transition_true", scale=2.0)
    coefficient = operator.transform(value, active)
    expected = torch.tensor(
        [[[20.0, 2.0, 0.0, 40.0, 6.0, 0.0, 14.0, 4.0]]],
        dtype=torch.float64,
    )
    _assert_close(coefficient, expected)
    reconstructed = operator.inverse(coefficient, active)
    _assert_close(reconstructed, torch.where(active, value, torch.zeros_like(value)))


def test_wrong_transition_has_frozen_outside_in_order_and_exact_inverse() -> None:
    assert wrong_adjacency_order(1) == (0,)
    assert wrong_adjacency_order(5) == (0, 4, 1, 3, 2)
    value = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]]], dtype=torch.float64
    )
    active = torch.ones_like(value, dtype=torch.bool)
    operator = MaskConditionedOperator("transition_wrong")
    coefficient = operator.transform(value, active)
    expected = torch.tensor(
        [[[1.0, -30.0, -12.0, 4.0, 14.0, 31.0]]],
        dtype=torch.float64,
    )
    _assert_close(coefficient, expected)
    _assert_close(operator.inverse(coefficient, active), value)


def test_wrong_adjacency_audit_reports_destroyed_edges_without_crossing_mask() -> None:
    active = torch.zeros((2, 3, 24), dtype=torch.bool)
    active[0, 0] = True
    active[0, 1, 2:11] = True
    active[0, 1, 14:24] = True
    active[0, 2, 4:6] = True  # Too short to enter the mechanism audit.
    active[1, 0, 1:8] = True
    active[1, 2, 3:20] = True
    audit = wrong_adjacency_audit(active)
    assert audit.eligible_segments == 5
    assert audit.eligible_edges == (23 + 8 + 9 + 6 + 16)
    assert 0 <= audit.retained_true_edges <= audit.eligible_edges
    assert audit.retained_true_edge_fraction <= 0.30


def test_segmentwise_dct_is_orthonormal_and_mask_reversible() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(91)
    value = torch.randn((3, 2, 9), generator=generator, dtype=torch.float64)
    active = torch.tensor(
        [
            [[True, True, True, False, True, True, False, False, True]] * 2,
            [[True, True, True, True, True, True, True, True, True]] * 2,
            [[False, True, True, True, False, True, True, True, True]] * 2,
        ],
        dtype=torch.bool,
    )
    operator = MaskConditionedOperator("orthogonal_dct")
    coefficient = operator.transform(value, active)
    reconstructed = operator.inverse(coefficient, active)
    masked = torch.where(active, value, torch.zeros_like(value))
    _assert_close(reconstructed, masked, atol=2e-12, rtol=2e-12)
    _assert_close(
        coefficient.square().sum(),
        masked.square().sum(),
        atol=2e-12,
        rtol=2e-12,
    )


def test_train_only_scale_matches_active_rms() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(92)
    value = torch.randn((17, 3, 12), generator=generator, dtype=torch.float64)
    active = torch.rand((17, 3, 12), generator=generator) > 0.15
    scale = fit_operator_scale(value, active, kind="transition_true")
    operator = MaskConditionedOperator("transition_true", scale=scale)
    coefficient = operator.transform(value, active)
    assert abs(fit_level_rms(value, active) - fit_level_rms(coefficient, active)) < 1e-12


def test_transform_supports_member_axis_and_preserves_autograd_mask() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(93)
    value = torch.randn(
        (2, 4, 3, 10), generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    active = torch.rand((2, 4, 3, 10), generator=generator) > 0.2
    operator = MaskConditionedOperator("transition_true", scale=0.73)
    coefficient = operator.transform(value, active)
    assert coefficient.shape == value.shape
    assert torch.equal(coefficient[~active], torch.zeros_like(coefficient[~active]))
    coefficient.square().sum().backward()
    assert value.grad is not None
    assert bool(torch.isfinite(value.grad).all())
    assert torch.equal(value.grad[~active], torch.zeros_like(value.grad[~active]))


def test_matched_noise_pullback_is_exact() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(94)
    clean = torch.randn((3, 2, 8), generator=generator, dtype=torch.float64)
    noise = torch.randn((3, 2, 8), generator=generator, dtype=torch.float64)
    active = torch.tensor(
        [
            [[True, True, False, True, True, True, False, True]] * 2,
            [[True, True, True, True, True, True, True, True]] * 2,
            [[False, True, True, False, True, True, True, True]] * 2,
        ],
        dtype=torch.bool,
    )
    alpha = torch.tensor([0.18, 0.51, 0.87], dtype=torch.float64)
    operator = MaskConditionedOperator("transition_true", scale=0.81)
    noisy_transition = transition_native_forward(
        clean, active, alpha, noise, operator
    )
    pulled_back = operator.inverse(noisy_transition, active)
    matched_level = matched_noise_level_forward(
        clean, active, alpha, noise, operator
    )
    _assert_close(pulled_back, matched_level, atol=2e-12, rtol=2e-12)


def test_induced_metric_matches_native_transition_error() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(95)
    prediction = torch.randn((5, 2, 7), generator=generator, dtype=torch.float64)
    target = torch.randn((5, 2, 7), generator=generator, dtype=torch.float64)
    active = torch.rand((5, 2, 7), generator=generator) > 0.25
    operator = MaskConditionedOperator("transition_true", scale=1.13)
    native_prediction = operator.transform(prediction, active)
    native_target = operator.transform(target, active)
    expected = (native_prediction - native_target).square().sum() / active.sum()
    _assert_close(operator.induced_mse(prediction, target, active), expected)


def test_oracle_ddim_step_commutes_with_transition_pullback() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(96)
    clean = torch.randn((4, 2, 9), generator=generator, dtype=torch.float64)
    noise = torch.randn((4, 2, 9), generator=generator, dtype=torch.float64)
    active = torch.rand((4, 2, 9), generator=generator) > 0.2
    current = torch.tensor([0.08, 0.22, 0.47, 0.71], dtype=torch.float64)
    previous = torch.tensor([0.19, 0.38, 0.66, 0.89], dtype=torch.float64)
    operator = MaskConditionedOperator("transition_true", scale=0.67)
    noisy_transition = transition_native_forward(
        clean, active, current, noise, operator
    )
    clean_transition = operator.transform(clean, active)
    previous_transition = ddim_direct_x0_step(
        noisy_transition,
        clean_transition,
        active,
        current,
        previous,
    )
    pulled_previous = operator.inverse(previous_transition, active)
    noisy_level = operator.inverse(noisy_transition, active)
    direct_previous = ddim_direct_x0_step(
        noisy_level,
        torch.where(active, clean, torch.zeros_like(clean)),
        active,
        current,
        previous,
    )
    _assert_close(pulled_previous, direct_previous, atol=3e-12, rtol=3e-12)


def test_numpy_and_torch_layout_contract_is_stable() -> None:
    active = torch.tensor(
        [[[True, False, True, True], [True, True, True, False]]],
        dtype=torch.bool,
    )
    operator = MaskConditionedOperator("transition_true")
    matrix = operator.matrix(active, dtype=torch.float64)
    assert matrix.shape == (1, 2, 4, 4)
    assert np.array_equal(
        matrix[0, 0].numpy(),
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        ),
    )
