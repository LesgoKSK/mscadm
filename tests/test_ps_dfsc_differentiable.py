from __future__ import annotations

import numpy as np
import pytest
import torch

cvxpy = pytest.importorskip("cvxpy")
cvxpylayers = pytest.importorskip("cvxpylayers")

from cvxpylayers.torch import CvxpyLayer

from ps_dfsc.differentiable_suc import (
    DifferentiableSUC,
    commitment_transitions,
)
from ps_dfsc.exact_suc import solve_two_stage_suc


def test_three_hour_qp_gradient_matches_finite_difference() -> None:
    probability = cvxpy.Parameter(2)
    schedule = cvxpy.Variable(3)
    scenario_linear_cost = np.asarray(
        [[1.0, 2.0, 3.0], [-1.0, -0.5, 0.2]]
    ) @ schedule
    problem = cvxpy.Problem(
        cvxpy.Minimize(
            probability @ scenario_linear_cost
            + 0.5 * cvxpy.sum_squares(schedule)
        ),
        [schedule >= -10.0, schedule <= 10.0],
    )
    assert problem.is_dpp()
    layer = CvxpyLayer(
        problem, parameters=[probability], variables=[schedule]
    )
    truth = torch.tensor([0.4, -0.2, 0.3], dtype=torch.double)

    def realized_loss(value: torch.Tensor) -> torch.Tensor:
        (solution,) = layer(value, solver_args={"eps": 1e-8})
        return (solution - truth).square().sum()

    weights = torch.tensor([0.4, 0.6], dtype=torch.double, requires_grad=True)
    loss = realized_loss(weights)
    loss.backward()
    direction = torch.tensor([1.0, -1.0], dtype=torch.double)
    automatic = float(weights.grad @ direction)
    epsilon = 1e-4
    plus = float(realized_loss(weights.detach() + epsilon * direction))
    minus = float(realized_loss(weights.detach() - epsilon * direction))
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isclose(automatic, finite_difference, rtol=2e-3, atol=2e-3)


def test_full_fixed_commitment_layer_has_nonzero_weight_gradient() -> None:
    wind = np.stack(
        [np.full((6, 24), 80.0), np.full((6, 24), 120.0)]
    )
    exact = solve_two_stage_suc(
        wind, np.asarray([0.5, 0.5]), mip_gap=0.02, time_limit=60
    )
    commitment = exact.first_stage.commitment
    startup, shutdown = commitment_transitions(commitment)
    layer = DifferentiableSUC(scenarios=2)
    wind_tensor = torch.tensor(wind, dtype=torch.double, requires_grad=True)
    probability = torch.tensor(
        [0.5, 0.5], dtype=torch.double, requires_grad=True
    )
    output = layer.plan(
        wind_tensor,
        probability,
        torch.tensor(commitment, dtype=torch.double),
        torch.tensor(startup, dtype=torch.double),
        torch.tensor(shutdown, dtype=torch.double),
        solver_args={"eps": 1e-4, "max_iters": 5000},
    )
    proxy = (
        output.day_ahead_dispatch.sum()
        + 0.1 * output.reserve_up.sum()
        + 0.1 * output.reserve_down.sum()
    )
    proxy.backward()
    assert torch.isfinite(probability.grad).all()
    assert torch.isfinite(wind_tensor.grad).all()
    assert float(probability.grad.abs().sum()) > 0.0
    assert float(wind_tensor.grad.abs().sum()) > 0.0
