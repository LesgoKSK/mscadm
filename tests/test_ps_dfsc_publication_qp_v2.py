from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.fast_differentiable_suc_publication_v2 import (
    ScaledPublicationDifferentiableSUC,
)


def test_scaled_publication_qp_is_finite_and_has_weight_gradient():
    commitment = np.ones((12, 24))
    startup, shutdown = commitment_transitions(commitment)
    layer = ScaledPublicationDifferentiableSUC(
        scenarios=2, strong_convexity=1e-4
    )
    wind = torch.tensor(
        np.stack(
            [np.full((6, 24), 80.0), np.full((6, 24), 120.0)]
        ),
        dtype=torch.double,
        requires_grad=True,
    )
    probability = torch.tensor(
        [0.5, 0.5], dtype=torch.double, requires_grad=True
    )
    output = layer.plan(
        wind,
        probability,
        torch.tensor(commitment, dtype=torch.double),
        torch.tensor(startup, dtype=torch.double),
        torch.tensor(shutdown, dtype=torch.double),
    )
    assert torch.isfinite(output.day_ahead_dispatch).all()
    assert float(output.day_ahead_dispatch.max()) > 1.0
    proxy = (
        output.day_ahead_dispatch.sum()
        + 0.1 * output.reserve_up.sum()
        + 0.1 * output.reserve_down.sum()
    )
    proxy.backward()
    assert torch.isfinite(probability.grad).all()
    assert torch.isfinite(wind.grad).all()
    assert float(probability.grad.abs().sum()) > 0.0
    assert float(wind.grad.abs().sum()) > 0.0


def test_scaled_publication_realized_layer_returns_mw():
    commitment = np.ones((12, 24))
    startup, shutdown = commitment_transitions(commitment)
    layer = ScaledPublicationDifferentiableSUC(
        scenarios=2, strong_convexity=1e-4
    )
    wind = torch.tensor(
        np.stack(
            [np.full((6, 24), 80.0), np.full((6, 24), 120.0)]
        ),
        dtype=torch.double,
    )
    probability = torch.tensor([0.5, 0.5], dtype=torch.double)
    planned = layer.plan(
        wind,
        probability,
        torch.tensor(commitment, dtype=torch.double),
        torch.tensor(startup, dtype=torch.double),
        torch.tensor(shutdown, dtype=torch.double),
    )
    realized = layer.realize(
        wind[0],
        torch.tensor(commitment, dtype=torch.double),
        torch.tensor(startup, dtype=torch.double),
        torch.tensor(shutdown, dtype=torch.double),
        planned.day_ahead_dispatch,
        planned.reserve_up,
        planned.reserve_down,
    )
    assert torch.isfinite(realized.dispatch).all()
    assert float(realized.dispatch.max()) > 1.0
    balance = (
        realized.dispatch.sum(dim=1)[0]
        + realized.used_wind.sum(dim=1)[0]
        + realized.load_shedding[0]
    )
    expected = torch.tensor(layer.system.load, dtype=torch.double)
    assert torch.allclose(balance, expected, rtol=2e-3, atol=2.0)

