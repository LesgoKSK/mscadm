from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("clarabel")
pytest.importorskip("cvxpylayers")

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.fast_differentiable_suc_publication_v3 import (
    ClarabelPublicationDifferentiableSUC,
)


def test_clarabel_publication_qp_forward_backward_and_realized():
    commitment = np.ones((12, 24))
    startup, shutdown = commitment_transitions(commitment)
    layer = ClarabelPublicationDifferentiableSUC(
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
    loss = (
        realized.dispatch.sum()
        + 0.1 * planned.reserve_up.sum()
        + 0.1 * planned.reserve_down.sum()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(probability.grad).all()
    assert torch.isfinite(wind.grad).all()
    assert float(probability.grad.abs().sum()) > 0.0
    assert float(wind.grad.abs().sum()) > 0.0
