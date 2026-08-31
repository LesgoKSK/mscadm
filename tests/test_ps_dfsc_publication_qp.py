from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from ps_dfsc.differentiable_suc import commitment_transitions
from ps_dfsc.fast_differentiable_suc_publication import (
    PublicationDifferentiableSUC,
)


def test_publication_qp_has_nonzero_probability_gradient():
    commitment = np.ones((12, 24))
    startup, shutdown = commitment_transitions(commitment)
    layer = PublicationDifferentiableSUC(scenarios=2, strong_convexity=1e-4)
    wind = torch.tensor(
        np.stack(
            [np.full((6, 24), 80.0), np.full((6, 24), 120.0)]
        ),
        dtype=torch.double,
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
        solver_args={"eps": 1e-4, "max_iters": 5000},
    )
    proxy = (
        output.day_ahead_dispatch.sum()
        + 0.1 * output.reserve_up.sum()
        + 0.1 * output.reserve_down.sum()
    )
    proxy.backward()
    assert torch.isfinite(probability.grad).all()
    assert float(probability.grad.abs().sum()) > 0.0
