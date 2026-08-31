import numpy as np
import pytest
import torch

from ps_dfsc.metrics import weighted_per_date_metrics
from ps_dfsc.stable_training_metrics import (
    smooth_energy_score,
    smooth_variogram_score,
)


def test_smoothed_scores_match_exact_values_and_have_finite_zero_atom_gradients():
    rng = np.random.default_rng(41)
    values = rng.uniform(size=(2, 8, 10, 24))
    values[:, 1] = values[:, 0]
    values[:, :, :, :3] = 0.0
    truth = rng.uniform(size=(2, 10, 24))
    truth[:, :, :3] = 0.0
    probability = np.full((2, 8), 1.0 / 8.0)

    scenarios = torch.tensor(
        values, dtype=torch.float64, requires_grad=True
    )
    weights = torch.tensor(
        probability, dtype=torch.float64, requires_grad=True
    )
    observations = torch.tensor(truth, dtype=torch.float64)
    energy = smooth_energy_score(scenarios, weights, observations)
    variogram = smooth_variogram_score(
        scenarios, weights, observations
    )
    (energy + variogram).backward()

    assert torch.isfinite(scenarios.grad).all()
    assert torch.isfinite(weights.grad).all()
    exact = weighted_per_date_metrics(values, probability, truth)
    assert float(energy) == pytest.approx(
        float(np.mean(exact["ES"])), abs=2.0e-5
    )
    assert float(variogram) == pytest.approx(
        float(np.mean(exact["VS"])), abs=2.0e-5
    )
