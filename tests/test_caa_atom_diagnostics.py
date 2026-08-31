import numpy as np

from caa_rahc.atom_diagnostics import (
    ensemble_atom_quantization,
    pooled_atom_diagnostics,
    reliability_bins,
)


def test_atom_reliability_and_quantization():
    p0 = np.linspace(0.01, 0.3, 24)[None, :]
    truth = np.zeros((1, 24))
    truth[:, 12:] = 0.2
    scenarios = np.broadcast_to(np.linspace(0, 1, 101)[None, :, None], (1, 101, 24)).copy()
    bins = reliability_bins(p0, truth == 0, bins=4)
    assert sum(item["count"] for item in bins) == 24
    quant = ensemble_atom_quantization(scenarios, p0, 0.001)
    assert quant["members"] == 101
    pooled = pooled_atom_diagnostics([p0], [0.001], [truth], [scenarios])
    assert "analytic_scores" in pooled and "finite_ensemble" in pooled
