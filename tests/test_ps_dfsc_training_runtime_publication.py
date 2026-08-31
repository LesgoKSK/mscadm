from __future__ import annotations

import numpy as np
import torch

from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.pipeline_runtime_publication import (
    _day_sha256,
    _model_sha256,
)


def test_publication_training_hashes_model_and_refresh_day():
    torch.manual_seed(3)
    model = PSDFSCNetwork()
    first = _model_sha256(model)
    assert first == _model_sha256(model)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] += 1.0
    assert first != _model_sha256(model)

    scenarios = np.zeros((100, 10, 24), dtype=np.float32)
    assignments = np.arange(100, dtype=np.int64) % 20
    day_hash = _day_sha256(scenarios, assignments)
    scenarios[0, 0, 0] = 1.0
    assert day_hash != _day_sha256(scenarios, assignments)
