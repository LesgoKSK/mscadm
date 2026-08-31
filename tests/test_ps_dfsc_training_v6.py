import numpy as np
import pytest
import torch

from ps_dfsc.training_v6 import _validate_state


def _state():
    return {
        "model_state": {"weight": torch.ones(2)},
        "history_epochs": [{"loss": 1.0}],
        "augmented_dual": {"CRPS": 0.0},
        "cached_commitment": np.ones((1, 12, 24)),
    }


def test_finite_checkpoint_state_is_accepted():
    _validate_state(_state())


@pytest.mark.parametrize(
    "field",
    ["model_state", "history_epochs", "augmented_dual", "cached_commitment"],
)
def test_nonfinite_checkpoint_state_fails_closed(field):
    state = _state()
    if field == "model_state":
        state[field]["weight"][0] = torch.nan
    elif field == "history_epochs":
        state[field][0]["loss"] = float("nan")
    elif field == "augmented_dual":
        state[field]["CRPS"] = float("inf")
    else:
        state[field][0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        _validate_state(state)
