import numpy as np

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.sampling import generate_torch_scenarios


def test_sampling_seed_is_recorded_and_repeatable_for_smoke_checkpoint() -> None:
    # The integration checkpoint is optional in clean test environments.
    checkpoint = "outputs/repro_smoke/vae/final.pt"
    from pathlib import Path

    if not Path(checkpoint).exists():
        return
    config = load_config("repro_configs/smoke.json")
    data = build_gefcom2014(config["data_dir"])
    first, first_metadata = generate_torch_scenarios(
        checkpoint, data, scenarios=2, day_batch=500, device="cpu", seed=19
    )
    second, second_metadata = generate_torch_scenarios(
        checkpoint, data, scenarios=2, day_batch=500, device="cpu", seed=19
    )
    assert np.array_equal(first, second)
    assert first_metadata["seed"] == second_metadata["seed"] == 19
