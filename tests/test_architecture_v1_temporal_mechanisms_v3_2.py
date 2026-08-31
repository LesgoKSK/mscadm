"""Fail-closed contracts for the frozen v3.2 execution matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from architecture_v1.model import (
    T0StableSourceRectifiedFlow,
    T1FeatureStableSourceRectifiedFlow,
    T1SourceStableRectifiedFlow,
    T1StableShuffleRectifiedFlow,
)
from architecture_v1.training import tensor_state_sha256
from repro_scripts.run_architecture_v1_temporal_mechanisms_v3_2 import (
    DEFAULT_CONFIG,
    _load_config,
    dry_run,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kwargs() -> dict[str, object]:
    return {
        "condition_dim": 4, "zones": 2, "hours": 24,
        "encoder_dim": 8, "encoder_depth": 1,
        "flow_dim": 8, "flow_depth": 1,
        "heads": 2, "ff_multiplier": 2, "dropout": 0.0,
        "atom_hidden_dim": 8, "stable_source_rho_max": 0.95,
    }


def test_config_sidecar_and_lineage_are_exact() -> None:
    config, _ = _load_config(DEFAULT_CONFIG)
    words = DEFAULT_CONFIG.with_name(DEFAULT_CONFIG.name + ".sha256").read_text().split()
    assert words == [_digest(DEFAULT_CONFIG), DEFAULT_CONFIG.name]
    assert config["role_access"]["selection_state"] == "sealed"
    assert config["role_access"]["calibration_state"] == "sealed"
    assert config["role_access"]["selection_access_authorized"] is False


def test_remaining_matrix_is_exactly_twelve_unique_runs() -> None:
    config, _ = _load_config(DEFAULT_CONFIG)
    matrix = [(str(name), int(seed)) for name, seed in config["execution_order_remaining_12"]]
    expected = {
        (name, seed)
        for seed in (0, 1, 2)
        for name in config["candidates"]
        if not (seed == 0 and name in config["seed0_imports"])
    }
    assert len(matrix) == len(set(matrix)) == 12
    assert set(matrix) == expected


def test_all_five_candidates_share_topology_and_initial_tensors() -> None:
    config, _ = _load_config(DEFAULT_CONFIG)
    order = config["shuffle_control"]["hour_order_zero_based"]
    torch.manual_seed(32001)
    reference = T0StableSourceRectifiedFlow(**_kwargs())
    state = reference.state_dict()
    models = [
        reference,
        T1FeatureStableSourceRectifiedFlow(**_kwargs()),
        T1SourceStableRectifiedFlow(**_kwargs()),
        T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="feature", **_kwargs()
        ),
        T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="source", **_kwargs()
        ),
    ]
    for model in models:
        model.load_state_dict(state, strict=True)
    assert len({tuple(model.state_dict()) for model in models}) == 1
    assert len({tensor_state_sha256(model.state_dict()) for model in models}) == 1
    assert len({model.parameter_count() for model in models}) == 1


def test_dry_run_is_nonmutating_and_sealed() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text())
    output = ROOT / config["output_root"]
    before = output.exists()
    result = dry_run(DEFAULT_CONFIG)
    assert output.exists() is before
    assert result["mode"] == "predictor_only_no_targets_loaded_no_files_created"
    assert result["new_training_count"] == 12
    assert result["selection_state"] == "sealed"
    assert result["calibration_state"] == "sealed"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
