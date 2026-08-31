from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from torch import nn

from architecture_v1.atom import INTERIOR_STATE, ONE_STATE, ZERO_STATE
from architecture_v1.evaluation import (
    evaluate_archive,
    load_scenario_archive,
    masked_per_day_metrics,
    validate_archive_metadata,
    write_scenario_archive,
)
from architecture_v1.model import R0JointRectifiedFlow, T0MemorylessRectifiedFlow
from architecture_v1.training import (
    ArchitectureBatch,
    ArchitectureTrainer,
    CHECKPOINT_SCHEMA,
    FAILURE_SCHEMA,
    load_checkpoint,
    parameter_manifest,
    shared_ea_state_sha256,
    validate_stage,
)


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert re.search(pattern, str(error)), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def _tiny_r0() -> R0JointRectifiedFlow:
    torch.manual_seed(17)
    return R0JointRectifiedFlow(
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
    )


def _batch(*, missing_fill: float = 0.37, batch_size: int = 1) -> ArchitectureBatch:
    condition = torch.linspace(
        -1.0,
        1.0,
        batch_size * 10 * 24 * 20,
        dtype=torch.float32,
    ).reshape(batch_size, 10, 24, 20)
    target = torch.full((batch_size, 10, 24), 0.4, dtype=torch.float32)
    target[:, 0, 0] = 0.0
    target[:, 0, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, 0, 5] = False
    target[:, 0, 5] = float(missing_fill)
    missing = ~observed
    state = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    state[observed & (target == 0.0)] = ZERO_STATE
    state[observed & (target == 1.0)] = ONE_STATE
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=missing,
        day_index=torch.arange(batch_size, dtype=torch.long) + 20_000,
    )


def _archive_arrays() -> dict[str, np.ndarray]:
    observations = np.full((2, 10, 24), 0.4, dtype=np.float32)
    observations[:, 0, 0] = 0.0
    observations[:, 0, 1] = 1.0
    observed = np.ones_like(observations, dtype=bool)
    observed[:, 0, 5] = False
    missing = ~observed
    scenarios = np.repeat(observations[:, None], 4, axis=1)
    states = np.full(scenarios.shape, INTERIOR_STATE, dtype=np.int64)
    states[scenarios == 0.0] = ZERO_STATE
    states[scenarios == 1.0] = ONE_STATE
    zero_probability = (observations == 0.0).astype(np.float32)
    one_probability = (observations == 1.0).astype(np.float32)
    return {
        "scenarios": scenarios,
        "observations": observations,
        "observed_mask": observed,
        "raw_missing_mask": missing,
        "states": states,
        "zero_probability": zero_probability,
        "one_probability": one_probability,
        "day": np.asarray(["2023-01-01", "2023-01-02"], dtype="datetime64[D]"),
        "zones": np.arange(1, 11, dtype=np.int64),
    }


def _metadata(*, split_role: str = "selection") -> dict[str, object]:
    return {
        "candidate_id": "R0",
        "run_id": "r0_seed17",
        "training_seed": 17,
        "sampling_seed": 117,
        "split_role": split_role,
        "config_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "data_bundle_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "shared_EA_state_sha256": "e" * 64,
        "integrator": "heun",
        "integration_steps": 2,
        "flow_nfe": 3,
        "batched_flow_forward_calls": 6,
        "member_chunk": 2,
        "allocation_semantics": "balanced_shared_priority_control",
        "estimand": "finite_dependent_scenario_set",
        "score_semantics": "empirical_v_stat",
        "common_random_numbers_group": "r0_t0_seed117",
        "plan_id": "architecture_v1_smoke_v1",
        "scientific_status": "smoke_only_non_scientific",
        "sampling_wall_seconds": 0.25,
        "peak_memory_allocated_bytes": 0,
    }


def test_architecture_batch_enforces_raw_mask_and_neutral_missing_state() -> None:
    batch = _batch()
    assert not bool(batch.ramp_observed_mask[0, 0, 4])
    assert not bool(batch.ramp_observed_mask[0, 0, 5])

    with _raises(ValueError, "complement"):
        ArchitectureBatch(
            condition=batch.condition,
            target=batch.target,
            state=batch.state,
            observed_mask=batch.observed_mask,
            raw_missing_mask=batch.raw_missing_mask.clone().fill_(False),
            day_index=batch.day_index,
        )
    invalid_state = batch.state.clone()
    invalid_state[batch.raw_missing_mask] = ZERO_STATE
    with _raises(ValueError, "neutral INTERIOR"):
        ArchitectureBatch(
            condition=batch.condition,
            target=batch.target,
            state=invalid_state,
            observed_mask=batch.observed_mask,
            raw_missing_mask=batch.raw_missing_mask,
            day_index=batch.day_index,
        )


def test_atom_and_flow_losses_ignore_filled_missing_target() -> None:
    model = _tiny_r0()
    first = _batch(missing_fill=0.1)
    second = _batch(missing_fill=0.9)
    atom_first = model.atom_loss(
        first.condition,
        states=first.state,
        observed_mask=first.observed_mask,
    )
    atom_second = model.atom_loss(
        second.condition,
        states=second.state,
        observed_mask=second.observed_mask,
    )
    assert torch.equal(atom_first["loss"], atom_second["loss"])

    generator1 = torch.Generator().manual_seed(99)
    generator2 = torch.Generator().manual_seed(99)
    flow_first = model.flow_loss(
        first.condition,
        first.target,
        states=first.state,
        observed_mask=first.observed_mask,
        generator=generator1,
    )
    flow_second = model.flow_loss(
        second.condition,
        second.target,
        states=second.state,
        observed_mask=second.observed_mask,
        generator=generator2,
    )
    assert torch.equal(flow_first["loss"], flow_second["loss"])


class _ValidationProbe(nn.Module):
    def flow_loss(
        self,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        states: torch.Tensor,
        observed_mask: torch.Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        del condition, states, observed_mask, generator
        return {"loss": observation[0, 0, 0].float()}


def _weighted_flow_batch(*, value: float, active_cells: int) -> ArchitectureBatch:
    condition = torch.zeros((1, 10, 24, 20), dtype=torch.float32)
    target = torch.zeros((1, 10, 24), dtype=torch.float32)
    target.reshape(-1)[:active_cells] = 0.4
    target[0, 0, 0] = value
    observed = torch.ones_like(target, dtype=torch.bool)
    state = torch.where(
        target == 0.0,
        torch.full_like(target, ZERO_STATE, dtype=torch.long),
        torch.full_like(target, INTERIOR_STATE, dtype=torch.long),
    )
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.tensor([20_001], dtype=torch.long),
    )


def test_flow_validation_is_weighted_by_observed_interior_cells() -> None:
    first = _weighted_flow_batch(value=0.2, active_cells=1)
    second = _weighted_flow_batch(value=0.8, active_cells=9)
    result = validate_stage(
        _ValidationProbe(), [first, second], stage="flow", seed=3
    )
    assert np.isclose(result["loss"], (0.2 + 9 * 0.8) / 10)


def test_short_atom_and_flow_stages_write_verified_checkpoints(tmp_path: Path) -> None:
    model = _tiny_r0()
    batch = _batch()
    trainer = ArchitectureTrainer(
        model,
        tmp_path,
        resolved_config={"candidate": "R0", "smoke": True},
        protocol_sha256="a" * 64,
        data_bundle_sha256="b" * 64,
        training_seed=17,
    )
    atom_report = trainer.fit_stage(
        "atom", [batch], [batch], steps=1, learning_rate=1e-3
    )
    shared_after_atom = shared_ea_state_sha256(model)
    flow_report = trainer.fit_stage(
        "flow", [batch], [batch], steps=1, learning_rate=1e-3
    )
    assert shared_ea_state_sha256(model) == shared_after_atom

    for report in (atom_report, flow_report):
        latest = Path(report["latest_safe"])
        best = Path(report["best"])
        assert latest.is_file() and best.is_file()
        assert latest.with_name(latest.name + ".sha256").is_file()
        assert best.with_name(best.name + ".sha256").is_file()

    payload = torch.load(flow_report["best"], map_location="cpu", weights_only=False)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["shared_EA_state_sha256"] == shared_after_atom
    assert payload["parameter_manifest"] == parameter_manifest(model)

    restored = _tiny_r0()
    loaded = load_checkpoint(
        flow_report["best"],
        restored,
        expected_config_sha256=trainer.config_sha256,
        expected_protocol_sha256="a" * 64,
        expected_data_bundle_sha256="b" * 64,
        expected_shared_EA_state_sha256=shared_after_atom,
    )
    assert loaded["global_step"] == 1
    assert shared_ea_state_sha256(restored) == shared_after_atom

    with _raises(ValueError, "shared E/A"):
        load_checkpoint(
            flow_report["best"],
            _tiny_r0(),
            expected_config_sha256=trainer.config_sha256,
            expected_protocol_sha256="a" * 64,
            expected_data_bundle_sha256="b" * 64,
            expected_shared_EA_state_sha256="f" * 64,
        )

    with _raises(ValueError, "parameter_manifest"):
        load_checkpoint(
            flow_report["best"],
            T0MemorylessRectifiedFlow(
                encoder_dim=4,
                encoder_depth=0,
                flow_dim=4,
                flow_depth=0,
                heads=1,
                ff_multiplier=1,
                atom_hidden_dim=4,
            ),
            expected_config_sha256=trainer.config_sha256,
            expected_protocol_sha256="a" * 64,
            expected_data_bundle_sha256="b" * 64,
        )

    sidecar = Path(str(flow_report["best"]) + ".sha256")
    sidecar.write_text(f"{'0' * 64}  best.pt\n", encoding="ascii")
    with _raises(RuntimeError, "file SHA256"):
        load_checkpoint(
            flow_report["best"],
            _tiny_r0(),
            expected_config_sha256=trainer.config_sha256,
            expected_protocol_sha256="a" * 64,
            expected_data_bundle_sha256="b" * 64,
        )


class _NonfiniteAtomModel(R0JointRectifiedFlow):
    def atom_loss(
        self,
        condition: torch.Tensor,
        *,
        observation: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del condition, observation, states, observed_mask
        anchor = next(self.encoder.parameters()).sum()
        return {"loss": anchor * torch.tensor(float("nan"))}


def test_nonfinite_step_emits_failure_bundle(tmp_path: Path) -> None:
    model = _NonfiniteAtomModel(
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        atom_hidden_dim=4,
    )
    trainer = ArchitectureTrainer(
        model,
        tmp_path,
        resolved_config={"candidate": "R0", "smoke": True},
        protocol_sha256="a" * 64,
        data_bundle_sha256="b" * 64,
        training_seed=2,
    )
    with _raises(FloatingPointError, "non-finite"):
        trainer.fit_stage(
            "atom", [_batch()], [_batch()], steps=1, learning_rate=1e-3
        )
    failure = next((tmp_path / "failures").glob("atom_step*/failure.json"))
    record = json.loads(failure.read_text(encoding="utf-8"))
    assert record["schema"] == FAILURE_SCHEMA
    assert len(record["shared_EA_state_sha256"]) == 64
    assert record["parameter_manifest"]["schema"].endswith("_v1")
    assert failure.with_name("batch.pt").is_file()
    assert failure.with_name("batch.pt.sha256").is_file()
    assert failure.with_name("traceback.txt").is_file()


def test_masked_metrics_ignore_filled_value_and_adjacent_ramps() -> None:
    arrays = _archive_arrays()
    original = masked_per_day_metrics(
        arrays["scenarios"],
        arrays["observations"],
        arrays["observed_mask"],
        zero_probability=arrays["zero_probability"],
        one_probability=arrays["one_probability"],
    )
    changed = arrays["observations"].copy()
    changed[:, 0, 5] = 0.93
    rescored = masked_per_day_metrics(
        arrays["scenarios"],
        changed,
        arrays["observed_mask"],
        zero_probability=arrays["zero_probability"],
        one_probability=arrays["one_probability"],
    )
    for name in (
        "level_CRPS",
        "ramp_CRPS",
        "zero_Brier",
        "one_Brier",
        "atom_state_Brier",
    ):
        np.testing.assert_array_equal(original[name], rescored[name])
        np.testing.assert_array_equal(original[name], np.zeros(2))
    np.testing.assert_array_equal(original["valid_level_cells"], [239, 239])
    np.testing.assert_array_equal(original["valid_ramp_cells"], [228, 228])


def test_scenario_archive_round_trip_hash_and_role_guards(tmp_path: Path) -> None:
    arrays = _archive_arrays()
    destination = tmp_path / "selection.npz"
    write_scenario_archive(destination, **arrays, metadata=_metadata())
    archive = load_scenario_archive(
        destination,
        expected_metadata={
            "candidate_id": "R0",
            "plan_id": "architecture_v1_smoke_v1",
        },
    )
    assert archive.path == destination.resolve()
    assert len(archive.sha256) == 64
    evaluated = evaluate_archive(destination)
    assert np.isclose(evaluated["summary"]["level_CRPS"], 0.0)
    assert np.isclose(evaluated["summary"]["ramp_CRPS"], 0.0)
    assert evaluated["metadata"]["estimand"] == "finite_dependent_scenario_set"

    with _raises(ValueError, "r_seen and final"):
        validate_archive_metadata(_metadata(split_role="r_seen"), members=4)
    with _raises(ValueError, "r_seen and final"):
        validate_archive_metadata(_metadata(split_role="final"), members=4)
    with _raises(ValueError, "mismatch"):
        load_scenario_archive(
            destination, expected_metadata={"checkpoint_sha256": "f" * 64}
        )
    manifest_path = destination.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with _raises(RuntimeError, "SHA256 mismatch"):
        load_scenario_archive(destination)


def test_archive_rejects_invalid_nfe_and_interior_boundary_values(tmp_path: Path) -> None:
    arrays = _archive_arrays()
    bad_metadata = _metadata()
    bad_metadata["flow_nfe"] = 4
    with _raises(ValueError, "flow_nfe"):
        write_scenario_archive(
            tmp_path / "bad_nfe.npz", **arrays, metadata=bad_metadata
        )

    bad_arrays = dict(arrays)
    scenarios = arrays["scenarios"].copy()
    scenarios[:, :, 2, 2] = 0.0
    bad_arrays["scenarios"] = scenarios
    with _raises(ValueError, "strictly"):
        write_scenario_archive(
            tmp_path / "bad_atom.npz", **bad_arrays, metadata=_metadata()
        )


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        if "tmp_path" in inspect.signature(test).parameters:
            with tempfile.TemporaryDirectory() as temporary:
                test(Path(temporary))
        else:
            test()
        print(f"PASS {test.__name__}")
