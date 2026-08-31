"""CPU contracts for the architecture-v1 formal training primitives.

The file deliberately avoids pytest-only fixtures so it can also be executed
directly with the registered environment::

    python tests/test_architecture_v1_formal_training.py
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from architecture_v1.atom import INTERIOR_STATE, ONE_STATE, ZERO_STATE
from architecture_v1.formal_training import (
    FORMAL_COMPLETION_SCHEMA,
    FormalEpochTrainer,
    FormalValidationBank,
    deterministic_epoch_permutation,
    write_training_freeze,
)
from architecture_v1.model import R0JointRectifiedFlow
from architecture_v1.training import ArchitectureBatch, tensor_state_sha256


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert pattern.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def _batch(days: list[int], *, seed: int) -> ArchitectureBatch:
    """Return deterministic joint-day data with atoms, interiors and a mask."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = len(days)
    condition = torch.randn(
        (count, 10, 24, 20), dtype=torch.float32, generator=generator
    )
    target = 0.05 + 0.90 * torch.rand(
        (count, 10, 24), dtype=torch.float32, generator=generator
    )
    target[:, 0, 0] = 0.0
    target[:, 1, 1] = 1.0
    observed = torch.ones((count, 10, 24), dtype=torch.bool)
    # The filled value is finite but cannot supervise either objective.
    observed[:, 2, 2] = False
    target[:, 2, 2] = 0.37
    state = torch.full((count, 10, 24), INTERIOR_STATE, dtype=torch.long)
    state[target == 0.0] = ZERO_STATE
    state[target == 1.0] = ONE_STATE
    state[~observed] = INTERIOR_STATE
    return ArchitectureBatch(
        condition=condition,
        target=target,
        state=state,
        observed_mask=observed,
        raw_missing_mask=~observed,
        day_index=torch.tensor(days, dtype=torch.long),
    )


def _subset(batch: ArchitectureBatch, indices: list[int]) -> ArchitectureBatch:
    selected = torch.tensor(indices, dtype=torch.long)
    return ArchitectureBatch(
        condition=batch.condition.index_select(0, selected),
        target=batch.target.index_select(0, selected),
        state=batch.state.index_select(0, selected),
        observed_mask=batch.observed_mask.index_select(0, selected),
        raw_missing_mask=batch.raw_missing_mask.index_select(0, selected),
        day_index=batch.day_index.index_select(0, selected),
    )


def _model(*, seed: int, auxiliary_weight: float = 0.0) -> R0JointRectifiedFlow:
    torch.manual_seed(seed)
    return R0JointRectifiedFlow(
        condition_dim=20,
        zones=10,
        hours=24,
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
        atom_location_auxiliary_weight=auxiliary_weight,
        atom_location_mean=0.0,
        atom_location_std=1.0,
        atom_location_smooth_l1_beta=1.0,
    )


NOISE_SEEDS = (31_001, 31_002, 31_003, 31_004)
TIME_SEEDS = (32_001, 32_002, 32_003, 32_004)


def _bank(validation: ArchitectureBatch) -> FormalValidationBank:
    return FormalValidationBank.from_explicit_seeds(
        validation.day_index,
        noise_seeds=NOISE_SEEDS,
        time_seeds=TIME_SEEDS,
        plan_id="architecture_v1_test_fixed_K4",
        evaluation_batch_days=2,
    )


def _state_hash(model: R0JointRectifiedFlow) -> str:
    return tensor_state_sha256(
        {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
    )


def _trainer(
    model: R0JointRectifiedFlow,
    output: Path,
    bank: FormalValidationBank,
) -> FormalEpochTrainer:
    return FormalEpochTrainer(
        model,
        output,
        resolved_config={
            "schema": "architecture_v1_formal_training_test_v1",
            "epochs": 2,
            "batch_days": 2,
            "validation_plan": bank.manifest,
        },
        protocol_sha256="a" * 64,
        data_bundle_sha256="b" * 64,
        code_sha256="c" * 64,
        validation_bank=bank,
        training_seed=7,
        run_id="architecture_v1_test_R0_seed7",
        device="cpu",
    )


def _fit_two_epochs(
    trainer: FormalEpochTrainer,
    train: ArchitectureBatch,
    validation: ArchitectureBatch,
    *,
    resume: bool = False,
    epoch_callback=None,
) -> dict[str, object]:
    return trainer.fit_stage(
        "flow",
        train,
        validation,
        max_epochs=2,
        batch_days=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        validate_every_epochs=1,
        early_stopping_patience=8,
        early_stopping_minimum_delta=0.0,
        early_stopping_relative_delta=0.0,
        minimum_epochs_before_early_stop=2,
        resume=resume,
        epoch_callback=epoch_callback,
        restore_best=True,
    )


def test_explicit_seed_validation_bank_is_order_and_partition_invariant() -> None:
    validation = _batch([20_001, 20_002, 20_003, 20_004], seed=101)
    bank = _bank(validation)
    repeated = _bank(validation)
    assert bank.replicates == 4
    assert bank.noise_seeds == NOISE_SEEDS
    assert bank.time_seeds == TIME_SEEDS
    assert bank.sha256 == repeated.sha256
    assert torch.equal(bank.noise, repeated.noise)
    assert torch.equal(bank.flow_time, repeated.flow_time)

    model = _model(seed=11)
    whole = bank.evaluate(model, validation, stage="flow", device="cpu")
    # Deliberately reverse both minibatch order and the order inside each batch.
    repartitioned = [
        _subset(validation, [3, 1]),
        _subset(validation, [2, 0]),
    ]
    changed = bank.evaluate(model, repartitioned, stage="flow", device="cpu")
    assert whole == changed
    assert whole["replicates"] == 4.0
    assert np.isfinite(whole["loss"])

    order = deterministic_epoch_permutation(
        7, training_seed=19, stage="flow", epoch=3
    )
    assert np.array_equal(np.sort(order), np.arange(7))
    assert np.array_equal(
        order,
        deterministic_epoch_permutation(
            7, training_seed=19, stage="flow", epoch=3
        ),
    )


def test_atom_auxiliary_validation_uses_observed_interior_cells() -> None:
    validation = _batch([21_001, 21_002, 21_003], seed=102)
    bank = _bank(validation)
    model = _model(seed=12, auxiliary_weight=0.25)
    metrics = bank.evaluate(model, validation, stage="atom", device="cpu")

    expected_interior = int(
        (
            validation.observed_mask
            & (validation.state == INTERIOR_STATE)
        ).sum().item()
    )
    assert metrics["interior_location_count"] == float(expected_interior)
    assert metrics["interior_location_smooth_l1"] > 0.0
    assert np.isclose(
        metrics["loss"],
        metrics["atom_nll"]
        + 0.25 * metrics["interior_location_smooth_l1"],
        rtol=1e-7,
        atol=1e-8,
    )
    assert 0.0 < metrics["observed_fraction"] < 1.0


def test_two_epoch_stage_writes_verified_artifacts() -> None:
    train = _batch([1_001, 1_002, 1_003, 1_004], seed=103)
    validation = _batch([2_001, 2_002], seed=104)
    bank = _bank(validation)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "formal"
        model = _model(seed=13)
        report = _fit_two_epochs(_trainer(model, output, bank), train, validation)
        assert report["status"] == "complete"
        assert report["epochs_completed"] == 2
        assert report["selection_target_accessed"] is False
        for name in ("latest_safe_checkpoint", "best_checkpoint", "history", "completion"):
            artifact = Path(str(report[name]))
            assert artifact.is_file(), artifact
            assert artifact.with_name(artifact.name + ".sha256").is_file()
        history = json.loads(Path(str(report["history"])).read_text(encoding="utf-8"))
        assert len(history["records"]) == 2
        assert history["records"][0]["permutation_sha256"]
        assert history["records"][1]["validation"] is not None


def test_interrupted_resume_matches_uninterrupted_final_tensor_hash() -> None:
    train = _batch([3_001, 3_002, 3_003, 3_004], seed=105)
    validation = _batch([4_001, 4_002], seed=106)
    bank = _bank(validation)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        uninterrupted_model = _model(seed=14)
        uninterrupted = _trainer(
            uninterrupted_model, root / "uninterrupted", bank
        )
        _fit_two_epochs(uninterrupted, train, validation)
        expected_hash = _state_hash(uninterrupted_model)

        interrupted_model = _model(seed=14)
        interrupted = _trainer(interrupted_model, root / "resumed", bank)

        def stop_after_first_complete_epoch(record):
            if int(record["epoch"]) == 0:
                raise RuntimeError("intentional epoch-boundary interruption")

        with _raises(RuntimeError, "intentional"):
            _fit_two_epochs(
                interrupted,
                train,
                validation,
                epoch_callback=stop_after_first_complete_epoch,
            )
        latest = root / "resumed" / "flow" / "latest_safe.pt"
        assert latest.is_file()
        assert latest.with_name(latest.name + ".sha256").is_file()

        # A fresh, even differently initialised model must be replaced by the
        # verified safe state before epoch two is evaluated.
        resumed_model = _model(seed=999)
        resumed = _trainer(resumed_model, root / "resumed", bank)
        report = _fit_two_epochs(resumed, train, validation, resume=True)
        assert report["epochs_completed"] == 2
        assert _state_hash(resumed_model) == expected_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _fake_completion(
    root: Path,
    *,
    seed: int,
    protocol_hash: str,
    data_hash: str,
    code_hash: str,
) -> Path:
    run = root / f"seed{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(f"verified fake checkpoint seed={seed}\n".encode("ascii"))
    checkpoint_hash = _sha256(checkpoint)
    completion = run / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "schema": FORMAL_COMPLETION_SCHEMA,
                "status": "complete",
                "run_id": f"fake_R0_seed{seed}",
                "candidate_id": "R0",
                "training_seed": seed,
                "stage": "flow",
                "stage_identity": {
                    "identity_sha256": hashlib.sha256(
                        f"identity-{seed}".encode("ascii")
                    ).hexdigest(),
                    "identity_hashes": {
                        "protocol_sha256": protocol_hash,
                        "data_bundle_sha256": data_hash,
                        "code_sha256": code_hash,
                    },
                },
                "best_checkpoint": str(checkpoint.resolve()),
                "best_checkpoint_sha256": checkpoint_hash,
                "best_epoch": 1,
                "best_validation_loss": 1.0 + 0.01 * seed,
                "shared_EA_state_sha256": hashlib.sha256(
                    f"shared-{seed}".encode("ascii")
                ).hexdigest(),
                "selection_target_accessed": False,
                "calibration_target_accessed": False,
                "r_seen_target_accessed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _sha256(completion)
    return completion


def test_training_freeze_accepts_exact_three_seed_set_and_rejects_missing_seed() -> None:
    protocol_hash = "a" * 64
    data_hash = "b" * 64
    code_hash = "c" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        completions = {
            seed: _fake_completion(
                root,
                seed=seed,
                protocol_hash=protocol_hash,
                data_hash=data_hash,
                code_hash=code_hash,
            )
            for seed in (0, 1, 2)
        }
        with _raises(ValueError, "exactly every registered seed"):
            write_training_freeze(
                root / "missing.freeze.json",
                completion_files={0: completions[0], 1: completions[1]},
                gate_config_sha256="d" * 64,
                selection_plan_sha256="e" * 64,
                protocol_sha256=protocol_hash,
                data_bundle_sha256=data_hash,
                code_sha256=code_hash,
            )
        assert not (root / "missing.freeze.json").exists()

        frozen = write_training_freeze(
            root / "training.freeze.json",
            completion_files=completions,
            gate_config_sha256="d" * 64,
            selection_plan_sha256="e" * 64,
            protocol_sha256=protocol_hash,
            data_bundle_sha256=data_hash,
            code_sha256=code_hash,
        )
        assert frozen["status"] == "training_frozen"
        assert frozen["registered_training_seeds"] == [0, 1, 2]
        assert frozen["training_closed"] is True
        assert frozen["selection_authorized"] is False
        freeze_path = Path(str(frozen["training_freeze"]))
        assert freeze_path.is_file()
        assert freeze_path.with_name(freeze_path.name + ".sha256").is_file()


def main() -> None:
    tests = (
        test_explicit_seed_validation_bank_is_order_and_partition_invariant,
        test_atom_auxiliary_validation_uses_observed_interior_cells,
        test_two_epoch_stage_writes_verified_artifacts,
        test_interrupted_resume_matches_uninterrupted_final_tensor_hash,
        test_training_freeze_accepts_exact_three_seed_set_and_rejects_missing_seed,
    )
    torch.set_num_threads(1)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
