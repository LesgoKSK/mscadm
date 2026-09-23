from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from architecture_v1.g0b_tiny_denoiser import (
    TinyEMA,
    module_state_sha256,
)
from architecture_v1.transition_atom import (
    TransitionAtomNuisance,
    train_only_atom_contract,
)
from architecture_v1.transition_probe import PATH_IDS
from repro_scripts import run_architecture_v1_transition_object_formal as formal
from repro_scripts import run_architecture_v1_transition_object_p0 as p0


def _atom_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7401)
    condition = torch.randn((4, 10, 24, 20), generator=generator)
    observation = torch.rand((4, 10, 24), generator=generator)
    observation[:, :, 0] = 0.0
    observation[0, 0, 1] = 1.0
    observed = torch.ones_like(observation, dtype=torch.bool)
    return condition, observation, observed


def _atom_model(
    observation: torch.Tensor, observed: torch.Tensor, *, seed: int
) -> TransitionAtomNuisance:
    contract = train_only_atom_contract(observation, observed)
    torch.manual_seed(seed)
    return TransitionAtomNuisance(
        fixed_one_probability=float(contract["fixed_one_probability"]),
        location_mean=float(contract["location_mean"]),
        location_std=float(contract["location_std"]),
    )


def test_formal_dry_run_is_target_free_and_has_exact_matrix() -> None:
    result = formal.dry_run(formal.DEFAULT_CONFIG, include_run_keys=True)
    assert result["mode"] == "target_free_no_files_created"
    assert result["expected_shared_atom_models"] == 6
    assert result["expected_retained_runs"] == 84
    assert 0 <= result["completed_retained_runs"] <= 84
    assert result["remaining_retained_runs"] == (
        84 - result["completed_retained_runs"]
    )
    assert len(result["run_keys"]) == 84
    assert result["formal_training_target_roles_if_executed"] == ["train"]
    assert result["outer_test_TGO_metrics_constructed"] is False


def test_formal_matrix_contains_every_fold_seed_path_once() -> None:
    config = formal._read_json(formal.DEFAULT_CONFIG)
    matrix = formal._matrix(config)
    observed = {
        (spec.outer_fold, spec.model_seed, spec.path_id) for spec in matrix
    }
    expected = {
        (fold, seed, path_id)
        for fold in range(6)
        for seed in (3, 4)
        for path_id in PATH_IDS
    }
    assert observed == expected


def test_atom_epoch_bank_is_deterministic_seeded_per_epoch() -> None:
    first, first_manifest = formal._atom_index_bank(
        fold=2, train_days=5, updates=3, batch_size=4
    )
    replay, replay_manifest = formal._atom_index_bank(
        fold=2, train_days=5, updates=3, batch_size=4
    )
    other, other_manifest = formal._atom_index_bank(
        fold=3, train_days=5, updates=3, batch_size=4
    )
    assert torch.equal(first, replay)
    assert first_manifest == replay_manifest
    assert first_manifest["epochs_touched"] == 3
    flat = first.flatten()
    assert torch.equal(torch.sort(flat[:5]).values, torch.arange(5))
    assert torch.equal(torch.sort(flat[5:10]).values, torch.arange(5))
    assert first_manifest["sha256"] != other_manifest["sha256"]
    assert not torch.equal(first, other)


def test_shared_atom_resume_replays_next_update_exactly() -> None:
    condition, observation, observed = _atom_data()
    bank, _manifest = formal._atom_index_bank(
        fold=0, train_days=4, updates=2, batch_size=4
    )
    model = _atom_model(observation, observed, seed=51000)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    ema = TinyEMA(model, decay=0.999)
    summary = formal._atom_summary()
    values = formal._atom_update(
        model,
        optimizer,
        ema,
        condition,
        observation,
        observed,
        bank,
        0,
        torch.device("cpu"),
        gradient_clip=1.0,
    )
    formal._update_atom_summary(
        summary,
        update=1,
        loss=values[0],
        atom_nll=values[1],
        location=values[2],
        gradient_norm=values[3],
    )
    identity = {
        "schema": formal.ATOM_IDENTITY_SCHEMA,
        "updates": 2,
        "checkpoint_interval_updates": 1,
    }
    identity_sha = formal._canonical_sha256(identity)
    with tempfile.TemporaryDirectory(prefix="tgo_atom_resume_", dir="/tmp") as raw:
        checkpoint = Path(raw) / "resume.pt"
        formal._atomic_torch(
            checkpoint,
            formal._atom_checkpoint_payload(
                identity, identity_sha, model, optimizer, ema, summary
            ),
        )
        reference_values = formal._atom_update(
            model,
            optimizer,
            ema,
            condition,
            observation,
            observed,
            bank,
            1,
            torch.device("cpu"),
            gradient_clip=1.0,
        )
        reference = (
            reference_values,
            module_state_sha256(model),
            p0._semantic_state_sha256(optimizer.state_dict()),
            p0._ema_hash(ema),
        )

        restored = _atom_model(observation, observed, seed=999)
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        restored_ema = TinyEMA(restored, decay=0.999)
        completed, restored_summary = formal._load_atom_resume(
            checkpoint,
            identity,
            identity_sha,
            restored,
            restored_optimizer,
            restored_ema,
        )
        assert completed == 1
        assert restored_summary == summary
        replay_values = formal._atom_update(
            restored,
            restored_optimizer,
            restored_ema,
            condition,
            observation,
            observed,
            bank,
            completed,
            torch.device("cpu"),
            gradient_clip=1.0,
        )
        replay = (
            replay_values,
            module_state_sha256(restored),
            p0._semantic_state_sha256(restored_optimizer.state_dict()),
            p0._ema_hash(restored_ema),
        )
        assert replay == reference


def test_final_ema_interruption_window_recovers_run_record() -> None:
    spec = formal.RunSpec(outer_fold=0, model_seed=3, path_id="LEVEL_IID")
    identity = {
        "schema": formal.RUN_IDENTITY_SCHEMA,
        **spec.manifest(),
        "updates": 1024,
    }
    identity_sha = formal._canonical_sha256(identity)
    system = formal.TransitionDenoisingSystem("LEVEL_IID", model_seed=3)
    state = {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
    }
    state_sha = formal.tensor_mapping_sha256(state)
    summary = {
        "updates_completed": 1024,
        "loss_first": 1.0,
        "loss_final": 0.9,
        "loss_min": 0.8,
        "loss_max": 1.1,
        "gradient_norm_max": 2.0,
        "wall_seconds_accumulated": 1.0,
    }
    with tempfile.TemporaryDirectory(prefix="tgo_final_recover_", dir="/tmp") as raw:
        root = Path(raw)
        final = root / "final_ema.pt"
        formal._atomic_torch(
            final,
            {
                "schema": formal.RUN_FINAL_SCHEMA,
                "identity": identity,
                "identity_sha256": identity_sha,
                "completed_updates": 1024,
                "EMA_updates": 1024,
                "EMA_system_state": state,
                "EMA_system_state_sha256": state_sha,
                "summary": summary,
            },
        )
        formal._atomic_torch(root / "resume.pt", {"discarded": True})
        record = formal._validate_run_completion(
            root, identity_sha=identity_sha, spec=spec
        )
        assert record is not None
        assert record["status"] == "complete"
        assert record["resume_checkpoint_deleted"] is True
        assert not (root / "resume.pt").exists()
        assert (root / "run.json").is_file()
