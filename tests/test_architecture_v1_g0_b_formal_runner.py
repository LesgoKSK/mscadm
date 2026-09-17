"""Protocol and persistence tests for the G0-B retained-training runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from architecture_v1.g0b_tiny_denoiser import (
    TinyDenoisingSystem,
    TinyEMA,
    module_state_sha256,
    tensor_mapping_sha256,
)
from repro_scripts import run_architecture_v1_g0_b_tiny_denoiser_formal as formal


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_formal_matrix_is_exact_unique_324_and_has_expected_endpoints() -> None:
    matrix = formal._matrix(_config())
    assert len(matrix) == len({spec.key for spec in matrix}) == 324
    assert matrix[0].key == "fold0__fraction0p25__IID__seed3"
    assert matrix[-1].key == "fold5__fraction1p0__PA_SHUFFLE__seed5"
    assert {spec.outer_fold for spec in matrix} == set(range(6))
    assert {spec.fraction for spec in matrix} == {0.25, 0.5, 1.0}
    assert {spec.path for spec in matrix} == set(formal.PATH_IDS)
    assert {spec.model_seed for spec in matrix} == {3, 4, 5}


def test_dry_run_validates_p0_without_creating_or_changing_files() -> None:
    output = ROOT / _config()["output_root"]
    formal_root = output / formal.FORMAL_ROOT_NAME
    completed_before = len(list(formal_root.rglob("run.json")))
    before = {
        str(path.relative_to(output)): formal._sha256(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    result = formal.dry_run(CONFIG, include_run_keys=False)
    after = {
        str(path.relative_to(output)): formal._sha256(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert result["mode"] == "target_free_no_files_created"
    assert result["P0"]["status"] == "G0_B_TINY_DENOISER_P0_GO"
    assert result["expected_retained_runs"] == 324
    assert result["completed_retained_runs"] == completed_before
    assert result["remaining_retained_runs"] == 324 - completed_before
    assert result["outer_test_reconstruction_evaluation_implemented"] is False


def test_cpu_formal_execution_requires_an_explicit_slow_run_override() -> None:
    try:
        formal._configure_device("cpu", allow_cpu=False)
    except RuntimeError as error:
        assert "--allow-cpu" in str(error)
    else:
        raise AssertionError("formal CPU execution was accepted without --allow-cpu")
    device, runtime = formal._configure_device("cpu", allow_cpu=True)
    assert device == torch.device("cpu")
    assert runtime["automatic_mixed_precision"] is False
    assert runtime["deterministic_algorithms"] is True


def test_rng_restore_normalizes_cuda_mapped_state_tensors_to_cpu() -> None:
    if not torch.cuda.is_available():
        return
    expected = formal.p0_runner._rng_state()
    cuda_mapped = {
        **expected,
        "torch_cpu": expected["torch_cpu"].to("cuda"),
        "torch_cuda": [value.to("cuda") for value in expected["torch_cuda"]],
    }
    formal.p0_runner._set_rng_state(cuda_mapped)
    assert torch.equal(torch.get_rng_state(), expected["torch_cpu"])
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(
            torch.cuda.get_rng_state_all(), expected["torch_cuda"], strict=True
        )
    )


def test_registered_resume_checkpoint_round_trip_restores_exact_state() -> None:
    config = _config()
    system = TinyDenoisingSystem("IID", model_seed=3)
    optimizer = torch.optim.AdamW(system.parameters(), lr=3e-4, weight_decay=1e-4)
    ema = TinyEMA(system, decay=0.999)
    loss = sum(parameter.square().sum() for parameter in system.parameters())
    loss.backward()
    optimizer.step()
    ema.update(system)
    summary = formal._running_summary()
    summary.update(
        {
            "updates_completed": 128,
            "first_loss": float(loss.detach()),
            "final_loss": float(loss.detach()),
            "loss_min": float(loss.detach()),
            "loss_max": float(loss.detach()),
            "gradient_norm_max_before_clipping": 1.0,
        }
    )
    identity = {"schema": formal.RUN_IDENTITY_SCHEMA, "run_key": "synthetic"}
    identity_sha = formal._canonical_sha256(identity)
    expected_system_sha = module_state_sha256(system)
    expected_ema_sha = tensor_mapping_sha256(ema.shadow)
    with tempfile.TemporaryDirectory(prefix="g0b_formal_resume_", dir="/tmp") as raw:
        path = Path(raw) / "resume.pt"
        formal._atomic_torch(
            path,
            formal._checkpoint_payload(
                identity, identity_sha, system, optimizer, ema, summary
            ),
        )
        restored = TinyDenoisingSystem("IID", model_seed=3)
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=3e-4, weight_decay=1e-4
        )
        restored_ema = TinyEMA(restored, decay=0.999)
        completed, restored_summary = formal._load_resume(
            path,
            identity,
            identity_sha,
            restored,
            restored_optimizer,
            restored_ema,
            config,
            torch.device("cpu"),
        )
        assert completed == restored_summary["updates_completed"] == 128
        assert module_state_sha256(restored) == expected_system_sha
        assert tensor_mapping_sha256(restored_ema.shadow) == expected_ema_sha
        assert formal._verified_sidecar(path) == formal._sha256(path)


def test_final_ema_interruption_window_recovers_completion_and_deletes_resume() -> None:
    spec = formal.RunSpec(0, 0.25, 0, "IID", 3)
    identity = {
        "schema": formal.RUN_IDENTITY_SCHEMA,
        **spec.manifest(),
        "updates": 1024,
        "randomness": {"batch_calendar_days": 16},
    }
    identity_sha = formal._canonical_sha256(identity)
    system = TinyDenoisingSystem("IID", model_seed=3)
    state = {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
    }
    state_sha = tensor_mapping_sha256(state)
    summary = {
        "updates_completed": 1024,
        "first_loss": 1.0,
        "final_loss": 0.5,
        "loss_min": 0.5,
        "loss_max": 1.0,
        "gradient_norm_max_before_clipping": 1.2,
        "wall_seconds_accumulated": 2.0,
        "MULAN_online_schedule_audits": [],
    }
    with tempfile.TemporaryDirectory(prefix="g0b_formal_final_", dir="/tmp") as raw:
        run_root = Path(raw)
        final_path = run_root / "final_ema.pt"
        formal._atomic_torch(
            final_path,
            {
                "schema": formal.FINAL_EMA_SCHEMA,
                "run_identity": identity,
                "run_identity_sha256": identity_sha,
                "completed_updates": 1024,
                "EMA_updates": 1024,
                "EMA_system_state": state,
                "EMA_system_state_sha256": state_sha,
                "runner_summary": summary,
                "MULAN_final_EMA_schedule_audit": None,
                "outer_test_reconstruction_evaluation_performed": False,
            },
        )
        formal._atomic_torch(run_root / "resume.pt", {"discarded": True})
        completion = formal._recover_or_validate_completion(
            run_root, identity_sha, spec
        )
        assert completion is not None
        assert completion["status"] == "complete"
        assert completion["resume_checkpoint_deleted"] is True
        assert not (run_root / "resume.pt").exists()
        assert formal._verified_sidecar(run_root / "run.json")


def test_training_freeze_is_impossible_with_missing_runs() -> None:
    matrix = formal._matrix(_config())
    with tempfile.TemporaryDirectory(prefix="g0b_formal_freeze_", dir="/tmp") as raw:
        assert formal._try_freeze(Path(raw), matrix, "synthetic") is None
        assert not (Path(raw) / "training.freeze.json").exists()
