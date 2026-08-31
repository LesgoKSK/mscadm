from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from caa_rahc.nested_data import NestedDataBundle
from repro.data import ArrayStandardizer, SplitData
from repro_scripts import run_caa_outer as runner


def _split(day: str, rows: int = 2) -> SplitData:
    target = np.full((rows, 24), 0.4, dtype=np.float32)
    return SplitData(
        condition=np.zeros((rows, 24, 20), dtype=np.float32),
        flat_condition=np.zeros((rows, 250), dtype=np.float32),
        target=target,
        target_standard=np.zeros_like(target),
        zone=np.arange(1, rows + 1, dtype=np.int64),
        day=np.full(rows, np.datetime64(day, "D")),
    )


def _bundle(outer: int) -> NestedDataBundle:
    protocol: dict[str, Any] = {
        "name": "test_nested_protocol",
        "outer": outer,
        "protocol_sha256": f"protocol-{outer}",
        "date_sha256": {
            "train": f"train-{outer}",
            "head_validation": "head-validation",
            "calibration": "calibration",
            "test": f"test-{outer}",
        },
        "input_fingerprint": {"combined_sha256": "i" * 64},
        "data_fingerprint": {"sha256": "d" * 64},
    }
    condition_scaler = ArrayStandardizer(
        mean=np.zeros((1, 24, 10), dtype=np.float32),
        std=np.ones((1, 24, 10), dtype=np.float32),
    )
    flat_scaler = ArrayStandardizer(
        mean=np.zeros((1, 250), dtype=np.float32),
        std=np.ones((1, 250), dtype=np.float32),
    )
    target_scaler = ArrayStandardizer(
        mean=np.zeros((1, 24), dtype=np.float32),
        std=np.ones((1, 24), dtype=np.float32),
    )
    return NestedDataBundle(
        train=_split("2012-01-01"),
        validation=_split("2012-01-02"),
        calibration=_split("2012-01-03"),
        test=_split(f"2012-01-0{3 + outer}"),
        condition_standardizer=condition_scaler,
        flat_condition_standardizer=flat_scaler,
        target_standardizer=target_scaler,
        protocol=protocol,
    )


def _config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    config: dict[str, Any] = {
        "experiment_name": "runner unit test",
        "data_dir": "Data",
        "output_root": str(tmp_path / "outputs"),
        "outer_splits": [1, 2, 3],
        "model_seeds": [0, 1, 2],
        "device": "cpu",
        "model": {"head": {}, "denoiser": {}},
        "diffusion": {"timesteps": 2},
        "training": {"resume": False, "diffusion_steps": 1},
        "sampling": {"scenarios": 2, "steps": 1, "eta": 0.0, "day_batch": 2},
        "scope_statement": "unit test",
    }
    path = tmp_path / "caa_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, config


def _manifest(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = runner.output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "caa_rahc_frozen_outer_manifest_v1",
        "config_path": runner._display_path(config_path),
        "config_sha256": runner.sha256_file(config_path),
        "config_canonical_sha256": runner.canonical_json_sha256(config),
        "code": runner.code_fingerprint(),
        "outer_splits": [1, 2, 3],
        "model_seeds": [0, 1, 2],
        "protocols": {f"outer{outer}": _bundle(outer).protocol for outer in (1, 2, 3)},
    }
    (root / runner.FROZEN_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _write_fake_checkpoint(
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    *,
    outer: int = 1,
    seed: int = 0,
) -> Path:
    checkpoint = runner.run_directory(runner.output_root(config), outer, seed) / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": "cr_mscadm",
            "variant": "full",
            "seed": seed,
            "step": config["training"]["diffusion_steps"],
            "config": config,
            "protocol": bundle.protocol,
            "model": {},
        },
        checkpoint,
    )
    audit = runner._checkpoint_expected_audit(
        checkpoint,
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
    )
    runner.checkpoint_audit_path(checkpoint).write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint


def _fake_generation(
    checkpoint: str | Path,
    data: NestedDataBundle,
    *,
    split_name: str,
    scenarios: int,
    steps: int,
    eta: float,
    day_batch: int,
    seed: int,
    device: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    split = getattr(data, split_name)
    values = np.full((len(split), scenarios, 24), 0.5, dtype=np.float32)
    return values, {
        "model": "cr_mscadm",
        "variant": "full",
        "training_seed": 0,
        "sampling_seed": seed,
        "split": split_name,
        "scenarios": scenarios,
        "sampling_steps": steps,
        "sampler": "ddim",
        "eta": eta,
        "checkpoint": str(Path(checkpoint).resolve()),
    }


def test_audit_splits_writes_and_revalidates_deterministic_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    monkeypatch.setattr(
        runner, "build_nested_gefcom2014", lambda _root, *, outer: _bundle(outer)
    )
    first = runner.audit_splits(config_path, config)
    second = runner.audit_splits(config_path, config)
    assert first == second
    manifest_path = runner.output_root(config) / runner.FROZEN_MANIFEST_NAME
    assert manifest_path.is_file()
    assert first["protocols"]["outer1"]["protocol_sha256"] == "protocol-1"
    loaded = runner.load_frozen_manifest(config_path, config)
    assert loaded == first


def test_train_is_fresh_only_and_checkpoint_reuse_requires_full_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    manifest = _manifest(config_path, config)
    bundle = _bundle(1)
    calls: list[dict[str, Any]] = []

    class FakeTrainer:
        def __init__(self, received, data, output_dir, **kwargs):
            assert received["training"]["resume"] is False
            assert data is bundle
            self.config = received
            self.output_dir = Path(output_dir)
            self.kwargs = kwargs
            calls.append(kwargs)

        def fit(self) -> Path:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "final.pt"
            torch.save(
                {
                    "name": "cr_mscadm",
                    "variant": "full",
                    "seed": self.kwargs["seed"],
                    "step": self.config["training"]["diffusion_steps"],
                    "config": self.config,
                    "protocol": bundle.protocol,
                    "model": {},
                },
                path,
            )
            return path

    monkeypatch.setattr(runner, "CRTrainer", FakeTrainer)
    checkpoint = runner.train_one(
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=1,
        seed=0,
        device="cpu",
    )
    assert len(calls) == 1
    assert runner.checkpoint_audit_path(checkpoint).is_file()
    runner.train_one(
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=1,
        seed=0,
        device="cpu",
    )
    assert len(calls) == 1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["protocol"] = {**bundle.protocol, "protocol_sha256": "old-protocol"}
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="protocol_sha256 mismatch"):
        runner.train_one(
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
            device="cpu",
        )


def test_partial_run_and_unaudited_checkpoint_are_rejected(tmp_path: Path) -> None:
    config_path, config = _config(tmp_path)
    manifest = _manifest(config_path, config)
    bundle = _bundle(1)
    run = runner.run_directory(runner.output_root(config), 1, 0)
    run.mkdir(parents=True)
    (run / "head_latest.pt").write_bytes(b"old weight")
    with pytest.raises(RuntimeError, match="resume is forbidden"):
        runner.train_one(
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
            device="cpu",
        )

    (run / "head_latest.pt").unlink()
    checkpoint = _write_fake_checkpoint(config_path, config, manifest, bundle)
    runner.checkpoint_audit_path(checkpoint).unlink()
    with pytest.raises(RuntimeError, match="unaudited"):
        runner.validate_checkpoint(
            checkpoint,
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
        )


def test_calibration_archive_is_hashed_validated_and_safely_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    manifest = _manifest(config_path, config)
    bundle = _bundle(1)
    _write_fake_checkpoint(config_path, config, manifest, bundle)
    calls = 0

    def generated(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _fake_generation(*args, **kwargs)

    monkeypatch.setattr(runner, "generate_cr_scenarios", generated)
    archive = runner.generate_one(
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=1,
        seed=0,
        split="calibration",
        device="cpu",
    )
    assert calls == 1
    assert archive.is_file()
    assert runner.archive_audit_path(archive).is_file()
    runner.generate_one(
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=1,
        seed=0,
        split="calibration",
        device="cpu",
    )
    assert calls == 1

    audit_path = runner.archive_audit_path(archive)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["archive_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash/audit"):
        runner.generate_one(
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
            split="calibration",
            device="cpu",
        )


def test_test_generation_requires_matching_selection_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    manifest = _manifest(config_path, config)
    root = runner.output_root(config)
    bundle = _bundle(1)
    _write_fake_checkpoint(config_path, config, manifest, bundle)
    monkeypatch.setattr(runner, "generate_cr_scenarios", _fake_generation)

    with pytest.raises(RuntimeError, match="missing selection lock"):
        runner.generate_one(
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
            split="test",
            device="cpu",
        )
    wrong = runner.selection_lock_identity(root, manifest)
    wrong["code_sha256"] = "wrong"
    (root / runner.SELECTION_LOCK_NAME).write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(RuntimeError, match="code_sha256"):
        runner.generate_one(
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=1,
            seed=0,
            split="test",
            device="cpu",
        )

    lock = {
        "schema": "caa_rahc_selection_lock_v1",
        **runner.selection_lock_identity(root, manifest),
        "selected_candidate": "unit-test-candidate",
    }
    (root / runner.SELECTION_LOCK_NAME).write_text(json.dumps(lock), encoding="utf-8")
    archive = runner.generate_one(
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=1,
        seed=0,
        split="test",
        device="cpu",
    )
    assert archive.is_file()
    with np.load(archive, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        assert metadata["checkpoint_sha256"] == runner.sha256_file(
            runner.run_directory(root, 1, 0) / "final.pt"
        )
        assert metadata["split_date_sha256"] == bundle.protocol["date_sha256"]["test"]
        assert metadata["protocol_sha256"] == bundle.protocol["protocol_sha256"]
        assert metadata["sampling_seed"] == runner.sampling_seed(1, 0, "test")


def test_sampling_seeds_and_registered_output_layout_are_collision_free(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    seeds = {
        runner.sampling_seed(outer, seed, split)
        for outer in (1, 2, 3)
        for seed in (0, 1, 2)
        for split in ("calibration", "test")
    }
    assert len(seeds) == 18
    assert runner.run_directory(root, 3, 2) == root / "outer3/runs/full/seed2"
    assert runner.scenario_archive_path(root, 2, 1, "calibration") == (
        root / "outer2/scenarios/full_seed1_calibration_raw.npz"
    )

