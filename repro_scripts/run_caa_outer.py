"""Audited base-model orchestration for the frozen CAA-RAHC outer study.

Only four stages live here:

``audit-splits``
    Materialize and freeze all three nested split protocols.
``train``
    Train fresh CR-MS-CADM full models.  Partial or unaudited weights are
    rejected; this runner never resumes training.
``generate-calibration``
    Generate the development/calibration archives before method selection.
``generate-test``
    Generate outer-test archives only after a matching ``selection.lock.json``
    exists at the experiment root.

The runner treats every existing artifact as untrusted until both its content
hash and its embedded provenance metadata have been verified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from caa_rahc.nested_data import NestedDataBundle, build_nested_gefcom2014
from cr_mscadm.experiment import generate_cr_scenarios
from cr_mscadm.training import CRTrainer
from repro.sampling import save_scenarios


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "repro_configs" / "caa_rahc_frozen.json"
FROZEN_MANIFEST_NAME = "frozen_manifest.json"
SELECTION_LOCK_NAME = "selection.lock.json"

BASE_CODE_PATHS = (
    "caa_rahc/nested_data.py",
    "cr_mscadm/model.py",
    "cr_mscadm/training.py",
    "cr_mscadm/experiment.py",
    "repro/data.py",
    "repro/torch_data.py",
    "repro/models/mscadm.py",
    "repro/diffusion/__init__.py",
    "repro/diffusion_core.py",
    "repro_scripts/run_caa_outer.py",
)

SELECTION_CODE_PATHS = BASE_CODE_PATHS + (
    "caa_rahc/structural_atom.py",
    "caa_rahc/gate.py",
    "caa_rahc/hinge_tail.py",
    "caa_rahc/baselines.py",
    "caa_rahc/metrics.py",
    "caa_rahc/selection.py",
    "caa_rahc/candidates.py",
    "caa_rahc/features.py",
    "cr_mscadm/calibration.py",
    "rahc/v2_calibration.py",
    "rahc/validation.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file blocks atomic write: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(path.resolve())


def load_experiment_config(path: str | Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_workspace_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "data_dir",
        "output_root",
        "outer_splits",
        "model_seeds",
        "model",
        "diffusion",
        "training",
        "sampling",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"CAA configuration is missing keys: {missing}")
    if config["training"].get("resume") is not False:
        raise ValueError("CAA outer training must set training.resume=false")
    if sorted(int(value) for value in config["outer_splits"]) != [1, 2, 3]:
        raise ValueError("CAA outer_splits must be exactly [1, 2, 3]")
    if sorted(int(value) for value in config["model_seeds"]) != [0, 1, 2]:
        raise ValueError("CAA model_seeds must be exactly [0, 1, 2]")
    return config_path, config


def output_root(config: dict[str, Any]) -> Path:
    return _resolve_workspace_path(config["output_root"])


def data_root(config: dict[str, Any]) -> Path:
    return _resolve_workspace_path(config["data_dir"])


def run_directory(root: Path, outer: int, seed: int) -> Path:
    return root / f"outer{outer}" / "runs" / "full" / f"seed{seed}"


def scenario_archive_path(root: Path, outer: int, seed: int, split: str) -> Path:
    if split not in {"calibration", "test"}:
        raise ValueError("scenario split must be calibration or test")
    return root / f"outer{outer}" / "scenarios" / f"full_seed{seed}_{split}_raw.npz"


def checkpoint_audit_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".audit.json")


def archive_audit_path(archive: Path) -> Path:
    return archive.with_suffix(archive.suffix + ".audit.json")


def sampling_seed(outer: int, model_seed: int, split: str) -> int:
    base = {"calibration": 30_000_000, "test": 40_000_000}.get(split)
    if base is None:
        raise ValueError("sampling split must be calibration or test")
    return int(base + outer * 10_000 + model_seed)


def _code_fingerprint(paths: Iterable[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = WORKSPACE / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered code file is missing: {path}")
        records.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "algorithm": "sha256",
        "combined_sha256": canonical_json_sha256(records),
        "files": records,
    }


def code_fingerprint() -> dict[str, Any]:
    """Fingerprint only code that can affect raw base-model training/sampling."""

    return _code_fingerprint(BASE_CODE_PATHS)


def selection_code_fingerprint() -> dict[str, Any]:
    """Fingerprint the full authoritative CAA selection and calibration stack."""

    return _code_fingerprint(SELECTION_CODE_PATHS)

def _manifest_identity(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": _display_path(config_path),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_json_sha256(config),
        "code": code_fingerprint(),
    }


def _manifest_payload(
    config_path: Path,
    config: dict[str, Any],
    bundles: dict[int, NestedDataBundle],
) -> dict[str, Any]:
    identity = _manifest_identity(config_path, config)
    protocols = {f"outer{outer}": bundle.protocol for outer, bundle in sorted(bundles.items())}
    input_hashes = {
        bundle.protocol["input_fingerprint"]["combined_sha256"] for bundle in bundles.values()
    }
    data_hashes = {
        bundle.protocol["data_fingerprint"]["sha256"] for bundle in bundles.values()
    }
    if len(input_hashes) != 1 or len(data_hashes) != 1:
        raise RuntimeError("outer builders disagree on input or parsed-data fingerprints")
    return {
        "schema": "caa_rahc_frozen_outer_manifest_v1",
        "experiment_name": config.get("experiment_name", "CAA-RAHC"),
        **identity,
        "input_sha256": next(iter(input_hashes)),
        "data_sha256": next(iter(data_hashes)),
        "outer_splits": [int(value) for value in config["outer_splits"]],
        "model_seeds": [int(value) for value in config["model_seeds"]],
        "protocols": protocols,
        "output_layout": "outer{1,2,3}/runs/full/seed{0,1,2}",
        "fresh_weight_policy": "resume=false; unaudited or partial checkpoints are rejected",
        "selection_lock_contract": {
            "path": SELECTION_LOCK_NAME,
            "required_before": "generate-test",
            "required_hashes": [
                "config_sha256",
                "base_code_sha256",
                "code_sha256",
                "manifest_sha256",
                "protocol_sha256",
            ],
        },
        "scope_statement": config.get("scope_statement"),
    }


def audit_splits(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    bundles = {
        int(outer): build_nested_gefcom2014(data_root(config), outer=int(outer))
        for outer in config["outer_splits"]
    }
    payload = _manifest_payload(config_path, config, bundles)
    path = output_root(config) / FROZEN_MANIFEST_NAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"existing frozen manifest does not match current config/code/data: {path}"
            )
    else:
        _write_json_atomic(path, payload)
    print(
        json.dumps(
            {
                "stage": "audit-splits",
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "protocol_sha256": {
                    key: value["protocol_sha256"] for key, value in payload["protocols"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def load_frozen_manifest(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = output_root(config) / FROZEN_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen manifest is missing; run --phase audit-splits first: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = _manifest_identity(config_path, config)
    for key in ("config_path", "config_sha256", "config_canonical_sha256", "code"):
        if manifest.get(key) != identity[key]:
            raise RuntimeError(f"frozen manifest {key} does not match current state")
    if manifest.get("outer_splits") != [int(value) for value in config["outer_splits"]]:
        raise RuntimeError("frozen manifest outer_splits mismatch")
    if manifest.get("model_seeds") != [int(value) for value in config["model_seeds"]]:
        raise RuntimeError("frozen manifest model_seeds mismatch")
    return manifest


def verify_bundle_against_manifest(
    bundle: NestedDataBundle, manifest: dict[str, Any], outer: int
) -> None:
    expected = manifest.get("protocols", {}).get(f"outer{outer}")
    if expected is None:
        raise RuntimeError(f"outer{outer} protocol is absent from frozen manifest")
    if bundle.protocol != expected:
        raise RuntimeError(f"outer{outer} runtime data protocol differs from frozen manifest")


def _checkpoint_expected_audit(
    checkpoint: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    outer: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "caa_rahc_checkpoint_audit_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "outer": int(outer),
        "model_seed": int(seed),
        "variant": "full",
        "training_step": int(config["training"]["diffusion_steps"]),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_json_sha256(config),
        "code_sha256": manifest["code"]["combined_sha256"],
        "protocol_sha256": bundle.protocol["protocol_sha256"],
        "resume": False,
    }


def _inspect_checkpoint_payload(
    checkpoint: Path,
    *,
    config: dict[str, Any],
    bundle: NestedDataBundle,
    seed: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("name") != "cr_mscadm":
        raise RuntimeError(f"checkpoint has wrong model name: {payload.get('name')!r}")
    if payload.get("variant") != "full" or int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError("checkpoint variant or model seed mismatch")
    if int(payload.get("step", -1)) != int(config["training"]["diffusion_steps"]):
        raise RuntimeError("checkpoint training step mismatch")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("protocol_sha256") != bundle.protocol.get(
        "protocol_sha256"
    ):
        raise RuntimeError("checkpoint protocol_sha256 mismatch")
    if protocol != bundle.protocol:
        raise RuntimeError("checkpoint contains a non-identical data protocol")
    if canonical_json_sha256(payload.get("config")) != canonical_json_sha256(config):
        raise RuntimeError("checkpoint training configuration mismatch")
    if payload["config"].get("training", {}).get("resume") is not False:
        raise RuntimeError("checkpoint was not trained under resume=false")
    return payload


def validate_checkpoint(
    checkpoint: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    outer: int,
    seed: int,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _inspect_checkpoint_payload(checkpoint, config=config, bundle=bundle, seed=seed)
    audit_path = checkpoint_audit_path(checkpoint)
    if not audit_path.is_file():
        raise RuntimeError(f"checkpoint is unaudited and cannot be reused: {audit_path}")
    recorded = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = _checkpoint_expected_audit(
        checkpoint,
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
    )
    if recorded != expected:
        raise RuntimeError("checkpoint audit/hash metadata mismatch; refusing weight reuse")
    return expected


def train_one(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    outer: int,
    seed: int,
    device: str | None,
) -> Path:
    root = output_root(config)
    run = run_directory(root, outer, seed)
    checkpoint = run / "final.pt"
    if checkpoint.exists():
        audit = validate_checkpoint(
            checkpoint,
            config_path=config_path,
            config=config,
            manifest=manifest,
            bundle=bundle,
            outer=outer,
            seed=seed,
        )
        print(json.dumps({"skip": "train", **audit}), flush=True)
        return checkpoint
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(
            f"partial/non-final training artifacts exist in {run}; resume is forbidden, "
            "so use a clean run directory"
        )
    training_config = copy.deepcopy(config)
    training_config["training"]["resume"] = False
    trained = CRTrainer(
        training_config,
        bundle,
        run,
        variant="full",
        seed=seed,
        device=device,
    ).fit()
    trained = Path(trained)
    if trained.resolve() != checkpoint.resolve() or not checkpoint.is_file():
        raise RuntimeError(f"trainer did not produce the registered final checkpoint: {checkpoint}")
    _inspect_checkpoint_payload(checkpoint, config=config, bundle=bundle, seed=seed)
    audit = _checkpoint_expected_audit(
        checkpoint,
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
    )
    _write_json_atomic(checkpoint_audit_path(checkpoint), audit)
    print(json.dumps({"trained": str(checkpoint.resolve()), **audit}), flush=True)
    return checkpoint


def selection_lock_identity(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / FROZEN_MANIFEST_NAME
    return {
        "config_sha256": manifest["config_sha256"],
        "base_code_sha256": manifest["code"]["combined_sha256"],
        "code_sha256": selection_code_fingerprint()["combined_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": {
            key: value["protocol_sha256"] for key, value in manifest["protocols"].items()
        },
    }


def validate_selection_lock(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = root / SELECTION_LOCK_NAME
    if not path.is_file():
        raise RuntimeError(
            f"outer test generation is locked; missing selection lock: {path}"
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = selection_lock_identity(root, manifest)
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"selection lock {key} does not match frozen experiment")
    return lock


def _archive_expected_metadata(
    *,
    generated_metadata: dict[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
    config_path: Path,
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    outer: int,
    seed: int,
    split: str,
    sample_seed: int,
) -> dict[str, Any]:
    metadata = dict(generated_metadata)
    metadata.update(
        {
            "model": "cr_mscadm",
            "variant": "full",
            "training_seed": int(seed),
            "sampling_seed": int(sample_seed),
            "split": split,
            "outer": int(outer),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "split_date_sha256": bundle.protocol["date_sha256"][split],
            "protocol_sha256": bundle.protocol["protocol_sha256"],
            "config_sha256": sha256_file(config_path),
            "code_sha256": manifest["code"]["combined_sha256"],
        }
    )
    return metadata


def _load_archive_metadata(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise RuntimeError("archive metadata must be a scalar JSON string")
    return json.loads(str(value.item()))


def _archive_expected_audit(
    archive: Path,
    *,
    metadata: dict[str, Any],
    scenarios: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": "caa_rahc_scenario_archive_audit_v1",
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "shape": list(scenarios.shape),
        "dtype": str(scenarios.dtype),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "split_date_sha256": metadata["split_date_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "config_sha256": metadata["config_sha256"],
        "code_sha256": metadata["code_sha256"],
        "sampling_seed": int(metadata["sampling_seed"]),
    }


def validate_archive(
    archive: Path,
    *,
    split_data: Any,
    expected_metadata: dict[str, Any],
    scenarios_count: int,
) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    audit_path = archive_audit_path(archive)
    if not audit_path.is_file():
        raise RuntimeError(f"archive hash sidecar is missing: {audit_path}")
    with np.load(archive, allow_pickle=False) as stored:
        required = {"scenarios", "observations", "zone", "day", "metadata"}
        if not required.issubset(stored.files):
            raise RuntimeError(f"archive lacks required arrays: {sorted(required - set(stored.files))}")
        scenarios = stored["scenarios"]
        observations = stored["observations"]
        zone = stored["zone"]
        day = stored["day"]
        metadata = _load_archive_metadata(stored["metadata"])
    expected_shape = (len(split_data), int(scenarios_count), 24)
    if scenarios.shape != expected_shape:
        raise RuntimeError(f"archive scenario shape {scenarios.shape} != {expected_shape}")
    if not np.isfinite(scenarios).all() or float(scenarios.min()) < 0.0 or float(
        scenarios.max()
    ) > 1.0:
        raise RuntimeError("archive scenarios are non-finite or outside [0,1]")
    if not np.array_equal(observations, split_data.target):
        raise RuntimeError("archive observations do not match frozen split")
    if not np.array_equal(zone, split_data.zone) or not np.array_equal(day, split_data.day):
        raise RuntimeError("archive zone/day identity does not match frozen split")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"archive metadata mismatch for {key}")
    recorded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_audit = _archive_expected_audit(
        archive, metadata=metadata, scenarios=scenarios
    )
    if recorded_audit != expected_audit:
        raise RuntimeError("archive content hash/audit sidecar mismatch")
    return expected_audit


def generate_one(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: NestedDataBundle,
    outer: int,
    seed: int,
    split: str,
    device: str | None,
) -> Path:
    if split not in {"calibration", "test"}:
        raise ValueError("split must be calibration or test")
    root = output_root(config)
    if split == "test":
        validate_selection_lock(root, manifest)
    checkpoint = run_directory(root, outer, seed) / "final.pt"
    checkpoint_audit = validate_checkpoint(
        checkpoint,
        config_path=config_path,
        config=config,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
    )
    sample_seed = sampling_seed(outer, seed, split)
    sampling = config["sampling"]
    generated_template = {
        "model": "cr_mscadm",
        "variant": "full",
        "training_seed": int(seed),
        "sampling_seed": int(sample_seed),
        "split": split,
        "scenarios": int(sampling["scenarios"]),
        "sampling_steps": int(sampling["steps"]),
        "sampler": "ddim",
        "eta": float(sampling["eta"]),
        "checkpoint": str(checkpoint.resolve()),
    }
    expected_metadata = _archive_expected_metadata(
        generated_metadata=generated_template,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_audit["checkpoint_sha256"],
        config_path=config_path,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
        split=split,
        sample_seed=sample_seed,
    )
    archive = scenario_archive_path(root, outer, seed, split)
    if archive.exists():
        audit = validate_archive(
            archive,
            split_data=getattr(bundle, split),
            expected_metadata=expected_metadata,
            scenarios_count=int(sampling["scenarios"]),
        )
        print(json.dumps({"skip": f"generate-{split}", **audit}), flush=True)
        return archive
    if archive_audit_path(archive).exists():
        raise RuntimeError(f"orphan archive audit sidecar blocks generation: {archive_audit_path(archive)}")
    scenarios, generated_metadata = generate_cr_scenarios(
        checkpoint,
        bundle,
        split_name=split,
        scenarios=int(sampling["scenarios"]),
        steps=int(sampling["steps"]),
        eta=float(sampling["eta"]),
        day_batch=int(sampling.get("day_batch", 8)),
        seed=sample_seed,
        device=device,
    )
    metadata = _archive_expected_metadata(
        generated_metadata=generated_metadata,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_audit["checkpoint_sha256"],
        config_path=config_path,
        manifest=manifest,
        bundle=bundle,
        outer=outer,
        seed=seed,
        split=split,
        sample_seed=sample_seed,
    )
    temporary = archive.with_suffix(".tmp.npz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        raise RuntimeError(f"stale temporary archive blocks generation: {temporary}")
    save_scenarios(temporary, scenarios, getattr(bundle, split), metadata)
    temporary.replace(archive)
    audit = _archive_expected_audit(archive, metadata=metadata, scenarios=scenarios)
    _write_json_atomic(archive_audit_path(archive), audit)
    validate_archive(
        archive,
        split_data=getattr(bundle, split),
        expected_metadata=expected_metadata,
        scenarios_count=int(sampling["scenarios"]),
    )
    print(json.dumps({"generated": str(archive.resolve()), **audit}), flush=True)
    return archive


def _selected_values(
    configured: Iterable[int], requested: int | None, *, label: str
) -> list[int]:
    values = [int(value) for value in configured]
    if requested is None:
        return values
    if requested not in values:
        raise ValueError(f"requested {label} {requested} is not registered in {values}")
    return [int(requested)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audited CAA-RAHC outer base-model stages")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--phase",
        required=True,
        choices=["audit-splits", "train", "generate-calibration", "generate-test"],
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path, config = load_experiment_config(args.config)
    if args.phase == "audit-splits":
        audit_splits(config_path, config)
        return
    manifest = load_frozen_manifest(config_path, config)
    outers = _selected_values(config["outer_splits"], args.outer, label="outer")
    seeds = _selected_values(config["model_seeds"], args.seed, label="seed")
    for outer in outers:
        bundle = build_nested_gefcom2014(data_root(config), outer=outer)
        verify_bundle_against_manifest(bundle, manifest, outer)
        for seed in seeds:
            if args.phase == "train":
                train_one(
                    config_path=config_path,
                    config=config,
                    manifest=manifest,
                    bundle=bundle,
                    outer=outer,
                    seed=seed,
                    device=args.device,
                )
            else:
                generate_one(
                    config_path=config_path,
                    config=config,
                    manifest=manifest,
                    bundle=bundle,
                    outer=outer,
                    seed=seed,
                    split="calibration" if args.phase == "generate-calibration" else "test",
                    device=args.device,
                )


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_MANIFEST_NAME",
    "SELECTION_LOCK_NAME",
    "archive_audit_path",
    "audit_splits",
    "canonical_json_sha256",
    "checkpoint_audit_path",
    "code_fingerprint",
    "generate_one",
    "load_experiment_config",
    "load_frozen_manifest",
    "run_directory",
    "sampling_seed",
    "scenario_archive_path",
    "selection_code_fingerprint",
    "selection_lock_identity",
    "sha256_file",
    "train_one",
    "validate_archive",
    "validate_checkpoint",
    "validate_selection_lock",
    "verify_bundle_against_manifest",
]



