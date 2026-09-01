#!/usr/bin/env python3
"""Run family-v1.2 Temporal Utility Probe P0 or retained adapter training.

The default command is read-only and prints the frozen 48-run matrix.  P0 is
train-only and disposable.  Retained adapter training is authorized only by a
verified P0_GO and never materializes validation, calibration, selection,
R-SEEN, or final targets.  Formal validation scenario generation is a separate
future boundary and is deliberately not implemented in this runner.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1_2_probe.json"
DATA_PROVENANCE_AMENDMENT = (
    ROOT
    / "repro_configs"
    / "architecture_v1_family_v1_2_data_provenance_amendment.json"
)
P0_SCHEMA = "architecture_v1_family_v1_2_P0_result_v1"
TRAINING_SCHEMA = "architecture_v1_family_v1_2_adapter_training_result_v1"
COMPLETION_SCHEMA = "architecture_v1_family_v1_2_adapter_run_completion_v1"
RUNNER_STATE_SCHEMA = "architecture_v1_family_v1_2_adapter_runner_state_v1"

from architecture_v1.data import build_architecture_v1_train_data
from architecture_v1.family_diffusion_v1_1 import VPredictionJointDDPM
from architecture_v1.family_v1_2_probe import (
    DDIMStageBlock,
    ProbeEMATrainer,
    TemporalContextResidualAdapter,
    TemporalUtilityProbe,
    fit_nwp_dynamicity_registry,
    load_frozen_d0_v_best_ema,
    permutation_bank_sha256,
    validate_permutation_bank,
    validate_stage_registry,
)
from architecture_v1.model import T0StableSourceRectifiedFlow
from architecture_v1.training import (
    ArchitectureBatch,
    canonical_sha256,
    file_sha256,
    tensor_state_sha256,
)
import repro_scripts.run_architecture_v1_family_v1 as family_v1
from repro_scripts.run_architecture_v1_family_v1_1_p0 import (
    _load_config as load_family_v1_1_config,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def _verified(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != str(expected):
        raise RuntimeError(f"frozen file SHA256 mismatch: {path}")
    return actual


def _verified_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"file or SHA256 sidecar missing: {path}")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
    return _verified(path, pieces[0])


def _diffusion(config: Mapping[str, Any], device: torch.device) -> VPredictionJointDDPM:
    contract = config["diffusion_contract"]
    return VPredictionJointDDPM(
        timesteps=int(contract["training_timesteps"]),
        cosine_offset=float(contract["cosine_offset"]),
        beta_min=float(contract["beta_clip"][0]),
        beta_max=float(contract["beta_clip"][1]),
    ).to(device)


def _load_data_provenance_amendment(config_path: Path) -> dict[str, Any]:
    """Validate the hashed bridge from current metadata to D0-v lineage.

    One R-SEEN source JSON acquired extra audit metadata after the D0-v runs.
    Its consumed date union is unchanged.  The amendment makes that fact an
    explicit, fail-closed provenance rule instead of weakening protocol checks.
    """

    _verified_sidecar(DATA_PROVENANCE_AMENDMENT)
    amendment = _read(DATA_PROVENANCE_AMENDMENT)
    if amendment.get("schema") != (
        "architecture_v1_family_v1_2_data_provenance_amendment_v1"
    ):
        raise ValueError("unexpected family-v1.2 data-provenance amendment schema")
    if amendment.get("status") != (
        "frozen_after_protocol_fail_closed_before_any_P0_optimizer_update"
    ):
        raise RuntimeError("family-v1.2 data-provenance amendment is not frozen")
    base = (ROOT / amendment["base_probe_config"]).resolve()
    if base != config_path.resolve():
        raise RuntimeError("data-provenance amendment binds a different probe config")
    _verified(base, amendment["base_probe_config_sha256"])

    legacy = amendment["legacy_D0_v_lineage"]
    p0_identity_path = ROOT / legacy["family_v1_1_P0_identity"]
    training_identity_path = ROOT / legacy["family_v1_1_training_identity"]
    _verified(p0_identity_path, legacy["family_v1_1_P0_identity_sha256"])
    _verified(
        training_identity_path,
        legacy["family_v1_1_training_identity_sha256"],
    )
    p0_identity = _read(p0_identity_path)
    training_identity = _read(training_identity_path)
    for actual, expected, label in (
        (
            p0_identity.get("protocol_sha256"),
            legacy["protocol_sha256"],
            "legacy P0 protocol",
        ),
        (
            p0_identity.get("train_only_data_bundle_sha256"),
            legacy["train_only_data_bundle_sha256"],
            "legacy train-only bundle",
        ),
        (
            p0_identity.get("train_split_array_sha256"),
            legacy["train_split_array_sha256"],
            "legacy train split",
        ),
        (
            training_identity.get("protocol_sha256"),
            legacy["protocol_sha256"],
            "legacy training protocol",
        ),
        (
            training_identity.get("fit_data_bundle_sha256"),
            legacy["fit_data_bundle_sha256"],
            "legacy fit bundle",
        ),
    ):
        if actual != expected:
            raise RuntimeError(f"{label} evidence drifted")

    drift = amendment["audit_source_drift"]
    _verified(ROOT / drift["path"], drift["current_repository_file_sha256"])
    evidence_path = ROOT / drift["legacy_hash_evidence"]
    _verified(evidence_path, drift["legacy_hash_evidence_sha256"])
    evidence = _read(evidence_path)
    matches = [
        value
        for value in evidence["r_seen_sources"]
        if value["path"] == drift["path"]
    ]
    if len(matches) != 1:
        raise RuntimeError("legacy R-SEEN hash evidence is missing or ambiguous")
    source = matches[0]
    for key, expected in (
        ("schema", drift["schema"]),
        ("file_sha256", drift["legacy_file_sha256"]),
        ("date_count", drift["extracted_date_count"]),
        ("date_sha256", drift["extracted_date_sha256"]),
    ):
        if source.get(key) != expected:
            raise RuntimeError(f"legacy R-SEEN hash evidence drifted: {key}")
    authorization = amendment["authorization"]
    if authorization[
        "P0_and_adapter_training_may_use_current_repository_protocol_after_all_bridge_checks_pass"
    ] is not True:
        raise RuntimeError("data-provenance amendment does not authorize P0/training")
    if any(
        authorization[key]
        for key in (
            "validation_access_authorized",
            "selection_access_authorized",
            "calibration_access_authorized",
            "external_final_access_authorized",
        )
    ):
        raise RuntimeError("data-provenance amendment opened a sealed target role")
    return amendment


def _validate_train_data_provenance_bridge(
    bundle: Any,
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact train-array compatibility without rewriting source data."""

    legacy = amendment["legacy_D0_v_lineage"]
    current = amendment["current_repository_identity"]
    drift = amendment["audit_source_drift"]
    projection = amendment["equivalence_projection"]
    protocol = bundle.protocol.manifest
    manifest = bundle.manifest
    source_records = [
        value for value in protocol["r_seen_sources"] if value["path"] == drift["path"]
    ]
    if len(source_records) != 1:
        raise RuntimeError("current protocol R-SEEN source is missing or ambiguous")
    source = source_records[0]

    checks = {
        "current_protocol_sha256": protocol["protocol_sha256"]
        == current["protocol_sha256"],
        "current_train_only_bundle_sha256": manifest[
            "train_only_data_bundle_sha256"
        ]
        == current["train_only_data_bundle_sha256"],
        "train_split_array_sha256": manifest["data_audit"]["split_array_sha256"][
            "train"
        ]
        == current["train_split_array_sha256"]
        == legacy["train_split_array_sha256"],
        "source_file_sha256": source["file_sha256"]
        == drift["current_repository_file_sha256"],
        "source_schema": source["schema"] == drift["schema"],
        "source_date_count": source["date_count"] == drift["extracted_date_count"],
        "source_date_sha256": source["date_sha256"]
        == drift["extracted_date_sha256"],
        "base_config_legacy_protocol": config["lineage"]["protocol_sha256"]
        == legacy["protocol_sha256"],
        "base_config_legacy_fit_bundle": config["lineage"][
            "fit_data_bundle_sha256"
        ]
        == legacy["fit_data_bundle_sha256"],
    }

    projected_protocol = copy.deepcopy(protocol)
    projected_source = next(
        value
        for value in projected_protocol["r_seen_sources"]
        if value["path"] == drift["path"]
    )
    projected_source["file_sha256"] = drift["legacy_file_sha256"]
    projected_protocol.pop("protocol_sha256")
    projected_protocol_sha = canonical_sha256(projected_protocol)
    checks["projected_protocol_sha256"] = (
        projected_protocol_sha
        == projection["projected_protocol_sha256"]
        == legacy["protocol_sha256"]
    )

    projected_manifest = copy.deepcopy(manifest)
    projected_bundle_source = next(
        value
        for value in projected_manifest["r_seen_sources"]
        if value["path"] == drift["path"]
    )
    projected_bundle_source["file_sha256"] = drift["legacy_file_sha256"]
    projected_manifest["protocol_sha256"] = projected_protocol_sha
    projected_manifest.pop("train_only_data_bundle_sha256")
    projected_bundle_sha = canonical_sha256(projected_manifest)
    checks["projected_train_only_bundle_sha256"] = (
        projected_bundle_sha
        == projection["projected_train_only_data_bundle_sha256"]
        == legacy["train_only_data_bundle_sha256"]
    )
    passed = all(checks.values())
    if not passed:
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"family-v1.2 data-provenance bridge failed: {failed}")
    return {
        "schema": "architecture_v1_family_v1_2_train_data_bridge_audit_v1",
        "passed": True,
        "amendment": str(DATA_PROVENANCE_AMENDMENT.resolve()),
        "amendment_sha256": file_sha256(DATA_PROVENANCE_AMENDMENT),
        "current_protocol_sha256": protocol["protocol_sha256"],
        "legacy_D0_v_protocol_sha256": legacy["protocol_sha256"],
        "current_train_only_data_bundle_sha256": manifest[
            "train_only_data_bundle_sha256"
        ],
        "projected_legacy_train_only_data_bundle_sha256": projected_bundle_sha,
        "train_split_array_sha256": manifest["data_audit"]["split_array_sha256"][
            "train"
        ],
        "changed_model_facing_arrays": 0,
        "checks": checks,
    }


def _load_config(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[DDIMStageBlock, ...],
    tuple[tuple[int, ...], ...],
]:
    config = _read(path)
    if config.get("schema") != "architecture_v1_family_v1_2_temporal_utility_probe":
        raise ValueError("unexpected family-v1.2 probe schema")
    if config.get("status") != (
        "frozen_before_family_v1_2_implementation_P0_or_retained_adapter_training"
    ):
        raise RuntimeError("family-v1.2 protocol is not frozen at the P0 boundary")
    sidecar = path.with_name(path.name + ".sha256")
    pieces = sidecar.read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1] != path.name:
        raise ValueError("family-v1.2 config sidecar is malformed")
    _verified(path, pieces[0])
    amendment = _load_data_provenance_amendment(path)

    lineage = config["lineage"]
    for path_key, hash_key in (
        ("base_family_v1_1_config", "base_family_v1_1_config_sha256"),
        ("family_v1_1_training_freeze", "family_v1_1_training_freeze_sha256"),
        ("family_v1_1_comparison", "family_v1_1_comparison_sha256"),
        ("data_protocol_config", "data_protocol_config_sha256"),
    ):
        _verified(ROOT / lineage[path_key], lineage[hash_key])
    freeze = _read(ROOT / lineage["family_v1_1_training_freeze"])
    comparison = _read(ROOT / lineage["family_v1_1_comparison"])
    required = lineage["required_family_v1_1_status"]
    if freeze.get("status") != required or comparison.get("status") != required:
        raise RuntimeError("family-v1.1 did not authorize the temporal probe")
    if freeze.get("training_closed") is not True:
        raise RuntimeError("family-v1.1 training/evaluation is not frozen")
    if comparison.get("selected_family") is not None:
        raise RuntimeError("family-v1.2 cannot overwrite a formal family winner")

    roles = config["role_access"]
    if roles["P0_allowed_target_roles"] != ["train"]:
        raise RuntimeError("P0 target-role allowlist changed")
    if roles["adapter_training_allowed_target_roles"] != ["train"]:
        raise RuntimeError("adapter-training target-role allowlist changed")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise RuntimeError("selection/calibration must remain sealed")
    if roles["selection_access_authorized"] or roles["calibration_access_authorized"]:
        raise RuntimeError("family-v1.2 cannot authorize sealed-role access")

    base_path = ROOT / lineage["base_family_v1_1_config"]
    base_config, model_config = load_family_v1_1_config(base_path)
    if base_config["lineage"]["protocol_sha256"] != lineage["protocol_sha256"]:
        raise RuntimeError("family-v1.2 protocol lineage drifted")
    if base_config["lineage"]["fit_data_bundle_sha256"] != lineage[
        "fit_data_bundle_sha256"
    ]:
        raise RuntimeError("family-v1.2 fit-data lineage drifted")
    if amendment["legacy_D0_v_lineage"]["protocol_sha256"] != lineage[
        "protocol_sha256"
    ]:
        raise RuntimeError("data amendment protocol lineage differs from probe config")
    if amendment["legacy_D0_v_lineage"]["fit_data_bundle_sha256"] != lineage[
        "fit_data_bundle_sha256"
    ]:
        raise RuntimeError("data amendment fit-data lineage differs from probe config")

    diffusion = _diffusion(config, torch.device("cpu"))
    stages = validate_stage_registry(
        diffusion,
        steps=int(config["diffusion_contract"]["DDIM_steps"]),
        records=config["DDIM_stage_blocks"],
    )
    reverse = [value for block in stages for value in block.timesteps]
    if reverse != [int(value) for value in config["diffusion_contract"]["reverse_grid"]]:
        raise RuntimeError("family-v1.2 reverse-grid registry drifted")
    shuffle = config["shuffle_control"]
    bank = validate_permutation_bank(
        shuffle["permutation_bank"],
        hours=int(config["adapter"]["hours"]),
        maximum_adjacent_edges=int(
            shuffle["true_or_reverse_adjacent_edges_allowed_per_permutation"]
        ),
    )
    used = tuple(int(value) for value in shuffle["training_permutation_indices"])
    held_out = tuple(
        int(value) for value in shuffle["same_checkpoint_inference_only_indices"]
    )
    if set(used) & set(held_out) or sorted((*used, *held_out)) != list(range(len(bank))):
        raise RuntimeError("training and inference-only permutation registries overlap or omit entries")
    adapter = TemporalContextResidualAdapter(
        int(config["adapter"]["context_dimension"]),
        hidden_dim=int(config["adapter"]["hidden_dimension"]),
        hours=int(config["adapter"]["hours"]),
        residual_scale=float(config["adapter"]["residual_scale"]),
    )
    if adapter.parameter_count() != int(config["adapter"]["trainable_parameters_expected"]):
        raise RuntimeError("family-v1.2 adapter parameter count drifted")
    if not adapter.output_is_exactly_zero():
        raise RuntimeError("family-v1.2 adapter no longer has zero initial output")
    matrix = _training_matrix(config)
    if len(matrix) != int(config["retained_adapter_training"]["total_runs"]):
        raise RuntimeError("family-v1.2 retained run matrix drifted")
    return config, base_config, model_config, stages, bank


def _training_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    training = config["retained_adapter_training"]
    return [
        {
            "backbone_seed": int(seed),
            "stage_block_index": int(block),
            "order_mode": str(order),
        }
        for seed in training["backbone_seeds"]
        for block in training["stage_block_indices"]
        for order in training["order_modes"]
    ]


def _code_manifest(config_path: Path) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        config_path.resolve(),
        DATA_PROVENANCE_AMENDMENT.resolve(),
        ROOT / "architecture_v1" / "family_v1_2_probe.py",
        ROOT / "architecture_v1" / "family_diffusion.py",
        ROOT / "architecture_v1" / "family_diffusion_v1_1.py",
        ROOT / "architecture_v1" / "model.py",
        ROOT / "architecture_v1" / "data.py",
        ROOT / "architecture_v1" / "training.py",
        ROOT / "tests" / "test_architecture_v1_family_v1_2_probe.py",
        ROOT / "tests" / "test_architecture_v1_family_v1_2_protocol.py",
    ]
    records = [
        {
            "path": value.relative_to(ROOT).as_posix(),
            "bytes": value.stat().st_size,
            "sha256": file_sha256(value),
        }
        for value in paths
    ]
    core = {"schema": "architecture_v1_family_v1_2_code_manifest_v1", "files": records}
    return {**core, "code_sha256": canonical_sha256(core)}


def _model_kwargs(
    base_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    return family_v1._model_kwargs(base_config, model_config)


def _load_backbone(
    *,
    seed: int,
    config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    device: torch.device,
) -> tuple[T0StableSourceRectifiedFlow, dict[str, Any]]:
    torch.manual_seed(1000 + int(seed))
    model = T0StableSourceRectifiedFlow(**kwargs)
    record = config["frozen_D0_v_backbones"][f"seed{int(seed)}"]
    report = load_frozen_d0_v_best_ema(
        model,
        ROOT / record["path"],
        expected_file_sha256=record["sha256"],
    )
    model = model.to(device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("loaded D0-v best EMA is not fully frozen")
    device_hash = tensor_state_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    if device_hash != report["best_EMA_backbone_tensor_sha256"]:
        raise RuntimeError("D0-v best EMA changed while moving to the runtime device")
    return model, report


def _new_probe(
    *,
    seed: int,
    stage: DDIMStageBlock,
    config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    device: torch.device,
) -> tuple[TemporalUtilityProbe, dict[str, Any]]:
    backbone, report = _load_backbone(
        seed=seed, config=config, kwargs=kwargs, device=device
    )
    initialization_seed = 271000 + 100 * int(seed) + int(stage.index)
    torch.manual_seed(initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(initialization_seed)
    adapter = config["adapter"]
    probe = TemporalUtilityProbe(
        backbone,
        _diffusion(config, device),
        stage,
        adapter_hidden_dim=int(adapter["hidden_dimension"]),
        residual_scale=float(adapter["residual_scale"]),
        registered_ddim_steps=int(config["diffusion_contract"]["DDIM_steps"]),
    ).to(device)
    if probe.adapter.parameter_count() != int(adapter["trainable_parameters_expected"]):
        raise RuntimeError("runtime adapter parameter count drifted")
    return probe, {**report, "adapter_initialization_seed": initialization_seed}


def _make_trainer(
    probe: TemporalUtilityProbe,
    *,
    config: Mapping[str, Any],
    backbone_identity: Mapping[str, Any],
) -> ProbeEMATrainer:
    training = config["retained_adapter_training"]
    return ProbeEMATrainer(
        probe,
        learning_rate=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        ema_decay=float(training["EMA_decay"]),
        backbone_identity=backbone_identity,
    )


def _take_batch(split: Any, indices: Sequence[int], device: torch.device) -> ArchitectureBatch:
    return ArchitectureBatch.from_split(split, indices).to(device)


def _epoch_order(length: int, *, epoch: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(int(seed) + int(epoch)).permutation(length)
    order = np.ascontiguousarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(length, dtype=np.int64)):
        raise RuntimeError("adapter epoch order is not a complete permutation")
    return order


def _batch_for_update(
    split: Any,
    *,
    update_zero_based: int,
    batch_days: int,
    order_seed: int,
    device: torch.device,
) -> tuple[ArchitectureBatch, np.ndarray, int, int, int]:
    batches = math.ceil(len(split) / int(batch_days))
    epoch = int(update_zero_based) // batches
    batch_index = int(update_zero_based) % batches
    order = _epoch_order(len(split), epoch=epoch, seed=int(order_seed))
    start = batch_index * int(batch_days)
    indices = order[start : start + int(batch_days)]
    if len(indices) < 1:
        raise RuntimeError("adapter update selected an empty calendar-day batch")
    return _take_batch(split, indices, device), indices, epoch, batch_index, start


def _paired_timestep_and_noise(
    batch: ArchitectureBatch,
    *,
    stage: DDIMStageBlock,
    backbone_seed: int,
    epoch: int,
    batch_start: int,
    train_days: int,
    update_zero_based: int,
    seed_root: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    choices = torch.as_tensor(
        stage.timesteps, dtype=torch.long, device=batch.condition.device
    )
    # Across the exact 120 full epochs, every block point receives exactly the
    # same number of calendar-day exposures (32040 is divisible by 3 and 4).
    offset = family_v1._stable_seed(
        "family-v1.2-stage-cycle", seed_root, backbone_seed, stage.index
    ) % len(choices)
    ordinal = (
        int(epoch) * int(train_days)
        + int(batch_start)
        + torch.arange(len(batch.condition), device=batch.condition.device)
    )
    timestep = choices.index_select(0, (ordinal + int(offset)).remainder(len(choices)).long())
    generator = torch.Generator(device=batch.condition.device)
    generator.manual_seed(
        family_v1._stable_seed(
            "family-v1.2-forward-noise",
            seed_root,
            backbone_seed,
            stage.index,
            update_zero_based,
        )
    )
    noise = torch.randn(
        batch.target.shape,
        dtype=batch.target.dtype,
        device=batch.target.device,
        generator=generator,
    )
    return timestep, noise


def _shuffle_order(
    *,
    bank: Sequence[Sequence[int]],
    training_indices: Sequence[int],
    backbone_seed: int,
    stage_index: int,
    update_zero_based: int,
) -> tuple[int, tuple[int, ...]]:
    selection = family_v1._stable_seed(
        "family-v1.2-shuffle-bank",
        backbone_seed,
        stage_index,
        update_zero_based,
    ) % len(training_indices)
    bank_index = int(training_indices[int(selection)])
    return bank_index, tuple(int(value) for value in bank[bank_index])


def _adapter_hash(probe: TemporalUtilityProbe) -> str:
    return tensor_state_sha256(
        {name: value.detach().cpu() for name, value in probe.adapter.state_dict().items()}
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _zero_initialization_sampling_audit(
    probe: TemporalUtilityProbe,
    batch: ArchitectureBatch,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    condition = batch.condition[:1]
    members = 3
    member_chunk = 1
    expected = (1, members, probe.backbone.zones, probe.backbone.hours)
    initial = torch.linspace(
        -1.25,
        1.25,
        int(np.prod(expected)),
        dtype=condition.dtype,
        device=condition.device,
    ).reshape(expected)
    seed = 28100
    baseline = probe.diffusion.sample_ddim(
        probe.backbone,
        condition,
        members=members,
        steps=probe.registered_ddim_steps,
        seed=seed,
        member_chunk=member_chunk,
        initial_noise=initial,
    )
    adapted = probe.sample_ddim(
        condition,
        members=members,
        steps=probe.registered_ddim_steps,
        seed=seed,
        member_chunk=member_chunk,
        allocation=baseline.atom_allocation,
        initial_noise=initial,
        hour_order=None,
        intervention=True,
    )
    chunked = probe.sample_ddim(
        condition,
        members=members,
        steps=probe.registered_ddim_steps,
        seed=seed,
        member_chunk=members,
        allocation=baseline.atom_allocation,
        initial_noise=initial,
        hour_order=None,
        intervention=True,
    )
    exact = all(
        (
            torch.equal(baseline.values, adapted.values),
            torch.equal(baseline.states, adapted.states),
            torch.equal(baseline.interior_latent, adapted.interior_latent),
            torch.equal(
                baseline.atom_statistics.probabilities,
                adapted.atom_statistics.probabilities,
            ),
        )
    )
    chunk_equivalent = torch.allclose(
        adapted.interior_latent, chunked.interior_latent, atol=2e-6, rtol=2e-6
    )
    # The probe records context calls across all member chunks.  With a
    # one-member chunk, each registered DDIM point is visited once per member.
    member_chunks = math.ceil(members / member_chunk)
    expected_intervention_calls = len(probe.stage_block.timesteps) * member_chunks
    expected_baseline_calls = (
        probe.registered_ddim_steps - len(probe.stage_block.timesteps)
    ) * member_chunks
    return {
        "zero_initialized_adapter_output_exact": probe.adapter.output_is_exactly_zero(),
        "full_31_step_scenario_exactly_matches_D0_v": exact,
        "member_chunk_equivalent": bool(chunk_equivalent),
        "context_residual_abs_max": adapted.context_residual_abs_max,
        "intervention_timesteps": list(adapted.intervention_timesteps),
        "expected_intervention_timesteps": list(probe.stage_block.timesteps),
        "member_chunks": member_chunks,
        "intervention_calls_all_chunks": adapted.context_intervention_calls,
        "expected_intervention_calls_all_chunks": expected_intervention_calls,
        "baseline_calls_all_chunks": adapted.context_baseline_calls,
        "expected_baseline_calls_all_chunks": expected_baseline_calls,
        "interior_decoded_boundary_count": int(
            torch.count_nonzero(
                (adapted.values[adapted.active_mask] <= 0.0)
                | (adapted.values[adapted.active_mask] >= 1.0)
            ).item()
        ),
        "passed": bool(
            probe.adapter.output_is_exactly_zero()
            and exact
            and chunk_equivalent
            and adapted.context_residual_abs_max == 0.0
            and adapted.intervention_timesteps == probe.stage_block.timesteps
            and adapted.context_intervention_calls == expected_intervention_calls
            and adapted.context_baseline_calls == expected_baseline_calls
        ),
    }


def _optimizer_adapter_only(trainer: ProbeEMATrainer) -> bool:
    adapter_ids = {id(parameter) for parameter in trainer.probe.adapter.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    return adapter_ids == optimizer_ids and all(
        not parameter.requires_grad for parameter in trainer.probe.backbone.parameters()
    )


def _execute_p0(config_path: Path) -> dict[str, Any]:
    config, base_config, model_config, stages, bank = _load_config(config_path)
    amendment = _load_data_provenance_amendment(config_path)
    output_root = ROOT / config["output_root"] / "P0_preflight"
    result_path = output_root / "P0_RESULT.json"
    if result_path.exists():
        _verified_sidecar(result_path)
        existing = _read(result_path)
        if existing.get("schema") == P0_SCHEMA and existing.get("status") == "P0_GO":
            return existing
        raise RuntimeError("existing family-v1.2 P0 is not a verified GO")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("non-empty family-v1.2 P0 output refuses overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    print("[family-v1.2 P0] validating frozen protocol, train-only data, and CUDA")
    device, runtime = family_v1._configure_cuda()
    code = _code_manifest(config_path)
    data_config = ROOT / config["lineage"]["data_protocol_config"]
    bundle = build_architecture_v1_train_data(config_path=data_config)
    if bundle.materialized_roles != ("train",):
        raise RuntimeError("family-v1.2 P0 materialized a forbidden target role")
    data_bridge = _validate_train_data_provenance_bridge(bundle, config, amendment)
    target_access = bundle.manifest["formal_train_only_target_access"]
    if target_access["materialized_roles"] != ["train"] or target_access[
        "forbidden_target_arrays_materialized"
    ]:
        raise RuntimeError("family-v1.2 P0 target-access audit failed")
    nwp_registry = fit_nwp_dynamicity_registry(bundle.train.raw_condition)
    nwp_registry_sha = _atomic_json(
        output_root / "NWP_DYNAMICITY_REGISTRY.json", nwp_registry
    )
    kwargs = _model_kwargs(base_config, model_config)
    p0 = config["P0_preflight"]
    seed = int(p0["pilot_backbone_seed"])
    stage = stages[int(p0["pilot_stage_block_index"])]
    updates = int(p0["discarded_adapter_updates_per_order_mode"])
    training = config["retained_adapter_training"]
    batch_days = int(training["batch_calendar_days"])
    order_seed = int(training["common_calendar_day_order_seed"])
    noise_root = int(training["common_timestep_and_noise_seed_root"])
    training_bank_indices = tuple(
        int(value) for value in config["shuffle_control"]["training_permutation_indices"]
    )
    held_out_index = int(
        config["shuffle_control"]["same_checkpoint_inference_only_indices"][0]
    )

    # Load and validate all three retained checkpoint files before training.
    print("[family-v1.2 P0] validating all three frozen D0-v best-EMA checkpoints")
    checkpoint_reports: dict[str, Any] = {}
    for backbone_seed in training["backbone_seeds"]:
        model, report = _load_backbone(
            seed=int(backbone_seed),
            config=config,
            kwargs=kwargs,
            device=device,
        )
        checkpoint_reports[str(backbone_seed)] = report
        del model
        torch.cuda.empty_cache()

    chronological, chrono_backbone = _new_probe(
        seed=seed, stage=stage, config=config, kwargs=kwargs, device=device
    )
    shuffled, shuffle_backbone = _new_probe(
        seed=seed, stage=stage, config=config, kwargs=kwargs, device=device
    )
    initial_adapter_hash = _adapter_hash(chronological)
    if _adapter_hash(shuffled) != initial_adapter_hash:
        raise RuntimeError("chronological/shuffle adapter initial tensors differ")
    chrono_trainer = _make_trainer(
        chronological, config=config, backbone_identity=chrono_backbone
    )
    shuffle_trainer = _make_trainer(
        shuffled, config=config, backbone_identity=shuffle_backbone
    )
    if not _optimizer_adapter_only(chrono_trainer) or not _optimizer_adapter_only(
        shuffle_trainer
    ):
        raise RuntimeError("P0 optimizer includes a non-adapter parameter")
    first_batch, _, _, _, _ = _batch_for_update(
        bundle.train,
        update_zero_based=0,
        batch_days=batch_days,
        order_seed=order_seed,
        device=device,
    )
    sampling = _zero_initialization_sampling_audit(
        chronological, first_batch, config
    )
    if sampling["passed"] is not True:
        raise RuntimeError("zero-initialized probe sampling contract failed")
    print("[family-v1.2 P0] zero-init DDIM equivalence passed; starting 50 paired updates")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    chrono_metrics: list[dict[str, float]] = []
    shuffle_metrics: list[dict[str, float]] = []
    permutation_indices: list[int] = []
    timestep_counts = {
        "chronological": {str(value): 0 for value in stage.timesteps},
        "shuffle": {str(value): 0 for value in stage.timesteps},
    }
    resume_report: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="family_v1_2_p0_", dir="/tmp") as directory:
        checkpoint = Path(directory) / "resume_audit.pt"
        resume_identity = {
            "schema": "family_v1_2_P0_resume_identity_v1",
            "backbone_seed": seed,
            "stage_block_index": stage.index,
            "order_mode": "chronological",
            "config_sha256": file_sha256(config_path),
        }
        resumed: ProbeEMATrainer | None = None
        for update_zero in range(updates):
            batch, _, epoch, _, batch_start = _batch_for_update(
                bundle.train,
                update_zero_based=update_zero,
                batch_days=batch_days,
                order_seed=order_seed,
                device=device,
            )
            timestep, noise = _paired_timestep_and_noise(
                batch,
                stage=stage,
                backbone_seed=seed,
                epoch=epoch,
                batch_start=batch_start,
                train_days=len(bundle.train),
                update_zero_based=update_zero,
                seed_root=noise_root,
            )
            bank_index, shuffled_order = _shuffle_order(
                bank=bank,
                training_indices=training_bank_indices,
                backbone_seed=seed,
                stage_index=stage.index,
                update_zero_based=update_zero,
            )
            permutation_indices.append(bank_index)
            for value in timestep.detach().cpu().tolist():
                timestep_counts["chronological"][str(int(value))] += 1
                timestep_counts["shuffle"][str(int(value))] += 1
            if update_zero == 25:
                resumed_probe, resumed_backbone = _new_probe(
                    seed=seed,
                    stage=stage,
                    config=config,
                    kwargs=kwargs,
                    device=device,
                )
                resumed = _make_trainer(
                    resumed_probe,
                    config=config,
                    backbone_identity=resumed_backbone,
                )
                payload = resumed.load_checkpoint(
                    checkpoint,
                    expected_identity=resume_identity,
                    restore_rng=True,
                )
                if int(payload["optimizer_updates"]) != 25:
                    raise RuntimeError("P0 resume checkpoint has the wrong update count")
            chrono_metric = chrono_trainer.train_step(
                batch,
                hour_order=None,
                timestep=timestep,
                noise=noise,
            )
            shuffle_metric = shuffle_trainer.train_step(
                batch,
                hour_order=shuffled_order,
                timestep=timestep,
                noise=noise,
            )
            chrono_metrics.append(chrono_metric)
            shuffle_metrics.append(shuffle_metric)
            if update_zero == 24:
                chrono_trainer.save_checkpoint(
                    checkpoint,
                    identity=resume_identity,
                    runner_state={"next_update_zero_based": 25},
                )
            if update_zero == 25:
                assert resumed is not None
                replay = resumed.train_step(
                    batch,
                    hour_order=None,
                    timestep=timestep,
                    noise=noise,
                )
                exact_metrics = all(
                    replay[name] == chrono_metric[name]
                    for name in (
                        "loss",
                        "v_mse",
                        "gradient_norm",
                        "output_projection_gradient_norm",
                        "optimizer_updates",
                    )
                )
                resume_report = {
                    "checkpoint_file_was_temporary": True,
                    "next_update_metrics_exact": exact_metrics,
                    "adapter_tensor_exact": _adapter_hash(resumed.probe)
                    == _adapter_hash(chronological),
                    "EMA_tensor_exact": resumed.ema.tensor_sha256()
                    == chrono_trainer.ema.tensor_sha256(),
                    "optimizer_state_exact": _nested_equal(
                        resumed.optimizer.state_dict(),
                        chrono_trainer.optimizer.state_dict(),
                    ),
                    "restored_update_count": resumed.optimizer_updates,
                }
                resume_report["passed"] = all(
                    value
                    for key, value in resume_report.items()
                    if key not in ("restored_update_count",)
                ) and resume_report["restored_update_count"] == 26
            if (update_zero + 1) % 10 == 0 or update_zero + 1 == updates:
                print(
                    f"[family-v1.2 P0] paired updates {update_zero + 1}/{updates}"
                )

    if resume_report is None or resume_report["passed"] is not True:
        raise RuntimeError("P0 checkpoint-resume audit failed")
    chronological.assert_backbone_unchanged()
    shuffled.assert_backbone_unchanged()
    if chrono_trainer.optimizer_updates != updates or shuffle_trainer.optimizer_updates != updates:
        raise RuntimeError("P0 paired adapter update counts differ")
    if timestep_counts["chronological"] != timestep_counts["shuffle"]:
        raise RuntimeError("P0 paired timestep exposure counts differ")
    if chrono_metrics[0]["output_projection_gradient_norm"] <= 0.0:
        raise RuntimeError("frozen flow did not preserve adapter context autograd")

    # Same-checkpoint counterfactual: only adjacency changes.
    with chrono_trainer.ema_weights():
        chronological.eval()
        with torch.no_grad():
            _, base_context, chrono_residual, _ = chronological._contexts(
                first_batch.condition[:1], hour_order=None
            )
            _, same_base, heldout_residual, _ = chronological._contexts(
                first_batch.condition[:1], hour_order=bank[held_out_index]
            )
    same_checkpoint = {
        "held_out_permutation_index": held_out_index,
        "base_context_exactly_same": bool(torch.equal(base_context, same_base)),
        "residual_shape_same": chrono_residual.shape == heldout_residual.shape,
        "residual_finite": bool(
            torch.isfinite(chrono_residual).all() and torch.isfinite(heldout_residual).all()
        ),
        "order_switch_changes_residual": bool(
            not torch.equal(chrono_residual, heldout_residual)
        ),
    }
    same_checkpoint["passed"] = all(
        same_checkpoint[key]
        for key in (
            "base_context_exactly_same",
            "residual_shape_same",
            "residual_finite",
            "order_switch_changes_residual",
        )
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    maximum_memory = int(float(p0["maximum_peak_GPU_memory_GiB"]) * (1024**3))
    bank_sha = permutation_bank_sha256(bank)
    checks = {
        "zero_initialization_sampling": sampling["passed"],
        "all_three_best_EMA_backbones_validated": len(checkpoint_reports) == 3
        and all(value["all_parameters_frozen"] for value in checkpoint_reports.values()),
        "backbone_hashes_unchanged": chronological.backbone_tensor_sha256()
        == chronological.frozen_backbone_sha256
        and shuffled.backbone_tensor_sha256() == shuffled.frozen_backbone_sha256,
        "optimizer_and_EMA_adapter_only": _optimizer_adapter_only(chrono_trainer)
        and _optimizer_adapter_only(shuffle_trainer)
        and set(chrono_trainer.ema.parameter_names) == set(chrono_trainer.parameter_names),
        "context_autograd_preserved": chrono_metrics[0][
            "output_projection_gradient_norm"
        ]
        > 0.0,
        "registered_stage_only": sampling["intervention_timesteps"]
        == sampling["expected_intervention_timesteps"],
        "atom_contract_unchanged": sampling["interior_decoded_boundary_count"] == 0,
        "paired_update_and_timestep_counts": chrono_trainer.optimizer_updates
        == shuffle_trainer.optimizer_updates
        and timestep_counts["chronological"] == timestep_counts["shuffle"],
        "permutation_bank_valid": bool(bank_sha),
        "same_checkpoint_switch": same_checkpoint["passed"],
        "checkpoint_resume_exact": resume_report["passed"],
        "peak_memory_within_gate": peak_memory <= maximum_memory,
        "target_roles_train_only": bundle.materialized_roles == ("train",),
        "data_provenance_bridge": data_bridge["passed"],
    }
    passed = all(checks.values())
    result = {
        "schema": P0_SCHEMA,
        "status": "P0_GO" if passed else "P0_NO_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "code_manifest": code,
        "runtime": runtime,
        "materialized_target_roles": ["train"],
        "data_provenance_bridge": data_bridge,
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
        "all_weights_discarded": True,
        "retained_adapter_training_authorized": passed,
        "pilot_backbone_seed": seed,
        "pilot_stage_block": stage.manifest(),
        "discarded_updates_per_order": updates,
        "initial_adapter_tensor_sha256": initial_adapter_hash,
        "checkpoint_reports": checkpoint_reports,
        "sampling_audit": sampling,
        "same_checkpoint_audit": same_checkpoint,
        "resume_audit": resume_report,
        "permutation_bank_sha256": bank_sha,
        "permutation_indices_used": sorted(set(permutation_indices)),
        "timestep_exposure_counts": timestep_counts,
        "NWP_dynamicity_registry": str(
            (output_root / "NWP_DYNAMICITY_REGISTRY.json").resolve()
        ),
        "NWP_dynamicity_registry_file_sha256": nwp_registry_sha,
        "NWP_dynamicity_registry_payload_sha256": nwp_registry["registry_sha256"],
        "peak_GPU_memory_bytes": peak_memory,
        "peak_GPU_memory_limit_bytes": maximum_memory,
        "checks": checks,
        "wall_seconds": float(time.perf_counter() - started),
        "next_action": (
            "freeze_P0_hashes_then_start_48_retained_adapter_runs"
            if passed
            else config["P0_preflight"]["failure"]
        ),
    }
    _atomic_json(result_path, result)
    if list(output_root.rglob("*.pt")) or list(output_root.rglob("*.pth")):
        raise RuntimeError("P0 retained a forbidden weight file")
    if not passed:
        raise RuntimeError("family-v1.2 P0 hard gate failed")
    print(
        "[family-v1.2 P0] GO: train-only, weights discarded, "
        f"peak GPU memory {peak_memory / (1024**3):.2f} GiB"
    )
    return result


def _validate_p0(config: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / config["output_root"] / "P0_preflight"
    path = root / "P0_RESULT.json"
    digest = _verified_sidecar(path)
    result = _read(path)
    required = {
        "schema": P0_SCHEMA,
        "status": "P0_GO",
        "passed": True,
        "all_weights_discarded": True,
        "retained_adapter_training_authorized": True,
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"family-v1.2 P0 authorization drifted: {key}")
    bridge = result.get("data_provenance_bridge", {})
    if bridge.get("passed") is not True:
        raise RuntimeError("family-v1.2 P0 data-provenance bridge is absent or invalid")
    if _verified_sidecar(DATA_PROVENANCE_AMENDMENT) != bridge.get(
        "amendment_sha256"
    ):
        raise RuntimeError("family-v1.2 P0 data-provenance amendment drifted")
    if list(root.rglob("*.pt")) or list(root.rglob("*.pth")):
        raise RuntimeError("family-v1.2 P0 retained forbidden weights")
    nwp_path = Path(result["NWP_dynamicity_registry"])
    if _verified_sidecar(nwp_path) != result["NWP_dynamicity_registry_file_sha256"]:
        raise RuntimeError("P0 NWP dynamicity registry drifted")
    return {"path": str(path.resolve()), "sha256": digest, "payload": result}


def _gradient_audit(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    values = np.asarray(
        [float(record["preclip_gradient_norm"]) for record in records],
        dtype=np.float64,
    )
    if len(values) < 1 or not np.isfinite(values).all():
        raise FloatingPointError("adapter gradient audit is empty or non-finite")
    training = config["retained_adapter_training"]
    clip = float(training["gradient_clip"])
    clip_fraction = float(np.mean(values > clip))
    maximum = float(values.max())
    gates = {
        "clip_fraction": clip_fraction
        <= float(training["gradient_clip_fraction_max"]),
        "maximum": maximum <= float(training["preclip_gradient_norm_max"]),
    }
    return {
        "updates": int(len(values)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": maximum,
        "clip_fraction": clip_fraction,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _run_one(
    *,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    stages: Sequence[DDIMStageBlock],
    bank: Sequence[Sequence[int]],
    kwargs: Mapping[str, Any],
    bundle: Any,
    p0: Mapping[str, Any],
    code: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    resume: bool,
    config_path: Path,
    data_bridge: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(record["backbone_seed"])
    stage = stages[int(record["stage_block_index"])]
    order_mode = str(record["order_mode"])
    if order_mode not in ("chronological", "shuffle"):
        raise ValueError("unknown probe order mode")
    run_root = (
        output_root
        / "runs"
        / f"seed{seed}"
        / f"block{stage.index}"
        / order_mode
    )
    completion_path = run_root / "completion.json"
    latest_path = run_root / "latest_safe.pt"
    final_path = run_root / "final_adapter.pt"
    if completion_path.exists():
        _verified_sidecar(completion_path)
        completion = _read(completion_path)
        if completion.get("schema") != COMPLETION_SCHEMA:
            raise ValueError("unexpected family-v1.2 completion schema")
        _verified(final_path, completion["final_adapter_checkpoint_sha256"])
        return completion
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FileExistsError(f"existing adapter run requires --resume: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    probe, backbone = _new_probe(
        seed=seed, stage=stage, config=config, kwargs=kwargs, device=device
    )
    trainer = _make_trainer(probe, config=config, backbone_identity=backbone)
    training = config["retained_adapter_training"]
    bank_sha = permutation_bank_sha256(bank)
    identity = {
        "schema": "architecture_v1_family_v1_2_adapter_run_identity_v1",
        "backbone_seed": seed,
        "backbone_checkpoint_sha256": backbone["checkpoint_sha256"],
        "best_EMA_backbone_tensor_sha256": backbone[
            "best_EMA_backbone_tensor_sha256"
        ],
        "stage_block": stage.manifest(),
        "order_mode": order_mode,
        "adapter_initialization_seed": backbone["adapter_initialization_seed"],
        "config_sha256": file_sha256(config_path),
        "code_sha256": code["code_sha256"],
        "P0_result_sha256": p0["sha256"],
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "legacy_D0_v_protocol_sha256": data_bridge[
            "legacy_D0_v_protocol_sha256"
        ],
        "data_provenance_amendment_sha256": data_bridge["amendment_sha256"],
        "permutation_bank_sha256": bank_sha,
        "fixed_optimizer_updates": int(training["fixed_optimizer_updates"]),
    }
    start_update = 0
    history: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    timestep_counts = {str(value): 0 for value in stage.timesteps}
    permutation_counts = {
        str(value): 0
        for value in config["shuffle_control"]["training_permutation_indices"]
    }
    if latest_path.exists():
        if not resume:
            raise FileExistsError(f"resume flag required: {latest_path}")
        payload = trainer.load_checkpoint(
            latest_path, expected_identity=identity, restore_rng=True
        )
        state = payload.get("runner_state", {})
        if state.get("schema") != RUNNER_STATE_SCHEMA:
            raise ValueError("adapter runner checkpoint state is missing")
        start_update = int(state["next_update_zero_based"])
        history = [dict(value) for value in state["history"]]
        gradients = [dict(value) for value in state["gradient_records"]]
        timestep_counts = {
            str(key): int(value) for key, value in state["timestep_counts"].items()
        }
        permutation_counts = {
            str(key): int(value)
            for key, value in state["permutation_counts"].items()
        }
    maximum_updates = int(training["fixed_optimizer_updates"])
    if not 0 <= start_update <= maximum_updates:
        raise RuntimeError("adapter resume update is outside the frozen budget")
    batch_days = int(training["batch_calendar_days"])
    order_seed = int(training["common_calendar_day_order_seed"])
    noise_root = int(training["common_timestep_and_noise_seed_root"])
    training_indices = tuple(
        int(value)
        for value in config["shuffle_control"]["training_permutation_indices"]
    )
    checkpoint_every = int(training["checkpoint_every_updates"])
    run_started = time.perf_counter()
    for update_zero in range(start_update, maximum_updates):
        batch, indices, epoch, batch_index, batch_start = _batch_for_update(
            bundle.train,
            update_zero_based=update_zero,
            batch_days=batch_days,
            order_seed=order_seed,
            device=device,
        )
        timestep, noise = _paired_timestep_and_noise(
            batch,
            stage=stage,
            backbone_seed=seed,
            epoch=epoch,
            batch_start=batch_start,
            train_days=len(bundle.train),
            update_zero_based=update_zero,
            seed_root=noise_root,
        )
        bank_index, shuffle_order = _shuffle_order(
            bank=bank,
            training_indices=training_indices,
            backbone_seed=seed,
            stage_index=stage.index,
            update_zero_based=update_zero,
        )
        hour_order = None if order_mode == "chronological" else shuffle_order
        metrics = trainer.train_step(
            batch,
            hour_order=hour_order,
            timestep=timestep,
            noise=noise,
        )
        for value in timestep.detach().cpu().tolist():
            timestep_counts[str(int(value))] += 1
        if order_mode == "shuffle":
            permutation_counts[str(bank_index)] += 1
        gradients.append(
            {
                "optimizer_update": trainer.optimizer_updates,
                "preclip_gradient_norm": metrics["gradient_norm"],
                "clipped": metrics["gradient_was_clipped"] == 1.0,
            }
        )
        if trainer.optimizer_updates == 1 or trainer.optimizer_updates % checkpoint_every == 0:
            history.append(
                {
                    "optimizer_update": trainer.optimizer_updates,
                    "epoch_1_based": epoch + 1,
                    "batch_index_zero_based": batch_index,
                    "calendar_day_indices_sha256": hashlib.sha256(
                        np.ascontiguousarray(indices, dtype=np.int64).view(np.uint8)
                    ).hexdigest(),
                    "loss": metrics["loss"],
                    "context_residual_abs_max": metrics[
                        "context_residual_abs_max"
                    ],
                    "gradient_norm": metrics["gradient_norm"],
                }
            )
        if (
            trainer.optimizer_updates % checkpoint_every == 0
            or trainer.optimizer_updates == maximum_updates
        ):
            runner_state = {
                "schema": RUNNER_STATE_SCHEMA,
                "next_update_zero_based": trainer.optimizer_updates,
                "history": history,
                "gradient_records": gradients,
                "timestep_counts": timestep_counts,
                "permutation_counts": permutation_counts,
            }
            trainer.save_checkpoint(
                latest_path, identity=identity, runner_state=runner_state
            )
    if trainer.optimizer_updates != maximum_updates:
        raise RuntimeError("adapter run did not finish the exact frozen update budget")
    probe.assert_backbone_unchanged()
    expected_exposures = int(training["calendar_day_timestep_exposures_expected"])
    if sum(timestep_counts.values()) != expected_exposures:
        raise RuntimeError("adapter timestep exposure count differs from frozen protocol")
    if len(set(timestep_counts.values())) != 1:
        raise RuntimeError("adapter block timesteps did not receive exactly balanced exposure")
    gradient = _gradient_audit(gradients, config)
    runner_state = {
        "schema": RUNNER_STATE_SCHEMA,
        "next_update_zero_based": trainer.optimizer_updates,
        "history": history,
        "gradient_records": gradients,
        "timestep_counts": timestep_counts,
        "permutation_counts": permutation_counts,
    }
    final_sha = trainer.save_checkpoint(
        final_path, identity=identity, runner_state=runner_state
    )
    latest_sha = _verified_sidecar(latest_path)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **dict(record),
        "optimizer_updates": trainer.optimizer_updates,
        "EMA_updates": trainer.ema.num_updates,
        "final_adapter_checkpoint": str(final_path.resolve()),
        "final_adapter_checkpoint_sha256": final_sha,
        "latest_safe_checkpoint": str(latest_path.resolve()),
        "latest_safe_checkpoint_sha256": latest_sha,
        "frozen_backbone_tensor_sha256": probe.frozen_backbone_sha256,
        "online_adapter_tensor_sha256": _adapter_hash(probe),
        "EMA_adapter_tensor_sha256": trainer.ema.tensor_sha256(),
        "timestep_counts": timestep_counts,
        "permutation_counts": permutation_counts,
        "gradient_audit": gradient,
        "training_gate_passed": gradient["passed"],
        "identity": identity,
        "materialized_target_roles": ["train"],
        "validation_target_accessed": False,
        "calibration_target_accessed": False,
        "selection_target_accessed": False,
        "r_seen_target_accessed": False,
        "final_target_accessed": False,
        "wall_seconds_this_process": float(time.perf_counter() - run_started),
    }
    _atomic_json(completion_path, completion)
    del trainer, probe
    torch.cuda.empty_cache()
    return completion


def _execute_training(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config, base_config, model_config, stages, bank = _load_config(config_path)
    amendment = _load_data_provenance_amendment(config_path)
    p0 = _validate_p0(config)
    device, runtime = family_v1._configure_cuda()
    code = _code_manifest(config_path)
    data_config = ROOT / config["lineage"]["data_protocol_config"]
    bundle = build_architecture_v1_train_data(config_path=data_config)
    if bundle.materialized_roles != ("train",):
        raise RuntimeError("retained adapter training materialized forbidden targets")
    data_bridge = _validate_train_data_provenance_bridge(bundle, config, amendment)
    output_root = ROOT / config["output_root"] / "formal_training"
    freeze_path = output_root / "training.freeze.json"
    result_path = output_root / "TRAINING_RESULT.json"
    if freeze_path.exists() and result_path.exists():
        _verified_sidecar(freeze_path)
        _verified_sidecar(result_path)
        return _read(result_path)
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = _model_kwargs(base_config, model_config)
    matrix = _training_matrix(config)
    identity = {
        "schema": "architecture_v1_family_v1_2_training_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "P0_result_sha256": p0["sha256"],
        "code_manifest": code,
        "runtime": runtime,
        "protocol_sha256": bundle.protocol.manifest["protocol_sha256"],
        "legacy_D0_v_protocol_sha256": data_bridge[
            "legacy_D0_v_protocol_sha256"
        ],
        "data_provenance_bridge": data_bridge,
        "materialized_target_roles": ["train"],
        "run_matrix": matrix,
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }
    identity_path = output_root / "identity.json"
    if identity_path.exists():
        previous = _read(identity_path)
        without_time = lambda value: {
            key: item for key, item in value.items() if key != "created_utc"
        }
        if without_time(previous) != without_time(identity):
            raise RuntimeError("family-v1.2 formal-training identity drifted")
    else:
        _atomic_json(identity_path, identity)
    completions: dict[str, Any] = {}
    try:
        for record in matrix:
            completion = _run_one(
                record=record,
                config=config,
                stages=stages,
                bank=bank,
                kwargs=kwargs,
                bundle=bundle,
                p0=p0,
                code=code,
                output_root=output_root,
                device=device,
                resume=resume,
                config_path=config_path,
                data_bridge=data_bridge,
            )
            key = (
                f"seed{record['backbone_seed']}_block{record['stage_block_index']}_"
                f"{record['order_mode']}"
            )
            completions[key] = {
                "completion": completion["final_adapter_checkpoint"].replace(
                    "final_adapter.pt", "completion.json"
                ),
                "final_adapter_checkpoint": completion["final_adapter_checkpoint"],
                "final_adapter_checkpoint_sha256": completion[
                    "final_adapter_checkpoint_sha256"
                ],
                "training_gate_passed": completion["training_gate_passed"],
            }
        all_passed = all(value["training_gate_passed"] for value in completions.values())
        result = {
            "schema": TRAINING_SCHEMA,
            "status": "FORTY_EIGHT_OF_FORTY_EIGHT_ADAPTER_RUNS_COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runs": completions,
            "all_training_gates_passed": all_passed,
            "materialized_target_roles": ["train"],
            "validation_target_accessed": False,
            "calibration_target_accessed": False,
            "selection_target_accessed": False,
            "r_seen_target_accessed": False,
            "final_target_accessed": False,
            "next_action": "freeze_training_then_generate_common_validation_scenarios",
        }
        result_sha = _atomic_json(result_path, result)
        freeze = {
            "schema": "architecture_v1_family_v1_2_adapter_training_freeze_v1",
            "status": result["status"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_result": str(result_path.resolve()),
            "training_result_sha256": result_sha,
            "completed_runs": len(completions),
            "all_training_gates_passed": all_passed,
            "final_adapter_checkpoint_sha256": {
                key: value["final_adapter_checkpoint_sha256"]
                for key, value in completions.items()
            },
            "validation_scenario_generation_authorized": all_passed,
            "selection_state": "sealed",
            "calibration_state": "sealed",
        }
        _atomic_json(freeze_path, freeze)
        return result
    except Exception as error:
        _atomic_json(
            output_root / "TRAINING_FAILURE.json",
            {
                "schema": "architecture_v1_family_v1_2_training_failure_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "resume_boundary": "latest_safe_registered_update",
                "validation_target_accessed": False,
                "calibration_target_accessed": False,
                "selection_target_accessed": False,
                "r_seen_target_accessed": False,
                "final_target_accessed": False,
            },
        )
        raise


def _dry_run(config_path: Path) -> dict[str, Any]:
    config, _, _, stages, bank = _load_config(config_path)
    amendment = _load_data_provenance_amendment(config_path)
    p0_path = ROOT / config["output_root"] / "P0_preflight" / "P0_RESULT.json"
    p0_status = "not_run"
    if p0_path.exists():
        try:
            p0_status = _validate_p0(config)["payload"]["status"]
        except Exception as error:  # dry-run reports but does not hide invalid state
            p0_status = f"invalid:{type(error).__name__}:{error}"
    matrix = _training_matrix(config)
    return {
        "schema": "architecture_v1_family_v1_2_dry_run_v1",
        "scientific_question": config["scientific_estimand"]["primary_question"],
        "P0_status": p0_status,
        "retained_training_runs": len(matrix),
        "matrix": matrix,
        "stage_blocks": [value.manifest() for value in stages],
        "permutation_bank_entries": len(bank),
        "permutation_bank_sha256": permutation_bank_sha256(bank),
        "trainable_parameters_per_run": int(
            config["adapter"]["trainable_parameters_expected"]
        ),
        "backbone_parameters_trainable": False,
        "formal_training_target_roles": ["train"],
        "data_provenance_amendment": str(DATA_PROVENANCE_AMENDMENT.resolve()),
        "data_provenance_amendment_sha256": file_sha256(
            DATA_PROVENANCE_AMENDMENT
        ),
        "current_repository_protocol_sha256": amendment[
            "current_repository_identity"
        ]["protocol_sha256"],
        "legacy_D0_v_protocol_sha256": amendment["legacy_D0_v_lineage"][
            "protocol_sha256"
        ],
        "validation_scenarios_implemented_by_this_runner": False,
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "next_flag": (
            "--execute-p0" if p0_status == "not_run" else "--execute-training"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute-p0", action="store_true")
    actions.add_argument("--execute-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.resume and not args.execute_training:
        parser.error("--resume is valid only with --execute-training")
    if args.execute_p0:
        result = _execute_p0(config_path)
    elif args.execute_training:
        result = _execute_training(config_path, resume=args.resume)
    else:
        result = _dry_run(config_path)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
