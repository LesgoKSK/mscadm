"""Minimal, fail-closed training primitives for architecture-v1.

This module intentionally stops short of providing a formal experiment
runner.  It supplies the small auditable unit that a runner can orchestrate:

* a strict joint-day batch contract carrying the raw-observation mask;
* one-step and short-stage atom/flow fitting in FP32;
* finite guards before and after every optimizer update;
* atomic, hash-addressed ``latest_safe`` and validation-selected checkpoints;
* a reproducible failure bundle when a step cannot be certified finite.

The target array is finite because :mod:`architecture_v1.data` fills gaps
inside a role/day only.  Filled cells are nevertheless *never supervision*:
all model losses receive ``observed_mask`` explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Literal

import numpy as np
import torch

from .atom import INTERIOR_STATE, ONE_STATE, ZERO_STATE
from .model import JointRectifiedFlowBase


CHECKPOINT_SCHEMA = "architecture_v1_checkpoint_v1"
FAILURE_SCHEMA = "architecture_v1_failure_bundle_v1"
Stage = Literal["atom", "flow"]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonically JSON serializable: {type(value)!r}")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return text


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _canonicalize(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_sidecar.replace(sidecar)
    return digest


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash named tensors independently of device and serialization format.

    ``torch.save`` bytes are deliberately not used for scientific identity:
    those bytes may change across PyTorch/container versions. Names, dtypes,
    shapes and raw contiguous CPU bytes are length-delimited before hashing so
    distinct state mappings cannot become ambiguous by concatenation.
    """

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise TypeError(f"state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def shared_ea_state_sha256(model: JointRectifiedFlowBase) -> str:
    """Return the stable identity of the common NWP encoder/atom shell."""

    shared: dict[str, torch.Tensor] = {}
    shared.update(
        {f"encoder.{name}": value for name, value in model.encoder.state_dict().items()}
    )
    shared.update(
        {f"atom.{name}": value for name, value in model.atom.state_dict().items()}
    )
    return tensor_state_sha256(shared)


def parameter_manifest(model: JointRectifiedFlowBase) -> dict[str, Any]:
    """Describe model topology without stage-dependent trainability flags."""

    entries: list[dict[str, Any]] = []
    component_counts: dict[str, int] = {}
    for name, value in model.named_parameters():
        component = name.split(".", 1)[0]
        count = int(value.numel())
        component_counts[component] = component_counts.get(component, 0) + count
        entries.append(
            {
                "name": name,
                "component": component,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": count,
            }
        )
    structural = {
        "entries": entries,
        "component_parameter_counts": component_counts,
        "total_parameters": int(sum(component_counts.values())),
    }
    return {
        "schema": "architecture_v1_parameter_manifest_v1",
        **structural,
        "structural_sha256": canonical_sha256(structural),
    }


@dataclass(frozen=True)
class ArchitectureBatch:
    """One minibatch of complete joint days with explicit supervision masks."""

    condition: torch.Tensor
    target: torch.Tensor
    state: torch.Tensor
    observed_mask: torch.Tensor
    raw_missing_mask: torch.Tensor
    day_index: torch.Tensor

    def __post_init__(self) -> None:
        if self.condition.ndim != 4 or self.condition.shape[1:] != (10, 24, 20):
            raise ValueError("condition must have shape [B,10,24,20]")
        expected = self.condition.shape[:3]
        for name, value in (
            ("target", self.target),
            ("state", self.state),
            ("observed_mask", self.observed_mask),
            ("raw_missing_mask", self.raw_missing_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must align with [B,10,24]")
        if self.day_index.shape != (len(self.condition),):
            raise ValueError("day_index must have shape [B]")
        if self.condition.dtype != torch.float32 or self.target.dtype != torch.float32:
            raise TypeError("condition and target must be float32")
        if self.state.dtype != torch.long:
            raise TypeError("state must use torch.long codes")
        if self.observed_mask.dtype != torch.bool or self.raw_missing_mask.dtype != torch.bool:
            raise TypeError("observation masks must be boolean")
        if self.day_index.dtype != torch.long:
            raise TypeError("day_index must use torch.long")
        device = self.condition.device
        for name, value in (
            ("target", self.target),
            ("state", self.state),
            ("observed_mask", self.observed_mask),
            ("raw_missing_mask", self.raw_missing_mask),
            ("day_index", self.day_index),
        ):
            if value.device != device:
                raise ValueError(f"{name} is not on the condition device")
        if not bool(torch.isfinite(self.condition).all()):
            raise ValueError("condition contains non-finite values")
        if not bool(torch.isfinite(self.target).all()):
            raise ValueError("target contains non-finite values")
        if bool(((self.target < 0.0) | (self.target > 1.0)).any()):
            raise ValueError("target must lie in [0,1]")
        if bool(((self.state < ZERO_STATE) | (self.state > ONE_STATE)).any()):
            raise ValueError("state must contain only 0/1/2")
        if not torch.equal(self.observed_mask, ~self.raw_missing_mask):
            raise ValueError("observed_mask must be the complement of raw_missing_mask")
        if not bool(self.observed_mask.any()):
            raise ValueError("a training batch must contain observed supervision")
        if bool((self.state[self.raw_missing_mask] != INTERIOR_STATE).any()):
            raise ValueError("raw-missing cells must use neutral INTERIOR state")
        expected_state = torch.full_like(self.state, INTERIOR_STATE)
        expected_state = expected_state.masked_fill(
            self.observed_mask & (self.target == 0.0), ZERO_STATE
        )
        expected_state = expected_state.masked_fill(
            self.observed_mask & (self.target == 1.0), ONE_STATE
        )
        if not torch.equal(self.state, expected_state):
            raise ValueError("state does not match observed exact-boundary targets")

    @property
    def ramp_observed_mask(self) -> torch.Tensor:
        return self.observed_mask[..., :-1] & self.observed_mask[..., 1:]

    def to(self, device: str | torch.device) -> "ArchitectureBatch":
        selected = torch.device(device)
        return ArchitectureBatch(
            condition=self.condition.to(selected),
            target=self.target.to(selected),
            state=self.state.to(selected),
            observed_mask=self.observed_mask.to(selected),
            raw_missing_mask=self.raw_missing_mask.to(selected),
            day_index=self.day_index.to(selected),
        )

    def cpu_payload(self) -> dict[str, torch.Tensor]:
        return {
            "condition": self.condition.detach().cpu(),
            "target": self.target.detach().cpu(),
            "state": self.state.detach().cpu(),
            "observed_mask": self.observed_mask.detach().cpu(),
            "raw_missing_mask": self.raw_missing_mask.detach().cpu(),
            "day_index": self.day_index.detach().cpu(),
        }

    @classmethod
    def from_split(
        cls,
        split: Any,
        indices: Sequence[int] | np.ndarray,
    ) -> "ArchitectureBatch":
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("indices must be a non-empty one-dimensional vector")
        days = np.asarray(split.day[selected]).astype("datetime64[D]").astype(np.int64)
        return cls(
            condition=torch.from_numpy(
                np.ascontiguousarray(split.condition[selected], dtype=np.float32)
            ),
            target=torch.from_numpy(
                np.ascontiguousarray(split.target[selected], dtype=np.float32)
            ),
            state=torch.from_numpy(
                np.ascontiguousarray(split.state[selected], dtype=np.int64)
            ),
            observed_mask=torch.from_numpy(
                np.ascontiguousarray(split.observed_mask[selected], dtype=bool)
            ),
            raw_missing_mask=torch.from_numpy(
                np.ascontiguousarray(split.raw_missing_mask[selected], dtype=bool)
            ),
            day_index=torch.from_numpy(np.ascontiguousarray(days, dtype=np.int64)),
        )


def configure_stage(model: JointRectifiedFlowBase, stage: Stage) -> list[torch.nn.Parameter]:
    """Select exactly the parameters owned by one registered training stage."""

    if stage not in ("atom", "flow"):
        raise ValueError(f"unknown architecture-v1 stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "atom":
        model._shared_frozen = False  # keep ``train()`` from forcing E/A to eval
        modules: list[torch.nn.Module] = [model.encoder, model.atom]
    else:
        model._shared_frozen = True
        modules = [model.flow]
        # T0/T1 use two parameter-matched temporal branches.  Both branches and
        # the common source adapter belong to the flow-stage transport law even
        # when a candidate deliberately disconnects one recurrent edge.
        for module in (
            model.feature_cell,
            model.source_cell,
            model.source_adapter,
            model.stable_source,
        ):
            if module is not None:
                modules.append(module)
    for module in modules:
        module.requires_grad_(True)
    model.train()
    if stage == "flow":
        model.encoder.eval()
        model.atom.eval()
    parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(f"stage {stage} selected no parameters")
    return parameters


def _stage_loss(
    model: JointRectifiedFlowBase,
    batch: ArchitectureBatch,
    stage: Stage,
    *,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    if stage == "atom":
        kwargs: dict[str, Any] = {
            "states": batch.state,
            "observed_mask": batch.observed_mask,
        }
        if float(getattr(model, "atom_location_auxiliary_weight", 0.0)) > 0.0:
            kwargs["location_target"] = batch.target
        return model.atom_loss(batch.condition, **kwargs)
    if stage == "flow":
        return model.flow_loss(
            batch.condition,
            batch.target,
            states=batch.state,
            observed_mask=batch.observed_mask,
            generator=generator,
        )
    raise ValueError(stage)


def _assert_finite_mapping(values: Mapping[str, torch.Tensor], *, where: str) -> None:
    if "loss" not in values:
        raise KeyError(f"{where} did not return a loss")
    for name, value in values.items():
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"{where}.{name} must be a scalar tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{where}.{name} must be reduced in FP32")
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"non-finite {where}.{name}")


def _nonfinite_named_tensors(values: Iterable[tuple[str, torch.Tensor]]) -> list[str]:
    result: list[str] = []
    for name, value in values:
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            result.append(name)
    return result


def _optimizer_named_tensors(optimizer: torch.optim.Optimizer) -> Iterable[tuple[str, torch.Tensor]]:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value):
                yield f"parameter{parameter_index}.{name}", value


def train_step(
    model: JointRectifiedFlowBase,
    batch: ArchitectureBatch,
    optimizer: torch.optim.Optimizer,
    *,
    stage: Stage,
    generator: torch.Generator | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Execute one certified-finite optimizer update."""

    if gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    losses = _stage_loss(model, batch, stage, generator=generator)
    _assert_finite_mapping(losses, where=f"{stage}_loss")
    losses["loss"].backward()
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"stage {stage} produced no parameter gradients")
    bad_gradients = _nonfinite_named_tensors(gradients)
    if bad_gradients:
        raise FloatingPointError(
            f"non-finite {stage} gradients: {bad_gradients[:8]}"
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError(f"non-finite {stage} gradient norm")
    optimizer.step()
    bad_parameters = _nonfinite_named_tensors(model.named_parameters())
    if bad_parameters:
        raise FloatingPointError(
            f"non-finite parameters after {stage} update: {bad_parameters[:8]}"
        )
    bad_optimizer = _nonfinite_named_tensors(_optimizer_named_tensors(optimizer))
    if bad_optimizer:
        raise FloatingPointError(
            f"non-finite optimizer state after {stage} update: {bad_optimizer[:8]}"
        )
    return {
        **{name: float(value.detach().cpu()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "wall_seconds": float(time.perf_counter() - started),
    }


@torch.no_grad()
def validate_stage(
    model: JointRectifiedFlowBase,
    batches: Sequence[ArchitectureBatch],
    *,
    stage: Stage,
    seed: int,
) -> dict[str, float]:
    if not batches:
        raise ValueError("validation requires at least one batch")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=batches[0].condition.device)
    generator.manual_seed(int(seed))
    totals: dict[str, float] = {}
    total_weight = 0
    try:
        for batch in batches:
            values = _stage_loss(model, batch, stage, generator=generator)
            _assert_finite_mapping(values, where=f"validation_{stage}")
            if stage == "atom":
                weight = int(batch.observed_mask.sum().item())
            else:
                weight = int(
                    (
                        batch.observed_mask
                        & (batch.state == INTERIOR_STATE)
                    ).sum().item()
                )
                if weight == 0:
                    raise ValueError(
                        "flow validation batch has no observed interior cells"
                    )
            total_weight += weight
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.cpu()) * weight
    finally:
        model.train(was_training)
    if total_weight <= 0:
        raise ValueError(f"validation contains no supervised {stage} cells")
    return {name: value / total_weight for name, value in totals.items()}


def _derived_seed(base_seed: int, stage: Stage, step: int, namespace: str) -> int:
    encoded = f"{int(base_seed)}:{stage}:{int(step)}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _checkpoint_payload(
    *,
    model: JointRectifiedFlowBase,
    optimizer: torch.optim.Optimizer,
    stage: Stage,
    step: int,
    training_seed: int,
    run_id: str,
    resolved_config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    data_bundle_sha256: str,
    validation: Mapping[str, float] | None,
    train_metrics: Mapping[str, float],
) -> dict[str, Any]:
    manifest = parameter_manifest(model)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "candidate_id": model.variant,
        "training_seed": int(training_seed),
        "stage": stage,
        "global_step": int(step),
        "resolved_config": _canonicalize(resolved_config),
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "data_bundle_sha256": data_bundle_sha256,
        "shared_EA_state_sha256": shared_ea_state_sha256(model),
        "parameter_manifest": manifest,
        "model_spec": model.model_spec(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation": dict(validation or {}),
        "train_metrics": dict(train_metrics),
        "finite_audit": {
            "loss_finite": True,
            "gradients_finite": True,
            "parameters_finite": True,
            "optimizer_state_finite": True,
        },
    }


def load_checkpoint(
    path: str | Path,
    model: JointRectifiedFlowBase,
    *,
    expected_config_sha256: str,
    expected_protocol_sha256: str,
    expected_data_bundle_sha256: str,
    expected_shared_EA_state_sha256: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected architecture-v1 checkpoint schema")
    checks = {
        "config_sha256": expected_config_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "data_bundle_sha256": expected_data_bundle_sha256,
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise ValueError(f"checkpoint {field} mismatch")
    checkpoint_manifest = payload.get("parameter_manifest")
    if not isinstance(checkpoint_manifest, Mapping):
        raise ValueError("checkpoint parameter_manifest is missing")
    if checkpoint_manifest != parameter_manifest(model):
        raise ValueError("checkpoint parameter_manifest/model topology mismatch")
    stored_shared_hash = _validate_sha256(
        payload.get("shared_EA_state_sha256", ""),
        name="shared_EA_state_sha256",
    )
    if (
        expected_shared_EA_state_sha256 is not None
        and stored_shared_hash
        != _validate_sha256(
            expected_shared_EA_state_sha256,
            name="expected_shared_EA_state_sha256",
        )
    ):
        raise ValueError("checkpoint shared E/A state SHA256 mismatch")
    sidecar = source.with_name(source.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint hash sidecar is missing: {sidecar}")
    declared = sidecar.read_text(encoding="ascii").split()[0]
    if declared != file_sha256(source):
        raise RuntimeError("checkpoint file SHA256 mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if shared_ea_state_sha256(model) != stored_shared_hash:
        raise RuntimeError("checkpoint shared E/A tensor state hash mismatch")
    bad = _nonfinite_named_tensors(model.named_parameters())
    if bad:
        raise FloatingPointError(f"checkpoint contains non-finite parameters: {bad[:8]}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        bad_optimizer = _nonfinite_named_tensors(_optimizer_named_tensors(optimizer))
        if bad_optimizer:
            raise FloatingPointError(
                f"checkpoint contains non-finite optimizer state: {bad_optimizer[:8]}"
            )
    return payload


class ArchitectureTrainer:
    """Short-stage trainer used by smoke tests and future experiment runners."""

    def __init__(
        self,
        model: JointRectifiedFlowBase,
        output_dir: str | Path,
        *,
        resolved_config: Mapping[str, Any],
        protocol_sha256: str,
        data_bundle_sha256: str,
        training_seed: int,
        run_id: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config = _canonicalize(resolved_config)
        self.config_sha256 = canonical_sha256(self.resolved_config)
        self.protocol_sha256 = _validate_sha256(
            protocol_sha256, name="protocol_sha256"
        )
        self.data_bundle_sha256 = _validate_sha256(
            data_bundle_sha256, name="data_bundle_sha256"
        )
        self.training_seed = int(training_seed)
        self.run_id = run_id or f"{model.variant.lower()}_seed{self.training_seed}"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _failure_directory(self, stage: Stage, step: int) -> Path:
        base = self.output_dir / "failures" / f"{stage}_step{step:06d}"
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _save_failure_bundle(
        self,
        *,
        error: Exception,
        stage: Stage,
        step: int,
        batch: ArchitectureBatch,
        optimizer: torch.optim.Optimizer,
    ) -> Path:
        destination = self._failure_directory(stage, step)
        bad_parameters = _nonfinite_named_tensors(self.model.named_parameters())
        bad_gradients = _nonfinite_named_tensors(
            (
                name,
                parameter.grad,
            )
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        )
        bad_optimizer = _nonfinite_named_tensors(_optimizer_named_tensors(optimizer))
        batch_path = destination / "batch.pt"
        _atomic_torch_save(batch.cpu_payload(), batch_path)
        (destination / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        _atomic_json(
            destination / "failure.json",
            {
                "schema": FAILURE_SCHEMA,
                "run_id": self.run_id,
                "candidate_id": self.model.variant,
                "training_seed": self.training_seed,
                "stage": stage,
                "step": int(step),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "config_sha256": self.config_sha256,
                "protocol_sha256": self.protocol_sha256,
                "data_bundle_sha256": self.data_bundle_sha256,
                "shared_EA_state_sha256": shared_ea_state_sha256(self.model),
                "parameter_manifest": parameter_manifest(self.model),
                "batch_sha256": file_sha256(batch_path),
                "nonfinite_parameters": bad_parameters,
                "nonfinite_gradients": bad_gradients,
                "nonfinite_optimizer_state": bad_optimizer,
                "scientific_status": "failed; no checkpoint from this step is publishable",
            },
        )
        _atomic_json(
            destination / "runtime.json",
            {
                "torch_version": torch.__version__,
                "device": str(self.device),
                "cuda_available": bool(torch.cuda.is_available()),
            },
        )
        return destination

    def fit_stage(
        self,
        stage: Stage,
        train_batches: Sequence[ArchitectureBatch],
        validation_batches: Sequence[ArchitectureBatch],
        *,
        steps: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        validate_every: int = 1,
    ) -> dict[str, Any]:
        if steps < 1 or validate_every < 1:
            raise ValueError("steps and validate_every must be positive")
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if not train_batches or not validation_batches:
            raise ValueError("train and validation batches must be non-empty")
        train = [batch.to(self.device) for batch in train_batches]
        validation = [batch.to(self.device) for batch in validation_batches]
        parameters = configure_stage(self.model, stage)
        optimizer = torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        checkpoint_dir = self.output_dir / "checkpoints" / stage
        latest_path = checkpoint_dir / "latest_safe.pt"
        best_path = checkpoint_dir / "best.pt"
        best_score = float("inf")
        best_step = 0
        latest_hash = ""
        best_hash = ""
        history: list[dict[str, Any]] = []
        for step in range(1, steps + 1):
            batch = train[(step - 1) % len(train)]
            seed = _derived_seed(self.training_seed, stage, step, "training_loss")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            try:
                metrics = train_step(
                    self.model,
                    batch,
                    optimizer,
                    stage=stage,
                    generator=generator,
                    gradient_clip=gradient_clip,
                )
                validation_metrics: dict[str, float] | None = None
                if step % validate_every == 0 or step == steps:
                    validation_metrics = validate_stage(
                        self.model,
                        validation,
                        stage=stage,
                        seed=_derived_seed(
                            self.training_seed, stage, step, "validation_loss"
                        ),
                    )
                payload = _checkpoint_payload(
                    model=self.model,
                    optimizer=optimizer,
                    stage=stage,
                    step=step,
                    training_seed=self.training_seed,
                    run_id=self.run_id,
                    resolved_config=self.resolved_config,
                    config_sha256=self.config_sha256,
                    protocol_sha256=self.protocol_sha256,
                    data_bundle_sha256=self.data_bundle_sha256,
                    validation=validation_metrics,
                    train_metrics=metrics,
                )
                latest_hash = _atomic_torch_save(payload, latest_path)
                if validation_metrics is not None:
                    score = float(validation_metrics["loss"])
                    if not np.isfinite(score):
                        raise FloatingPointError("non-finite validation selection score")
                    if score < best_score:
                        best_score = score
                        best_step = step
                        best_hash = _atomic_torch_save(payload, best_path)
                history.append(
                    {
                        "stage": stage,
                        "step": step,
                        "train": metrics,
                        "validation": validation_metrics,
                    }
                )
            except Exception as error:
                self._save_failure_bundle(
                    error=error,
                    stage=stage,
                    step=step,
                    batch=batch,
                    optimizer=optimizer,
                )
                raise
        if not best_path.is_file():
            raise RuntimeError("stage finished without a validation-selected checkpoint")
        _atomic_json(
            self.output_dir / f"{stage}_history.json",
            {
                "schema": "architecture_v1_training_history_v1",
                "run_id": self.run_id,
                "stage": stage,
                "records": history,
            },
        )
        return {
            "stage": stage,
            "steps": int(steps),
            "latest_safe": str(latest_path),
            "latest_safe_sha256": latest_hash,
            "best": str(best_path),
            "best_sha256": best_hash,
            "best_step": int(best_step),
            "best_validation_loss": float(best_score),
        }


__all__ = [
    "ArchitectureBatch",
    "ArchitectureTrainer",
    "CHECKPOINT_SCHEMA",
    "FAILURE_SCHEMA",
    "canonical_sha256",
    "configure_stage",
    "file_sha256",
    "load_checkpoint",
    "parameter_manifest",
    "shared_ea_state_sha256",
    "tensor_state_sha256",
    "train_step",
    "validate_stage",
]
