from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mm_jdwind.data import JointDataBundle, JointSplitData, JointWindDataset

from .graph import SpectralArtifacts, fit_spectral_artifacts
from .model import STGFFlow


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


class STGFTrainer:
    STAGES = ("center", "flow")

    def __init__(
        self,
        config: dict[str, Any],
        data: JointDataBundle,
        output_dir: str | Path,
        *,
        seed: int,
        transform_mode: str,
        device: str | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.transform_mode = str(transform_mode)
        requested = device or config.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        seed_everything(self.seed)
        graph = self.config["graph"]
        self.artifacts = fit_spectral_artifacts(
            data.train,
            transform_mode=self.transform_mode,
            neighbors=int(graph.get("neighbors", 3)),
            correlation_power=float(graph.get("correlation_power", 2.0)),
            epsilon=float(graph.get("logit_epsilon", 1e-4)),
        )
        model_config = dict(self.config["model"])
        model_config["logit_epsilon"] = float(
            graph.get("logit_epsilon", 1e-4)
        )
        self.model = STGFFlow(self.artifacts, **model_config).to(self.device)
        self.history: dict[str, list[dict[str, Any]]] = {
            stage: [] for stage in self.STAGES
        }
        self.output_dir.joinpath("run_config.json").write_text(
            json.dumps(
                {
                    "name": "stgf_flow",
                    "seed": self.seed,
                    "transform_mode": self.transform_mode,
                    "config": self.config,
                    "artifact_metadata": self.artifacts.metadata,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.output_dir.joinpath("protocol.json").write_text(
            json.dumps(data.protocol, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _loader(
        self, split: JointSplitData, *, shuffle: bool, seed_offset: int
    ) -> DataLoader:
        training = self.config["training"]
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        return DataLoader(
            JointWindDataset(split),
            batch_size=int(training.get("batch_size", 16)),
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=int(training.get("workers", 0)),
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _move(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch["condition"].to(self.device, non_blocking=True),
            batch["target"].to(self.device, non_blocking=True),
        )

    def _set_stage(self, stage: str) -> list[torch.nn.Parameter]:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        module = getattr(self.model, stage)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        self.model.eval()
        module.train()
        return list(module.parameters())

    def _loss(
        self,
        stage: str,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if stage == "center":
            return self.model.center_loss(condition, observation)
        if stage == "flow":
            return self.model.flow_loss(
                condition, observation, generator=generator
            )
        raise ValueError(stage)

    @torch.no_grad()
    def _validate(self, stage: str) -> dict[str, float]:
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(
            90_000 + self.seed + (0 if stage == "center" else 1_000)
        )
        totals: dict[str, float] = {}
        count = 0
        for batch in self._loader(
            self.data.validation, shuffle=False, seed_offset=900
        ):
            condition, observation = self._move(batch)
            values = self._loss(
                stage,
                condition,
                observation,
                generator=generator if stage == "flow" else None,
            )
            count += len(condition)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * len(
                    condition
                )
        return {
            f"val_{key}": value / count for key, value in totals.items()
        }

    def _payload(self, stage: str, step: int) -> dict[str, Any]:
        return {
            "name": "stgf_flow",
            "seed": self.seed,
            "transform_mode": self.transform_mode,
            "stage": stage,
            "step": int(step),
            "config": self.config,
            "protocol": self.data.protocol,
            "artifacts": self.artifacts.torch_payload(),
            "model": self.model.state_dict(),
            "history": self.history,
        }

    def fit_stage(self, stage: str) -> Path:
        training = self.config["training"]
        total_steps = int(training[f"{stage}_steps"])
        parameters = self._set_stage(stage)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training[f"{stage}_learning_rate"]),
            weight_decay=float(training.get(f"{stage}_weight_decay", 0.0)),
        )
        batches = _infinite(
            self._loader(
                self.data.train,
                shuffle=True,
                seed_offset=100 + len(stage),
            )
        )
        validate_every = int(training.get("validate_every", 250))
        log_every = int(training.get("log_every", 100))
        patience = int(training.get("early_stopping_patience_validations", 8))
        minimum_delta = float(
            training.get("early_stopping_minimum_delta", 1e-4)
        )
        best = float("inf")
        best_step = 0
        stale = 0
        best_path = self.output_dir / f"{stage}_best.pt"
        started = time.perf_counter()
        rolling: dict[str, float] = {}
        rolling_count = 0
        stopped_step = total_steps
        for step in range(1, total_steps + 1):
            condition, observation = self._move(next(batches))
            optimizer.zero_grad(set_to_none=True)
            values = self._loss(stage, condition, observation)
            if not torch.isfinite(values["loss"]):
                raise FloatingPointError(
                    f"non-finite {stage} loss at step {step}"
                )
            values["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(training.get("gradient_clip", 1.0))
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite {stage} gradient at step {step}"
                )
            optimizer.step()
            if any(not torch.isfinite(value).all() for value in parameters):
                raise FloatingPointError(
                    f"non-finite {stage} parameter at step {step}"
                )
            rolling_count += 1
            for key, value in values.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if step % log_every == 0 or step == total_steps:
                record = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    **{
                        key: value / rolling_count
                        for key, value in rolling.items()
                    },
                }
                self.history[stage].append(record)
                print(json.dumps({"stage": stage, **record}), flush=True)
                rolling, rolling_count = {}, 0
            if step % validate_every == 0 or step == total_steps:
                validation = self._validate(stage)
                score = float(validation["val_loss"])
                if not np.isfinite(score):
                    raise FloatingPointError(
                        f"non-finite {stage} validation at step {step}"
                    )
                payload = self._payload(stage, step)
                payload["validation"] = validation
                payload["optimizer"] = optimizer.state_dict()
                _atomic_save(
                    payload, self.output_dir / f"{stage}_latest.pt"
                )
                if score < best - minimum_delta:
                    best = score
                    best_step = step
                    stale = 0
                    _atomic_save(payload, best_path)
                else:
                    stale += 1
                print(
                    json.dumps(
                        {
                            "stage": f"{stage}_validation",
                            "step": step,
                            "best_step": best_step,
                            "patience_used": stale,
                            **validation,
                        }
                    ),
                    flush=True,
                )
                self._set_stage(stage)
                if stale >= patience:
                    stopped_step = step
                    break
        selected = torch.load(
            best_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(selected["model"])
        self.history[stage].append(
            {
                "step": stopped_step,
                "status": (
                    "early_stopped"
                    if stopped_step < total_steps
                    else "max_steps"
                ),
                "best_step": best_step,
                "best_validation_loss": best,
            }
        )
        return best_path

    def fit(self) -> Path:
        self.fit_stage("center")
        self.fit_stage("flow")
        flow = torch.load(
            self.output_dir / "flow_best.pt",
            map_location=self.device,
            weights_only=False,
        )
        final = self.output_dir / "final.pt"
        payload = self._payload("flow_validation_selected", int(flow["step"]))
        payload["validation"] = flow.get("validation", {})
        _atomic_save(payload, final)
        self.output_dir.joinpath("history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return final


def load_stgf_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[dict[str, Any], STGFFlow]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("name") != "stgf_flow":
        raise ValueError("not an STGF-Flow checkpoint")
    artifacts = SpectralArtifacts.from_torch_payload(payload["artifacts"])
    graph = payload["config"]["graph"]
    model_config = dict(payload["config"]["model"])
    model_config["logit_epsilon"] = float(
        graph.get("logit_epsilon", 1e-4)
    )
    model = STGFFlow(artifacts, **model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return payload, model


__all__ = ["STGFTrainer", "load_stgf_checkpoint", "seed_everything"]
