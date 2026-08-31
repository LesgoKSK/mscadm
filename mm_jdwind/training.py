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

from .data import INTERIOR_STATE, JointDataBundle, JointSplitData, JointWindDataset
from .model import MMJDWind, MixedMeasureOutput, mixed_measure_head_loss
from .sampling import integrate_rectified_flow, sample_independent_states


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def _statistics_repeat(
    statistics: MixedMeasureOutput, members: int
) -> MixedMeasureOutput:
    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, members, -1, -1).reshape(-1, 10, 24)

    return MixedMeasureOutput(
        location=repeat(statistics.location),
        scale=repeat(statistics.scale),
        zero_logit=repeat(statistics.zero_logit),
        upper_conditional_logit=repeat(statistics.upper_conditional_logit),
    )


def ensemble_proper_loss(
    model: MMJDWind,
    condition: torch.Tensor,
    observation: torch.Tensor,
    *,
    members: int,
    flow_steps: int,
    crps_weight: float,
    energy_weight: float,
    variogram_weight: float,
    ramp_weight: float,
) -> dict[str, torch.Tensor]:
    """Differentiable proper-score fine-tuning with frozen sampled atom states."""

    with torch.no_grad():
        statistics = model.statistics(condition)
        states, _ = sample_independent_states(
            statistics.state_probabilities,
            members=members,
            seed=int(torch.randint(0, 2**31 - 1, ()).item()),
        )
    repeated_condition = condition[:, None].expand(
        -1, members, -1, -1, -1
    ).reshape(-1, 10, 24, condition.shape[-1])
    repeated_state = states.reshape(-1, 10, 24)
    noise = torch.randn_like(repeated_state, dtype=condition.dtype)
    residual = integrate_rectified_flow(
        model,
        noise,
        repeated_condition,
        repeated_state,
        steps=flow_steps,
        method="euler",
    )
    generated = model.reconstruct(
        residual, repeated_state, _statistics_repeat(statistics, members)
    ).reshape(len(condition), members, 10, 24)
    first = (generated - observation[:, None]).abs().mean(dim=1)
    pairwise = (
        generated[:, :, None] - generated[:, None, :]
    ).abs().mean(dim=(1, 2))
    crps = (first - 0.5 * pairwise).mean()
    flat_generated = generated.flatten(2)
    flat_observation = observation.flatten(1)
    energy_first = torch.linalg.vector_norm(
        flat_generated - flat_observation[:, None], dim=-1
    ).mean(dim=1)
    energy_pairwise = torch.linalg.vector_norm(
        flat_generated[:, :, None] - flat_generated[:, None, :], dim=-1
    ).mean(dim=(1, 2))
    energy = (energy_first - 0.5 * energy_pairwise).mean()
    observed_time = (observation[:, :, 1:] - observation[:, :, :-1]).abs().sqrt()
    generated_time = (
        generated[:, :, :, 1:] - generated[:, :, :, :-1]
    ).abs().sqrt().mean(dim=1)
    observed_zone = (observation[:, 1:] - observation[:, :-1]).abs().sqrt()
    generated_zone = (
        generated[:, :, 1:] - generated[:, :, :-1]
    ).abs().sqrt().mean(dim=1)
    variogram = (
        (generated_time - observed_time).square().mean()
        + (generated_zone - observed_zone).square().mean()
    )
    generated_ramp = generated[..., 1:] - generated[..., :-1]
    observed_ramp = observation[..., 1:] - observation[..., :-1]
    ramp_first = (generated_ramp - observed_ramp[:, None]).abs().mean(dim=1)
    ramp_pairwise = (
        generated_ramp[:, :, None] - generated_ramp[:, None, :]
    ).abs().mean(dim=(1, 2))
    ramp_crps = (ramp_first - 0.5 * ramp_pairwise).mean()
    total = (
        crps_weight * crps
        + energy_weight * energy
        + variogram_weight * variogram
        + ramp_weight * ramp_crps
    )
    return {
        "loss": total,
        "CRPS": crps,
        "energy": energy,
        "variogram": variogram,
        "ramp_CRPS": ramp_crps,
    }


class MMTrainer:
    STAGES = ("head", "jump", "flow", "proper")

    def __init__(
        self,
        config: dict[str, Any],
        data: JointDataBundle,
        output_dir: str | Path,
        *,
        seed: int,
        device: str | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        requested = device or self.config.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        seed_everything(self.seed)
        self.model = MMJDWind(**self.config["model"]).to(self.device)
        train_state = data.train.state
        nonzero = train_state != 0
        upper_count = int((train_state == 2).sum())
        nonzero_count = int(nonzero.sum())
        upper_prior = (upper_count + 0.5) / (nonzero_count + 1.0)
        self.model.head.set_upper_probability_prior(upper_prior)
        self.history: dict[str, list[dict[str, float | int]]] = {
            stage: [] for stage in self.STAGES
        }
        self.output_dir.joinpath("run_config.json").write_text(
            json.dumps(
                {
                    "name": "mm_jdwind",
                    "seed": self.seed,
                    "upper_conditional_jeffreys_prior": upper_prior,
                    "config": self.config,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.output_dir.joinpath("protocol.json").write_text(
            json.dumps(data.protocol, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _loader(
        self, split: JointSplitData, *, shuffle: bool, seed_offset: int
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        training = self.config["training"]
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch["condition"].to(self.device, non_blocking=True),
            batch["target"].to(self.device, non_blocking=True),
            batch["state"].to(self.device, non_blocking=True),
        )

    def _parameters(self, stage: str) -> Iterator[nn.Parameter]:
        module = getattr(self.model, stage if stage != "proper" else "flow")
        return module.parameters()

    def _set_stage(self, stage: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        module = getattr(self.model, stage if stage != "proper" else "flow")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        self.model.eval()
        module.train()

    def _stage_loss(
        self,
        stage: str,
        condition: torch.Tensor,
        observation: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        training = self.config["training"]
        if stage == "head":
            return mixed_measure_head_loss(
                observation,
                state,
                self.model.statistics(condition),
                mean_weight=float(training.get("head_mean_weight", 0.05)),
            )
        with torch.no_grad():
            statistics = self.model.statistics(condition)
        if stage == "jump":
            return self.model.jump.loss(
                condition, state, statistics.state_probabilities
            )
        if stage == "flow":
            residual = self.model.residual(observation, state, statistics)
            return self.model.flow.loss(
                residual,
                condition,
                state,
                atom_weight=float(training.get("flow_atom_weight", 0.05)),
            )
        if stage == "proper":
            proper = training["proper"]
            return ensemble_proper_loss(
                self.model,
                condition,
                observation,
                members=int(proper.get("members", 6)),
                flow_steps=int(proper.get("flow_steps", 4)),
                crps_weight=float(proper.get("crps_weight", 1.0)),
                energy_weight=float(proper.get("energy_weight", 0.05)),
                variogram_weight=float(proper.get("variogram_weight", 0.05)),
                ramp_weight=float(proper.get("ramp_weight", 0.1)),
            )
        raise ValueError(stage)

    @torch.no_grad()
    def _validate(self, stage: str) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self._loader(
            self.data.validation, shuffle=False, seed_offset=900
        ):
            condition, observation, state = self._move(batch)
            values = self._stage_loss(stage, condition, observation, state)
            batch_size = len(condition)
            count += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        return {f"val_{key}": value / count for key, value in totals.items()}

    def fit_stage(self, stage: str) -> Path:
        if stage not in self.STAGES:
            raise ValueError(stage)
        training = self.config["training"]
        steps = int(training[f"{stage}_steps"])
        if steps == 0:
            return self.output_dir / f"{stage}_skipped.pt"
        self._set_stage(stage)
        optimizer = torch.optim.AdamW(
            self._parameters(stage),
            lr=float(training[f"{stage}_learning_rate"]),
            weight_decay=float(training.get(f"{stage}_weight_decay", 0.0)),
        )
        loader = _infinite(
            self._loader(self.data.train, shuffle=True, seed_offset=100 + len(stage))
        )
        log_every = int(training.get("log_every", 100))
        validate_every = int(training.get("validate_every", 250))
        best = float("inf")
        started = time.perf_counter()
        rolling: dict[str, float] = {}
        rolling_count = 0
        best_path = self.output_dir / f"{stage}_best.pt"
        for step in range(1, steps + 1):
            condition, observation, state = self._move(next(loader))
            optimizer.zero_grad(set_to_none=True)
            losses = self._stage_loss(stage, condition, observation, state)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._parameters(stage)),
                float(training.get("gradient_clip", 1.0)),
            )
            optimizer.step()
            rolling_count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if step % log_every == 0 or step == steps:
                record: dict[str, float | int] = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    **{key: value / rolling_count for key, value in rolling.items()},
                }
                self.history[stage].append(record)
                print(json.dumps({"stage": stage, **record}), flush=True)
                rolling, rolling_count = {}, 0
            if step % validate_every == 0 or step == steps:
                validation = self._validate(stage)
                score = validation["val_loss"]
                payload = self._payload(stage=stage, step=step)
                payload["validation"] = validation
                payload["optimizer"] = optimizer.state_dict()
                _atomic_torch_save(payload, self.output_dir / f"{stage}_latest.pt")
                if score < best:
                    best = score
                    _atomic_torch_save(payload, best_path)
                print(
                    json.dumps({"stage": f"{stage}_validation", "step": step, **validation}),
                    flush=True,
                )
                self._set_stage(stage)
        best_payload = torch.load(
            best_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(best_payload["model"])
        return best_path

    def _payload(self, *, stage: str, step: int) -> dict[str, Any]:
        return {
            "name": "mm_jdwind",
            "seed": self.seed,
            "stage": stage,
            "step": int(step),
            "config": self.config,
            "protocol": self.data.protocol,
            "model": self.model.state_dict(),
            "history": self.history,
        }

    def fit(self) -> Path:
        for stage in self.STAGES:
            self.fit_stage(stage)
        final = self.output_dir / "final.pt"
        _atomic_torch_save(
            self._payload(
                stage="proper" if int(self.config["training"]["proper_steps"]) else "flow",
                step=int(
                    self.config["training"][
                        "proper_steps"
                        if int(self.config["training"]["proper_steps"])
                        else "flow_steps"
                    ]
                ),
            ),
            final,
        )
        self.output_dir.joinpath("history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return final


def load_mm_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[dict[str, Any], MMJDWind]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("name") != "mm_jdwind":
        raise ValueError("not an MM-JDWind checkpoint")
    model = MMJDWind(**payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return payload, model


__all__ = [
    "MMTrainer",
    "ensemble_proper_loss",
    "load_mm_checkpoint",
    "seed_everything",
]
