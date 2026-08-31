from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DataBundle
from .diffusion import GaussianDiffusion
from .registry import build_diffusion, build_torch_model, uses_flat_condition
from .torch_data import WindScenarioDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class ExperimentTrainer:
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        data: DataBundle,
        output_dir: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.data = data
        requested = device or config.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training = dict(config.get("training", {}))
        self.seed = int(config.get("seed", 0))
        seed_everything(self.seed)
        self.model = build_torch_model(name, config)
        self.diffusion: GaussianDiffusion | None = None
        if name in {"mscadm", "ddpm"}:
            self.diffusion = build_diffusion(name, config).to(self.device)
        if isinstance(self.model, dict):
            self.model = {key: value.to(self.device) for key, value in self.model.items()}
        else:
            self.model = self.model.to(self.device)
        self.history: list[dict[str, float | int]] = []
        self.step = 0
        (self.output_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.output_dir / "protocol.json").write_text(
            json.dumps(data.protocol, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _loader(self) -> DataLoader:
        dataset = WindScenarioDataset(self.data.train, flat_condition=uses_flat_condition(self.name))
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=int(self.training.get("batch_size", 256)),
            shuffle=True,
            drop_last=True,
            num_workers=int(self.training.get("workers", 0)),
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _checkpoint_payload(self, optimizers: dict[str, torch.optim.Optimizer]) -> dict[str, Any]:
        if isinstance(self.model, dict):
            model_state: Any = {key: value.state_dict() for key, value in self.model.items()}
        else:
            model_state = self.model.state_dict()
        return {
            "name": self.name,
            "config": self.config,
            "step": self.step,
            "model": model_state,
            "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
            "history": self.history,
        }

    def _save(self, optimizers: dict[str, torch.optim.Optimizer], *, final: bool = False) -> None:
        atomic_torch_save(self._checkpoint_payload(optimizers), self.output_dir / "latest.pt")
        if final:
            atomic_torch_save(self._checkpoint_payload(optimizers), self.output_dir / "final.pt")
        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

    def _restore(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        path = self.output_dir / "latest.pt"
        if not path.exists() or not bool(self.training.get("resume", True)):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["name"] != self.name:
            raise ValueError(f"Checkpoint is for {checkpoint['name']}, not {self.name}")
        if isinstance(self.model, dict):
            for key, module in self.model.items():
                module.load_state_dict(checkpoint["model"][key])
        else:
            self.model.load_state_dict(checkpoint["model"])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
        self.step = int(checkpoint["step"])
        self.history = list(checkpoint.get("history", []))

    def fit(self) -> Path:
        if self.name == "wgan":
            return self._fit_wgan()
        assert isinstance(self.model, nn.Module)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.training.get("learning_rate", 1e-4)),
            weight_decay=float(self.training.get("weight_decay", 0.0)),
        )
        optimizers = {"model": optimizer}
        self._restore(optimizers)
        batches = infinite(self._loader())
        total_steps = int(self.training.get("steps", 26_000))
        checkpoint_every = int(self.training.get("checkpoint_every", 1_000))
        log_every = int(self.training.get("log_every", 100))
        started = time.perf_counter()
        rolling: dict[str, float] = {}
        rolling_count = 0
        self.model.train()
        while self.step < total_steps:
            batch = move(next(batches), self.device)
            optimizer.zero_grad(set_to_none=True)
            if self.name in {"mscadm", "ddpm"}:
                condition = batch["condition"]
                masking = float(self.training.get("condition_mask_probability", 0.0))
                if masking > 0:
                    mask = (torch.rand(len(condition), 1, 1, device=self.device) < masking)
                    condition = condition.masked_fill(mask, 0)
                assert self.diffusion is not None
                losses = self.diffusion.loss(self.model, batch["target"], condition)
            else:
                losses = self.model.loss(batch["target"].squeeze(-1), batch["condition"])
            losses["loss"].backward()
            clip = float(self.training.get("gradient_clip", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
            self.step += 1
            rolling_count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if self.step % log_every == 0 or self.step == total_steps:
                record: dict[str, float | int] = {
                    "step": self.step,
                    "seconds": time.perf_counter() - started,
                }
                record.update({key: value / rolling_count for key, value in rolling.items()})
                self.history.append(record)
                print(json.dumps(record), flush=True)
                rolling, rolling_count = {}, 0
            if self.step % checkpoint_every == 0:
                self._save(optimizers)
        self._save(optimizers, final=True)
        return self.output_dir / "final.pt"

    def _fit_wgan(self) -> Path:
        assert isinstance(self.model, dict)
        generator = self.model["generator"]
        critic = self.model["critic"]
        lr = float(self.training.get("learning_rate", 2e-4))
        betas = tuple(float(value) for value in self.training.get("betas", [0.0, 0.9]))
        weight_decay = float(self.training.get("weight_decay", 1e-4))
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizers = {"generator": generator_optimizer, "critic": critic_optimizer}
        self._restore(optimizers)
        batches = infinite(self._loader())
        total_steps = int(self.training.get("steps", 26_000))
        critic_steps = int(self.training.get("critic_steps", 5))
        gp_weight = float(self.training.get("gradient_penalty", 10.0))
        checkpoint_every = int(self.training.get("checkpoint_every", 1_000))
        log_every = int(self.training.get("log_every", 100))
        started = time.perf_counter()
        accum_critic = 0.0
        accum_generator = 0.0
        for _ in range(self.step, total_steps):
            for _critic_index in range(critic_steps):
                batch = move(next(batches), self.device)
                target, condition = batch["target"].squeeze(-1), batch["condition"]
                noise = torch.randn(len(target), generator.latent_dim, device=self.device)
                fake = generator(noise, condition).detach()
                critic_optimizer.zero_grad(set_to_none=True)
                penalty = critic.gradient_penalty(target, fake, condition)
                critic_loss = critic(fake, condition).mean() - critic(target, condition).mean() + gp_weight * penalty
                critic_loss.backward()
                critic_optimizer.step()
                accum_critic += float(critic_loss.detach())
            batch = move(next(batches), self.device)
            condition = batch["condition"]
            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(len(condition), generator.latent_dim, device=self.device)
            generator_loss = -critic(generator(noise, condition), condition).mean()
            generator_loss.backward()
            generator_optimizer.step()
            accum_generator += float(generator_loss.detach())
            self.step += 1
            if self.step % log_every == 0 or self.step == total_steps:
                record = {
                    "step": self.step,
                    "seconds": time.perf_counter() - started,
                    "critic": accum_critic / (log_every * critic_steps),
                    "generator": accum_generator / log_every,
                }
                self.history.append(record)
                print(json.dumps(record), flush=True)
                accum_critic = accum_generator = 0.0
            if self.step % checkpoint_every == 0:
                self._save(optimizers)
        self._save(optimizers, final=True)
        return self.output_dir / "final.pt"


__all__ = ["ExperimentTrainer", "seed_everything"]
