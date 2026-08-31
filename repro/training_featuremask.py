from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from .training import ExperimentTrainer, infinite, move


class FeatureMaskTrainer(ExperimentTrainer):
    """MS-CADM trainer that masks random condition elements, not whole samples."""

    def fit(self) -> Path:
        if self.name != "mscadm" or not isinstance(self.model, nn.Module) or self.diffusion is None:
            raise ValueError("FeatureMaskTrainer is only defined for MS-CADM")
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
        probability = float(self.training.get("condition_mask_probability", 0.1))
        started = time.perf_counter(); rolling: dict[str, float] = {}; count = 0
        self.model.train()
        while self.step < total_steps:
            batch = move(next(batches), self.device)
            condition = batch["condition"]
            if probability > 0:
                # One decision per feature and hour. Zone identity can also be
                # partially hidden, matching the article's broad wording.
                condition = condition.masked_fill(torch.rand_like(condition) < probability, 0)
            optimizer.zero_grad(set_to_none=True)
            losses = self.diffusion.loss(self.model, batch["target"], condition)
            losses["loss"].backward()
            clip = float(self.training.get("gradient_clip", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step(); self.step += 1; count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if self.step % log_every == 0 or self.step == total_steps:
                record: dict[str, float | int] = {"step": self.step, "seconds": time.perf_counter() - started}
                record.update({key: value / count for key, value in rolling.items()})
                self.history.append(record); print(json.dumps(record), flush=True)
                rolling, count = {}, 0
            if self.step % checkpoint_every == 0:
                self._save(optimizers)
        self._save(optimizers, final=True)
        return self.output_dir / "final.pt"


__all__ = ["FeatureMaskTrainer"]
