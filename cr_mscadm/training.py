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

from repro.data import DataBundle, SplitData
from repro.diffusion import GaussianDiffusion
from repro.torch_data import WindScenarioDataset

from .model import ResidualMSCADM, statistics_loss


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


class CRTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        data: DataBundle,
        output_dir: str | Path,
        *,
        variant: str,
        seed: int,
        device: str | None = None,
    ) -> None:
        if variant not in {"fixed", "hetero_no_crps", "full"}:
            raise ValueError(f"unknown variant: {variant}")
        self.config = deepcopy(config)
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant = variant
        self.seed = int(seed)
        requested = device or self.config.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        seed_everything(self.seed)
        mode = "fixed" if variant == "fixed" else "hetero"
        self.model = ResidualMSCADM(
            head_config=self.config["model"]["head"],
            denoiser_config=self.config["model"]["denoiser"],
            mode=mode,
        ).to(self.device)
        self.diffusion = GaussianDiffusion(**self.config["diffusion"]).to(self.device)
        self.history: dict[str, list[dict[str, float | int]]] = {"head": [], "diffusion": []}
        self.output_dir.joinpath("run_config.json").write_text(
            json.dumps(
                {"variant": variant, "seed": seed, "config": self.config},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.output_dir.joinpath("protocol.json").write_text(
            json.dumps(data.protocol, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _loader(self, split: SplitData, *, shuffle: bool, seed_offset: int = 0) -> DataLoader:
        training = self.config["training"]
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        return DataLoader(
            WindScenarioDataset(split),
            batch_size=int(training.get("batch_size", 256)),
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=int(training.get("workers", 0)),
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _move(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        condition = batch["condition"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        return condition, target

    @torch.no_grad()
    def _validate_head(self) -> dict[str, float]:
        self.model.statistics.eval()
        totals: dict[str, float] = {}
        count = 0
        crps_weight = float(self.config["training"].get("crps_weight", 0.5))
        if self.variant == "hetero_no_crps":
            crps_weight = 0.0
        for batch in self._loader(self.data.validation, shuffle=False):
            condition, target = self._move(batch)
            location, scale = self.model.statistics(condition)
            if self.variant == "fixed":
                values = {
                    "loss": torch.nn.functional.smooth_l1_loss(location, target),
                    "mean_loss": torch.nn.functional.smooth_l1_loss(location, target),
                }
            else:
                values = statistics_loss(
                    target,
                    location,
                    scale,
                    crps_weight=crps_weight,
                    mean_weight=float(self.config["training"].get("mean_weight", 0.1)),
                )
            batch_size = len(condition)
            count += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        return {f"val_{key}": value / count for key, value in totals.items()}

    def fit_head(self) -> Path:
        training = self.config["training"]
        total_steps = int(training.get("head_steps", 4_000))
        validate_every = int(training.get("head_validate_every", 250))
        log_every = int(training.get("log_every", 100))
        optimizer = torch.optim.AdamW(
            self.model.statistics.parameters(),
            lr=float(training.get("head_learning_rate", 3e-4)),
            weight_decay=float(training.get("head_weight_decay", 1e-4)),
        )
        latest = self.output_dir / "head_latest.pt"
        step, best = 0, float("inf")
        if latest.exists() and bool(training.get("resume", True)):
            payload = torch.load(latest, map_location=self.device, weights_only=False)
            self.model.statistics.load_state_dict(payload["statistics"])
            optimizer.load_state_dict(payload["optimizer"])
            step, best = int(payload["step"]), float(payload["best"])
            self.history["head"] = list(payload.get("history", []))
        batches = _infinite(self._loader(self.data.train, shuffle=True, seed_offset=11))
        crps_weight = float(training.get("crps_weight", 0.5))
        if self.variant == "hetero_no_crps":
            crps_weight = 0.0
        rolling: dict[str, float] = {}
        rolling_count = 0
        started = time.perf_counter()
        self.model.statistics.train()
        while step < total_steps:
            condition, target = self._move(next(batches))
            optimizer.zero_grad(set_to_none=True)
            location, scale = self.model.statistics(condition)
            if self.variant == "fixed":
                loss = torch.nn.functional.smooth_l1_loss(location, target)
                losses = {"loss": loss, "mean_loss": loss}
            else:
                losses = statistics_loss(
                    target,
                    location,
                    scale,
                    crps_weight=crps_weight,
                    mean_weight=float(training.get("mean_weight", 0.1)),
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.statistics.parameters(), float(training.get("gradient_clip", 1.0))
            )
            optimizer.step()
            step += 1
            rolling_count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if step % log_every == 0:
                record: dict[str, float | int] = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    **{key: value / rolling_count for key, value in rolling.items()},
                }
                self.history["head"].append(record)
                print(json.dumps({"stage": "head", "variant": self.variant, **record}), flush=True)
                rolling, rolling_count = {}, 0
            if step % validate_every == 0 or step == total_steps:
                validation = self._validate_head()
                score = validation["val_loss"]
                payload = {
                    "step": step,
                    "best": min(best, score),
                    "statistics": self.model.statistics.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": self.history["head"],
                    "validation": validation,
                }
                _atomic_save(payload, latest)
                if score < best:
                    best = score
                    payload["best"] = best
                    _atomic_save(payload, self.output_dir / "head_best.pt")
                print(
                    json.dumps({"stage": "head_validation", "step": step, **validation}),
                    flush=True,
                )
                self.model.statistics.train()
        best_payload = torch.load(
            self.output_dir / "head_best.pt", map_location=self.device, weights_only=False
        )
        self.model.statistics.load_state_dict(best_payload["statistics"])
        if self.variant == "fixed":
            self._estimate_fixed_scale()
        return self.output_dir / "head_best.pt"

    @torch.no_grad()
    def _estimate_fixed_scale(self) -> None:
        self.model.statistics.eval()
        sum_error = torch.zeros(24, device=self.device)
        sum_square = torch.zeros(24, device=self.device)
        count = 0
        for batch in self._loader(self.data.train, shuffle=False):
            condition, target = self._move(batch)
            location, _ = self.model.statistics(condition)
            error = (target - location).squeeze(-1)
            sum_error += error.sum(0)
            sum_square += error.square().sum(0)
            count += len(error)
        variance = (sum_square / count - (sum_error / count).square()).clamp_min(0.05**2)
        self.model.fixed_scale.copy_(variance.sqrt()[None, :, None])

    def _final_payload(self, step: int) -> dict[str, Any]:
        return {
            "name": "cr_mscadm",
            "variant": self.variant,
            "seed": self.seed,
            "config": self.config,
            "step": step,
            "model": self.model.state_dict(),
            "history": self.history,
            "protocol": self.data.protocol,
        }

    def fit_diffusion(self) -> Path:
        training = self.config["training"]
        total_steps = int(training.get("diffusion_steps", 18_000))
        checkpoint_every = int(training.get("checkpoint_every", 1_000))
        log_every = int(training.get("log_every", 100))
        for parameter in self.model.statistics.parameters():
            parameter.requires_grad_(False)
        self.model.statistics.eval()
        optimizer = torch.optim.AdamW(
            self.model.denoiser.parameters(),
            lr=float(training.get("diffusion_learning_rate", 1e-4)),
            weight_decay=float(training.get("diffusion_weight_decay", 0.0)),
        )
        step = 0
        latest = self.output_dir / "diffusion_latest.pt"
        if latest.exists() and bool(training.get("resume", True)):
            payload = torch.load(latest, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            step = int(payload["step"])
            self.history = payload.get("history", self.history)
        batches = _infinite(self._loader(self.data.train, shuffle=True, seed_offset=29))
        rolling: dict[str, float] = {}
        rolling_count = 0
        started = time.perf_counter()
        self.model.denoiser.train()
        while step < total_steps:
            condition, target = self._move(next(batches))
            with torch.no_grad():
                residual = self.model.standardize(target, condition).clamp(-8.0, 8.0)
            masking = float(training.get("condition_mask_probability", 0.1))
            if masking > 0:
                mask = torch.rand_like(condition) < masking
                diffusion_condition = condition.masked_fill(mask, 0.0)
            else:
                diffusion_condition = condition
            optimizer.zero_grad(set_to_none=True)
            losses = self.diffusion.loss(self.model, residual, diffusion_condition)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.denoiser.parameters(), float(training.get("gradient_clip", 1.0))
            )
            optimizer.step()
            step += 1
            rolling_count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if step % log_every == 0 or step == total_steps:
                record: dict[str, float | int] = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    **{key: value / rolling_count for key, value in rolling.items()},
                }
                self.history["diffusion"].append(record)
                print(
                    json.dumps({"stage": "diffusion", "variant": self.variant, **record}),
                    flush=True,
                )
                rolling, rolling_count = {}, 0
            if step % checkpoint_every == 0 or step == total_steps:
                _atomic_save(
                    {
                        **self._final_payload(step),
                        "optimizer": optimizer.state_dict(),
                    },
                    latest,
                )
        final = self.output_dir / "final.pt"
        _atomic_save(self._final_payload(step), final)
        self.output_dir.joinpath("history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return final

    def fit(self) -> Path:
        head_best = self.output_dir / "head_best.pt"
        diffusion_latest = self.output_dir / "diffusion_latest.pt"
        if diffusion_latest.exists():
            payload = torch.load(diffusion_latest, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model"])
        else:
            self.fit_head()
        if not head_best.exists():
            raise FileNotFoundError("head training did not produce head_best.pt")
        return self.fit_diffusion()


def load_cr_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[dict[str, Any], ResidualMSCADM]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("name") != "cr_mscadm":
        raise ValueError("not a CR-MS-CADM checkpoint")
    mode = "fixed" if payload["variant"] == "fixed" else "hetero"
    config = payload["config"]
    model = ResidualMSCADM(
        head_config=config["model"]["head"],
        denoiser_config=config["model"]["denoiser"],
        mode=mode,
    )
    model.load_state_dict(payload["model"])
    return payload, model.to(device).eval()


__all__ = ["CRTrainer", "load_cr_checkpoint", "seed_everything"]
