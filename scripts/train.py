from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mscadm.config import load_config
from mscadm.data import build_datasets
from mscadm.diffusion import GaussianDiffusion
from mscadm.model import MSCADM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_condition(condition: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return condition
    condition = condition.clone()
    keep = torch.rand_like(condition[..., :10]) >= probability
    condition[..., :10] *= keep
    return condition


@torch.no_grad()
def validation_loss(
    model: MSCADM,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    model.eval()
    values = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        target = batch["target"].to(device)
        condition = batch["condition"].to(device)
        values.append(float(diffusion.training_loss(model, target, condition)["loss"]))
    model.train()
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_config = config["data"]
    train_set, validation_set, test_set, stats = build_datasets(
        data_config["data_dir"],
        validation_days=int(data_config["validation_days"]),
        target_scale=data_config["target_scale"],
        zones=tuple(data_config["zones"]),
    )
    training_config = config["training"]
    train_loader = DataLoader(
        train_set,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    model_config = dict(config["model"])
    model_config["learn_variance"] = bool(config["diffusion"]["learn_variance"])
    model = MSCADM(**model_config)
    diffusion = GaussianDiffusion(**config["diffusion"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    diffusion.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(json.dumps({
        "device": str(device),
        "train_sequences": len(train_set),
        "validation_sequences": len(validation_set),
        "test_sequences": len(test_set),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }))
    iterator = iter(train_loader)
    for step in range(1, int(training_config["steps"]) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        target = batch["target"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        condition = mask_condition(condition, float(training_config.get("condition_mask_probability", 0.1)))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            losses = diffusion.training_loss(model, target, condition)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % int(training_config["log_every"]) == 0:
            print(json.dumps({
                "step": step,
                "loss": float(losses["loss"]),
                "simple": float(losses["simple"]),
                "vlb": float(losses["vlb"]),
            }))
        if step % int(training_config["validate_every"]) == 0:
            value = validation_loss(model, diffusion, validation_loader, device)
            print(json.dumps({"step": step, "validation_loss": value}))
        if step % int(training_config["checkpoint_every"]) == 0:
            checkpoint = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "feature_stats": asdict(stats),
            }
            torch.save(checkpoint, output_dir / f"checkpoint_{step:07d}.pt")
            torch.save(checkpoint, output_dir / "latest.pt")


if __name__ == "__main__":
    main()

