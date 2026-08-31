from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import DataBundle, SplitData
from .registry import build_diffusion, build_torch_model


def load_trained_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[str, dict[str, Any], torch.nn.Module | dict[str, torch.nn.Module]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    name, config = checkpoint["name"], checkpoint["config"]
    model = build_torch_model(name, config)
    if isinstance(model, dict):
        for key, module in model.items():
            module.load_state_dict(checkpoint["model"][key])
            module.to(device).eval()
    else:
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
    return name, config, model


@torch.no_grad()
def generate_torch_scenarios(
    checkpoint_path: str | Path,
    data: DataBundle,
    *,
    split_name: str = "test",
    scenarios: int = 100,
    sampling_steps: int | None = None,
    sampler: str = "ddim",
    eta: float = 1.0,
    day_batch: int = 8,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name, config, model = load_trained_model(checkpoint_path, device=selected_device)
    split: SplitData = getattr(data, split_name)
    generated_chunks: list[np.ndarray] = []
    if name in {"mscadm", "ddpm"}:
        assert isinstance(model, torch.nn.Module)
        diffusion = build_diffusion(name, config).to(selected_device)
        steps = diffusion.timesteps if sampling_steps is None else sampling_steps
        for start in range(0, len(split), day_batch):
            condition = torch.from_numpy(split.condition[start : start + day_batch]).to(selected_device)
            repeated = condition[:, None].expand(-1, scenarios, -1, -1).reshape(
                -1, condition.shape[1], condition.shape[2]
            )
            if sampler == "ancestral":
                standard = diffusion.sample_ancestral(model, repeated)
            elif sampler == "ddim":
                standard = diffusion.sample_ddim(model, repeated, steps=steps, eta=eta)
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
            standard_array = standard.squeeze(-1).cpu().numpy().reshape(len(condition), scenarios, 24)
            generated_chunks.append(standard_array)
    else:
        flat = torch.from_numpy(split.flat_condition)
        for start in range(0, len(split), day_batch):
            condition = flat[start : start + day_batch].to(selected_device)
            if name == "wgan":
                assert isinstance(model, dict)
                standard = model["generator"].sample(condition, scenarios)
            else:
                assert isinstance(model, torch.nn.Module)
                standard = model.sample(condition, scenarios)
            generated_chunks.append(standard.cpu().numpy())
    standardized = np.concatenate(generated_chunks, axis=0)
    generated = np.clip(data.target_standardizer.inverse(standardized), 0.0, 1.0)
    metadata = {
        "model": name,
        "split": split_name,
        "scenarios": scenarios,
        "sampling_steps": sampling_steps,
        "sampler": sampler if name in {"mscadm", "ddpm"} else None,
        "eta": eta if name in {"mscadm", "ddpm"} else None,
        "checkpoint": str(Path(checkpoint_path).resolve()),
    }
    return generated, metadata


def save_scenarios(
    path: str | Path,
    scenarios: np.ndarray,
    split: SplitData,
    metadata: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scenarios=scenarios.astype(np.float32),
        observations=split.target.astype(np.float32),
        zone=split.zone,
        day=split.day,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output


__all__ = ["load_trained_model", "generate_torch_scenarios", "save_scenarios"]
