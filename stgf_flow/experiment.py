from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mm_jdwind.data import JointDataBundle

from .sampling import sample_stgf
from .training import load_stgf_checkpoint


@torch.no_grad()
def generate_stgf_scenarios(
    checkpoint: str | Path,
    data: JointDataBundle,
    *,
    split_name: str,
    members: int,
    flow_steps: int,
    day_batch: int,
    member_chunk: int,
    seed: int,
    device: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload, model = load_stgf_checkpoint(
        checkpoint, device=selected_device
    )
    split = getattr(data, split_name)
    chunks: list[np.ndarray] = []
    for start in range(0, len(split), day_batch):
        condition = torch.from_numpy(
            split.condition[start : start + day_batch]
        ).to(selected_device)
        generated = sample_stgf(
            model,
            condition,
            members=members,
            steps=flow_steps,
            member_chunk=member_chunk,
            seed=seed + start * 10_000,
        )
        chunks.append(generated.cpu().numpy())
    scenarios = np.concatenate(chunks, axis=0).astype(np.float32)
    metadata = {
        "name": "stgf_flow",
        "transform_mode": payload["transform_mode"],
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split_name,
        "members": int(members),
        "flow_steps": int(flow_steps),
        "integrator": "heun",
        "seed": int(seed),
        "protocol_sha256": data.protocol.get("protocol_sha256"),
        "artifact_metadata": payload["artifacts"].get("metadata", {}),
    }
    return {
        "scenarios": scenarios,
        "observations": split.target.astype(np.float32),
        "day": split.day,
        "zones": split.zones,
    }, metadata


def save_stgf_archive(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


__all__ = ["generate_stgf_scenarios", "save_stgf_archive"]
