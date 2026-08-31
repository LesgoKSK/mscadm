from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import JointDataBundle, JointSplitData
from .sampling import sample_joint
from .training import load_mm_checkpoint, seed_everything


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def generate_joint_scenarios(
    checkpoint: str | Path,
    data: JointDataBundle,
    *,
    split_name: str,
    members: int,
    jump_steps: int,
    flow_steps: int,
    state_mode: str,
    day_batch: int,
    member_chunk: int,
    seed: int,
    device: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload, model = load_mm_checkpoint(checkpoint, device=selected_device)
    split: JointSplitData = getattr(data, split_name)
    seed_everything(seed)
    scenarios: list[np.ndarray] = []
    zero_probability: list[np.ndarray] = []
    one_probability: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for start in range(0, len(split), day_batch):
        condition = torch.from_numpy(
            split.condition[start : start + day_batch]
        ).to(selected_device)
        generated, statistics, sampled_state = sample_joint(
            model,
            condition,
            members=members,
            jump_steps=jump_steps,
            flow_steps=flow_steps,
            seed=seed + 1009 * start,
            state_mode=state_mode,
            member_chunk=member_chunk,
        )
        scenarios.append(generated.cpu().numpy())
        zero_probability.append(statistics.zero_probability.cpu().numpy())
        one_probability.append(statistics.one_probability.cpu().numpy())
        states.append(sampled_state.cpu().numpy().astype(np.int8))
    arrays = {
        "scenarios": np.concatenate(scenarios).astype(np.float32),
        "observations": split.target.astype(np.float32),
        "zero_probability": np.concatenate(zero_probability).astype(np.float32),
        "one_probability": np.concatenate(one_probability).astype(np.float32),
        "states": np.concatenate(states).astype(np.int8),
        "day": split.day.astype("datetime64[D]"),
        "zones": split.zones.astype(np.int64),
    }
    metadata = {
        "model": "mm_jdwind",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_seed": int(payload["seed"]),
        "sampling_seed": int(seed),
        "split": split_name,
        "members": int(members),
        "jump_steps": int(jump_steps),
        "flow_steps": int(flow_steps),
        "state_mode": state_mode,
        "protocol_sha256": data.protocol.get("protocol_sha256"),
    }
    return arrays, metadata


def save_joint_archive(
    path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(destination)


__all__ = ["generate_joint_scenarios", "save_joint_archive", "sha256_file"]
