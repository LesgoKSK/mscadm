from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from caa_rahc.nested_data import NestedDataBundle, build_nested_gefcom2014
from repro.data import SplitData


ZERO_STATE = 0
INTERIOR_STATE = 1
ONE_STATE = 2
MASK_STATE = 3


@dataclass(frozen=True)
class JointSplitData:
    """One sample per calendar day with all ten zones kept together."""

    condition: np.ndarray
    target: np.ndarray
    day: np.ndarray
    zones: np.ndarray

    def __post_init__(self) -> None:
        if self.condition.ndim != 4:
            raise ValueError("condition must have shape [day,zone,hour,feature]")
        if self.target.shape != self.condition.shape[:3]:
            raise ValueError("target must align with [day,zone,hour]")
        if self.condition.shape[1:3] != (10, 24):
            raise ValueError("MM-JDWind requires exactly ten zones and 24 hours")
        if self.day.shape != (len(self.target),):
            raise ValueError("day labels do not align with joint samples")
        if self.zones.shape != (10,) or not np.array_equal(
            self.zones, np.arange(1, 11, dtype=np.int64)
        ):
            raise ValueError("zones must be the ordered identifiers 1..10")
        if not np.isfinite(self.condition).all() or not np.isfinite(self.target).all():
            raise ValueError("joint split contains non-finite values")
        if float(self.target.min()) < 0.0 or float(self.target.max()) > 1.0:
            raise ValueError("wind power targets must lie in [0,1]")

    def __len__(self) -> int:
        return len(self.target)

    @property
    def state(self) -> np.ndarray:
        result = np.full(self.target.shape, INTERIOR_STATE, dtype=np.int64)
        result[self.target == 0.0] = ZERO_STATE
        result[self.target == 1.0] = ONE_STATE
        return result


@dataclass(frozen=True)
class JointDataBundle:
    train: JointSplitData
    validation: JointSplitData
    calibration: JointSplitData
    test: JointSplitData
    protocol: dict[str, Any]
    source: NestedDataBundle


def jointify_split(split: SplitData) -> JointSplitData:
    days = np.sort(np.unique(split.day.astype("datetime64[D]")))
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for day in days:
        mask = split.day.astype("datetime64[D]") == day
        indices = np.flatnonzero(mask)
        if len(indices) != 10:
            raise ValueError(f"{day} contains {len(indices)} zone records, expected 10")
        order = np.argsort(split.zone[indices])
        selected = indices[order]
        if not np.array_equal(split.zone[selected], np.arange(1, 11)):
            raise ValueError(f"{day} does not contain ordered zones 1..10")
        conditions.append(split.condition[selected])
        targets.append(split.target[selected])
    return JointSplitData(
        condition=np.ascontiguousarray(np.stack(conditions), dtype=np.float32),
        target=np.ascontiguousarray(np.stack(targets), dtype=np.float32),
        day=days,
        zones=np.arange(1, 11, dtype=np.int64),
    )


def build_joint_nested_gefcom2014(
    data_dir: str = "Data", *, outer: int
) -> JointDataBundle:
    source = build_nested_gefcom2014(data_dir, outer=outer)
    protocol = dict(source.protocol)
    protocol["joint_layout"] = {
        "sample": "calendar day",
        "shape": ["day", "zone", "hour", "feature"],
        "zones": list(range(1, 11)),
        "hours": 24,
    }
    return JointDataBundle(
        train=jointify_split(source.train),
        validation=jointify_split(source.validation),
        calibration=jointify_split(source.calibration),
        test=jointify_split(source.test),
        protocol=protocol,
        source=source,
    )


class JointWindDataset(Dataset):
    def __init__(self, split: JointSplitData) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = self.split.target[index]
        state = np.full(target.shape, INTERIOR_STATE, dtype=np.int64)
        state[target == 0.0] = ZERO_STATE
        state[target == 1.0] = ONE_STATE
        return {
            "condition": torch.from_numpy(self.split.condition[index].copy()),
            "target": torch.from_numpy(target.copy()),
            "state": torch.from_numpy(state),
        }


def flatten_joint_cases(
    scenarios: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert [day,member,zone,hour] into legacy [zone-day,member,hour]."""

    values = np.asarray(scenarios)
    truth = np.asarray(observations)
    if values.ndim != 4:
        raise ValueError("scenarios must have shape [day,member,zone,hour]")
    if truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("observations do not align with joint scenarios")
    flattened = values.transpose(0, 2, 1, 3).reshape(
        values.shape[0] * values.shape[2], values.shape[1], values.shape[3]
    )
    flat_truth = truth.reshape(truth.shape[0] * truth.shape[1], truth.shape[2])
    return flattened, flat_truth


__all__ = [
    "INTERIOR_STATE",
    "JointDataBundle",
    "JointSplitData",
    "JointWindDataset",
    "MASK_STATE",
    "ONE_STATE",
    "ZERO_STATE",
    "build_joint_nested_gefcom2014",
    "flatten_joint_cases",
    "jointify_split",
]
