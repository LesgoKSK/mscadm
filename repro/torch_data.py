from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .data import SplitData


class WindScenarioDataset(Dataset):
    def __init__(self, split: SplitData, *, flat_condition: bool = False) -> None:
        self.split = split
        self.use_flat_condition = flat_condition

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        condition = (
            self.split.flat_condition[index]
            if self.use_flat_condition
            else self.split.condition[index]
        )
        return {
            "condition": torch.from_numpy(condition.copy()),
            "target": torch.from_numpy(self.split.target_standard[index, :, None].copy()),
            "target_raw": torch.from_numpy(self.split.target[index].copy()),
            "zone": torch.tensor(self.split.zone[index], dtype=torch.long),
        }

