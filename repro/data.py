from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NWP_FEATURES = (
    "U10",
    "V10",
    "U100",
    "V100",
    "WS10",
    "WS100",
    "WE10",
    "WE100",
    "WD10",
    "WD100",
)


@dataclass(frozen=True)
class ArrayStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axes: int | tuple[int, ...] = 0) -> "ArrayStandardizer":
        mean = np.mean(values, axis=axes, keepdims=True, dtype=np.float64)
        std = np.std(values, axis=axes, keepdims=True, dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


@dataclass(frozen=True)
class SplitData:
    condition: np.ndarray
    flat_condition: np.ndarray
    target: np.ndarray
    target_standard: np.ndarray
    zone: np.ndarray
    day: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


@dataclass(frozen=True)
class DataBundle:
    train: SplitData
    validation: SplitData
    test: SplitData
    condition_standardizer: ArrayStandardizer
    flat_condition_standardizer: ArrayStandardizer
    target_standardizer: ArrayStandardizer
    protocol: dict[str, int | str]


def _parse(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["TIMESTAMP"], format="%Y%m%d %H:%M")
    frame["day"] = (frame["timestamp"] - pd.Timedelta(hours=1)).dt.normalize()
    return frame


def _derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature formulas from the public Dumas et al. reference code."""
    frame = frame.copy()
    frame["WS10"] = np.hypot(frame["U10"], frame["V10"])
    frame["WS100"] = np.hypot(frame["U100"], frame["V100"])
    frame["WE10"] = 0.5 * frame["WS10"] ** 3
    frame["WE100"] = 0.5 * frame["WS100"] ** 3
    frame["WD10"] = np.arctan2(frame["U10"], frame["V10"]) * 180.0 / np.pi
    frame["WD100"] = np.arctan2(frame["U100"], frame["V100"]) * 180.0 / np.pi
    return frame


def load_complete_hours(data_dir: str | Path, zones: Sequence[int] = tuple(range(1, 11))) -> pd.DataFrame:
    """Join official train/test files and reproduce reference forward filling."""
    root = Path(data_dir)
    test_targets = pd.read_csv(root / "TestTar_W.csv")
    frames = []
    for zone in zones:
        training = pd.read_csv(root / f"Train_W_Zone{zone}.csv")
        test_predictors = pd.read_csv(root / f"TestPred_W_Zone{zone}.csv")
        target = test_targets.loc[test_targets["ZONEID"] == zone]
        testing = test_predictors.merge(
            target,
            on=["ZONEID", "TIMESTAMP"],
            how="left",
            validate="one_to_one",
        )
        zone_frame = pd.concat([training, testing], ignore_index=True)
        zone_frame = _parse(zone_frame).sort_values("timestamp")
        zone_frame["TARGETVAR"] = zone_frame["TARGETVAR"].ffill()
        if len(zone_frame) != 731 * 24:
            raise ValueError(f"Zone {zone} has {len(zone_frame)} rows, expected 17544")
        if zone_frame.isna().any().any():
            missing = zone_frame.isna().sum()
            raise ValueError(f"Unresolved missing values in zone {zone}: {missing[missing > 0].to_dict()}")
        frames.append(_derive_features(zone_frame))
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["ZONEID", "timestamp"]).reset_index(drop=True)


def _daily_arrays(hours: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conditions = []
    targets = []
    zones = []
    days = []
    for (zone, day), group in hours.groupby(["ZONEID", "day"], sort=True):
        group = group.sort_values("timestamp")
        if len(group) != 24:
            raise ValueError(f"Zone {zone} day {day} does not contain 24 hours")
        conditions.append(group.loc[:, NWP_FEATURES].to_numpy(np.float32))
        targets.append(group["TARGETVAR"].to_numpy(np.float32))
        zones.append(int(zone))
        days.append(np.datetime64(day, "D"))
    return (
        np.stack(conditions),
        np.stack(targets),
        np.asarray(zones, dtype=np.int64),
        np.asarray(days),
    )


def _flat_feature_major(condition: np.ndarray, zone: np.ndarray) -> np.ndarray:
    # Dumas concatenates 24 values of each feature, then ten zone indicators.
    flattened = condition.transpose(0, 2, 1).reshape(len(condition), -1)
    one_hot = np.eye(10, dtype=np.float32)[zone - 1]
    return np.concatenate([flattened, one_hot], axis=1)


def _split_days(days: np.ndarray, split_days: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    learning, test = train_test_split(days, test_size=split_days, random_state=seed, shuffle=True)
    train, validation = train_test_split(
        learning, test_size=split_days, random_state=seed, shuffle=True
    )
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def build_gefcom2014(
    data_dir: str | Path = "Data",
    *,
    zones: Sequence[int] = tuple(range(1, 11)),
    split_days: int = 50,
    seed: int = 0,
) -> DataBundle:
    """Build the exact random-day LS/VS/TEST protocol inherited by the paper."""
    hours = load_complete_hours(data_dir, zones)
    condition, target, zone, day = _daily_arrays(hours)
    unique_days = np.sort(np.unique(day))
    train_days, validation_days, test_days = _split_days(unique_days, split_days, seed)
    masks = {
        "train": np.isin(day, train_days),
        "validation": np.isin(day, validation_days),
        "test": np.isin(day, test_days),
    }
    train_condition = condition[masks["train"]]
    condition_scaler = ArrayStandardizer.fit(train_condition, axes=0)
    raw_flat = _flat_feature_major(condition, zone)
    flat_scaler = ArrayStandardizer.fit(raw_flat[masks["train"]], axes=0)
    target_scaler = ArrayStandardizer.fit(target[masks["train"]], axes=0)
    scaled_condition = condition_scaler.transform(condition)
    zone_one_hot = np.eye(10, dtype=np.float32)[zone - 1]
    condition_with_zone = np.concatenate(
        [scaled_condition, np.repeat(zone_one_hot[:, None, :], 24, axis=1)], axis=-1
    )
    scaled_flat = flat_scaler.transform(raw_flat)
    scaled_target = target_scaler.transform(target)

    def select(name: str) -> SplitData:
        mask = masks[name]
        return SplitData(
            condition=condition_with_zone[mask],
            flat_condition=scaled_flat[mask],
            target=target[mask],
            target_standard=scaled_target[mask],
            zone=zone[mask],
            day=day[mask],
        )

    return DataBundle(
        train=select("train"),
        validation=select("validation"),
        test=select("test"),
        condition_standardizer=condition_scaler,
        flat_condition_standardizer=flat_scaler,
        target_standardizer=target_scaler,
        protocol={
            "name": "dumas_random_50_per_zone",
            "seed": seed,
            "split_days": split_days,
            "total_days": len(unique_days),
        },
    )

