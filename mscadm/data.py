from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BASE_FEATURES = ("U10", "U100", "V10", "V100")
DERIVED_FEATURES = ("WS10", "WS100", "WE10", "WE100", "WD10", "WD100")
CONTINUOUS_FEATURES = BASE_FEATURES + DERIVED_FEATURES


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    names: tuple[str, ...] = CONTINUOUS_FEATURES

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


@dataclass(frozen=True)
class DailyRecord:
    zone: int
    day: np.datetime64
    condition: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray


def _parse_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["TIMESTAMP"], format="%Y%m%d %H:%M")
    # The files run from 01:00 through the next day's 00:00. Shifting one
    # hour gives the intended 24-hour forecast-day key.
    frame["forecast_day"] = (frame["timestamp"] - pd.Timedelta(hours=1)).dt.normalize()
    return frame


def add_paper_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the six NWP features named in Eq. (13) of the paper.

    The paper does not specify the exact wind-energy or direction convention.
    We use speed cubed (the omitted physical constant is removed by
    standardisation) and the mathematical atan2(V, U) direction in radians.
    Both choices are recorded here rather than hidden in training code.
    """
    frame = frame.copy()
    frame["WS10"] = np.hypot(frame["U10"], frame["V10"])
    frame["WS100"] = np.hypot(frame["U100"], frame["V100"])
    frame["WE10"] = frame["WS10"] ** 3
    frame["WE100"] = frame["WS100"] ** 3
    frame["WD10"] = np.arctan2(frame["V10"], frame["U10"])
    frame["WD100"] = np.arctan2(frame["V100"], frame["U100"])
    return frame


def load_training_hours(data_dir: str | Path, zones: Iterable[int] = range(1, 11)) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for zone in zones:
        path = data_dir / f"Train_W_Zone{zone}.csv"
        frame = pd.read_csv(path)
        if set(frame["ZONEID"].unique()) != {zone}:
            raise ValueError(f"Unexpected zone id in {path}")
        frames.append(add_paper_features(_parse_timestamp(frame)))
    return pd.concat(frames, ignore_index=True)


def load_test_hours(data_dir: str | Path, zones: Iterable[int] = range(1, 11)) -> pd.DataFrame:
    data_dir = Path(data_dir)
    targets = pd.read_csv(data_dir / "TestTar_W.csv")
    frames = []
    for zone in zones:
        predictors = pd.read_csv(data_dir / f"TestPred_W_Zone{zone}.csv")
        zone_targets = targets.loc[targets["ZONEID"] == zone]
        merged = predictors.merge(
            zone_targets,
            on=["ZONEID", "TIMESTAMP"],
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(predictors):
            raise ValueError(f"Test predictor/target alignment failed for zone {zone}")
        frames.append(add_paper_features(_parse_timestamp(merged)))
    return pd.concat(frames, ignore_index=True)


def fit_feature_stats(training_hours: pd.DataFrame) -> FeatureStats:
    values = training_hours.loc[:, CONTINUOUS_FEATURES].to_numpy(np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return FeatureStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def make_daily_records(
    hours: pd.DataFrame,
    stats: FeatureStats,
    *,
    drop_incomplete: bool,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> list[DailyRecord]:
    records: list[DailyRecord] = []
    for (zone, day), group in hours.groupby(["ZONEID", "forecast_day"], sort=True):
        if int(zone) not in zones:
            continue
        group = group.sort_values("timestamp")
        if len(group) != 24:
            continue
        target = group["TARGETVAR"].to_numpy(np.float32)
        mask = np.isfinite(target)
        if drop_incomplete and not mask.all():
            continue
        continuous = group.loc[:, stats.names].to_numpy(np.float32)
        continuous = stats.transform(continuous).astype(np.float32)
        one_hot = np.zeros((24, 10), dtype=np.float32)
        one_hot[:, int(zone) - 1] = 1.0
        condition = np.concatenate([continuous, one_hot], axis=-1)
        records.append(
            DailyRecord(
                zone=int(zone),
                day=np.datetime64(day),
                condition=condition,
                target=np.nan_to_num(target, nan=0.0)[:, None],
                target_mask=mask[:, None],
            )
        )
    return records


class WindDailyDataset(Dataset):
    def __init__(self, records: Sequence[DailyRecord], target_scale: str = "minus_one_one") -> None:
        self.records = list(records)
        self.target_scale = target_scale
        if target_scale not in {"zero_one", "minus_one_one"}:
            raise ValueError(f"Unknown target scale: {target_scale}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        target = torch.from_numpy(record.target.copy())
        if self.target_scale == "minus_one_one":
            target = target * 2.0 - 1.0
        return {
            "target": target,
            "condition": torch.from_numpy(record.condition.copy()),
            "target_mask": torch.from_numpy(record.target_mask.copy()),
            "zone": record.zone,
            "day": str(record.day),
        }

    def inverse_target(self, value: torch.Tensor) -> torch.Tensor:
        if self.target_scale == "minus_one_one":
            value = (value + 1.0) / 2.0
        return value.clamp(0.0, 1.0)


def build_datasets(
    data_dir: str | Path,
    *,
    validation_days: int = 70,
    target_scale: str = "minus_one_one",
    zones: Sequence[int] = tuple(range(1, 11)),
) -> tuple[WindDailyDataset, WindDailyDataset, WindDailyDataset, FeatureStats]:
    training_hours = load_training_hours(data_dir, zones)
    unique_days = np.sort(training_hours["forecast_day"].unique())
    if validation_days <= 0 or validation_days >= len(unique_days):
        raise ValueError("validation_days must leave non-empty train and validation periods")
    validation_start = unique_days[-validation_days]

    train_hours = training_hours.loc[training_hours["forecast_day"] < validation_start]
    validation_hours = training_hours.loc[training_hours["forecast_day"] >= validation_start]
    stats = fit_feature_stats(train_hours)

    train_records = make_daily_records(train_hours, stats, drop_incomplete=True, zones=zones)
    validation_records = make_daily_records(validation_hours, stats, drop_incomplete=True, zones=zones)
    test_hours = load_test_hours(data_dir, zones)
    test_records = make_daily_records(test_hours, stats, drop_incomplete=False, zones=zones)
    return (
        WindDailyDataset(train_records, target_scale),
        WindDailyDataset(validation_records, target_scale),
        WindDailyDataset(test_records, target_scale),
        stats,
    )

