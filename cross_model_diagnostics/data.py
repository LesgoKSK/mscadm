"""Leakage-controlled raw NWP loading and deterministic regime definitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_HASHES = {
    "train": "420c87109bb239dd07312c5dd602dc4ee4af973704b6cfc1da393794e193e673",
    "validation": "ccb58fc472247170c3a88bbae94081b9c26a6156abb5fd1ecb40a0d25a860dcd",
    "test": "cecc2c1ab04779ecbc8ca762238e7359e8291b5585b0c37bd875b371be5291dd",
}


def date_sha256(values: np.ndarray) -> str:
    dates = np.sort(np.asarray(values).astype("datetime64[D]")).astype(str)
    return hashlib.sha256("\n".join(dates.tolist()).encode("ascii")).hexdigest()


def legacy_day_split(days: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Reproduce sklearn ShuffleSplit used by the repository without sklearn."""

    dates = np.asarray(days).astype("datetime64[D]")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("legacy split input contains duplicate days")
    first = np.random.RandomState(seed).permutation(len(dates))
    test = dates[first[:50]]
    learning = dates[first[50:]]
    second = np.random.RandomState(seed).permutation(len(learning))
    validation = learning[second[:50]]
    train = learning[second[50:]]
    result = {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }
    for name, expected in LEGACY_HASHES.items():
        actual = date_sha256(result[name])
        if actual != expected:
            raise RuntimeError(f"legacy {name} split hash drifted: {actual} != {expected}")
    return result


def load_raw_joint_days(data_dir: str | Path) -> dict[str, np.ndarray]:
    """Load raw U/V NWP and targets into [day,zone,hour] arrays."""

    root = Path(data_dir)
    test_targets = pd.read_csv(root / "TestTar_W.csv")
    zone_records: list[pd.DataFrame] = []
    for zone in range(1, 11):
        training = pd.read_csv(root / f"Train_W_Zone{zone}.csv")
        predictors = pd.read_csv(root / f"TestPred_W_Zone{zone}.csv")
        target = test_targets.loc[test_targets["ZONEID"] == zone]
        testing = predictors.merge(
            target,
            on=["ZONEID", "TIMESTAMP"],
            how="left",
            validate="one_to_one",
        )
        frame = pd.concat((training, testing), ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["TIMESTAMP"], format="%Y%m%d %H:%M")
        frame["day"] = (frame["timestamp"] - pd.Timedelta(hours=1)).dt.normalize()
        frame = frame.sort_values("timestamp")
        frame["target_was_missing"] = frame["TARGETVAR"].isna()
        frame["TARGETVAR"] = frame["TARGETVAR"].ffill()
        if len(frame) != 731 * 24 or frame.isna().any().any():
            raise ValueError(f"zone {zone} is incomplete after repository-consistent forward fill")
        zone_records.append(frame)

    full = pd.concat(zone_records, ignore_index=True).sort_values(
        ["day", "ZONEID", "timestamp"]
    )
    days = np.sort(np.unique(full["day"].values.astype("datetime64[D]")))
    if len(days) != 731:
        raise ValueError(f"expected 731 days, found {len(days)}")
    arrays: dict[str, list[np.ndarray]] = {
        "U10": [],
        "V10": [],
        "U100": [],
        "V100": [],
        "target": [],
        "target_was_missing": [],
    }
    for day in days:
        selected = full.loc[full["day"].values.astype("datetime64[D]") == day]
        selected = selected.sort_values(["ZONEID", "timestamp"])
        if len(selected) != 10 * 24:
            raise ValueError(f"{day} does not contain 240 zone-hours")
        for source, column in (
            ("U10", "U10"),
            ("V10", "V10"),
            ("U100", "U100"),
            ("V100", "V100"),
            ("target", "TARGETVAR"),
            ("target_was_missing", "target_was_missing"),
        ):
            dtype = bool if source == "target_was_missing" else np.float32
            arrays[source].append(selected[column].to_numpy(dtype).reshape(10, 24))
    return {
        "day": days,
        **{name: np.stack(values) for name, values in arrays.items()},
    }


def nwp_day_features(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    u100 = np.asarray(raw["U100"], dtype=np.float64)
    v100 = np.asarray(raw["V100"], dtype=np.float64)
    u10 = np.asarray(raw["U10"], dtype=np.float64)
    v10 = np.asarray(raw["V10"], dtype=np.float64)
    speed = np.hypot(u100, v100)
    speed10 = np.hypot(u10, v10)
    zone_mean_speed = speed.mean(axis=1)
    vector_u = u100.mean(axis=1)
    vector_v = v100.mean(axis=1)
    direction = np.arctan2(vector_u, vector_v)
    direction_delta = np.angle(np.exp(1j * np.diff(direction, axis=1)))
    dot = u100[..., 1:] * u100[..., :-1] + v100[..., 1:] * v100[..., :-1]
    denominator = speed[..., 1:] * speed[..., :-1]
    cosine = np.divide(
        dot,
        denominator,
        out=np.ones_like(dot),
        where=denominator > 1e-12,
    )
    turn_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    turn_weight = np.minimum(speed[..., 1:], speed[..., :-1])
    weighted_turn = np.divide(
        np.sum(turn_angle * turn_weight, axis=(1, 2)),
        np.sum(turn_weight, axis=(1, 2)),
        out=np.zeros(speed.shape[0], dtype=np.float64),
        where=np.sum(turn_weight, axis=(1, 2)) > 1e-12,
    )
    return {
        "mean_ws100": speed.mean(axis=(1, 2)),
        "rms_dws100": np.sqrt(np.mean(np.diff(speed, axis=-1) ** 2, axis=(1, 2))),
        "weighted_turn100": weighted_turn,
        "nwp_ramp": np.max(np.abs(np.diff(zone_mean_speed, axis=1)), axis=1),
        "direction_shift": np.max(np.abs(direction_delta), axis=1),
        "spatial_dispersion": np.mean(np.std(speed, axis=1), axis=1),
        "mean_shear100_10": np.mean(speed - speed10, axis=(1, 2)),
    }


def fit_regime_thresholds(features: dict[str, np.ndarray], train_mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 1 or int(mask.sum()) < 100:
        raise ValueError("regime thresholds require at least 100 training days")
    wind_quartiles = np.quantile(features["mean_ws100"][mask], [0.25, 0.75])
    return {
        "mean_ws100_q25": float(wind_quartiles[0]),
        "mean_ws100_q75": float(wind_quartiles[1]),
        "rms_dws100_q75": float(np.quantile(features["rms_dws100"][mask], 0.75)),
        "weighted_turn100_q75": float(
            np.quantile(features["weighted_turn100"][mask], 0.75)
        ),
        "spatial_dispersion_q75": float(np.quantile(features["spatial_dispersion"][mask], 0.75)),
        "mean_shear100_10_q75": float(
            np.quantile(features["mean_shear100_10"][mask], 0.75)
        ),
        # Backwards-compatible exploratory definitions.  The five binary
        # labels below are the pre-specified primary regime panel.
        "wind_level_tertiles": np.quantile(features["mean_ws100"][mask], [1 / 3, 2 / 3]).tolist(),
        "nwp_ramp_q75": float(np.quantile(features["nwp_ramp"][mask], 0.75)),
        "direction_shift_q75": float(np.quantile(features["direction_shift"][mask], 0.75)),
        "fit_days": int(mask.sum()),
        "fit_scope": "caller-supplied training/reference dates only",
    }


def assign_regimes(features: dict[str, np.ndarray], thresholds: dict[str, Any]) -> dict[str, np.ndarray]:
    lower, upper = thresholds["wind_level_tertiles"]
    mean_speed = features["mean_ws100"]
    wind_level = np.full(len(mean_speed), "mid", dtype="U16")
    wind_level[mean_speed <= lower] = "low"
    wind_level[mean_speed >= upper] = "high"
    calm = mean_speed <= thresholds["mean_ws100_q25"]
    strong = mean_speed >= thresholds["mean_ws100_q75"]
    volatile = features["rms_dws100"] >= thresholds["rms_dws100_q75"]
    turning = features["weighted_turn100"] >= thresholds["weighted_turn100_q75"]
    heterogeneous = features["spatial_dispersion"] >= thresholds["spatial_dispersion_q75"]
    return {
        "calm": np.where(calm, "calm", "not_calm"),
        "strong": np.where(strong, "strong", "not_strong"),
        "speed_volatile": np.where(volatile, "speed_volatile", "ordinary"),
        "turning": np.where(turning, "turning", "ordinary"),
        "spatial_heterogeneous": np.where(
            heterogeneous, "spatial_heterogeneous", "ordinary"
        ),
        "wind_level": wind_level,
        "nwp_dynamics": np.where(
            features["nwp_ramp"] >= thresholds["nwp_ramp_q75"], "rampy", "ordinary"
        ),
        "direction_regime": np.where(
            features["direction_shift"] >= thresholds["direction_shift_q75"], "shifting", "ordinary"
        ),
        "spatial_regime": np.where(
            features["spatial_dispersion"] >= thresholds["spatial_dispersion_q75"],
            "heterogeneous",
            "coherent",
        ),
    }


def panel_train_dates(
    all_days: np.ndarray,
    *,
    panel: str,
    outer: int,
    mm_registry: str | Path,
    stgf_registry: str | Path,
) -> np.ndarray:
    legacy = legacy_day_split(all_days)
    if panel == "mm_ddpm":
        registry = json.loads(Path(mm_registry).read_text(encoding="utf-8"))
        current_test = np.asarray(registry["outer_test_dates"][str(outer)], dtype="datetime64[D]")
        return np.setdiff1d(legacy["train"], current_test, assume_unique=True)
    if panel == "stgf_ddpm":
        registry = json.loads(Path(stgf_registry).read_text(encoding="utf-8"))
        all_tests = np.concatenate(
            [np.asarray(value, dtype="datetime64[D]") for value in registry["outer_test_dates"].values()]
        )
        return np.setdiff1d(legacy["train"], np.unique(all_tests), assume_unique=True)
    raise ValueError(f"unknown diagnostic panel: {panel}")


def train_zero_epsilon(target: np.ndarray, train_mask: np.ndarray) -> float:
    values = np.asarray(target, dtype=np.float64)[np.asarray(train_mask, dtype=bool)]
    positive = values[(values > 0.0) & (values < 1.0)]
    if positive.size == 0:
        raise ValueError("training target contains no interior positive values")
    return float(np.clip(np.quantile(positive, 0.01), 1e-4, 0.02))
