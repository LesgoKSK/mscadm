"""Leakage-safe GEFCom2014 loader for architecture-v1.

Important differences from :mod:`repro.data`:

* NWP predictors can be loaded without opening any target file.
* Calendar roles are assigned before target missing values are touched.
* Missing targets are filled only inside the same role, zone, and calendar
  day.  The original NaNs and their boolean mask remain in every split.
* Standardizers are fitted only on raw-observed training cells.
* The 300 architecture-diagnostic dates remain a separate ``r_seen`` split;
  no local final-test split is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import (
    ALL_ROLES,
    ArchitectureProtocol,
    DEFAULT_CONFIG_PATH,
    LOCAL_ROLES,
    build_architecture_protocol,
    canonical_sha256,
    date_list_sha256,
    file_sha256,
)


BASE_NWP_FEATURES = ("U10", "V10", "U100", "V100")
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
ZERO_STATE = 0
INTERIOR_STATE = 1
ONE_STATE = 2


def _resolve_data_dir(
    data_dir: str | Path | None, protocol: ArchitectureProtocol
) -> Path:
    value = data_dir if data_dir is not None else protocol.config["data_dir"]
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    project_candidate = protocol.config_path.parents[1] / candidate
    if project_candidate.exists():
        return project_candidate
    return (protocol.config_path.parent / candidate).resolve()


def _parse_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["TIMESTAMP"], format="%Y%m%d %H:%M"
    )
    # GEFCom hours are labelled 01:00..24:00; subtracting one hour gives the
    # intended forecast calendar day also for the final 00:00 timestamp.
    result["day"] = (result["timestamp"] - pd.Timedelta(hours=1)).dt.normalize()
    return result


def _derive_nwp(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["WS10"] = np.hypot(result["U10"], result["V10"])
    result["WS100"] = np.hypot(result["U100"], result["V100"])
    result["WE10"] = 0.5 * result["WS10"] ** 3
    result["WE100"] = 0.5 * result["WS100"] ** 3
    result["WD10"] = np.arctan2(result["U10"], result["V10"]) * 180.0 / np.pi
    result["WD100"] = np.arctan2(result["U100"], result["V100"]) * 180.0 / np.pi
    return result


def load_nwp_hours(
    data_dir: str | Path,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> pd.DataFrame:
    """Load NWP predictors without reading ``TestTar_W.csv``.

    ``TARGETVAR`` is excluded with ``usecols`` even from the training CSVs.
    This function is therefore safe for predictor-only inference and for
    constructing split registries before target access.
    """

    root = Path(data_dir)
    frames: list[pd.DataFrame] = []
    predictor_columns = ["ZONEID", "TIMESTAMP", *BASE_NWP_FEATURES]
    for zone in zones:
        parts = []
        for name in (f"Train_W_Zone{zone}.csv", f"TestPred_W_Zone{zone}.csv"):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            parts.append(pd.read_csv(path, usecols=predictor_columns))
        frame = _parse_keys(pd.concat(parts, ignore_index=True))
        if not (frame["ZONEID"] == zone).all():
            raise ValueError(f"predictor ZONEID mismatch for zone {zone}")
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"duplicate predictor timestamp for zone {zone}")
        if len(frame) != 731 * 24:
            raise ValueError(
                f"zone {zone} has {len(frame)} predictor rows, expected {731 * 24}"
            )
        if frame.loc[:, BASE_NWP_FEATURES].isna().any().any():
            raise ValueError(f"zone {zone} contains missing base NWP predictors")
        frame = _derive_nwp(frame)
        if not np.isfinite(frame.loc[:, NWP_FEATURES].to_numpy()).all():
            raise ValueError(f"zone {zone} contains non-finite derived NWP predictors")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["day", "ZONEID", "timestamp"]).reset_index(drop=True)


def load_target_hours(
    data_dir: str | Path,
    zones: Sequence[int] = tuple(range(1, 11)),
    *,
    allowed_days: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Load raw targets only, retaining NaNs exactly as published.

    When ``allowed_days`` is supplied, target rows outside that frozen date
    set are discarded inside the CSV chunk loop and are never returned or
    materialized in the scientific data bundle.  Formal fitting uses this
    mode so calibration/selection targets cannot leak into trainer memory.
    """

    root = Path(data_dir)
    test_path = root / "TestTar_W.csv"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    columns = ["ZONEID", "TIMESTAMP", "TARGETVAR"]
    allowed = None
    if allowed_days is not None:
        allowed = np.asarray(list(allowed_days), dtype="datetime64[D]")
        if allowed.ndim != 1 or not len(allowed):
            raise ValueError("allowed_days must be a non-empty one-dimensional set")
        if len(np.unique(allowed)) != len(allowed):
            raise ValueError("allowed_days contains duplicates")
        allowed = np.sort(allowed)

    def read_selected(path: Path) -> pd.DataFrame:
        if allowed is None:
            return _parse_keys(pd.read_csv(path, usecols=columns))
        selected: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=columns, chunksize=8192):
            parsed = _parse_keys(chunk)
            keep = np.isin(
                parsed["day"].to_numpy(dtype="datetime64[D]"), allowed
            )
            if bool(keep.any()):
                selected.append(parsed.loc[keep].copy())
        if not selected:
            return pd.DataFrame(
                {
                    "ZONEID": pd.Series(dtype="int64"),
                    "TIMESTAMP": pd.Series(dtype="object"),
                    "TARGETVAR": pd.Series(dtype="float64"),
                    "timestamp": pd.Series(dtype="datetime64[ns]"),
                    "day": pd.Series(dtype="datetime64[ns]"),
                }
            )
        return pd.concat(selected, ignore_index=True)

    test = read_selected(test_path)
    frames: list[pd.DataFrame] = []
    for zone in zones:
        train_path = root / f"Train_W_Zone{zone}.csv"
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        train = read_selected(train_path)
        zone_test = test.loc[test["ZONEID"] == zone]
        frame = pd.concat([train, zone_test], ignore_index=True)
        if not (frame["ZONEID"] == zone).all():
            raise ValueError(f"target ZONEID mismatch for zone {zone}")
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"duplicate target timestamp for zone {zone}")
        expected_days = 731 if allowed is None else len(allowed)
        if len(frame) != expected_days * 24:
            raise ValueError(
                f"zone {zone} has {len(frame)} target rows, "
                f"expected {expected_days * 24}"
            )
        observed = frame["TARGETVAR"].dropna().to_numpy(np.float64)
        if not np.isfinite(observed).all():
            raise ValueError(f"zone {zone} contains non-finite, non-NaN targets")
        if observed.size and (observed.min() < 0.0 or observed.max() > 1.0):
            raise ValueError(f"zone {zone} targets are outside [0,1]")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["day", "ZONEID", "timestamp"]).reset_index(drop=True)


def assign_calendar_roles(
    nwp_hours: pd.DataFrame,
    dates: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Attach mutually exclusive roles to predictor rows before target access."""

    result = nwp_hours.copy()
    result["role"] = pd.Series(pd.NA, index=result.index, dtype="string")
    day_values = result["day"].to_numpy(dtype="datetime64[D]")
    unknown = set(dates).difference(ALL_ROLES)
    if unknown:
        raise ValueError(f"unknown local calendar roles: {sorted(unknown)}")
    for role in LOCAL_ROLES:
        if role not in dates:
            continue
        mask = np.isin(day_values, dates[role].astype("datetime64[D]"))
        if result.loc[mask, "role"].notna().any():
            raise ValueError(f"calendar role assignment overlaps at {role}")
        result.loc[mask, "role"] = role
    # In a smoke view, most full-protocol dates are intentionally absent.
    return result.loc[result["role"].notna()].reset_index(drop=True)


def fill_targets_split_first(
    hours: pd.DataFrame,
    *,
    target_column: str = "TARGETVAR",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill target gaps without allowing information to cross a role or day.

    The input must already contain a ``role`` column.  Each role is processed
    in a separate loop and the fill group is exactly
    ``(ZONEID, calendar day)``.  The returned ``TARGETVAR_RAW`` column and
    ``raw_missing_mask`` preserve the source data for clean-mask loss/scoring.
    """

    required = {"role", "ZONEID", "day", "timestamp", target_column}
    missing_columns = required.difference(hours.columns)
    if missing_columns:
        raise ValueError(f"missing target-fill columns: {sorted(missing_columns)}")
    if hours["role"].isna().any():
        raise ValueError("every target row must receive a role before filling")

    parts: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    affected_zone_days: dict[str, int] = {}
    for role in sorted(hours["role"].astype(str).unique().tolist()):
        part = hours.loc[hours["role"] == role].copy()
        part["TARGETVAR_RAW"] = part[target_column].astype(np.float64)
        part["raw_missing_mask"] = part["TARGETVAR_RAW"].isna()
        counts[role] = int(part["raw_missing_mask"].sum())
        affected_zone_days[role] = int(
            part.loc[part["raw_missing_mask"], ["ZONEID", "day"]]
            .drop_duplicates()
            .shape[0]
        )
        part[target_column] = part.groupby(
            ["ZONEID", "day"], sort=False, observed=True
        )["TARGETVAR_RAW"].transform(lambda value: value.ffill().bfill())
        if part[target_column].isna().any():
            unresolved = (
                part.loc[part[target_column].isna(), ["ZONEID", "day"]]
                .drop_duplicates()
                .astype(str)
                .to_dict("records")
            )
            raise ValueError(
                f"role {role} contains an all-missing zone-day that cannot be filled: "
                f"{unresolved}"
            )
        parts.append(part)
    result = pd.concat(parts).sort_index().reset_index(drop=True)
    audit = {
        "policy": "split-first; forward then backward within (role,zone,calendar_day)",
        "cross_role_fill": False,
        "cross_day_fill": False,
        "raw_missing_cells_by_role": counts,
        "affected_zone_days_by_role": affected_zone_days,
        "total_raw_missing_cells": int(sum(counts.values())),
    }
    return result, audit


@dataclass(frozen=True)
class MaskedStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        axes: int | tuple[int, ...],
        observed_mask: np.ndarray | None = None,
    ) -> "MaskedStandardizer":
        source = np.asarray(values, dtype=np.float64)
        if observed_mask is not None:
            mask = np.asarray(observed_mask, dtype=bool)
            if mask.shape != source.shape:
                raise ValueError("standardizer mask shape does not match values")
            source = np.where(mask, source, np.nan)
            mean = np.nanmean(source, axis=axes, keepdims=True)
            std = np.nanstd(source, axis=axes, keepdims=True)
        else:
            if not np.isfinite(source).all():
                raise ValueError("unmasked standardizer input contains non-finite values")
            mean = np.mean(source, axis=axes, keepdims=True)
            std = np.std(source, axis=axes, keepdims=True)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("a standardizer coordinate has no observed training values")
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


@dataclass(frozen=True)
class ArchitectureSplitData:
    """One joint ten-zone sample per calendar day."""

    condition: np.ndarray
    raw_condition: np.ndarray
    target: np.ndarray
    target_standard: np.ndarray
    target_raw: np.ndarray
    raw_missing_mask: np.ndarray
    day: np.ndarray
    zones: np.ndarray
    role: str

    def __post_init__(self) -> None:
        if self.condition.ndim != 4 or self.condition.shape[1:3] != (10, 24):
            raise ValueError("condition must have shape [day,10,24,feature]")
        if self.condition.shape[-1] != 20:
            raise ValueError("architecture-v1 condition dimension must be 20")
        expected = self.condition.shape[:3]
        for name, value in (
            ("target", self.target),
            ("target_standard", self.target_standard),
            ("target_raw", self.target_raw),
            ("raw_missing_mask", self.raw_missing_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must align with [day,zone,hour]")
        if self.raw_condition.shape != (*expected, len(NWP_FEATURES)):
            raise ValueError("raw_condition must contain the ten NWP features")
        if self.day.shape != (expected[0],):
            raise ValueError("day vector does not align with samples")
        if not np.array_equal(self.zones, np.arange(1, 11, dtype=np.int64)):
            raise ValueError("zones must be ordered 1..10")
        if not np.isfinite(self.condition).all() or not np.isfinite(self.target).all():
            raise ValueError("model-facing arrays must be finite")
        if not np.array_equal(np.isnan(self.target_raw), self.raw_missing_mask):
            raise ValueError("raw target NaNs and raw_missing_mask disagree")
        if float(self.target.min()) < 0.0 or float(self.target.max()) > 1.0:
            raise ValueError("filled targets must remain in [0,1]")

    def __len__(self) -> int:
        return len(self.day)

    @property
    def observed_mask(self) -> np.ndarray:
        return ~self.raw_missing_mask

    @property
    def ramp_observed_mask(self) -> np.ndarray:
        return self.observed_mask[..., 1:] & self.observed_mask[..., :-1]

    @property
    def state(self) -> np.ndarray:
        """Return 0/1/2 states, neutralizing imputed cells to interior.

        The neutralization prevents an atom identity inferred from an imputed
        target from leaking into model inputs for neighbouring observed cells.
        Trainers must still pass ``observed_mask`` and exclude missing cells
        from every supervised loss.
        """

        result = np.full(self.target.shape, INTERIOR_STATE, dtype=np.int64)
        observed = self.observed_mask
        result[observed & (self.target == 0.0)] = ZERO_STATE
        result[observed & (self.target == 1.0)] = ONE_STATE
        return result


@dataclass(frozen=True)
class ArchitectureDataBundle:
    train: ArchitectureSplitData
    validation: ArchitectureSplitData
    calibration: ArchitectureSplitData
    selection: ArchitectureSplitData
    r_seen: ArchitectureSplitData
    final: None
    condition_standardizer: MaskedStandardizer
    target_standardizer: MaskedStandardizer
    protocol: ArchitectureProtocol
    manifest: Mapping[str, Any]

    def role(self, name: str) -> ArchitectureSplitData:
        if name == "final":
            self.protocol.require_final_available()
        if name not in LOCAL_ROLES:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class ArchitectureFitDataBundle:
    """Stage-scoped train/validation view for formal candidate fitting.

    Calibration, selection, R-SEEN and final targets are deliberately absent
    from this type.  A formal fitting runner therefore cannot reach them by
    accidentally calling ``bundle.selection`` or indexing a generic role map.
    """

    train: ArchitectureSplitData
    validation: ArchitectureSplitData
    condition_standardizer: MaskedStandardizer
    target_standardizer: MaskedStandardizer
    protocol: ArchitectureProtocol
    manifest: Mapping[str, Any]

    @property
    def materialized_roles(self) -> tuple[str, str]:
        return ("train", "validation")

    def role(self, name: str) -> ArchitectureSplitData:
        if name not in self.materialized_roles:
            raise RuntimeError(
                f"formal fitting bundle does not materialize role {name!r}; "
                "calibration/selection remain sealed"
            )
        return getattr(self, name)


@dataclass(frozen=True)
class ArchitectureTrainDataBundle:
    """Train-only target view for discarded engineering preflights.

    Unlike :class:`ArchitectureFitDataBundle`, this type cannot expose even
    validation targets.  It is used by formal-v2.2 P0, where validation-bank
    construction and checkpoint selection are forbidden by protocol.
    """

    train: ArchitectureSplitData
    condition_standardizer: MaskedStandardizer
    target_standardizer: MaskedStandardizer
    protocol: ArchitectureProtocol
    manifest: Mapping[str, Any]

    @property
    def materialized_roles(self) -> tuple[str]:
        return ("train",)

    def role(self, name: str) -> ArchitectureSplitData:
        if name != "train":
            raise RuntimeError(
                f"train-only engineering bundle does not materialize role {name!r}"
            )
        return self.train


def _join_roles_and_targets(
    assigned_nwp: pd.DataFrame,
    target_hours: pd.DataFrame,
) -> pd.DataFrame:
    target_keys = target_hours.loc[
        :, ["ZONEID", "TIMESTAMP", "timestamp", "day", "TARGETVAR"]
    ]
    result = assigned_nwp.merge(
        target_keys,
        on=["ZONEID", "TIMESTAMP", "timestamp", "day"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (result["_merge"] == "both").all():
        raise ValueError("a predictor row has no matching target row")
    return result.drop(columns="_merge")


def _role_raw_arrays(hours: pd.DataFrame, role: str) -> dict[str, np.ndarray]:
    selected = hours.loc[hours["role"] == role].sort_values(
        ["day", "ZONEID", "timestamp"]
    )
    days = np.sort(selected["day"].to_numpy(dtype="datetime64[D]").astype("datetime64[D]"))
    days = np.unique(days)
    expected_rows = len(days) * 10 * 24
    if len(selected) != expected_rows:
        raise ValueError(
            f"role {role} has {len(selected)} hourly rows, expected {expected_rows}"
        )
    for day, block in selected.groupby("day", sort=True):
        counts = block.groupby("ZONEID", sort=True).size().to_numpy()
        if len(counts) != 10 or not np.array_equal(counts, np.full(10, 24)):
            raise ValueError(f"role {role}, day {day} is not a complete 10x24 predictor grid")
    shape = (len(days), 10, 24)
    return {
        "day": days,
        "condition": selected.loc[:, NWP_FEATURES]
        .to_numpy(np.float32)
        .reshape(*shape, len(NWP_FEATURES)),
        "target": selected["TARGETVAR"].to_numpy(np.float32).reshape(shape),
        "target_raw": selected["TARGETVAR_RAW"].to_numpy(np.float32).reshape(shape),
        "missing": selected["raw_missing_mask"].to_numpy(bool).reshape(shape),
    }


def _array_fingerprint(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if np.issubdtype(value.dtype, np.datetime64):
            value = value.astype("datetime64[D]").astype(np.int64)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _input_file_manifest(data_dir: Path, zones: Sequence[int]) -> dict[str, Any]:
    predictor_names = [
        name
        for zone in zones
        for name in (f"Train_W_Zone{zone}.csv", f"TestPred_W_Zone{zone}.csv")
    ]
    target_names = [f"Train_W_Zone{zone}.csv" for zone in zones]
    target_names.append("TestTar_W.csv")

    def records(names: Iterable[str]) -> list[dict[str, Any]]:
        result = []
        for name in sorted(set(names)):
            path = data_dir / name
            result.append(
                {
                    "path": name,
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
        return result

    predictor = records(predictor_names)
    target = records(target_names)
    return {
        "predictor_files": predictor,
        "target_files": target,
        "predictor_files_sha256": canonical_sha256({"files": predictor}),
        "target_files_sha256": canonical_sha256({"files": target}),
    }


def build_architecture_v1_data(
    data_dir: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    zones: Sequence[int] = tuple(range(1, 11)),
    smoke: bool | None = None,
) -> ArchitectureDataBundle:
    """Build the leakage-safe joint data bundle; this function never trains."""

    if tuple(zones) != tuple(range(1, 11)):
        raise ValueError("architecture-v1 joint data require ordered zones 1..10")

    # First load predictors and freeze roles.  Target access happens only after
    # these two steps, which makes the split-first ordering executable rather
    # than merely documentary.
    provisional_protocol = build_architecture_protocol(
        config_path, data_dir=data_dir, zones=zones, smoke=smoke
    )
    root = _resolve_data_dir(data_dir, provisional_protocol)
    nwp = load_nwp_hours(root, zones)
    available = np.sort(nwp["day"].to_numpy(dtype="datetime64[D]").astype("datetime64[D]"))
    available = np.unique(available)
    if date_list_sha256(available) != provisional_protocol.manifest[
        "predictor_calendar"
    ]["date_sha256"]:
        raise ValueError("loaded NWP calendar drifted after predictor-only protocol audit")
    protocol = provisional_protocol
    assigned = assign_calendar_roles(nwp, protocol.dates)

    targets = load_target_hours(root, zones)
    joined = _join_roles_and_targets(assigned, targets)
    filled, missing_audit = fill_targets_split_first(joined)

    raw = {role: _role_raw_arrays(filled, role) for role in LOCAL_ROLES}
    train_raw = raw["train"]
    condition_standardizer = MaskedStandardizer.fit(
        train_raw["condition"], axes=(0, 1)
    )
    target_standardizer = MaskedStandardizer.fit(
        train_raw["target_raw"],
        axes=(0, 1),
        observed_mask=~train_raw["missing"],
    )
    zone_one_hot = np.eye(10, dtype=np.float32)
    zone_one_hot = np.broadcast_to(zone_one_hot[None, :, None, :], (1, 10, 24, 10))

    def make_split(role: str) -> ArchitectureSplitData:
        values = raw[role]
        scaled_nwp = condition_standardizer.transform(values["condition"])
        repeated_zone = np.broadcast_to(
            zone_one_hot, (len(values["day"]), 10, 24, 10)
        )
        condition = np.concatenate([scaled_nwp, repeated_zone], axis=-1)
        return ArchitectureSplitData(
            condition=np.ascontiguousarray(condition, dtype=np.float32),
            raw_condition=np.ascontiguousarray(values["condition"], dtype=np.float32),
            target=np.ascontiguousarray(values["target"], dtype=np.float32),
            target_standard=np.ascontiguousarray(
                target_standardizer.transform(values["target"]), dtype=np.float32
            ),
            target_raw=np.ascontiguousarray(values["target_raw"], dtype=np.float32),
            raw_missing_mask=np.ascontiguousarray(values["missing"], dtype=bool),
            day=np.ascontiguousarray(values["day"], dtype="datetime64[D]"),
            zones=np.arange(1, 11, dtype=np.int64),
            role=role,
        )

    splits = {role: make_split(role) for role in LOCAL_ROLES}
    split_arrays_sha = {
        role: _array_fingerprint(
            day=split.day,
            condition=split.raw_condition,
            target_raw=split.target_raw,
            raw_missing_mask=split.raw_missing_mask,
        )
        for role, split in splits.items()
    }
    manifest = dict(protocol.manifest)
    manifest["data_audit"] = {
        "input_files": _input_file_manifest(root, zones),
        "split_first_missing": missing_audit,
        "split_array_sha256": split_arrays_sha,
        "condition_standardizer": {
            "fit_role": "train",
            "mean_sha256": hashlib.sha256(
                condition_standardizer.mean.tobytes()
            ).hexdigest(),
            "std_sha256": hashlib.sha256(
                condition_standardizer.std.tobytes()
            ).hexdigest(),
        },
        "target_standardizer": {
            "fit_role": "raw-observed train cells only",
            "mean_sha256": hashlib.sha256(target_standardizer.mean.tobytes()).hexdigest(),
            "std_sha256": hashlib.sha256(target_standardizer.std.tobytes()).hexdigest(),
        },
        "model_facing_missing_target_state": "neutral interior state plus observed_mask",
        "clean_ramp_mask": "observed[t] AND observed[t+1]",
    }
    manifest["data_bundle_sha256"] = canonical_sha256(manifest)
    return ArchitectureDataBundle(
        train=splits["train"],
        validation=splits["validation"],
        calibration=splits["calibration"],
        selection=splits["selection"],
        r_seen=splits["r_seen"],
        final=None,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        protocol=protocol,
        manifest=manifest,
    )


def build_architecture_v1_fit_data(
    data_dir: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> ArchitectureFitDataBundle:
    """Materialize only formal train/validation targets, fail closed.

    This is the only data entry point authorized for formal R0/T0/T1 fitting.
    It never constructs calibration, selection, R-SEEN, or final target
    arrays.  The full protocol remains available as predictor-only date
    metadata so disjointness and frozen hashes can still be audited.
    """

    if tuple(zones) != tuple(range(1, 11)):
        raise ValueError("architecture-v1 joint data require ordered zones 1..10")

    protocol = build_architecture_protocol(
        config_path, data_dir=data_dir, zones=zones, smoke=False
    )
    if protocol.smoke:
        raise RuntimeError("formal fitting may not use a smoke protocol view")
    root = _resolve_data_dir(data_dir, protocol)
    nwp = load_nwp_hours(root, zones)
    available = np.unique(
        nwp["day"].to_numpy(dtype="datetime64[D]").astype("datetime64[D]")
    )
    if date_list_sha256(available) != protocol.manifest["predictor_calendar"][
        "date_sha256"
    ]:
        raise ValueError("loaded NWP calendar drifted after predictor-only protocol audit")

    fit_dates = {
        role: protocol.role_dates(role, full=True)
        for role in ("train", "validation")
    }
    assigned = assign_calendar_roles(nwp, fit_dates)
    allowed_days = np.sort(np.concatenate(list(fit_dates.values())))
    targets = load_target_hours(root, zones, allowed_days=allowed_days)
    joined = _join_roles_and_targets(assigned, targets)
    filled, missing_audit = fill_targets_split_first(joined)

    raw = {
        role: _role_raw_arrays(filled, role)
        for role in ("train", "validation")
    }
    train_raw = raw["train"]
    condition_standardizer = MaskedStandardizer.fit(
        train_raw["condition"], axes=(0, 1)
    )
    target_standardizer = MaskedStandardizer.fit(
        train_raw["target_raw"],
        axes=(0, 1),
        observed_mask=~train_raw["missing"],
    )
    zone_one_hot = np.eye(10, dtype=np.float32)
    zone_one_hot = np.broadcast_to(zone_one_hot[None, :, None, :], (1, 10, 24, 10))

    def make_split(role: str) -> ArchitectureSplitData:
        values = raw[role]
        scaled_nwp = condition_standardizer.transform(values["condition"])
        repeated_zone = np.broadcast_to(
            zone_one_hot, (len(values["day"]), 10, 24, 10)
        )
        condition = np.concatenate([scaled_nwp, repeated_zone], axis=-1)
        return ArchitectureSplitData(
            condition=np.ascontiguousarray(condition, dtype=np.float32),
            raw_condition=np.ascontiguousarray(values["condition"], dtype=np.float32),
            target=np.ascontiguousarray(values["target"], dtype=np.float32),
            target_standard=np.ascontiguousarray(
                target_standardizer.transform(values["target"]), dtype=np.float32
            ),
            target_raw=np.ascontiguousarray(values["target_raw"], dtype=np.float32),
            raw_missing_mask=np.ascontiguousarray(values["missing"], dtype=bool),
            day=np.ascontiguousarray(values["day"], dtype="datetime64[D]"),
            zones=np.arange(1, 11, dtype=np.int64),
            role=role,
        )

    train = make_split("train")
    validation = make_split("validation")
    split_arrays_sha = {
        role: _array_fingerprint(
            day=split.day,
            condition=split.raw_condition,
            target_raw=split.target_raw,
            raw_missing_mask=split.raw_missing_mask,
        )
        for role, split in (("train", train), ("validation", validation))
    }
    manifest = dict(protocol.manifest)
    forbidden_roles = ("calibration", "selection", "r_seen", "final")
    manifest["formal_fit_target_access"] = {
        "schema": "architecture_v1_target_access_v1",
        "materialized_roles": ["train", "validation"],
        "materialized_date_count": int(len(allowed_days)),
        "materialized_date_sha256": date_list_sha256(allowed_days),
        "materialized_role_date_sha256": {
            role: date_list_sha256(fit_dates[role])
            for role in ("train", "validation")
        },
        "forbidden_roles": list(forbidden_roles),
        "forbidden_target_arrays_materialized": False,
        "csv_filtering": "chunk_filter_by_frozen_day_before_bundle_concatenation",
    }
    manifest["data_audit"] = {
        "input_files": _input_file_manifest(root, zones),
        "split_first_missing": missing_audit,
        "split_array_sha256": split_arrays_sha,
        "condition_standardizer": {
            "fit_role": "train",
            "mean_sha256": hashlib.sha256(
                condition_standardizer.mean.tobytes()
            ).hexdigest(),
            "std_sha256": hashlib.sha256(
                condition_standardizer.std.tobytes()
            ).hexdigest(),
        },
        "target_standardizer": {
            "fit_role": "raw-observed train cells only",
            "mean_sha256": hashlib.sha256(target_standardizer.mean.tobytes()).hexdigest(),
            "std_sha256": hashlib.sha256(target_standardizer.std.tobytes()).hexdigest(),
        },
        "model_facing_missing_target_state": "neutral interior state plus observed_mask",
        "clean_ramp_mask": "observed[t] AND observed[t+1]",
    }
    manifest["fit_data_bundle_sha256"] = canonical_sha256(manifest)
    return ArchitectureFitDataBundle(
        train=train,
        validation=validation,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        protocol=protocol,
        manifest=manifest,
    )


def build_architecture_v1_train_data(
    data_dir: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    zones: Sequence[int] = tuple(range(1, 11)),
) -> ArchitectureTrainDataBundle:
    """Materialize only formal train targets for discarded P0 preflights.

    The returned object has no validation/calibration/selection/R-SEEN/final
    attributes.  Predictor-only protocol metadata still describes the full
    frozen calendar, but target arrays are constructed only for the train
    dates and no validation bank is built.
    """

    if tuple(zones) != tuple(range(1, 11)):
        raise ValueError("architecture-v1 joint data require ordered zones 1..10")

    protocol = build_architecture_protocol(
        config_path, data_dir=data_dir, zones=zones, smoke=False
    )
    if protocol.smoke:
        raise RuntimeError("formal train-only preflight may not use a smoke view")
    root = _resolve_data_dir(data_dir, protocol)
    nwp = load_nwp_hours(root, zones)
    available = np.unique(
        nwp["day"].to_numpy(dtype="datetime64[D]").astype("datetime64[D]")
    )
    if date_list_sha256(available) != protocol.manifest["predictor_calendar"][
        "date_sha256"
    ]:
        raise ValueError("loaded NWP calendar drifted after predictor-only audit")

    train_dates = protocol.role_dates("train", full=True)
    assigned = assign_calendar_roles(nwp, {"train": train_dates})
    targets = load_target_hours(root, zones, allowed_days=train_dates)
    joined = _join_roles_and_targets(assigned, targets)
    filled, missing_audit = fill_targets_split_first(joined)
    raw = _role_raw_arrays(filled, "train")

    condition_standardizer = MaskedStandardizer.fit(
        raw["condition"], axes=(0, 1)
    )
    target_standardizer = MaskedStandardizer.fit(
        raw["target_raw"],
        axes=(0, 1),
        observed_mask=~raw["missing"],
    )
    zone_one_hot = np.eye(10, dtype=np.float32)
    zone_one_hot = np.broadcast_to(zone_one_hot[None, :, None, :], (1, 10, 24, 10))
    scaled_nwp = condition_standardizer.transform(raw["condition"])
    repeated_zone = np.broadcast_to(
        zone_one_hot, (len(raw["day"]), 10, 24, 10)
    )
    train = ArchitectureSplitData(
        condition=np.ascontiguousarray(
            np.concatenate([scaled_nwp, repeated_zone], axis=-1),
            dtype=np.float32,
        ),
        raw_condition=np.ascontiguousarray(raw["condition"], dtype=np.float32),
        target=np.ascontiguousarray(raw["target"], dtype=np.float32),
        target_standard=np.ascontiguousarray(
            target_standardizer.transform(raw["target"]), dtype=np.float32
        ),
        target_raw=np.ascontiguousarray(raw["target_raw"], dtype=np.float32),
        raw_missing_mask=np.ascontiguousarray(raw["missing"], dtype=bool),
        day=np.ascontiguousarray(raw["day"], dtype="datetime64[D]"),
        zones=np.arange(1, 11, dtype=np.int64),
        role="train",
    )

    split_sha = _array_fingerprint(
        day=train.day,
        condition=train.raw_condition,
        target_raw=train.target_raw,
        raw_missing_mask=train.raw_missing_mask,
    )
    manifest = dict(protocol.manifest)
    manifest["formal_train_only_target_access"] = {
        "schema": "architecture_v1_target_access_v1",
        "materialized_roles": ["train"],
        "materialized_date_count": int(len(train_dates)),
        "materialized_date_sha256": date_list_sha256(train_dates),
        "forbidden_roles": [
            "validation",
            "calibration",
            "selection",
            "r_seen",
            "final",
        ],
        "forbidden_target_arrays_materialized": False,
        "validation_bank_constructed": False,
        "checkpoint_selection": "none",
        "csv_filtering": "chunk_filter_by_frozen_day_before_bundle_concatenation",
    }
    manifest["data_audit"] = {
        "input_files": _input_file_manifest(root, zones),
        "split_first_missing": missing_audit,
        "split_array_sha256": {"train": split_sha},
        "condition_standardizer": {
            "fit_role": "train",
            "mean_sha256": hashlib.sha256(
                condition_standardizer.mean.tobytes()
            ).hexdigest(),
            "std_sha256": hashlib.sha256(
                condition_standardizer.std.tobytes()
            ).hexdigest(),
        },
        "target_standardizer": {
            "fit_role": "raw-observed train cells only",
            "mean_sha256": hashlib.sha256(
                target_standardizer.mean.tobytes()
            ).hexdigest(),
            "std_sha256": hashlib.sha256(
                target_standardizer.std.tobytes()
            ).hexdigest(),
        },
        "model_facing_missing_target_state": "neutral interior state plus observed_mask",
        "clean_ramp_mask": "observed[t] AND observed[t+1]",
    }
    manifest["train_only_data_bundle_sha256"] = canonical_sha256(manifest)
    return ArchitectureTrainDataBundle(
        train=train,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        protocol=protocol,
        manifest=manifest,
    )


__all__ = [
    "ArchitectureDataBundle",
    "ArchitectureFitDataBundle",
    "ArchitectureSplitData",
    "BASE_NWP_FEATURES",
    "INTERIOR_STATE",
    "MaskedStandardizer",
    "NWP_FEATURES",
    "ONE_STATE",
    "ZERO_STATE",
    "assign_calendar_roles",
    "build_architecture_v1_data",
    "build_architecture_v1_fit_data",
    "fill_targets_split_first",
    "load_nwp_hours",
    "load_target_hours",
]
