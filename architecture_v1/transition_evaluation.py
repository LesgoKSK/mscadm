"""Physical-scale archive and daily metrics for the TGO-v1 Probe.

This module is array-only.  It cannot load a target role, select a checkpoint,
or train a model.  It validates outer-held-out scenario archives and computes
the registered physical-power endpoints with calendar day as the inference
unit.  Ensemble members are finite predictive members, never replicates for
statistical inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .formal_evaluation import validation_per_day_metrics


ARCHIVE_SCHEMA = "architecture_v1_tgo_v1_scenario_archive_v1_3"
ARCHIVE_MANIFEST_SCHEMA = "architecture_v1_tgo_v1_scenario_manifest_v1_3"
METRIC_NAMES = (
    "level_CRPS",
    "ramp_CRPS",
    "normalized_joint_ES",
    "lagged_increment_variogram_score",
    "coverage90",
    "width90",
    "daily_mean_power_CRPS",
    "late_horizon_level_CRPS",
    "three_hour_window_energy_score",
    "max_up_ramp_CRPS",
    "max_down_ramp_CRPS",
    "up_ramp_threshold_Brier",
    "down_ramp_threshold_Brier",
    "ramp_threshold_Brier",
    "zero_Brier",
    "one_Brier",
    "atom_state_Brier",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_float(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must be floating point")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _empirical_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = _finite_float(samples, name="CRPS samples")
    observed = _finite_float(truth, name="CRPS truth")
    if values.ndim < 2 or observed.shape != (values.shape[0], *values.shape[2:]):
        raise ValueError("CRPS samples and truth do not align")
    members = values.shape[1]
    if members < 2:
        raise ValueError("CRPS requires at least two members")
    first = np.mean(np.abs(values - observed[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    weights = 2.0 * np.arange(1, members + 1) - members - 1.0
    shape = (1, members, *([1] * (values.ndim - 2)))
    pair = np.sum(ordered * weights.reshape(shape), axis=1) / (members * members)
    result = first - pair
    if np.any(result < -1e-10):
        raise FloatingPointError("empirical CRPS became negative")
    return np.maximum(result, 0.0)


def _daily_masked_mean(
    values: np.ndarray, mask: np.ndarray, *, name: str
) -> np.ndarray:
    score = _finite_float(values, name=name)
    valid = np.asarray(mask, dtype=bool)
    if score.shape != valid.shape:
        raise ValueError(f"{name} and mask do not align")
    count = valid.reshape(len(valid), -1).sum(axis=1)
    if np.any(count == 0):
        raise ValueError(f"a day has no valid {name} cells")
    return np.where(valid, score, 0.0).reshape(len(valid), -1).sum(axis=1) / count


def fit_train_only_ramp_thresholds(
    observation: np.ndarray,
    observed_mask: np.ndarray,
    *,
    upper_quantile: float = 0.95,
    lower_quantile: float = 0.05,
) -> dict[str, float | int]:
    truth = _finite_float(observation, name="ramp-threshold observations")
    mask = np.asarray(observed_mask, dtype=bool)
    if truth.ndim != 3 or truth.shape[1:] != (10, 24) or mask.shape != truth.shape:
        raise ValueError("ramp-threshold arrays must have shape [day,10,24]")
    if not 0.0 < lower_quantile < 0.5 < upper_quantile < 1.0:
        raise ValueError("ramp threshold quantiles are invalid")
    ramps = np.diff(truth, axis=-1)
    valid = mask[..., :-1] & mask[..., 1:]
    selected = ramps[valid]
    if len(selected) < 100:
        raise ValueError("too few observed train-only ramps")
    down = float(np.quantile(selected, lower_quantile))
    up = float(np.quantile(selected, upper_quantile))
    if not down < 0.0 < up:
        raise RuntimeError("train-only ramp thresholds do not straddle zero")
    return {
        "observed_train_ramps": int(len(selected)),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "down_threshold": down,
        "up_threshold": up,
    }


def _window_energy_score(
    scenarios: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    days, members, zones, hours = scenarios.shape
    if window < 1 or window > hours:
        raise ValueError("energy-score window is invalid")
    result = np.zeros(days, dtype=np.float64)
    counts = np.zeros(days, dtype=np.int64)
    for start in range(hours - window + 1):
        stop = start + window
        local_mask = mask[..., start:stop]
        for day in range(days):
            valid = local_mask[day].reshape(-1)
            dimension = int(valid.sum())
            if dimension == 0:
                continue
            sample = scenarios[day, :, :, start:stop].reshape(members, -1)[:, valid]
            target = truth[day, :, start:stop].reshape(-1)[valid]
            scale = np.sqrt(float(dimension))
            first = np.linalg.norm(sample - target[None], axis=1).mean() / scale
            pair = sample[:, None] - sample[None, :]
            second = 0.5 * np.linalg.norm(pair, axis=-1).mean() / scale
            result[day] += first - second
            counts[day] += 1
    if np.any(counts == 0):
        raise ValueError("a day has no valid local energy-score window")
    return np.maximum(result / counts, 0.0)


def transition_per_day_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    observed_mask: np.ndarray,
    *,
    zero_probability: np.ndarray,
    one_probability: np.ndarray,
    ramp_thresholds: Mapping[str, float | int],
) -> dict[str, np.ndarray]:
    sample = _finite_float(scenarios, name="scenarios")
    truth = _finite_float(observations, name="observations")
    mask = np.asarray(observed_mask, dtype=bool)
    if sample.ndim != 4 or sample.shape[2:] != (10, 24):
        raise ValueError("scenarios must have shape [day,member,10,24]")
    if truth.shape != (len(sample), 10, 24) or mask.shape != truth.shape:
        raise ValueError("truth/mask do not align with scenarios")
    base = validation_per_day_metrics(
        sample,
        truth,
        mask,
        zero_probability=zero_probability,
        one_probability=one_probability,
    )

    count = mask.reshape(len(mask), -1).sum(axis=1)
    member_mean = np.where(mask[:, None], sample, 0.0).reshape(
        len(sample), sample.shape[1], -1
    ).sum(axis=-1) / count[:, None]
    truth_mean = np.where(mask, truth, 0.0).reshape(len(truth), -1).sum(axis=-1) / count
    daily_mean = _empirical_crps(member_mean, truth_mean)

    late_sample = sample[..., -6:]
    late_truth = truth[..., -6:]
    late_mask = mask[..., -6:]
    late_cell = _empirical_crps(late_sample, late_truth)
    late_valid = late_mask.reshape(len(late_mask), -1).any(axis=1)
    late = np.full(len(sample), np.nan, dtype=np.float64)
    if bool(late_valid.any()):
        late[late_valid] = _daily_masked_mean(
            late_cell[late_valid], late_mask[late_valid], name="late-horizon level CRPS"
        )

    ramp = np.diff(sample, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)
    ramp_mask = mask[..., :-1] & mask[..., 1:]
    max_up_sample = np.empty((len(sample), sample.shape[1]), dtype=np.float64)
    max_down_sample = np.empty_like(max_up_sample)
    max_up_truth = np.empty(len(sample), dtype=np.float64)
    max_down_truth = np.empty(len(sample), dtype=np.float64)
    for day in range(len(sample)):
        valid = ramp_mask[day].reshape(-1)
        if not bool(valid.any()):
            raise ValueError("a day has no observed ramp for maximum-ramp CRPS")
        member_ramp = ramp[day].reshape(sample.shape[1], -1)[:, valid]
        target_ramp = truth_ramp[day].reshape(-1)[valid]
        max_up_sample[day] = member_ramp.max(axis=1)
        max_down_sample[day] = (-member_ramp).max(axis=1)
        max_up_truth[day] = target_ramp.max()
        max_down_truth[day] = (-target_ramp).max()

    up_threshold = float(ramp_thresholds["up_threshold"])
    down_threshold = float(ramp_thresholds["down_threshold"])
    forecast_up = np.mean(ramp >= up_threshold, axis=1)
    forecast_down = np.mean(ramp <= down_threshold, axis=1)
    truth_up = truth_ramp >= up_threshold
    truth_down = truth_ramp <= down_threshold
    up_brier = _daily_masked_mean(
        (forecast_up - truth_up) ** 2,
        ramp_mask,
        name="upper-ramp threshold Brier",
    )
    down_brier = _daily_masked_mean(
        (forecast_down - truth_down) ** 2,
        ramp_mask,
        name="lower-ramp threshold Brier",
    )

    result = {
        name: np.asarray(base[name], dtype=np.float64)
        for name in (
            "level_CRPS",
            "ramp_CRPS",
            "normalized_joint_ES",
            "lagged_increment_variogram_score",
            "coverage90",
            "width90",
            "zero_Brier",
            "one_Brier",
            "atom_state_Brier",
        )
    }
    result.update(
        {
            "daily_mean_power_CRPS": daily_mean,
            "late_horizon_level_CRPS": late,
            "three_hour_window_energy_score": _window_energy_score(
                sample, truth, mask, window=3
            ),
            "max_up_ramp_CRPS": _empirical_crps(max_up_sample, max_up_truth),
            "max_down_ramp_CRPS": _empirical_crps(
                max_down_sample, max_down_truth
            ),
            "up_ramp_threshold_Brier": up_brier,
            "down_ramp_threshold_Brier": down_brier,
            "ramp_threshold_Brier": 0.5 * (up_brier + down_brier),
        }
    )
    ordered = {name: result[name] for name in METRIC_NAMES}
    if set(result) != set(METRIC_NAMES):
        raise RuntimeError("TGO-v1 metric registry drifted")
    if any(value.shape != (len(sample),) for value in ordered.values()):
        raise RuntimeError("TGO-v1 metric is not a per-day vector")
    if not np.array_equal(np.isnan(ordered["late_horizon_level_CRPS"]), ~late_valid):
        raise FloatingPointError("late-horizon metric missingness drifted")
    if any(
        not np.isfinite(value).all()
        for name, value in ordered.items()
        if name != "late_horizon_level_CRPS"
    ):
        raise FloatingPointError("TGO-v1 metric contains non-finite values")
    return ordered


def validate_transition_archive_arrays(
    *,
    scenarios: Any,
    observations: Any,
    observed_mask: Any,
    raw_missing_mask: Any,
    states: Any,
    zero_probability: Any,
    one_probability: Any,
    day: Any,
    zones: Any,
) -> dict[str, np.ndarray]:
    if np.asarray(scenarios).dtype != np.float64:
        raise TypeError("TGO-v1.1 scenario archive requires FP64 values")
    sample = _finite_float(scenarios, name="scenarios")
    truth = _finite_float(observations, name="observations")
    if sample.ndim != 4 or sample.shape[2:] != (10, 24) or sample.shape[1] < 2:
        raise ValueError("scenario archive shape is invalid")
    expected = (len(sample), 10, 24)
    if truth.shape != expected:
        raise ValueError("archive truth shape is invalid")
    if sample.min() < 0.0 or sample.max() > 1.0:
        raise ValueError("archive scenarios lie outside [0,1]")
    observed = np.asarray(observed_mask)
    missing = np.asarray(raw_missing_mask)
    if (
        observed.dtype != np.bool_
        or missing.dtype != np.bool_
        or observed.shape != expected
        or missing.shape != expected
        or not np.array_equal(observed, ~missing)
    ):
        raise ValueError("archive observation masks are invalid")
    state = np.asarray(states)
    if state.shape != sample.shape or not np.isin(state, (0, 1, 2)).all():
        raise ValueError("archive atom states are invalid")
    if not np.all(sample[state == 0] == 0.0) or not np.all(sample[state == 2] == 1.0):
        raise ValueError("archive boundary states are not exact")
    interior = sample[state == 1]
    if np.any(interior <= 0.0) or np.any(interior >= 1.0):
        raise ValueError("archive interior values reached an exact boundary")
    zero = _finite_float(zero_probability, name="zero probability")
    one = _finite_float(one_probability, name="one probability")
    if zero.shape != expected or one.shape != expected:
        raise ValueError("archive atom probabilities do not align")
    if np.any(zero < 0.0) or np.any(one < 0.0) or np.any(zero + one > 1.0 + 1e-6):
        raise ValueError("archive atom probabilities are invalid")
    dates = np.asarray(day).astype("datetime64[D]")
    zone_ids = np.asarray(zones, dtype=np.int64)
    if dates.shape != (len(sample),) or len(np.unique(dates)) != len(dates):
        raise ValueError("archive calendar days are invalid")
    if not np.array_equal(zone_ids, np.arange(1, 11, dtype=np.int64)):
        raise ValueError("archive zone order is invalid")
    return {
        "scenarios": np.ascontiguousarray(sample, dtype=np.float64),
        "observations": truth.astype(np.float32),
        "observed_mask": observed,
        "raw_missing_mask": missing,
        "states": state.astype(np.int8),
        "zero_probability": zero.astype(np.float32),
        "one_probability": one.astype(np.float32),
        "day": dates,
        "zones": zone_ids,
    }


def write_transition_archive(
    path: Path,
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[str, Path]:
    if path.suffix != ".npz":
        raise ValueError("TGO-v1 scenario archive must end in .npz")
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite TGO-v1 archive: {path}")
    validated = validate_transition_archive_arrays(**arrays)
    meta = dict(_jsonable(metadata))
    required = {
        "schema",
        "config_sha256",
        "training_freeze_sha256",
        "atom_checkpoint_sha256",
        "denoiser_checkpoint_sha256",
        "outer_fold",
        "model_seed",
        "path_id",
        "sampling_seed",
        "members",
        "DDIM_steps",
        "member_chunk",
        "atom_allocation_sha256",
        "native_epsilon_sha256",
        "target_state_argument_used_for_sampling",
    }
    if set(meta) < required or meta["schema"] != ARCHIVE_SCHEMA:
        raise ValueError("TGO-v1 archive metadata is incomplete")
    if meta["target_state_argument_used_for_sampling"] is not False:
        raise ValueError("held-out target atom states cannot condition sampling")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **validated,
        metadata=np.asarray(
            json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
    )
    temporary.replace(path)
    digest = _file_sha256(path)
    _atomic_json(
        manifest_path,
        {
            "schema": ARCHIVE_MANIFEST_SCHEMA,
            "archive": path.name,
            "sha256": digest,
            "bytes": int(path.stat().st_size),
            "days": int(len(validated["day"])),
            "members": int(validated["scenarios"].shape[1]),
            "metadata": meta,
        },
    )
    return digest, manifest_path


def load_transition_archive(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _file_sha256(path)
    if (
        manifest.get("schema") != ARCHIVE_MANIFEST_SCHEMA
        or manifest.get("archive") != path.name
        or manifest.get("sha256") != digest
        or int(manifest.get("bytes", -1)) != path.stat().st_size
    ):
        raise RuntimeError("TGO-v1 archive manifest drifted")
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        arrays = validate_transition_archive_arrays(
            scenarios=stored["scenarios"],
            observations=stored["observations"],
            observed_mask=stored["observed_mask"],
            raw_missing_mask=stored["raw_missing_mask"],
            states=stored["states"],
            zero_probability=stored["zero_probability"],
            one_probability=stored["one_probability"],
            day=stored["day"],
            zones=stored["zones"],
        )
    if metadata != manifest["metadata"]:
        raise RuntimeError("TGO-v1 archive metadata copies differ")
    return arrays, metadata, digest


def aggregate_metric_replicates(
    replicates: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not replicates:
        raise ValueError("at least one TGO-v1 metric replicate is required")
    if any(tuple(item) != METRIC_NAMES for item in replicates):
        raise ValueError("TGO-v1 metric replicate schema drifted")
    result: dict[str, np.ndarray] = {}
    for name in METRIC_NAMES:
        values = [np.asarray(item[name], dtype=np.float64) for item in replicates]
        if any(value.shape != values[0].shape or value.ndim != 1 for value in values):
            raise ValueError(f"metric replicate {name} does not align")
        result[name] = np.mean(np.stack(values, axis=0), axis=0)
    return result


__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA",
    "ARCHIVE_SCHEMA",
    "METRIC_NAMES",
    "aggregate_metric_replicates",
    "fit_train_only_ramp_thresholds",
    "load_transition_archive",
    "transition_per_day_metrics",
    "validate_transition_archive_arrays",
    "write_transition_archive",
]
