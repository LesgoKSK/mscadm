"""Forecast-only features for atom, gate, and interior CAA components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "raw_ensemble_mean",
    "log_raw_ensemble_sd",
    "absolute_raw_mean_ramp",
    "distance_to_physical_boundary",
    "base_interval_width_90",
    "smoothed_logit_raw_p0",
    "smoothed_logit_raw_p1",
)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def build_features(raw_scenarios: np.ndarray, base_scenarios: np.ndarray) -> np.ndarray:
    """Return inference-available features with shape ``[case,24,7]``."""

    raw = np.asarray(raw_scenarios, dtype=np.float64)
    base = np.asarray(base_scenarios, dtype=np.float64)
    if raw.ndim != 3 or raw.shape != base.shape or raw.shape[2] != 24:
        raise ValueError("raw and base scenarios must share shape [case,member,24]")
    members = raw.shape[1]
    mean = raw.mean(axis=1)
    sd = raw.std(axis=1)
    ramp = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
    boundary = np.minimum(mean, 1.0 - mean)
    lower, upper = np.quantile(base, (0.05, 0.95), axis=1)
    width = upper - lower
    count_zero = np.sum(raw == 0.0, axis=1)
    count_one = np.sum(raw == 1.0, axis=1)
    p0 = (count_zero + 0.5) / (members + 1.0)
    p1 = (count_one + 0.5) / (members + 1.0)
    result = np.stack(
        (
            mean,
            np.log(sd + 1e-3),
            ramp,
            boundary,
            width,
            _logit(p0),
            _logit(p1),
        ),
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise ValueError("CAA features contain non-finite values")
    return result.astype(np.float32)


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        flat = np.asarray(values, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureScaler":
        if tuple(payload["names"]) != FEATURE_NAMES:
            raise ValueError("CAA feature schema mismatch")
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
        )


__all__ = ["FEATURE_NAMES", "FeatureScaler", "build_features"]
