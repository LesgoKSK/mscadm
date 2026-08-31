"""Leakage-safe, low-dimensional regime features for the RAHC experiment.

Only quantities available from the *raw* forecast ensemble are used.  The
small feature set is deliberate: the calibration sample contains only 50
unique validation dates, so a large MLP would be difficult to justify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "raw_ensemble_mean",
    "log_raw_ensemble_sd",
    "absolute_mean_ramp",
    "distance_to_physical_boundary",
)


def build_regime_features(scenarios: np.ndarray) -> np.ndarray:
    """Return forecast-only features with shape ``[case, hour, 4]``."""

    if scenarios.ndim != 3 or scenarios.shape[2] != 24:
        raise ValueError("scenarios must have shape [case, member, 24]")
    values = scenarios.astype(np.float64, copy=False)
    mean = values.mean(axis=1)
    sd = values.std(axis=1)
    ramp = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
    boundary = np.minimum(mean, 1.0 - mean)
    result = np.stack((mean, np.log(sd + 1e-3), ramp, boundary), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("regime features contain non-finite values")
    return result.astype(np.float32)


@dataclass(frozen=True)
class FeatureScaler:
    """Training-fold-only standardization parameters."""

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

    def to_dict(self) -> dict:
        return {
            "names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureScaler":
        if tuple(payload["names"]) != FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
        )


__all__ = ["FEATURE_NAMES", "FeatureScaler", "build_regime_features"]
