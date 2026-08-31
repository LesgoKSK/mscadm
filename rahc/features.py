from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "forecast_mean",
    "log_ensemble_sd",
    "interval_width_90",
    "absolute_mean_ramp",
    "distance_to_boundary",
    "ensemble_skew_proxy",
    "nwp_ws10_standardized",
    "nwp_ws100_standardized",
    "day_of_year_sin",
    "day_of_year_cos",
)


@dataclass(frozen=True)
class RegimeFeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RegimeFeatureScaler":
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "names": list(FEATURE_NAMES)}

    @classmethod
    def from_dict(cls, payload: dict) -> "RegimeFeatureScaler":
        if tuple(payload["names"]) != FEATURE_NAMES:
            raise ValueError("feature names do not match the RAHC feature protocol")
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))


class RegimeFeatureBuilder:
    """Build inference-available regime features without using observations."""

    @staticmethod
    def build(
        scenarios: np.ndarray,
        condition: np.ndarray,
        day: np.ndarray,
    ) -> np.ndarray:
        if scenarios.ndim != 3 or scenarios.shape[2] != 24:
            raise ValueError("scenarios must have shape [days, members, 24]")
        if condition.shape[:2] != (len(scenarios), 24) or condition.shape[2] < 6:
            raise ValueError("condition must have shape [days, 24, >=6]")
        if len(day) != len(scenarios):
            raise ValueError("day must align with scenarios")
        mean = scenarios.mean(axis=1, dtype=np.float64)
        sd = scenarios.std(axis=1, dtype=np.float64)
        lower, median, upper = np.quantile(scenarios, [0.05, 0.5, 0.95], axis=1)
        width = upper - lower
        ramp = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
        boundary = np.minimum(mean, 1.0 - mean)
        skew = (upper + lower - 2.0 * median) / np.maximum(width, 1e-4)
        date = day.astype("datetime64[D]")
        year = date.astype("datetime64[Y]")
        doy = (date - year).astype("timedelta64[D]").astype(np.float64)
        year_length = ((year + np.timedelta64(1, "Y")).astype("datetime64[D]") - year.astype("datetime64[D]"))
        year_length = year_length.astype("timedelta64[D]").astype(np.float64)
        angle = 2.0 * np.pi * doy / year_length
        sine = np.repeat(np.sin(angle)[:, None], 24, axis=1)
        cosine = np.repeat(np.cos(angle)[:, None], 24, axis=1)
        values = np.stack(
            [
                mean,
                np.log(sd + 1e-3),
                width,
                ramp,
                boundary,
                skew,
                condition[..., 4],
                condition[..., 5],
                sine,
                cosine,
            ],
            axis=-1,
        )
        if not np.isfinite(values).all():
            raise ValueError("regime features contain non-finite values")
        return values.astype(np.float32)


__all__ = ["FEATURE_NAMES", "RegimeFeatureBuilder", "RegimeFeatureScaler"]
