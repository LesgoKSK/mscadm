from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import betaincinv

from .features import FEATURE_NAMES, RegimeFeatureScaler
from .model import HierarchicalBetaBinomial, beta_binomial_nll


def finite_ensemble_rank(scenarios: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Integer mid-rank of each observation among finite ensemble members."""

    if scenarios.ndim != 3 or observations.shape != (len(scenarios), scenarios.shape[2]):
        raise ValueError("scenario/observation shapes do not align")
    less = np.sum(scenarios < observations[:, None, :], axis=1)
    equal = np.sum(scenarios == observations[:, None, :], axis=1)
    rank = less + equal // 2
    return np.clip(rank, 0, scenarios.shape[1]).astype(np.int64)


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    days = np.arange(values.shape[0])[:, None, None]
    hours = np.arange(values.shape[2])[None, None, :]
    ranks[days, order, hours] = np.arange(values.shape[1])[None, :, None]
    return ranks


def _quantile_with_linear_tails(probability: np.ndarray, source: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.interp(probability, source, values)
    left = probability < source[0]
    right = probability > source[-1]
    if np.any(left):
        slope = (values[1] - values[0]) / (source[1] - source[0])
        result[left] = values[0] + slope * (probability[left] - source[0])
    if np.any(right):
        slope = (values[-1] - values[-2]) / (source[-1] - source[-2])
        result[right] = values[-1] + slope * (probability[right] - source[-1])
    return result


def strict_adjacent_rank_reversals(before: np.ndarray, after: np.ndarray) -> int:
    """Exact inversion audit: a monotone adjacent sequence implies no pair reversal."""

    if before.shape != after.shape:
        raise ValueError("before and after must have identical shapes")
    order = np.argsort(before, axis=1, kind="stable")
    sorted_after = np.take_along_axis(after, order, axis=1)
    return int(np.sum(np.diff(sorted_after, axis=1) < -1e-12))


@dataclass
class FitSummary:
    epochs: int
    initial_nll: float
    final_nll: float
    final_regularization: float
    final_loss: float
    seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


class RAHCalibrator:
    def __init__(
        self,
        *,
        variant: str,
        scaler: RegimeFeatureScaler,
        hidden_dim: int = 24,
        maximum_log_shape: float = 3.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.variant = variant
        self.scaler = scaler
        self.hidden_dim = int(hidden_dim)
        self.maximum_log_shape = float(maximum_log_shape)
        self.device = torch.device(device)
        self.model = HierarchicalBetaBinomial(
            variant=variant,
            regime_dim=len(FEATURE_NAMES),
            hidden_dim=self.hidden_dim,
            maximum_log_shape=self.maximum_log_shape,
        ).to(self.device)
        self.fit_summary: FitSummary | None = None

    @staticmethod
    def _flatten(
        features: np.ndarray, zone: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if features.ndim != 3 or features.shape[1:] != (24, len(FEATURE_NAMES)):
            raise ValueError("features must have shape [days,24,regime_dim]")
        if zone.shape != (len(features),):
            raise ValueError("zone must align with features")
        regime = features.reshape(-1, features.shape[-1])
        hour = np.tile(np.arange(24, dtype=np.int64), len(features))
        zones = np.repeat(zone.astype(np.int64) - 1, 24)
        if np.any((zones < 0) | (zones >= 10)):
            raise ValueError("zones must be numbered 1..10")
        return regime, hour, zones

    @classmethod
    def fit(
        cls,
        scenarios: np.ndarray,
        observations: np.ndarray,
        features: np.ndarray,
        zone: np.ndarray,
        *,
        variant: str,
        regularization: float,
        epochs: int = 400,
        learning_rate: float = 0.03,
        hidden_dim: int = 24,
        maximum_log_shape: float = 3.5,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> "RAHCalibrator":
        import time

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        scaler = RegimeFeatureScaler.fit(features)
        calibrator = cls(
            variant=variant,
            scaler=scaler,
            hidden_dim=hidden_dim,
            maximum_log_shape=maximum_log_shape,
            device=selected_device,
        )
        regime_np, hour_np, zone_np = cls._flatten(scaler.transform(features), zone)
        rank_np = finite_ensemble_rank(scenarios, observations).reshape(-1)
        regime = torch.from_numpy(regime_np).to(selected_device)
        hour = torch.from_numpy(hour_np).to(selected_device)
        zones = torch.from_numpy(zone_np).to(selected_device)
        rank = torch.from_numpy(rank_np).to(selected_device)
        optimizer = torch.optim.Adam(calibrator.model.parameters(), lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=learning_rate * 0.05)
        started = time.perf_counter()
        initial_nll = float("nan")
        final_values = (float("nan"), float("nan"), float("nan"))
        calibrator.model.train()
        for epoch in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            alpha, beta, log_shapes = calibrator.model(regime, hour, zones)
            nll = beta_binomial_nll(rank, scenarios.shape[1], alpha, beta)
            penalty = calibrator.model.regularization(log_shapes)
            loss = nll + float(regularization) * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite RAHC training loss")
            if epoch == 0:
                initial_nll = float(nll.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            final_values = (float(nll.detach()), float(penalty.detach()), float(loss.detach()))
        calibrator.model.eval()
        calibrator.fit_summary = FitSummary(
            epochs=int(epochs),
            initial_nll=initial_nll,
            final_nll=final_values[0],
            final_regularization=final_values[1],
            final_loss=final_values[2],
            seconds=time.perf_counter() - started,
        )
        return calibrator

    @torch.no_grad()
    def predict_shapes(
        self,
        features: np.ndarray,
        zone: np.ndarray,
        *,
        strength: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.scaler.transform(features)
        regime_np, hour_np, zone_np = self._flatten(transformed, zone)
        regime = torch.from_numpy(regime_np).to(self.device)
        hour = torch.from_numpy(hour_np).to(self.device)
        zones = torch.from_numpy(zone_np).to(self.device)
        self.model.eval()
        alpha, beta, _ = self.model(regime, hour, zones, strength=strength)
        shape = (len(features), 24)
        return alpha.cpu().numpy().reshape(shape), beta.cpu().numpy().reshape(shape)

    def transform(
        self,
        scenarios: np.ndarray,
        features: np.ndarray,
        zone: np.ndarray,
        *,
        strength: float = 1.0,
    ) -> np.ndarray:
        if scenarios.ndim != 3 or scenarios.shape[2] != 24:
            raise ValueError("scenarios must have shape [days,members,24]")
        alpha, beta = self.predict_shapes(features, zone, strength=strength)
        days, members, hours = scenarios.shape
        desired = (np.arange(members, dtype=np.float64) + 0.5) / members
        raw_probability = betaincinv(alpha[:, None, :], beta[:, None, :], desired[None, :, None])
        raw_probability = np.nan_to_num(raw_probability, nan=0.5, posinf=1.0, neginf=0.0)
        raw_probability = np.clip(raw_probability, 0.0, 1.0)
        source_probability = (np.arange(members, dtype=np.float64) + 0.5) / members
        sorted_values = np.sort(scenarios.astype(np.float64), axis=1, kind="stable")
        calibrated_sorted = np.empty_like(sorted_values)
        for day_index in range(days):
            for hour_index in range(hours):
                calibrated_sorted[day_index, :, hour_index] = _quantile_with_linear_tails(
                    raw_probability[day_index, :, hour_index],
                    source_probability,
                    sorted_values[day_index, :, hour_index],
                )
        calibrated_sorted = np.maximum.accumulate(calibrated_sorted, axis=1)
        calibrated_sorted = np.clip(calibrated_sorted, 0.0, 1.0)
        calibrated = np.take_along_axis(calibrated_sorted, _ordinal_ranks(scenarios), axis=1)
        return calibrated.astype(scenarios.dtype, copy=False)

    def state_payload(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "maximum_log_shape": self.maximum_log_shape,
            "feature_scaler": self.scaler.to_dict(),
            "model": self.model.state_dict(),
            "fit_summary": None if self.fit_summary is None else self.fit_summary.to_dict(),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_payload()
        payload["metadata"] = metadata or {}
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> tuple["RAHCalibrator", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        calibrator = cls(
            variant=payload["variant"],
            scaler=RegimeFeatureScaler.from_dict(payload["feature_scaler"]),
            hidden_dim=int(payload["hidden_dim"]),
            maximum_log_shape=float(payload["maximum_log_shape"]),
            device=device,
        )
        calibrator.model.load_state_dict(payload["model"])
        summary = payload.get("fit_summary")
        calibrator.fit_summary = None if summary is None else FitSummary(**summary)
        calibrator.model.eval()
        return calibrator, dict(payload.get("metadata", {}))

    def describe(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "maximum_log_shape": self.maximum_log_shape,
            "feature_scaler": self.scaler.to_dict(),
            "fit_summary": None if self.fit_summary is None else self.fit_summary.to_dict(),
            "definition": "G(y|x)=BetaCDF(F_raw(y|x); alpha(x), beta(x))",
            "inverse_sampling": "F_raw^-1(BetaPPF(q; alpha(x), beta(x)))",
            "finite_ensemble_likelihood": "Beta-Binomial rank likelihood with deterministic integer mid-ranks",
        }


__all__ = [
    "RAHCalibrator",
    "FitSummary",
    "finite_ensemble_rank",
    "strict_adjacent_rank_reversals",
]
