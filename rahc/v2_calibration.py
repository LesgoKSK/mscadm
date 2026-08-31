"""Tie-aware RAHC fitting, monotone marginal transformation, and rank audit."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import betaincinv

from .v2_features import FEATURE_NAMES, FeatureScaler, build_regime_features
from .v2_model import HierarchicalRankModel, beta_binomial_log_pmf


TAIL_RULES = ("bounded", "linear", "clamp")


def rank_intervals(scenarios: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive pooled-rank intervals induced by forecast/target ties."""

    values = np.asarray(scenarios)
    truth = np.asarray(observations)
    if values.ndim != 3 or truth.shape != (len(values), values.shape[2]):
        raise ValueError("scenario and observation shapes do not align")
    lower = np.sum(values < truth[:, None, :], axis=1)
    upper = lower + np.sum(values == truth[:, None, :], axis=1)
    return lower.astype(np.int64), upper.astype(np.int64)


def _censored_nll_fast(
    lower: torch.Tensor,
    upper: torch.Tensor,
    members: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Exact interval-censored NLL, expanding the rank grid only for ties."""

    tied = upper != lower
    pieces: list[torch.Tensor] = []
    if torch.any(~tied):
        pieces.append(beta_binomial_log_pmf(lower[~tied], members, alpha[~tied], beta[~tied]).sum())
    if torch.any(tied):
        tied_lower = lower[tied]
        tied_upper = upper[tied]
        tied_alpha = alpha[tied]
        tied_beta = beta[tied]
        grid = torch.arange(members + 1, device=alpha.device, dtype=alpha.dtype)
        log_probability = beta_binomial_log_pmf(
            grid[None, :], members, tied_alpha[:, None], tied_beta[:, None]
        )
        valid = (grid[None, :] >= tied_lower[:, None]) & (grid[None, :] <= tied_upper[:, None])
        pieces.append(torch.logsumexp(log_probability.masked_fill(~valid, -torch.inf), dim=1).sum())
    return -torch.stack(pieces).sum() / len(lower)


def _flatten_features(
    scaled_features: np.ndarray, zone: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaled_features.ndim != 3 or scaled_features.shape[1:] != (24, len(FEATURE_NAMES)):
        raise ValueError("features must have shape [case,24,4]")
    zones = np.asarray(zone, dtype=np.int64)
    if zones.shape != (len(scaled_features),) or np.any((zones < 1) | (zones > 10)):
        raise ValueError("zone must align and be numbered 1..10")
    regime = scaled_features.reshape(-1, len(FEATURE_NAMES))
    hour = np.tile(np.arange(24, dtype=np.int64), len(scaled_features))
    zone_flat = np.repeat(zones - 1, 24)
    return regime, hour, zone_flat


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    case = np.arange(values.shape[0])[:, None, None]
    hour = np.arange(values.shape[2])[None, None, :]
    ranks[case, order, hour] = np.arange(values.shape[1])[None, :, None]
    return ranks


def _quantile_map(probabilities: np.ndarray, sorted_values: np.ndarray, tail_rule: str) -> np.ndarray:
    members = len(sorted_values)
    source = (np.arange(members, dtype=np.float64) + 0.5) / members
    if tail_rule == "bounded":
        # Physical-boundary pseudo-knots are less volatile than extrapolating
        # from only the two extreme order statistics.
        return np.interp(
            probabilities,
            np.concatenate(([0.0], source, [1.0])),
            np.concatenate(([0.0], sorted_values, [1.0])),
        )
    if tail_rule == "clamp":
        return np.interp(probabilities, source, sorted_values)
    if tail_rule != "linear":
        raise ValueError(f"unknown tail rule {tail_rule!r}")
    result = np.interp(probabilities, source, sorted_values)
    left = probabilities < source[0]
    right = probabilities > source[-1]
    if np.any(left):
        slope = (sorted_values[1] - sorted_values[0]) / (source[1] - source[0])
        result[left] = sorted_values[0] + slope * (probabilities[left] - source[0])
    if np.any(right):
        slope = (sorted_values[-1] - sorted_values[-2]) / (source[-1] - source[-2])
        result[right] = sorted_values[-1] + slope * (probabilities[right] - source[-1])
    return np.clip(result, 0.0, 1.0)


def apply_probability_map(
    scenarios: np.ndarray,
    raw_probabilities: np.ndarray,
    *,
    tail_rule: str,
) -> np.ndarray:
    """Apply case/hour probability grids while preserving stable member ranks."""

    values = np.asarray(scenarios)
    if values.ndim != 3:
        raise ValueError("scenarios must have shape [case,member,hour]")
    cases, members, hours = values.shape
    if raw_probabilities.shape != (cases, members, hours):
        raise ValueError("probability grid does not align with scenarios")
    if tail_rule not in TAIL_RULES:
        raise ValueError(f"tail_rule must be one of {TAIL_RULES}")
    sorted_values = np.sort(values.astype(np.float64), axis=1, kind="stable")
    calibrated_sorted = np.empty_like(sorted_values)
    for case in range(cases):
        for hour in range(hours):
            calibrated_sorted[case, :, hour] = _quantile_map(
                raw_probabilities[case, :, hour], sorted_values[case, :, hour], tail_rule
            )
    calibrated_sorted = np.maximum.accumulate(calibrated_sorted, axis=1)
    calibrated = np.take_along_axis(calibrated_sorted, _ordinal_ranks(values), axis=1)
    return calibrated.astype(values.dtype, copy=False)


@dataclass
class FitSummary:
    epochs: int
    initial_nll: float
    final_nll: float
    final_penalty: float
    final_loss: float
    seconds: float
    tied_cells: int
    total_cells: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


class RAHCalibratorV2:
    """Regime-adaptive hierarchical calibration with censored rank likelihood."""

    def __init__(
        self,
        variant: str,
        scaler: FeatureScaler,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.variant = variant
        self.scaler = scaler
        self.device = torch.device(device)
        self.model = HierarchicalRankModel(variant=variant, regime_dim=len(FEATURE_NAMES)).to(self.device)
        self.fit_summary: FitSummary | None = None

    @classmethod
    def fit(
        cls,
        scenarios: np.ndarray,
        observations: np.ndarray,
        zone: np.ndarray,
        *,
        variant: str,
        regularization: float,
        epochs: int = 160,
        learning_rate: float = 0.03,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> "RAHCalibratorV2":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        raw_features = build_regime_features(scenarios)
        scaler = FeatureScaler.fit(raw_features)
        calibrator = cls(variant, scaler, device=selected_device)
        regime_np, hour_np, zone_np = _flatten_features(scaler.transform(raw_features), zone)
        lower_np, upper_np = rank_intervals(scenarios, observations)
        regime = torch.from_numpy(regime_np).to(selected_device)
        hour = torch.from_numpy(hour_np).to(selected_device)
        zones = torch.from_numpy(zone_np).to(selected_device)
        lower = torch.from_numpy(lower_np.reshape(-1)).to(selected_device)
        upper = torch.from_numpy(upper_np.reshape(-1)).to(selected_device)
        optimizer = torch.optim.Adam(calibrator.model.parameters(), lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs), 1), eta_min=float(learning_rate) * 0.05
        )
        started = time.perf_counter()
        initial_nll = float("nan")
        final = (float("nan"), float("nan"), float("nan"))
        calibrator.model.train()
        for epoch in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            alpha, beta, _ = calibrator.model(regime, hour, zones)
            nll = _censored_nll_fast(lower, upper, scenarios.shape[1], alpha, beta)
            penalty = calibrator.model.penalty()
            loss = nll + float(regularization) * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite RAHC loss")
            if epoch == 0:
                initial_nll = float(nll.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            final = (
                float(nll.detach().cpu()),
                float(penalty.detach().cpu()),
                float(loss.detach().cpu()),
            )
        calibrator.model.eval()
        calibrator.fit_summary = FitSummary(
            epochs=int(epochs),
            initial_nll=initial_nll,
            final_nll=final[0],
            final_penalty=final[1],
            final_loss=final[2],
            seconds=time.perf_counter() - started,
            tied_cells=int(np.sum(lower_np != upper_np)),
            total_cells=int(lower_np.size),
        )
        return calibrator

    @torch.no_grad()
    def predict_shapes(
        self, scenarios: np.ndarray, zone: np.ndarray, *, strength: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_features = build_regime_features(scenarios)
        regime_np, hour_np, zone_np = _flatten_features(self.scaler.transform(raw_features), zone)
        regime = torch.from_numpy(regime_np).to(self.device)
        hour = torch.from_numpy(hour_np).to(self.device)
        zones = torch.from_numpy(zone_np).to(self.device)
        alpha, beta, _ = self.model(regime, hour, zones, strength=strength)
        shape = (len(scenarios), 24)
        return alpha.cpu().numpy().reshape(shape), beta.cpu().numpy().reshape(shape)

    def transform(
        self,
        scenarios: np.ndarray,
        zone: np.ndarray,
        *,
        strength: float = 1.0,
        tail_rule: str = "bounded",
    ) -> np.ndarray:
        alpha, beta = self.predict_shapes(scenarios, zone, strength=strength)
        members = scenarios.shape[1]
        desired = (np.arange(members, dtype=np.float64) + 0.5) / members
        probability = betaincinv(alpha[:, None, :], beta[:, None, :], desired[None, :, None])
        probability = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
        probability = np.maximum.accumulate(np.clip(probability, 0.0, 1.0), axis=1)
        return apply_probability_map(scenarios, probability, tail_rule=tail_rule)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "implementation": "RAHCalibratorV2",
            "variant": self.variant,
            "scaler": self.scaler.to_dict(),
            "model": self.model.state_dict(),
            "fit_summary": None if self.fit_summary is None else self.fit_summary.to_dict(),
            "metadata": metadata or {},
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> tuple["RAHCalibratorV2", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        calibrator = cls(payload["variant"], FeatureScaler.from_dict(payload["scaler"]), device=device)
        calibrator.model.load_state_dict(payload["model"])
        summary = payload.get("fit_summary")
        calibrator.fit_summary = None if summary is None else FitSummary(**summary)
        calibrator.model.eval()
        return calibrator, dict(payload.get("metadata", {}))


def rank_audit(before: np.ndarray, after: np.ndarray) -> dict[str, float | int]:
    """Audit strict reversals, tie changes, stable ranks, and boundaries."""

    raw = np.asarray(before)
    calibrated = np.asarray(after)
    if raw.shape != calibrated.shape or raw.ndim != 3:
        raise ValueError("before and after must have identical [case,member,hour] shapes")
    reversals = ties_broken = strict_collapsed = raw_ties = calibrated_ties = 0
    cases, members, hours = raw.shape
    triangle = np.triu(np.ones((members, members), dtype=bool), k=1)
    for case in range(cases):
        for hour in range(hours):
            left = raw[case, :, hour][:, None] - raw[case, :, hour][None, :]
            right = calibrated[case, :, hour][:, None] - calibrated[case, :, hour][None, :]
            first = left[triangle]
            second = right[triangle]
            reversals += int(np.sum(first * second < 0.0))
            ties_broken += int(np.sum((first == 0.0) & (second != 0.0)))
            strict_collapsed += int(np.sum((first != 0.0) & (second == 0.0)))
            raw_ties += int(np.sum(first == 0.0))
            calibrated_ties += int(np.sum(second == 0.0))
    raw_ranks = _ordinal_ranks(raw)
    calibrated_ranks = _ordinal_ranks(calibrated)
    rank_equal = raw_ranks == calibrated_ranks
    total_pairs = cases * hours * members * (members - 1) // 2
    return {
        "strict_reversals": reversals,
        "raw_ties_broken": ties_broken,
        "strict_pairs_collapsed": strict_collapsed,
        "raw_tied_pairs": raw_ties,
        "calibrated_tied_pairs": calibrated_ties,
        "stable_ordinal_rank_matches": int(rank_equal.sum()),
        "stable_ordinal_rank_total": int(rank_equal.size),
        "stable_ordinal_rank_fraction": float(rank_equal.mean()),
        "raw_boundary_fraction": float(np.mean((raw == 0.0) | (raw == 1.0))),
        "calibrated_boundary_fraction": float(
            np.mean((calibrated == 0.0) | (calibrated == 1.0))
        ),
        "pair_comparisons": int(total_pairs),
    }


__all__ = [
    "TAIL_RULES",
    "FitSummary",
    "RAHCalibratorV2",
    "apply_probability_map",
    "rank_audit",
    "rank_intervals",
]
