"""CAA fitting and atom-aware, gated logit-tail scenario transformation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import betaincinv, expit, logit
from torch.nn import functional as F

from .features import FEATURE_NAMES, FeatureScaler, build_features
from .model import CAAStateModels


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    case = np.arange(values.shape[0])[:, None, None]
    hour = np.arange(values.shape[2])[None, None, :]
    ranks[case, order, hour] = np.arange(values.shape[1])[None, :, None]
    return ranks


def _flatten_features(
    scaled: np.ndarray, zone: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaled.ndim != 3 or scaled.shape[1:] != (24, len(FEATURE_NAMES)):
        raise ValueError("scaled features must have shape [case,24,feature]")
    zones = np.asarray(zone, dtype=np.int64)
    if zones.shape != (len(scaled),) or np.any((zones < 1) | (zones > 10)):
        raise ValueError("zone must align and be numbered 1..10")
    return (
        scaled.reshape(-1, len(FEATURE_NAMES)),
        np.tile(np.arange(24, dtype=np.int64), len(scaled)),
        np.repeat(zones - 1, 24),
    )


def _base_atom_probabilities(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    members = base.shape[1]
    count_zero = np.sum(base == 0.0, axis=1)
    count_one = np.sum(base == 1.0, axis=1)
    # Jeffreys smoothing is used only for the estimable zero atom.  The upper
    # atom is retained exactly because development validation has no y=1
    # observations and cannot support a state-dependent upper-atom model.
    zero = (count_zero + 0.5) / (members + 1.0)
    nonzero = members - count_zero
    conditional_one = np.divide(
        count_one,
        nonzero,
        out=np.zeros_like(count_one, dtype=np.float64),
        where=nonzero > 0,
    )
    return zero.astype(np.float64), conditional_one.astype(np.float64)


def _interior_rank_intervals(
    base: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(base)
    truth = np.asarray(observations)
    interior_member = (values > 0.0) & (values < 1.0)
    members = interior_member.sum(axis=1)
    lower = np.sum(interior_member & (values < truth[:, None, :]), axis=1)
    upper = np.sum(interior_member & (values <= truth[:, None, :]), axis=1)
    valid = (truth > 0.0) & (truth < 1.0) & (members >= 2)
    return (
        lower.astype(np.int64),
        upper.astype(np.int64),
        members.astype(np.int64),
        valid,
    )


def _variable_beta_binomial_log_pmf(
    rank: torch.Tensor,
    members: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    value = rank.to(alpha.dtype)
    total = members.to(alpha.dtype)
    log_choose = (
        torch.lgamma(total + 1.0)
        - torch.lgamma(value + 1.0)
        - torch.lgamma(total - value + 1.0)
    )
    log_beta_num = (
        torch.lgamma(value + alpha)
        + torch.lgamma(total - value + beta)
        - torch.lgamma(total + alpha + beta)
    )
    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return log_choose + log_beta_num - log_beta_den


def variable_censored_beta_binomial_nll(
    lower: torch.Tensor,
    upper: torch.Tensor,
    members: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    maximum_members: int,
) -> torch.Tensor:
    """Exact tie-interval likelihood with a cell-specific interior count."""

    tied = lower != upper
    values: list[torch.Tensor] = []
    if torch.any(~tied):
        values.append(
            _variable_beta_binomial_log_pmf(
                lower[~tied], members[~tied], alpha[~tied], beta[~tied]
            ).sum()
        )
    if torch.any(tied):
        grid = torch.arange(
            maximum_members + 1, dtype=alpha.dtype, device=alpha.device
        )
        log_probability = _variable_beta_binomial_log_pmf(
            grid[None, :],
            members[tied, None],
            alpha[tied, None],
            beta[tied, None],
        )
        valid = (
            (grid[None, :] >= lower[tied, None])
            & (grid[None, :] <= upper[tied, None])
            & (grid[None, :] <= members[tied, None])
        )
        values.append(
            torch.logsumexp(log_probability.masked_fill(~valid, -torch.inf), dim=1).sum()
        )
    return -torch.stack(values).sum() / len(lower)


def _empirical_logit_quantile(
    probability: np.ndarray, interior_values: np.ndarray, *, epsilon: float
) -> np.ndarray:
    values = np.sort(np.clip(interior_values.astype(np.float64), epsilon, 1.0 - epsilon))
    source = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return np.interp(probability, source, logit(values))


def _robust_logit_tail_quantile(
    probability: np.ndarray,
    interior_values: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    values = np.sort(np.clip(interior_values.astype(np.float64), epsilon, 1.0 - epsilon))
    z = logit(values)
    source = (np.arange(len(z), dtype=np.float64) + 0.5) / len(z)
    result = np.interp(probability, source, z)
    if len(z) >= 8:
        q = np.asarray((0.05, 0.15, 0.85, 0.95))
        knots = np.quantile(z, q)
        left = probability < q[0]
        right = probability > q[-1]
        if np.any(left):
            slope = (knots[1] - knots[0]) / (q[1] - q[0])
            result[left] = knots[0] + slope * (probability[left] - q[0])
        if np.any(right):
            slope = (knots[3] - knots[2]) / (q[3] - q[2])
            result[right] = knots[3] + slope * (probability[right] - q[3])
    bound = abs(float(logit(epsilon)))
    return np.clip(result, -bound, bound)


@dataclass(frozen=True)
class CAAConfig:
    atom_strength: float
    interior_strength: float
    gate_maximum: float
    gate_scale: float
    logit_delta_cap: float
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if not 0.0 <= self.atom_strength <= 1.0:
            raise ValueError("atom_strength must be in [0,1]")
        if not 0.0 <= self.interior_strength <= 1.0:
            raise ValueError("interior_strength must be in [0,1]")
        if not 0.0 <= self.gate_maximum <= 1.0:
            raise ValueError("gate_maximum must be in [0,1]")
        if self.gate_scale <= 0.0 or self.logit_delta_cap <= 0.0:
            raise ValueError("gate_scale and logit_delta_cap must be positive")
        if not 0.0 < self.epsilon < 0.01:
            raise ValueError("epsilon must be in (0,0.01)")

    def to_dict(self) -> dict[str, float]:
        return {
            "atom_strength": self.atom_strength,
            "interior_strength": self.interior_strength,
            "gate_maximum": self.gate_maximum,
            "gate_scale": self.gate_scale,
            "logit_delta_cap": self.logit_delta_cap,
            "epsilon": self.epsilon,
        }


@dataclass
class FitSummary:
    epochs: int
    seconds: float
    initial_atom_bce: float
    final_atom_bce: float
    initial_gate_bce: float
    final_gate_bce: float
    initial_interior_nll: float
    final_interior_nll: float
    zero_events: int
    gate_training_cells: int
    interior_rank_cells: int
    upper_events: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


class CAACalibrator:
    """CRPS-constrained atom-aware local-tail calibration model."""

    def __init__(
        self,
        scaler: FeatureScaler,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.scaler = scaler
        self.device = torch.device(device)
        self.models = CAAStateModels(len(FEATURE_NAMES)).to(self.device)
        self.fit_summary: FitSummary | None = None

    @classmethod
    def fit(
        cls,
        raw_scenarios: np.ndarray,
        base_scenarios: np.ndarray,
        observations: np.ndarray,
        zone: np.ndarray,
        *,
        regularization_atom: float = 0.1,
        regularization_gate: float = 0.2,
        regularization_interior: float = 0.1,
        epochs: int = 220,
        learning_rate: float = 0.03,
        seed: int = 20260719,
        device: str | torch.device | None = None,
    ) -> "CAACalibrator":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        features = build_features(raw_scenarios, base_scenarios)
        scaler = FeatureScaler.fit(features)
        calibrator = cls(scaler, device=selected_device)
        regime_np, hour_np, zone_np = _flatten_features(scaler.transform(features), zone)
        regime = torch.from_numpy(regime_np).to(selected_device)
        hour = torch.from_numpy(hour_np).to(selected_device)
        zones = torch.from_numpy(zone_np).to(selected_device)
        truth = np.asarray(observations).reshape(-1)

        base_zero, _ = _base_atom_probabilities(base_scenarios)
        base_zero_flat = torch.from_numpy(base_zero.reshape(-1).astype(np.float32)).to(selected_device)
        target_zero = torch.from_numpy((truth == 0.0).astype(np.float32)).to(selected_device)

        lower90, upper90 = np.quantile(base_scenarios, (0.05, 0.95), axis=1)
        interior_truth = (truth > 0.0) & (truth < 1.0)
        missed = ((observations < lower90) | (observations > upper90)).reshape(-1)
        gate_mask_np = interior_truth
        gate_mask = torch.from_numpy(gate_mask_np).to(selected_device)
        gate_target = torch.from_numpy(missed.astype(np.float32)).to(selected_device)

        lower_np, upper_np, members_np, rank_valid_np = _interior_rank_intervals(
            base_scenarios, observations
        )
        rank_valid = torch.from_numpy(rank_valid_np.reshape(-1)).to(selected_device)
        lower = torch.from_numpy(lower_np.reshape(-1)).to(selected_device)
        upper = torch.from_numpy(upper_np.reshape(-1)).to(selected_device)
        members = torch.from_numpy(members_np.reshape(-1)).to(selected_device)

        optimizer = torch.optim.Adam(calibrator.models.parameters(), lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs)), eta_min=float(learning_rate) * 0.05
        )
        initial = None
        final = None
        started = time.perf_counter()
        calibrator.models.train()
        base_logit = torch.logit(base_zero_flat.clamp(1e-5, 1.0 - 1e-5))
        risk_offset = float(logit(0.10))
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            zero_delta = calibrator.models.zero_atom(regime, hour, zones).squeeze(-1)
            zero_probability = torch.sigmoid(base_logit + zero_delta)
            atom_bce = F.binary_cross_entropy(zero_probability, target_zero)

            gate_delta = calibrator.models.miss_gate(regime, hour, zones).squeeze(-1)
            miss_probability = torch.sigmoid(risk_offset + gate_delta)
            gate_bce = F.binary_cross_entropy(
                miss_probability[gate_mask], gate_target[gate_mask]
            )

            alpha, beta, _ = calibrator.models.interior_rank(regime, hour, zones)
            interior_nll = variable_censored_beta_binomial_nll(
                lower[rank_valid],
                upper[rank_valid],
                members[rank_valid],
                alpha[rank_valid],
                beta[rank_valid],
                maximum_members=raw_scenarios.shape[1],
            )
            loss = (
                atom_bce
                + gate_bce
                + interior_nll
                + float(regularization_atom) * calibrator.models.penalty("zero_atom")
                + float(regularization_gate) * calibrator.models.penalty("miss_gate")
                + float(regularization_interior) * calibrator.models.penalty("interior_rank")
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CAA training loss")
            values = (
                float(atom_bce.detach().cpu()),
                float(gate_bce.detach().cpu()),
                float(interior_nll.detach().cpu()),
            )
            if initial is None:
                initial = values
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.models.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            final = values
        calibrator.models.eval()
        assert initial is not None and final is not None
        calibrator.fit_summary = FitSummary(
            epochs=int(epochs),
            seconds=time.perf_counter() - started,
            initial_atom_bce=initial[0],
            final_atom_bce=final[0],
            initial_gate_bce=initial[1],
            final_gate_bce=final[1],
            initial_interior_nll=initial[2],
            final_interior_nll=final[2],
            zero_events=int(np.sum(truth == 0.0)),
            gate_training_cells=int(np.sum(gate_mask_np)),
            interior_rank_cells=int(np.sum(rank_valid_np)),
            upper_events=int(np.sum(truth == 1.0)),
        )
        return calibrator

    @torch.no_grad()
    def _predict(
        self,
        raw_scenarios: np.ndarray,
        base_scenarios: np.ndarray,
        zone: np.ndarray,
        config: CAAConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features = build_features(raw_scenarios, base_scenarios)
        regime_np, hour_np, zone_np = _flatten_features(
            self.scaler.transform(features), zone
        )
        regime = torch.from_numpy(regime_np).to(self.device)
        hour = torch.from_numpy(hour_np).to(self.device)
        zones = torch.from_numpy(zone_np).to(self.device)
        base_zero, conditional_one = _base_atom_probabilities(base_scenarios)
        base_logit = torch.from_numpy(
            logit(np.clip(base_zero.reshape(-1), 1e-5, 1.0 - 1e-5)).astype(np.float32)
        ).to(self.device)
        zero_delta = self.models.zero_atom(
            regime, hour, zones, strength=config.atom_strength
        ).squeeze(-1)
        zero = torch.sigmoid(base_logit + zero_delta).cpu().numpy().reshape(base_zero.shape)
        one = (1.0 - zero) * conditional_one
        gate_delta = self.models.miss_gate(regime, hour, zones).squeeze(-1)
        risk = torch.sigmoid(float(logit(0.10)) + gate_delta).cpu().numpy().reshape(base_zero.shape)
        gate = np.clip((risk - 0.10) / config.gate_scale, 0.0, 1.0) * config.gate_maximum
        alpha, beta, _ = self.models.interior_rank(
            regime, hour, zones, strength=config.interior_strength
        )
        shape = base_zero.shape
        return (
            zero,
            one,
            gate,
            alpha.cpu().numpy().reshape(shape),
            beta.cpu().numpy().reshape(shape),
        )

    def transform(
        self,
        raw_scenarios: np.ndarray,
        base_scenarios: np.ndarray,
        zone: np.ndarray,
        *,
        config: CAAConfig,
    ) -> np.ndarray:
        raw = np.asarray(raw_scenarios)
        base = np.asarray(base_scenarios)
        if raw.shape != base.shape or raw.ndim != 3:
            raise ValueError("raw and base scenarios must share [case,member,hour] shape")
        zero, one, gate, alpha, beta = self._predict(raw, base, zone, config)
        cases, members, hours = base.shape
        desired = (np.arange(members, dtype=np.float64) + 0.5) / members
        sorted_base = np.sort(base.astype(np.float64), axis=1, kind="stable")
        calibrated_sorted = np.empty_like(sorted_base)
        for case in range(cases):
            for hour_index in range(hours):
                q = desired
                p0 = float(np.clip(zero[case, hour_index], 0.0, 0.999))
                p1 = float(np.clip(one[case, hour_index], 0.0, 1.0 - p0))
                result = np.empty(members, dtype=np.float64)
                lower_atom = q <= p0
                upper_atom = q >= 1.0 - p1
                continuous = ~(lower_atom | upper_atom)
                result[lower_atom] = 0.0
                result[upper_atom] = 1.0
                if np.any(continuous):
                    denominator = max(1.0 - p0 - p1, 1e-8)
                    qc = np.clip((q[continuous] - p0) / denominator, 1e-8, 1.0 - 1e-8)
                    cell = sorted_base[case, :, hour_index]
                    interior_values = cell[(cell > 0.0) & (cell < 1.0)]
                    if len(interior_values) < 4:
                        interior_values = np.clip(cell, config.epsilon, 1.0 - config.epsilon)
                    z0 = _empirical_logit_quantile(
                        qc, interior_values, epsilon=config.epsilon
                    )
                    state_probability = betaincinv(
                        float(alpha[case, hour_index]),
                        float(beta[case, hour_index]),
                        qc,
                    )
                    state_probability = np.nan_to_num(
                        state_probability, nan=qc, posinf=1.0, neginf=0.0
                    )
                    state_probability = np.clip(state_probability, 1e-8, 1.0 - 1e-8)
                    z1 = _robust_logit_tail_quantile(
                        state_probability, interior_values, epsilon=config.epsilon
                    )
                    delta = np.clip(
                        z1 - z0, -config.logit_delta_cap, config.logit_delta_cap
                    )
                    z = z0 + float(gate[case, hour_index]) * delta
                    result[continuous] = expit(z)
                calibrated_sorted[case, :, hour_index] = np.maximum.accumulate(result)
        calibrated = np.take_along_axis(
            calibrated_sorted, _ordinal_ranks(base), axis=1
        )
        return calibrated.astype(base.dtype, copy=False)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "caa_rahc",
            "scaler": self.scaler.to_dict(),
            "models": self.models.state_dict(),
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
    ) -> tuple["CAACalibrator", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("name") != "caa_rahc":
            raise ValueError("not a CAA-RAHC checkpoint")
        calibrator = cls(FeatureScaler.from_dict(payload["scaler"]), device=device)
        calibrator.models.load_state_dict(payload["models"])
        summary = payload.get("fit_summary")
        calibrator.fit_summary = None if summary is None else FitSummary(**summary)
        calibrator.models.eval()
        return calibrator, dict(payload.get("metadata", {}))


__all__ = [
    "CAAConfig",
    "CAACalibrator",
    "FitSummary",
    "variable_censored_beta_binomial_nll",
]
