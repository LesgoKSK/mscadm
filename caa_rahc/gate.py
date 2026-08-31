"""Low-capacity asymmetric tail gate fitted with a proper quantile loss.

The gate is deliberately independent of realized test outcomes.  It maps
forecast-only features, hour, and zone to two non-negative dilation factors.
The factors are learned by pinball loss on the lower/upper ten percent of a
baseline ensemble's continuous logit-quantile curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .features import FEATURE_NAMES, FeatureScaler


def _logit(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


class AsymmetricTailGate(nn.Module):
    """Centered additive hour/zone/forecast gate with two tail outputs."""

    def __init__(self, feature_dim: int = len(FEATURE_NAMES), base_logit: float = -3.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.base_logit = float(base_logit)
        self.global_effect = nn.Parameter(torch.zeros(2))
        self.feature_effect = nn.Parameter(torch.zeros(self.feature_dim, 2))
        self.hour_effect = nn.Parameter(torch.zeros(24, 2))
        self.zone_effect = nn.Parameter(torch.zeros(10, 2))
        self.zone_spread_effect = nn.Parameter(torch.zeros(10, 2))

    @staticmethod
    def _center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=0, keepdim=True)

    def logits(
        self, features: torch.Tensor, hour: torch.Tensor, zone: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must be [cells,feature_dim]")
        value = self.global_effect + features @ self.feature_effect
        value = value + self._center(self.hour_effect)[hour]
        value = value + self._center(self.zone_effect)[zone]
        # Index 1 is log ensemble spread in the frozen feature schema.
        value = value + self._center(self.zone_spread_effect)[zone] * features[:, 1:2]
        return (value + self.base_logit).clamp(-10.0, 6.0)

    def forward(
        self,
        features: torch.Tensor,
        hour: torch.Tensor,
        zone: torch.Tensor,
        *,
        maximum_lambda: float = 1.0,
    ) -> torch.Tensor:
        return float(maximum_lambda) * torch.sigmoid(self.logits(features, hour, zone))

    def penalty(self) -> torch.Tensor:
        hour = self._center(self.hour_effect)
        cyclic = hour - torch.roll(hour, shifts=1, dims=0)
        return (
            0.10 * self.global_effect.square().mean()
            + self.feature_effect.square().mean()
            + 0.25 * hour.square().mean()
            + cyclic.square().mean()
            + self._center(self.zone_effect).square().mean()
            + 2.0 * self._center(self.zone_spread_effect).square().mean()
        )


@dataclass
class FittedTailGate:
    model: AsymmetricTailGate
    scaler: FeatureScaler
    maximum_lambda: float
    fit_summary: dict[str, Any]

    def predict(self, features: np.ndarray, zone: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be in [0,1]")
        values = np.asarray(features)
        zones = np.asarray(zone, dtype=np.int64)
        if values.ndim != 3 or values.shape[1:] != (24, len(FEATURE_NAMES)):
            raise ValueError("features must have shape [case,24,7]")
        if zones.shape != (len(values),) or np.any((zones < 1) | (zones > 10)):
            raise ValueError("zone must have one value in 1..10 per case")
        scaled = self.scaler.transform(values).reshape(-1, len(FEATURE_NAMES))
        hour = np.broadcast_to(np.arange(24)[None, :], (len(values), 24)).reshape(-1)
        cell_zone = np.broadcast_to((zones - 1)[:, None], (len(values), 24)).reshape(-1)
        self.model.eval()
        with torch.no_grad():
            result = self.model(
                torch.from_numpy(scaled),
                torch.from_numpy(hour.astype(np.int64)),
                torch.from_numpy(cell_zone.astype(np.int64)),
                maximum_lambda=self.maximum_lambda * float(strength),
            )
        return result.numpy().reshape(len(values), 24, 2).astype(np.float64)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "caa_asymmetric_tail_gate",
            "feature_names": list(FEATURE_NAMES),
            "model_config": {
                "feature_dim": self.model.feature_dim,
                "base_logit": self.model.base_logit,
            },
            "state_dict": self.model.state_dict(),
            "feature_scaler": self.scaler.to_dict(),
            "maximum_lambda": self.maximum_lambda,
            "fit_summary": self.fit_summary,
            "metadata": metadata or {},
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["FittedTailGate", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("name") != "caa_asymmetric_tail_gate":
            raise ValueError("not a CAA asymmetric tail-gate checkpoint")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("tail-gate feature schema mismatch")
        model = AsymmetricTailGate(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        fitted = cls(
            model=model,
            scaler=FeatureScaler.from_dict(payload["feature_scaler"]),
            maximum_lambda=float(payload["maximum_lambda"]),
            fit_summary=dict(payload["fit_summary"]),
        )
        return fitted, dict(payload.get("metadata", {}))


def _tail_training_arrays(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    features_by_seed: Sequence[np.ndarray],
    zone: np.ndarray,
    *,
    anchor: float,
) -> dict[str, np.ndarray]:
    if len(scenarios_by_seed) == 0 or len(scenarios_by_seed) != len(features_by_seed):
        raise ValueError("scenarios/features need the same nonzero seed count")
    truth = np.asarray(observations, dtype=np.float64)
    zones = np.asarray(zone, dtype=np.int64)
    if truth.ndim != 2 or truth.shape[1] != 24 or zones.shape != (len(truth),):
        raise ValueError("observations/zone are not aligned daily arrays")
    feature_parts: list[np.ndarray] = []
    zone_parts: list[np.ndarray] = []
    hour_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    hinge_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    side_parts: list[np.ndarray] = []
    for scenarios, features in zip(scenarios_by_seed, features_by_seed):
        values = np.asarray(scenarios, dtype=np.float64)
        forecast = np.asarray(features)
        if values.ndim != 3 or values.shape[0] != len(truth) or values.shape[2] != 24:
            raise ValueError("each scenario array must be [case,member,24]")
        if forecast.shape != (len(truth), 24, len(FEATURE_NAMES)):
            raise ValueError("each feature array must be [case,24,7]")
        members = values.shape[1]
        q = (np.arange(members, dtype=np.float64) + 0.5) / members
        selected = (q < anchor) | (q > 1.0 - anchor)
        tail_q = q[selected]
        side = (tail_q > 0.5).astype(np.int64)
        # Cells first, members last.
        sorted_z = _logit(np.sort(values, axis=1, kind="stable")).transpose(0, 2, 1).reshape(-1, members)
        lower_anchor = np.quantile(sorted_z, anchor, axis=1)
        upper_anchor = np.quantile(sorted_z, 1.0 - anchor, axis=1)
        tail_z = sorted_z[:, selected]
        hinge = np.where(
            side[None, :] == 0,
            np.maximum(lower_anchor[:, None] - tail_z, 0.0),
            np.maximum(tail_z - upper_anchor[:, None], 0.0),
        )
        cells = len(truth) * 24
        feature_parts.append(forecast.reshape(cells, -1))
        zone_parts.append(np.broadcast_to((zones - 1)[:, None], (len(zones), 24)).reshape(-1))
        hour_parts.append(np.broadcast_to(np.arange(24)[None, :], (len(zones), 24)).reshape(-1))
        truth_parts.append(truth.reshape(-1))
        z_parts.append(tail_z)
        hinge_parts.append(hinge)
        q_parts.append(np.broadcast_to(tail_q[None, :], (cells, len(tail_q))))
        side_parts.append(np.broadcast_to(side[None, :], (cells, len(side))))
    return {
        "features": np.concatenate(feature_parts).astype(np.float32),
        "zone": np.concatenate(zone_parts).astype(np.int64),
        "hour": np.concatenate(hour_parts).astype(np.int64),
        "truth": np.concatenate(truth_parts).astype(np.float32),
        "z": np.concatenate(z_parts).astype(np.float32),
        "hinge": np.concatenate(hinge_parts).astype(np.float32),
        "q": np.concatenate(q_parts).astype(np.float32),
        "side": np.concatenate(side_parts).astype(np.int64),
    }


def fit_tail_gate(
    scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    features_by_seed: Sequence[np.ndarray],
    zone: np.ndarray,
    *,
    anchor: float = 0.10,
    maximum_lambda: float = 1.0,
    regularization: float = 0.03,
    steps: int = 800,
    batch_cells: int = 4096,
    learning_rate: float = 0.03,
    random_seed: int = 20260719,
    device: str = "cpu",
) -> FittedTailGate:
    """Fit a two-sided gate by lower/upper-tail pinball loss.

    All feature scaling and optimization use only the arrays passed here, so a
    caller can enforce date-grouped cross-fitting by passing the training fold.
    """

    if not 0.0 < float(anchor) < 0.5:
        raise ValueError("anchor must be in (0,0.5)")
    if maximum_lambda < 0.0 or regularization < 0.0 or steps < 1 or batch_cells < 1:
        raise ValueError("invalid tail-gate optimization parameter")
    arrays = _tail_training_arrays(
        scenarios_by_seed, observations, features_by_seed, zone, anchor=float(anchor)
    )
    scaler = FeatureScaler.fit(arrays["features"].reshape(-1, 1, len(FEATURE_NAMES)))
    scaled = scaler.transform(arrays["features"])
    selected_device = torch.device(device)
    torch.manual_seed(int(random_seed))
    model = AsymmetricTailGate().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    tensors = {
        "features": torch.from_numpy(scaled).to(selected_device),
        "zone": torch.from_numpy(arrays["zone"]).to(selected_device),
        "hour": torch.from_numpy(arrays["hour"]).to(selected_device),
        "truth": torch.from_numpy(arrays["truth"]).to(selected_device),
        "z": torch.from_numpy(arrays["z"]).to(selected_device),
        "hinge": torch.from_numpy(arrays["hinge"]).to(selected_device),
        "q": torch.from_numpy(arrays["q"]).to(selected_device),
        "side": torch.from_numpy(arrays["side"]).to(selected_device),
    }
    generator = torch.Generator(device=selected_device).manual_seed(int(random_seed) + 17)
    cells = len(arrays["truth"])
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, int(steps) + 1):
        if batch_cells >= cells:
            index = torch.arange(cells, device=selected_device)
        else:
            index = torch.randint(
                cells, (int(batch_cells),), generator=generator, device=selected_device
            )
        gate = model(
            tensors["features"][index],
            tensors["hour"][index],
            tensors["zone"][index],
            maximum_lambda=float(maximum_lambda),
        )
        side = tensors["side"][index]
        chosen_gate = torch.gather(gate, 1, side)
        direction = torch.where(side == 0, -1.0, 1.0)
        prediction = torch.sigmoid(
            tensors["z"][index] + direction * chosen_gate * tensors["hinge"][index]
        )
        error = tensors["truth"][index, None] - prediction
        q = tensors["q"][index]
        pinball = torch.maximum(q * error, (q - 1.0) * error).mean()
        penalty = model.penalty()
        loss = pinball + float(regularization) * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pinball": float(pinball.detach().cpu()),
                    "penalty": float(penalty.detach().cpu()),
                }
            )
    fitted = FittedTailGate(
        model=model.cpu().eval(),
        scaler=scaler,
        maximum_lambda=float(maximum_lambda),
        fit_summary={},
    )
    predictions = []
    offset = 0
    for features in features_by_seed:
        count = len(features)
        predictions.append(fitted.predict(features, np.asarray(zone), strength=1.0))
        offset += count
    all_gate = np.concatenate(predictions, axis=0)
    fitted.fit_summary = {
        "loss": history[-1]["loss"],
        "tail_pinball": history[-1]["pinball"],
        "penalty": history[-1]["penalty"],
        "history": history,
        "cells_per_seed": int(np.asarray(observations).size),
        "model_seeds_pooled": len(scenarios_by_seed),
        "tail_quantiles_per_cell": int(arrays["q"].shape[1]),
        "anchor": float(anchor),
        "maximum_lambda": float(maximum_lambda),
        "regularization": float(regularization),
        "steps": int(steps),
        "random_seed": int(random_seed),
        "mean_lower_gate": float(all_gate[..., 0].mean()),
        "mean_upper_gate": float(all_gate[..., 1].mean()),
        "maximum_fitted_gate": float(all_gate.max()),
        "objective": "unweighted proper pinball loss on q<anchor or q>1-anchor",
        "inference_features_only": True,
    }
    return fitted


__all__ = ["AsymmetricTailGate", "FittedTailGate", "fit_tail_gate"]
