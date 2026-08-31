"""Train-only structural-zero model and atom-aware mixture utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


def zero_model_features(condition: np.ndarray) -> np.ndarray:
    """Low-capacity NWP/zone/hour design matrix available before realization."""

    values = np.asarray(condition, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (24, 20):
        raise ValueError("condition must have shape [case,24,20]")
    nwp = values[..., :10]
    zone = values[..., 10:20]
    hour = np.arange(24, dtype=np.float64)
    sine = np.broadcast_to(np.sin(2.0 * np.pi * hour / 24.0)[None, :, None], (*values.shape[:2], 1))
    cosine = np.broadcast_to(np.cos(2.0 * np.pi * hour / 24.0)[None, :, None], (*values.shape[:2], 1))
    design = np.concatenate((nwp, nwp.square(), zone, sine, cosine), axis=-1)
    return design.reshape(-1, design.shape[-1])


@dataclass
class StructuralZeroModel:
    coefficient: np.ndarray
    intercept: float
    upper_conditional_probability: float
    regularization_c: float
    train_zero_events: int
    train_upper_events: int
    train_cells: int

    @classmethod
    def fit(
        cls,
        condition: np.ndarray,
        target: np.ndarray,
        *,
        regularization_c: float = 0.1,
        maximum_iterations: int = 1000,
    ) -> "StructuralZeroModel":
        truth = np.asarray(target)
        if truth.shape != condition.shape[:2]:
            raise ValueError("target must align with condition cases/hours")
        design = zero_model_features(condition)
        zero = (truth.reshape(-1) == 0.0).astype(np.int64)
        if zero.sum() == 0 or zero.sum() == len(zero):
            raise ValueError("zero model training requires both zero and nonzero outcomes")
        fitted = LogisticRegression(
            C=float(regularization_c),
            penalty="l2",
            solver="lbfgs",
            max_iter=int(maximum_iterations),
            class_weight=None,
            random_state=20260719,
        ).fit(design, zero)
        upper = int(np.sum(truth == 1.0))
        nonzero = int(np.sum(truth > 0.0))
        # Strong global Jeffreys estimate.  With 100 members this probability
        # is normally below the representable 0.5% half-grid threshold and
        # therefore creates no artificial upper-atom member.
        upper_probability = (upper + 0.5) / (nonzero + 1.0)
        return cls(
            coefficient=fitted.coef_.reshape(-1).astype(np.float64),
            intercept=float(fitted.intercept_.item()),
            upper_conditional_probability=float(upper_probability),
            regularization_c=float(regularization_c),
            train_zero_events=int(zero.sum()),
            train_upper_events=upper,
            train_cells=int(len(zero)),
        )

    def predict_zero(self, condition: np.ndarray) -> np.ndarray:
        design = zero_model_features(condition)
        probability = expit(design @ self.coefficient + self.intercept)
        return probability.reshape(condition.shape[:2])

    def analytic_scores(self, condition: np.ndarray, target: np.ndarray) -> dict[str, float]:
        probability = np.clip(self.predict_zero(condition), 1e-8, 1.0 - 1e-8)
        zero = np.asarray(target) == 0.0
        return {
            "zero_probability_mean": float(probability.mean()),
            "zero_event_rate": float(zero.mean()),
            "zero_Brier": float(np.mean((probability - zero) ** 2)),
            "zero_log_loss": float(
                -np.mean(zero * np.log(probability) + (~zero) * np.log1p(-probability))
            ),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "caa_structural_zero",
            "coefficient": torch.from_numpy(self.coefficient),
            "intercept": self.intercept,
            "upper_conditional_probability": self.upper_conditional_probability,
            "regularization_c": self.regularization_c,
            "train_zero_events": self.train_zero_events,
            "train_upper_events": self.train_upper_events,
            "train_cells": self.train_cells,
            "metadata": metadata or {},
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["StructuralZeroModel", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("name") != "caa_structural_zero":
            raise ValueError("not a structural-zero checkpoint")
        model = cls(
            coefficient=payload["coefficient"].numpy().astype(np.float64),
            intercept=float(payload["intercept"]),
            upper_conditional_probability=float(payload["upper_conditional_probability"]),
            regularization_c=float(payload["regularization_c"]),
            train_zero_events=int(payload["train_zero_events"]),
            train_upper_events=int(payload["train_upper_events"]),
            train_cells=int(payload["train_cells"]),
        )
        return model, dict(payload.get("metadata", {}))


def scenario_zero_probability(scenarios: np.ndarray) -> np.ndarray:
    values = np.asarray(scenarios)
    members = values.shape[1]
    count = np.sum(values == 0.0, axis=1)
    return (count + 0.5) / (members + 1.0)


def blend_zero_probability(
    base_scenarios: np.ndarray,
    structural_probability: np.ndarray,
    strength: float,
) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("atom blend strength must be in [0,1]")
    base = np.clip(scenario_zero_probability(base_scenarios), 1e-6, 1.0 - 1e-6)
    structural = np.clip(structural_probability, 1e-6, 1.0 - 1e-6)
    return expit((1.0 - strength) * logit(base) + strength * logit(structural))


__all__ = [
    "StructuralZeroModel",
    "blend_zero_probability",
    "scenario_zero_probability",
    "zero_model_features",
]
