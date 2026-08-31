"""Analytic versus finite-ensemble boundary-event diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .metrics import analytic_atom_scores


def reliability_bins(
    probability: np.ndarray,
    event: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(event, dtype=bool).reshape(-1)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability/event arrays are invalid")
    if bins < 2:
        raise ValueError("bins must be at least two")
    # Equal-count bins remain informative for a rare, strongly skewed event.
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, int(bins))
    result = []
    for index, selected in enumerate(chunks):
        if len(selected) == 0:
            continue
        result.append(
            {
                "bin": index + 1,
                "count": int(len(selected)),
                "predicted_mean": float(p[selected].mean()),
                "observed_rate": float(y[selected].mean()),
                "predicted_minimum": float(p[selected].min()),
                "predicted_maximum": float(p[selected].max()),
            }
        )
    return result


def ensemble_atom_quantization(
    scenarios: np.ndarray,
    analytic_pi0: np.ndarray,
    analytic_pi1: np.ndarray | float,
) -> dict[str, float | int]:
    values = np.asarray(scenarios)
    if values.ndim != 3:
        raise ValueError("scenarios must be [case,member,hour]")
    shape = (values.shape[0], values.shape[2])
    p0 = np.broadcast_to(np.asarray(analytic_pi0, dtype=np.float64), shape)
    p1 = np.broadcast_to(np.asarray(analytic_pi1, dtype=np.float64), shape)
    realized0 = np.mean(values == 0.0, axis=1)
    realized1 = np.mean(values == 1.0, axis=1)
    error0 = np.abs(realized0 - p0)
    error1 = np.abs(realized1 - p1)
    return {
        "members": int(values.shape[1]),
        "probability_resolution": float(1.0 / values.shape[1]),
        "zero_quantization_MAE": float(error0.mean()),
        "zero_quantization_max_abs": float(error0.max()),
        "one_quantization_MAE": float(error1.mean()),
        "one_quantization_max_abs": float(error1.max()),
        "realized_zero_fraction": float(realized0.mean()),
        "realized_one_fraction": float(realized1.mean()),
        "analytic_zero_probability_mean": float(p0.mean()),
        "analytic_one_probability_mean": float(p1.mean()),
    }


def pooled_atom_diagnostics(
    zero_probability_parts: Sequence[np.ndarray],
    upper_probability_parts: Sequence[np.ndarray | float],
    observation_parts: Sequence[np.ndarray],
    scenario_parts: Sequence[np.ndarray] | None = None,
) -> dict[str, object]:
    if not (
        len(zero_probability_parts)
        == len(upper_probability_parts)
        == len(observation_parts)
        and len(observation_parts) > 0
    ):
        raise ValueError("atom diagnostic part catalogs differ")
    p0 = np.concatenate([np.asarray(value) for value in zero_probability_parts])
    truth = np.concatenate([np.asarray(value) for value in observation_parts])
    p1_parts = [
        np.broadcast_to(np.asarray(value, dtype=np.float64), np.asarray(obs).shape)
        for value, obs in zip(upper_probability_parts, observation_parts)
    ]
    p1 = np.concatenate(p1_parts)
    result: dict[str, object] = {
        "analytic_scores": analytic_atom_scores(p0, p1, truth),
        "zero_reliability": reliability_bins(p0, truth == 0.0),
        "one_reliability": reliability_bins(p1, truth == 1.0),
    }
    if scenario_parts is not None:
        if len(scenario_parts) != len(observation_parts):
            raise ValueError("scenario part catalog differs")
        scenarios = np.concatenate([np.asarray(value) for value in scenario_parts])
        result["finite_ensemble"] = ensemble_atom_quantization(scenarios, p0, p1)
    return result


__all__ = [
    "ensemble_atom_quantization",
    "pooled_atom_diagnostics",
    "reliability_bins",
]
