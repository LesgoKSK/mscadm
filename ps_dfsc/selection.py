from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .safety import GateDecision


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    beta: float
    proxy_objective: float
    gate: GateDecision
    mean_ess: float
    mean_transport: float
    checkpoint: str
    exact_mean_cost: float | None = None
    exact_cvar90: float | None = None


@dataclass(frozen=True)
class SelectionDecision:
    candidate: CandidateRecord | None
    used_identity_fallback: bool
    reason: str
    exact_shortlist: tuple[str, ...]


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    start = int(np.floor(alpha * len(array)))
    return float(array[min(start, len(array) - 1) :].mean())


def proxy_shortlist(
    candidates: list[CandidateRecord], *, limit: int = 3
) -> tuple[CandidateRecord, ...]:
    feasible = [candidate for candidate in candidates if candidate.gate.passed]
    feasible.sort(
        key=lambda item: (
            item.proxy_objective,
            -item.mean_ess,
            item.mean_transport,
            item.candidate_id,
        )
    )
    return tuple(feasible[:limit])


def choose_exact_candidate(
    candidates: list[CandidateRecord],
    *,
    baseline_mean_cost: float,
    baseline_cvar90: float,
    risk_weight: float = 0.25,
    tie_tolerance: float = 0.001,
) -> SelectionDecision:
    shortlist = proxy_shortlist(candidates)
    evaluated = [
        item
        for item in shortlist
        if item.exact_mean_cost is not None and item.exact_cvar90 is not None
    ]
    baseline_objective = 1.0 + risk_weight
    eligible: list[tuple[float, CandidateRecord]] = []
    for item in evaluated:
        objective = (
            float(item.exact_mean_cost) / baseline_mean_cost
            + risk_weight * float(item.exact_cvar90) / baseline_cvar90
        )
        if objective <= baseline_objective + 1e-12:
            eligible.append((objective, item))
    if not eligible:
        reason = (
            "no_safety_feasible_candidate"
            if not shortlist
            else "no_exact_candidate_improves_validation_objective"
        )
        return SelectionDecision(
            candidate=None,
            used_identity_fallback=True,
            reason=reason,
            exact_shortlist=tuple(item.candidate_id for item in shortlist),
        )
    eligible.sort(key=lambda pair: pair[0])
    best_objective = eligible[0][0]
    tied = [
        item
        for objective, item in eligible
        if objective <= best_objective * (1.0 + tie_tolerance)
    ]
    tied.sort(
        key=lambda item: (-item.mean_ess, item.mean_transport, item.candidate_id)
    )
    return SelectionDecision(
        candidate=tied[0],
        used_identity_fallback=False,
        reason="selected_exact_validation_objective",
        exact_shortlist=tuple(item.candidate_id for item in shortlist),
    )


__all__ = [
    "CandidateRecord",
    "SelectionDecision",
    "choose_exact_candidate",
    "empirical_cvar",
    "proxy_shortlist",
]
