from __future__ import annotations

import math

import numpy as np
import torch

from .mapping import WindFarmMapping
from .model import PSDFSCNetwork
from .reduction import fit_fixed_assignments, weighted_cluster_reduction
from .types import CalibratedDistribution


def _distribution(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    mapping: WindFarmMapping,
    *,
    clusters: int,
    assignments: np.ndarray | None,
    ess: float,
    entropy: float,
    transport_cost: float,
    used_fallback: bool,
    reason: str,
) -> CalibratedDistribution:
    base_for_labels = mapping.transform(scenarios)
    labels = (
        fit_fixed_assignments(base_for_labels, clusters)
        if assignments is None
        else np.asarray(assignments, dtype=np.int64)
    )
    reduced, reduced_probability = weighted_cluster_reduction(
        base_for_labels, probabilities, labels
    )
    return CalibratedDistribution(
        full_scenarios=np.asarray(scenarios, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        suc_scenarios=reduced,
        suc_probabilities=reduced_probability,
        ess=float(ess),
        entropy=float(entropy),
        transport_cost=float(transport_cost),
        used_fallback=used_fallback,
        fallback_reason=reason,
    )


def calibrate(
    model: PSDFSCNetwork,
    base_scenarios: np.ndarray,
    mapping: WindFarmMapping,
    *,
    clusters: int = 20,
    assignments: np.ndarray | None = None,
    ess_floor: float = 50.0,
    normalized_entropy_floor: float = 0.85,
    transport_budget: float = 0.02,
    device: str | torch.device | None = None,
) -> CalibratedDistribution:
    """Calibrate one day, with a truth-independent exact identity fallback."""

    base = np.asarray(base_scenarios, dtype=np.float64)
    if base.ndim != 3 or base.shape[1:] != (10, 24):
        raise ValueError("base_scenarios must have shape [member,10,24]")
    if not np.isfinite(base).all() or base.min() < 0.0 or base.max() > 1.0:
        raise ValueError("base_scenarios must be finite and in [0,1]")
    members = len(base)
    uniform = np.full(members, 1.0 / members)
    identity_entropy = math.log(members)
    target_device = next(model.parameters()).device if device is None else torch.device(device)
    try:
        model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(base, dtype=torch.float32, device=target_device)[None]
            output = model(tensor)
        moved = output.scenarios[0].cpu().numpy().astype(np.float64)
        probability = output.probabilities[0].cpu().numpy().astype(np.float64)
        ess = float(output.ess[0].cpu())
        entropy = float(output.entropy[0].cpu())
        transport_cost = float(output.transport_cost[0].cpu())
        violations: list[str] = []
        if not np.isfinite(moved).all() or not np.isfinite(probability).all():
            violations.append("non_finite_output")
        if ess < ess_floor:
            violations.append("ess_below_floor")
        if entropy / math.log(members) < normalized_entropy_floor:
            violations.append("entropy_below_floor")
        if transport_cost > transport_budget + 1e-12:
            violations.append("transport_budget_exceeded")
        if violations:
            return _distribution(
                base,
                uniform,
                mapping,
                clusters=clusters,
                assignments=assignments,
                ess=members,
                entropy=identity_entropy,
                transport_cost=0.0,
                used_fallback=True,
                reason=";".join(violations),
            )
        return _distribution(
            moved,
            probability / probability.sum(),
            mapping,
            clusters=clusters,
            assignments=assignments,
            ess=ess,
            entropy=entropy,
            transport_cost=transport_cost,
            used_fallback=False,
            reason="",
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return _distribution(
            base,
            uniform,
            mapping,
            clusters=clusters,
            assignments=assignments,
            ess=members,
            entropy=identity_entropy,
            transport_cost=0.0,
            used_fallback=True,
            reason=f"calibration_failure:{type(error).__name__}",
        )


__all__ = ["calibrate"]
