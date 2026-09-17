"""Formal metrics and inference for the frozen G0-B denoising-utility Probe.

This module contains no dataset or checkpoint loader.  It operates on arrays
or already-materialized tensors so the executable runner remains responsible
for the target-role boundary and training-freeze checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from architecture_v1.g0b_tiny_denoiser import ModeProjectorBank


ENDPOINTS = ("cell_MSE", "increment_MSE", "joint_day_normalized_SSE")
CONTRAST_IDS = (
    "PA_RWF_vs_IID_balanced_risk_at_fraction_0.5",
    "PA_RWF_vs_FIXED_BAND_balanced_risk_at_fraction_0.5",
    "PA_RWF_vs_CW_GROUP_balanced_risk_at_fraction_0.5",
    "PA_RWF_vs_PA_SHUFFLE_balanced_risk_at_fraction_0.5",
    "PA_RWF_vs_MULAN_LITE_balanced_risk_noninferiority_at_fraction_0.5",
    "PA_RWF_vs_MULAN_LITE_learning_curve_AULC",
    "PA_RWF_vs_IID_oracle_efficiency_at_fraction_0.5",
    "PA_RWF_vs_IID_cell_MSE_noninferiority_at_fraction_0.5",
    "PA_RWF_vs_IID_increment_MSE_noninferiority_at_fraction_0.5",
    "PA_RWF_vs_IID_joint_day_normalized_SSE_noninferiority_at_fraction_0.5",
    "FIXED_BAND_vs_IID_balanced_risk_at_fraction_0.5",
    "CW_GROUP_vs_IID_balanced_risk_at_fraction_0.5",
    "MULAN_LITE_vs_IID_balanced_risk_at_fraction_0.5",
)


def _finite(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


@torch.no_grad()
def sample_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    projectors: ModeProjectorBank,
    effective_rank: torch.Tensor,
    *,
    outer_train_second_moment: float,
) -> dict[str, torch.Tensor]:
    """Return one value per corruption sample for the frozen raw endpoints.

    All tensors use a common leading sample axis.  Inactive errors are zeroed
    before both cell metrics and the six orthogonal projections.
    """

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must align as [sample,10,24]")
    if tuple(prediction.shape[1:]) != (10, 24):
        raise ValueError("reconstruction fields must have shape [sample,10,24]")
    if active_mask.shape != target.shape or active_mask.dtype != torch.bool:
        raise ValueError("active mask must be boolean and align with target")
    if effective_rank.shape != (len(target), 6):
        raise ValueError("effective rank must have shape [sample,6]")
    if not np.isfinite(outer_train_second_moment) or outer_train_second_moment <= 0:
        raise ValueError("outer-train second moment must be finite and positive")

    active = active_mask.to(target.dtype)
    active_count = active.sum(dim=(1, 2))
    if bool((active_count <= 0).any()):
        raise RuntimeError("a reconstruction day has no active cell")
    error = (prediction - target) * active
    squared = error.square()
    active_sse = squared.sum(dim=(1, 2))
    cell = active_sse / active_count

    pair = active_mask[:, :, 1:] & active_mask[:, :, :-1]
    pair_count = pair.sum(dim=(1, 2))
    if bool((pair_count <= 0).any()):
        raise RuntimeError("a reconstruction day has no active hourly increment")
    increment_error = error[:, :, 1:] - error[:, :, :-1]
    increment = (
        increment_error.square() * pair.to(increment_error.dtype)
    ).sum(dim=(1, 2)) / pair_count.to(increment_error.dtype)

    projected = projectors.project(error)
    group = projected.square().sum(dim=(2, 3)) / effective_rank
    joint = active_sse / (
        active_count * float(outer_train_second_moment)
    )
    for name, value in (
        ("cell_MSE", cell),
        ("increment_MSE", increment),
        ("joint_day_normalized_SSE", joint),
        ("six_group_MSE", group),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} became non-finite")
    return {
        "cell_MSE": cell,
        "increment_MSE": increment,
        "joint_day_normalized_SSE": joint,
        "six_group_MSE": group,
    }


def balanced_reconstruction_risk(
    endpoint_values: np.ndarray,
    iid_denominators: np.ndarray,
) -> np.ndarray:
    """Equal-weight the three endpoints after frozen IID normalization."""

    values = _finite(endpoint_values, name="endpoint values")
    denominator = _finite(iid_denominators, name="IID denominators")
    if values.shape[-1] != len(ENDPOINTS) or denominator.shape != (
        values.shape[-2],
        len(ENDPOINTS),
    ):
        raise ValueError(
            "endpoint values must end in [fraction,3] and denominators in [fraction,3]"
        )
    if np.any(denominator <= 0.0):
        raise ValueError("IID endpoint denominators must be positive")
    expansion = (1,) * (values.ndim - 2) + denominator.shape
    return np.mean(values / denominator.reshape(expansion), axis=-1)


def learning_curve_aulc(values: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    """Normalized trapezoidal area over log data fraction (lower is better)."""

    risk = _finite(values, name="learning-curve values")
    fraction = _finite(fractions, name="fractions", ndim=1)
    if risk.shape[-1] != len(fraction) or len(fraction) < 2:
        raise ValueError("learning curve and fraction grid do not align")
    if np.any(fraction <= 0.0) or np.any(np.diff(fraction) <= 0.0):
        raise ValueError("fractions must be strictly increasing and positive")
    x = np.log(fraction)
    width = np.diff(x)
    area = np.sum(
        0.5 * (risk[..., :-1] + risk[..., 1:]) * width,
        axis=-1,
    )
    return area / float(x[-1] - x[0])


def relative_benefit(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Paired daily relative benefit, positive when the candidate is better.

    The full-sample reference mean is fixed before resampling, so the mean of
    the returned contributions is exactly the aggregate relative improvement.
    """

    left = _finite(reference, name="reference", ndim=1)
    right = _finite(candidate, name="candidate", ndim=1)
    if left.shape != right.shape:
        raise ValueError("paired daily metrics do not align")
    denominator = float(left.mean())
    if denominator <= 0.0:
        raise ValueError("relative-benefit reference mean must be positive")
    return np.ascontiguousarray((left - right) / denominator)


def month_cluster_max_t_bands(
    contributions: np.ndarray,
    days: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Calendar-month paired cluster bootstrap with joint max-T bands."""

    values = _finite(contributions, name="contrast contributions", ndim=2)
    dates = np.asarray(days, dtype="datetime64[D]")
    if values.shape[0] != len(dates) or len(dates) < 2:
        raise ValueError("contributions and calendar days do not align")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("calendar-day inference units must be unique")
    if repetitions < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid bootstrap repetitions or confidence")
    months = dates.astype("datetime64[M]")
    unique_months = np.unique(months)
    if len(unique_months) < 12:
        raise RuntimeError("fewer than twelve calendar-month clusters")
    cluster_sums = np.stack(
        [values[months == month].sum(axis=0) for month in unique_months]
    )
    cluster_counts = np.asarray(
        [int(np.sum(months == month)) for month in unique_months],
        dtype=np.int64,
    )
    generator = np.random.default_rng(int(seed))
    selected = generator.integers(
        0,
        len(unique_months),
        size=(int(repetitions), len(unique_months)),
    )
    numerator = cluster_sums[selected].sum(axis=1)
    denominator = cluster_counts[selected].sum(axis=1)[:, None]
    draws = numerator / denominator
    estimate = values.mean(axis=0)
    standard_error = draws.std(axis=0, ddof=1)
    centered = draws - estimate[None]
    usable = standard_error > 1e-15
    studentized = np.zeros_like(centered)
    studentized[:, usable] = centered[:, usable] / standard_error[None, usable]
    if bool((~usable).any()) and not np.allclose(
        centered[:, ~usable], 0.0, atol=1e-14, rtol=0.0
    ):
        raise FloatingPointError("zero-SE bootstrap endpoint has varying draws")
    maximum = np.max(np.abs(studentized), axis=1)
    critical = float(np.quantile(maximum, float(confidence)))
    return {
        "unit_of_inference": "calendar_day",
        "dependence_cluster": "calendar_month",
        "n_days": int(len(dates)),
        "month_clusters": int(len(unique_months)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "confidence": float(confidence),
        "endpoint_count": int(values.shape[1]),
        "critical_value": critical,
        "estimate": estimate,
        "standard_error": standard_error,
        "simultaneous_low": estimate - critical * standard_error,
        "simultaneous_high": estimate + critical * standard_error,
    }


def _gate(
    passed: bool,
    *,
    value: float | int,
    requirement: str,
) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "requirement": requirement}


def adjudicate(
    contrast_bands: Mapping[str, Mapping[str, float]],
    *,
    positive_outer_folds: int,
    positive_model_seeds: int,
    technical_eligibility: bool,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen G0-B decision order without inspecting raw targets."""

    if tuple(contrast_bands) != CONTRAST_IDS:
        raise ValueError("confirmatory contrast registry or order drifted")

    def estimate(name: str) -> float:
        return float(contrast_bands[name]["estimate"])

    def low(name: str) -> float:
        return float(contrast_bands[name]["simultaneous_low"])

    def superior(name: str, minimum: float) -> bool:
        return estimate(name) >= float(minimum) and low(name) > 0.0

    def noninferior(name: str, margin: float) -> bool:
        return low(name) >= -float(margin)

    names = CONTRAST_IDS
    pa_iid = superior(
        names[0], thresholds["PA_vs_IID_balanced_risk_min_relative_improvement_at_50pct"]
    )
    pa_fixed = superior(
        names[1], thresholds["PA_vs_FIXED_balanced_risk_min_relative_improvement_at_50pct"]
    )
    pa_cw = superior(
        names[2], thresholds["PA_vs_CW_balanced_risk_min_relative_improvement_at_50pct"]
    )
    pa_shuffle = superior(
        names[3], thresholds["PA_vs_SHUFFLE_balanced_risk_min_relative_improvement_at_50pct"]
    )
    pa_mulan_ni = noninferior(
        names[4], thresholds["PA_vs_MULAN_balanced_risk_noninferiority_margin_at_50pct"]
    )
    pa_mulan_aulc = superior(
        names[5], thresholds["PA_vs_MULAN_AULC_min_relative_improvement"]
    )
    pa_oracle = superior(
        names[6], thresholds["PA_vs_IID_oracle_efficiency_min_relative_improvement_at_50pct"]
    )
    endpoint_margin = float(
        thresholds["PA_vs_IID_each_endpoint_noninferiority_margin_at_50pct"]
    )
    endpoint_ni = [noninferior(name, endpoint_margin) for name in names[7:10]]
    endpoint_material = [estimate(name) >= 0.02 for name in names[7:10]]
    enough_endpoints = sum(endpoint_material) >= int(
        thresholds["PA_vs_IID_minimum_endpoints_with_point_improvement_at_least_2pct"]
    )
    enough_folds = positive_outer_folds >= int(
        thresholds["minimum_positive_outer_folds_for_PA_vs_IID_balanced_risk"]
    )
    enough_seeds = positive_model_seeds >= int(
        thresholds["minimum_positive_model_seeds_for_PA_vs_IID_balanced_risk"]
    )
    control_minimum = float(
        thresholds["control_path_vs_IID_material_relative_improvement_at_50pct"]
    )
    fixed_material = superior(names[10], control_minimum)
    cw_material = superior(names[11], control_minimum)
    mulan_material = superior(names[12], control_minimum)

    gates = {
        "technical_eligibility": _gate(
            technical_eligibility, value=int(technical_eligibility), requirement="true"
        ),
        "PA_vs_IID_balanced_risk": _gate(
            pa_iid,
            value=estimate(names[0]),
            requirement=(
                f"point >= {thresholds['PA_vs_IID_balanced_risk_min_relative_improvement_at_50pct']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_FIXED_balanced_risk": _gate(
            pa_fixed,
            value=estimate(names[1]),
            requirement=(
                f"point >= {thresholds['PA_vs_FIXED_balanced_risk_min_relative_improvement_at_50pct']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_CW_balanced_risk": _gate(
            pa_cw,
            value=estimate(names[2]),
            requirement=(
                f"point >= {thresholds['PA_vs_CW_balanced_risk_min_relative_improvement_at_50pct']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_SHUFFLE_balanced_risk": _gate(
            pa_shuffle,
            value=estimate(names[3]),
            requirement=(
                f"point >= {thresholds['PA_vs_SHUFFLE_balanced_risk_min_relative_improvement_at_50pct']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_MULAN_balanced_risk_noninferiority": _gate(
            pa_mulan_ni,
            value=low(names[4]),
            requirement=(
                "simultaneous lower >= -"
                f"{thresholds['PA_vs_MULAN_balanced_risk_noninferiority_margin_at_50pct']}"
            ),
        ),
        "PA_vs_MULAN_AULC": _gate(
            pa_mulan_aulc,
            value=estimate(names[5]),
            requirement=(
                f"point >= {thresholds['PA_vs_MULAN_AULC_min_relative_improvement']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_IID_oracle_efficiency": _gate(
            pa_oracle,
            value=estimate(names[6]),
            requirement=(
                f"point >= {thresholds['PA_vs_IID_oracle_efficiency_min_relative_improvement_at_50pct']} "
                "and simultaneous lower > 0"
            ),
        ),
        "PA_vs_IID_endpoint_noninferiority": _gate(
            all(endpoint_ni),
            value=min(low(name) for name in names[7:10]),
            requirement=f"every simultaneous lower >= -{endpoint_margin}",
        ),
        "PA_vs_IID_material_endpoint_count": _gate(
            enough_endpoints,
            value=sum(endpoint_material),
            requirement=(
                ">= "
                f"{thresholds['PA_vs_IID_minimum_endpoints_with_point_improvement_at_least_2pct']} "
                "endpoints with point improvement >= 0.02"
            ),
        ),
        "positive_outer_folds": _gate(
            enough_folds,
            value=int(positive_outer_folds),
            requirement=(
                f">= {thresholds['minimum_positive_outer_folds_for_PA_vs_IID_balanced_risk']}"
            ),
        ),
        "positive_model_seeds": _gate(
            enough_seeds,
            value=int(positive_model_seeds),
            requirement=(
                f">= {thresholds['minimum_positive_model_seeds_for_PA_vs_IID_balanced_risk']}"
            ),
        ),
    }
    all_go = all(item["passed"] for item in gates.values())

    if not technical_eligibility:
        status = "G0_B_TECHNICAL_NO_GO"
        reason = "technical eligibility failed"
    elif not pa_iid:
        if mulan_material:
            status = "G0_B_GENERIC_ADAPTIVITY_SUFFICIENT"
            reason = "PA failed IID while MuLAN-lite materially beat IID"
        elif cw_material:
            status = "G0_B_WHITENING_SUFFICIENT"
            reason = "PA failed IID while CW-group materially beat IID"
        elif fixed_material:
            status = "G0_B_FIXED_BAND_SUFFICIENT"
            reason = "PA failed IID while Fixed-band materially beat IID"
        else:
            status = "G0_B_STRUCTURED_SCHEDULE_NO_GO"
            reason = "PA and registered structured controls did not clear IID"
    elif not pa_shuffle:
        status = "G0_B_NWP_ALIGNMENT_NO_GO"
        reason = "PA did not beat its wrong-day schedule control"
    elif not pa_fixed:
        status = "G0_B_FIXED_BAND_SUFFICIENT"
        reason = "PA did not beat the static non-isotropic control"
    elif not pa_cw:
        status = "G0_B_WHITENING_SUFFICIENT"
        reason = "PA did not beat conditional whitening"
    elif not (pa_mulan_ni and pa_mulan_aulc):
        if mulan_material:
            status = "G0_B_GENERIC_ADAPTIVITY_SUFFICIENT"
            reason = "generic learned adaptivity was sufficient"
        else:
            status = "G0_B_UNRESOLVED"
            reason = "PA versus MuLAN-lite requirements were unresolved"
    elif not (all(endpoint_ni) and enough_endpoints and enough_folds and enough_seeds):
        status = "G0_B_UNRESOLVED"
        reason = "raw endpoint or stability gates were not all satisfied"
    elif not pa_oracle:
        status = "G0_B_ORACLE_ONLY_NO_LEARNING_GAIN"
        reason = "raw reconstruction gates passed but oracle efficiency did not"
    elif all_go:
        status = str(thresholds["GO_status"])
        reason = "all registered technical and scientific gates passed"
    else:
        status = "G0_B_UNRESOLVED"
        reason = "the frozen gate conjunction was not satisfied"

    return {
        "status": status,
        "reason": reason,
        "all_GO_gates_passed": bool(all_go),
        "gates": gates,
        "control_materiality": {
            "FIXED_BAND_vs_IID": fixed_material,
            "CW_GROUP_vs_IID": cw_material,
            "MULAN_LITE_vs_IID": mulan_material,
        },
    }


__all__ = [
    "CONTRAST_IDS",
    "ENDPOINTS",
    "adjudicate",
    "balanced_reconstruction_risk",
    "learning_curve_aulc",
    "month_cluster_max_t_bands",
    "relative_benefit",
    "sample_reconstruction_metrics",
]
