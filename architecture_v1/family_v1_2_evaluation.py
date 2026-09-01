"""Frozen calendar-day inference for the family-v1.2 temporal-utility probe.

The functions in this module are deliberately array-only.  They cannot open a
target split, select a checkpoint, or generate a scenario.  Their first axis is
always the paired calendar-day inference unit; ensemble members, sampling
replicates, and training seeds must already have been averaged by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PRIMARY_METRICS = ("ramp_CRPS", "lagged_increment_variogram_score")
SAFETY_METRICS = (
    "level_CRPS",
    "normalized_joint_ES",
    "coverage90",
    "width90",
)
REGIME_LABELS = ("stable", "moderate", "dynamic")


def _finite_array(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def daily_contribution(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    relative: bool,
) -> np.ndarray:
    """Return paired daily benefit contributions (positive means better).

    For an absolute endpoint the contribution is ``reference-candidate``.  A
    relative endpoint divides that daily difference by the full-validation
    reference mean, which is fixed before resampling.  Consequently the mean
    contribution is exactly the registered aggregate relative improvement.
    """

    left = _finite_array(reference, name="reference")
    right = _finite_array(candidate, name="candidate")
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("reference and candidate daily arrays do not align")
    result = left - right
    if relative:
        denominator = np.mean(left, axis=0)
        if np.any(np.abs(denominator) < 1e-15):
            raise ZeroDivisionError("relative contribution reference mean is zero")
        result = result / denominator
    return np.ascontiguousarray(result, dtype=np.float64)


def daily_harm(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    relative: bool,
) -> np.ndarray:
    """Return paired daily deterioration contributions (positive means worse)."""

    left = _finite_array(candidate, name="candidate")
    right = _finite_array(reference, name="reference")
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("candidate and reference daily arrays do not align")
    result = left - right
    if relative:
        denominator = np.mean(right, axis=0)
        if np.any(np.abs(denominator) < 1e-15):
            raise ZeroDivisionError("relative harm reference mean is zero")
        result = result / denominator
    return np.ascontiguousarray(result, dtype=np.float64)


def max_t_simultaneous_bands(
    contributions: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Joint paired-day bootstrap max-T bands for arbitrary endpoints.

    ``contributions`` has shape ``[day, ...endpoints]``.  The same resampled
    calendar-day indices are used for every endpoint, preserving all paired
    block and metric dependence.
    """

    values = _finite_array(contributions, name="contributions")
    if values.ndim < 2 or len(values) < 2:
        raise ValueError("max-T requires at least two days and one endpoint axis")
    if repetitions < 100:
        raise ValueError("max-T requires at least 100 bootstrap repetitions")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5,1)")
    endpoint_shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    estimate = flat.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(flat), size=(int(repetitions), len(flat)))
    draws = flat[indices].mean(axis=1)
    standard_error = draws.std(axis=0, ddof=1)
    centered = draws - estimate[None]
    usable = standard_error > 1e-15
    studentized = np.zeros_like(centered)
    studentized[:, usable] = centered[:, usable] / standard_error[None, usable]
    if bool((~usable).any()) and not np.allclose(
        centered[:, ~usable], 0.0, rtol=0.0, atol=1e-14
    ):
        raise FloatingPointError("a nominally zero-SE endpoint has varying draws")
    maximum = np.max(np.abs(studentized), axis=1)
    critical = float(np.quantile(maximum, float(confidence)))
    low = estimate - critical * standard_error
    high = estimate + critical * standard_error
    return {
        "unit_of_inference": "calendar_day",
        "n_days": int(len(values)),
        "endpoint_shape": list(endpoint_shape),
        "endpoint_count": int(flat.shape[1]),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "confidence": float(confidence),
        "critical_value": critical,
        "estimate": estimate.reshape(endpoint_shape),
        "standard_error": standard_error.reshape(endpoint_shape),
        "simultaneous_low": low.reshape(endpoint_shape),
        "simultaneous_high": high.reshape(endpoint_shape),
    }


def _quadratic_statistic(values: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sum((values / scale) ** 2))


def stage_homogeneity_omnibus(
    contributions: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Joint paired-day centered-bootstrap test of equal block means.

    Input shape is ``[day, primary_metric, DDIM_block]``.
    """

    values = _finite_array(contributions, name="stage contributions", ndim=3)
    days, metrics, blocks = values.shape
    if days < 2 or metrics < 1 or blocks < 2 or repetitions < 100:
        raise ValueError("invalid stage-omnibus dimensions or repetitions")
    within_day = values - values.mean(axis=2, keepdims=True)
    observed = within_day.mean(axis=0)
    scale = within_day.std(axis=0, ddof=1) / np.sqrt(float(days))
    scale = np.where(scale > 1e-15, scale, 1.0)
    observed_statistic = _quadratic_statistic(observed, scale)
    null_values = within_day - observed[None]
    rng = np.random.default_rng(int(seed))
    exceed = 0
    draw_statistics = np.empty(int(repetitions), dtype=np.float64)
    for draw in range(int(repetitions)):
        index = rng.integers(0, days, size=days)
        null_mean = null_values[index].mean(axis=0)
        statistic = _quadratic_statistic(null_mean, scale)
        draw_statistics[draw] = statistic
        exceed += int(statistic >= observed_statistic - 1e-15)
    return {
        "null": "equal ordered-benefit means across DDIM blocks",
        "unit_of_inference": "paired_calendar_day",
        "n_days": int(days),
        "metrics": int(metrics),
        "blocks": int(blocks),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "block_mean": values.mean(axis=0),
        "centered_block_mean": observed,
        "statistic": observed_statistic,
        "null_q95": float(np.quantile(draw_statistics, 0.95)),
        "p_value": float((exceed + 1) / (int(repetitions) + 1)),
    }


def _validate_regimes(
    labels: Sequence[str] | np.ndarray,
    days: int,
    regimes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    values = np.asarray(labels, dtype="U16")
    ordered = tuple(str(value) for value in regimes)
    if values.shape != (days,) or len(set(ordered)) != len(ordered):
        raise ValueError("regime labels do not align with calendar days")
    unknown = sorted(set(values.tolist()).difference(ordered))
    if unknown:
        raise ValueError(f"unknown NWP regimes: {unknown}")
    counts = np.asarray([np.count_nonzero(values == name) for name in ordered])
    if np.any(counts < 2):
        raise ValueError("every NWP regime requires at least two validation days")
    return values, ordered, counts.astype(np.int64)


def _group_means(
    values: np.ndarray,
    labels: np.ndarray,
    regimes: Sequence[str],
) -> np.ndarray:
    return np.stack([values[labels == name].mean(axis=0) for name in regimes], axis=0)


def nwp_regime_homogeneity_omnibus(
    contributions: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    *,
    repetitions: int,
    seed: int,
    regimes: Sequence[str] = REGIME_LABELS,
) -> dict[str, Any]:
    """Permutation test for the block-averaged NWP-regime main effect."""

    values = _finite_array(contributions, name="NWP contributions", ndim=3)
    days, metrics, _ = values.shape
    if repetitions < 100:
        raise ValueError("NWP omnibus requires at least 100 permutations")
    label, ordered, counts = _validate_regimes(labels, days, regimes)
    collapsed = values.mean(axis=2)
    grand = collapsed.mean(axis=0)
    scale = collapsed.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-15, scale, 1.0)

    def statistic(permuted: np.ndarray) -> tuple[float, np.ndarray]:
        means = _group_means(collapsed, permuted, ordered)
        weighted = counts[:, None] * ((means - grand[None]) / scale[None]) ** 2
        return float(weighted.sum()), means

    observed_statistic, means = statistic(label)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    exceed = 0
    for draw in range(int(repetitions)):
        value, _ = statistic(label[rng.permutation(days)])
        draws[draw] = value
        exceed += int(value >= observed_statistic - 1e-15)
    return {
        "null": "block-averaged ordered benefit is homogeneous across NWP regimes",
        "unit_of_inference": "calendar_day",
        "regimes": list(ordered),
        "counts": counts,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "regime_mean": means,
        "grand_mean": grand,
        "statistic": observed_statistic,
        "null_q95": float(np.quantile(draws, 0.95)),
        "p_value": float((exceed + 1) / (int(repetitions) + 1)),
    }


def block_by_nwp_interaction_omnibus(
    contributions: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    *,
    repetitions: int,
    seed: int,
    regimes: Sequence[str] = REGIME_LABELS,
) -> dict[str, Any]:
    """Residual-permutation test of the block-by-NWP interaction."""

    values = _finite_array(contributions, name="interaction contributions", ndim=3)
    days, metrics, blocks = values.shape
    if blocks < 2 or repetitions < 100:
        raise ValueError("interaction test requires blocks and 100 permutations")
    label, ordered, counts = _validate_regimes(labels, days, regimes)
    grand = values.mean(axis=(0, 2))
    block_mean = values.mean(axis=0)
    regime_main = np.stack(
        [values[label == name].mean(axis=(0, 2)) for name in ordered], axis=0
    )
    regime_lookup = {name: index for index, name in enumerate(ordered)}
    label_index = np.asarray([regime_lookup[str(name)] for name in label])
    fitted = (
        block_mean[None]
        + regime_main[label_index, :, None]
        - grand[None, :, None]
    )
    residual = values - fitted
    scale = residual.reshape(-1, metrics, blocks).std(axis=(0, 2), ddof=1)
    scale = np.where(scale > 1e-15, scale, 1.0)

    def interaction_and_statistic(sample: np.ndarray) -> tuple[np.ndarray, float]:
        sample_grand = sample.mean(axis=(0, 2))
        sample_block = sample.mean(axis=0)
        sample_regime = np.stack(
            [sample[label == name].mean(axis=(0, 2)) for name in ordered], axis=0
        )
        cell = _group_means(sample, label, ordered)
        interaction = (
            cell
            - sample_block[None]
            - sample_regime[:, :, None]
            + sample_grand[None, :, None]
        )
        weighted = counts[:, None, None] * (
            interaction / scale[None, :, None]
        ) ** 2
        return interaction, float(weighted.sum())

    interaction, observed_statistic = interaction_and_statistic(values)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    exceed = 0
    for draw in range(int(repetitions)):
        synthetic = fitted + residual[rng.permutation(days)]
        _, statistic = interaction_and_statistic(synthetic)
        draws[draw] = statistic
        exceed += int(statistic >= observed_statistic - 1e-15)
    return {
        "null": "additive DDIM-block plus NWP-regime effects",
        "unit_of_inference": "calendar_day residual vector",
        "regimes": list(ordered),
        "counts": counts,
        "blocks": int(blocks),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "cell_interaction": interaction,
        "statistic": observed_statistic,
        "null_q95": float(np.quantile(draws, 0.95)),
        "p_value": float((exceed + 1) / (int(repetitions) + 1)),
    }


def adjacent_supported_pairs(flags: Sequence[bool]) -> list[list[int]]:
    values = [bool(value) for value in flags]
    return [
        [index, index + 1]
        for index in range(len(values) - 1)
        if values[index] and values[index + 1]
    ]


def adjudicate(
    *,
    primary_bands: Mapping[str, Any],
    same_checkpoint_bands: Mapping[str, Any],
    safety_bands: Mapping[str, Any],
    stage_omnibus: Mapping[str, Any],
    nwp_omnibus: Mapping[str, Any],
    interaction_omnibus: Mapping[str, Any],
    margins: Mapping[str, float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply the frozen block-support rules and family-v1.2 decision tree.

    Band arrays use shape ``[metric, block]``.  Primary metric order is ramp,
    lagged variogram; safety order is level, joint ES, coverage, width.
    """

    primary_estimate = _finite_array(
        primary_bands["estimate"], name="primary estimate", ndim=2
    )
    primary_low = _finite_array(
        primary_bands["simultaneous_low"], name="primary low", ndim=2
    )
    same_low = _finite_array(
        same_checkpoint_bands["simultaneous_low"], name="same-checkpoint low", ndim=2
    )
    safety_low = _finite_array(
        safety_bands["simultaneous_low"], name="safety low", ndim=2
    )
    safety_high = _finite_array(
        safety_bands["simultaneous_high"], name="safety high", ndim=2
    )
    if (
        primary_estimate.shape[0] != 2
        or primary_estimate.shape != primary_low.shape
        or same_low.shape != primary_estimate.shape
        or safety_low.shape != safety_high.shape
        or safety_low.shape[0] != 4
        or safety_low.shape[1] != primary_estimate.shape[1]
    ):
        raise ValueError("adjudication band shapes do not match the frozen endpoints")
    blocks = primary_estimate.shape[1]
    primary = (
        (primary_low[0] > 0.0)
        & (primary_low[1] > 0.0)
        & (primary_estimate[0] >= float(margins["ramp_CRPS_improvement_min"]))
        & (
            primary_estimate[1]
            >= float(margins["lagged_variogram_relative_improvement_min"])
        )
    )
    same_checkpoint = (same_low[0] > 0.0) & (same_low[1] > 0.0)
    coverage_abs = np.maximum(np.abs(safety_low[2]), np.abs(safety_high[2]))
    safety = (
        (safety_high[0] <= float(margins["level_CRPS_noninferiority_margin"]))
        & (
            safety_high[1]
            <= float(margins["normalized_joint_ES_relative_noninferiority_margin"])
        )
        & (coverage_abs <= float(margins["coverage90_absolute_difference_max"]))
        & (
            safety_high[3]
            <= float(margins["width90_relative_increase_max"])
        )
    )
    mechanism = primary & same_checkpoint
    full = mechanism & safety
    primary_pairs = adjacent_supported_pairs(primary.tolist())
    mechanism_pairs = adjacent_supported_pairs(mechanism.tolist())
    full_pairs = adjacent_supported_pairs(full.tolist())
    persistent_damage = bool(mechanism_pairs and not full_pairs)
    stage_significant = float(stage_omnibus["p_value"]) < float(alpha)
    nwp_significant = float(nwp_omnibus["p_value"]) < float(alpha)
    interaction_significant = float(interaction_omnibus["p_value"]) < float(alpha)

    if not mechanism_pairs:
        status = "FAMILY_V1_2_ORDERED_RECURRENCE_NO_GO"
    elif persistent_damage:
        status = "FAMILY_V1_2_NO_GO_REVISE_INTERVENTION_SCALE"
    elif interaction_significant:
        status = "FAMILY_V1_3_DUAL_GATED_TEMPORAL_CONTEXT_RESIDUAL"
    elif stage_significant and nwp_significant:
        status = "FAMILY_V1_3_ADDITIVE_STAGE_AND_NWP_CONTROLS"
    elif stage_significant:
        status = "FAMILY_V1_3_SNR_STAGE_GATE"
    elif nwp_significant:
        status = "FAMILY_V1_3_NWP_GATE"
    else:
        status = "FAMILY_V1_2_TEMPORAL_GATING_NO_GO"

    return {
        "status": status,
        "blocks": int(blocks),
        "primary_block_support": primary,
        "same_checkpoint_block_support": same_checkpoint,
        "distribution_safety_block_support": safety,
        "mechanism_block_support": mechanism,
        "full_block_support": full,
        "primary_adjacent_pairs": primary_pairs,
        "mechanism_adjacent_pairs": mechanism_pairs,
        "full_adjacent_pairs": full_pairs,
        "persistent_distribution_damage": persistent_damage,
        "stage_omnibus_significant": stage_significant,
        "NWP_omnibus_significant": nwp_significant,
        "interaction_omnibus_significant": interaction_significant,
        "alpha": float(alpha),
        "validation_only": True,
    }


__all__ = [
    "PRIMARY_METRICS",
    "REGIME_LABELS",
    "SAFETY_METRICS",
    "adjacent_supported_pairs",
    "adjudicate",
    "block_by_nwp_interaction_omnibus",
    "daily_contribution",
    "daily_harm",
    "max_t_simultaneous_bands",
    "nwp_regime_homogeneity_omnibus",
    "stage_homogeneity_omnibus",
]
