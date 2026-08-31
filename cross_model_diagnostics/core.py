"""Pure-NumPy cross-model trajectory diagnostics.

All public functions treat a calendar day as the independent observational
unit.  Scenario members are a finite forecast representation, never
pseudo-replicates for uncertainty intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ZERO_STATE = 0
INTERIOR_STATE = 1
ONE_STATE = 2
TRANSITION_NAMES = ("interior_to_interior", "atom_to_interior", "atom_to_atom")


def _as_float(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("diagnostic input contains non-finite values")
    return result


def validate_archive_arrays(
    scenarios: np.ndarray, observations: np.ndarray, days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _as_float(scenarios)
    truth = _as_float(observations)
    dates = np.asarray(days).astype("datetime64[D]")
    if values.ndim != 4:
        raise ValueError("scenarios must have shape [day,member,zone,hour]")
    if truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("observations do not align with scenario day/zone/hour axes")
    if dates.shape != (values.shape[0],):
        raise ValueError("day labels do not align with scenario days")
    if values.shape[2:] != (10, 24):
        raise ValueError("diagnostics expect ten zones and 24 hourly leads")
    if values.shape[1] < 2:
        raise ValueError("at least two scenario members are required")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("scenario values must lie in [0,1]")
    if truth.min() < 0.0 or truth.max() > 1.0:
        raise ValueError("observations must lie in [0,1]")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("archive contains duplicate calendar days")
    return values, truth, dates


def empirical_crps(samples: np.ndarray, truth: np.ndarray, *, member_axis: int = 1) -> np.ndarray:
    """Empirical-distribution CRPS without materialising an M x M tensor.

    The returned array has the member axis removed.  This is the exact V-stat
    score of the finite equal-weight ensemble.
    """

    values = _as_float(samples)
    observed = _as_float(truth)
    values = np.moveaxis(values, member_axis, -1)
    if values.shape[:-1] != observed.shape:
        raise ValueError("truth shape does not equal samples with member axis removed")
    members = values.shape[-1]
    if members < 2:
        raise ValueError("CRPS requires at least two members")
    first = np.mean(np.abs(values - observed[..., None]), axis=-1)
    ordered = np.sort(values, axis=-1)
    weights = 2.0 * np.arange(1, members + 1, dtype=np.float64) - members - 1.0
    half_pairwise = np.sum(ordered * weights, axis=-1) / float(members * members)
    result = first - half_pairwise
    if np.any(result < -1e-12):
        raise RuntimeError("computed a negative empirical CRPS")
    return np.maximum(result, 0.0)


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim < 1:
        raise ValueError("run mask must have an hour axis")
    flat = binary.reshape(-1, binary.shape[-1])
    result = np.zeros(len(flat), dtype=np.int16)
    for index, row in enumerate(flat):
        current = 0
        best = 0
        for item in row:
            current = current + 1 if bool(item) else 0
            if current > best:
                best = current
        result[index] = best
    return result.reshape(binary.shape[:-1])


def _run_histogram(mask: np.ndarray, edges: Sequence[int] = (1, 2, 3, 5, 9, 17, 25)) -> np.ndarray:
    """Count all positive runs in bins 1, 2, 3-4, 5-8, 9-16, 17-24."""

    binary = np.asarray(mask, dtype=bool).reshape(-1, np.asarray(mask).shape[-1])
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for row in binary:
        length = 0
        runs: list[int] = []
        for item in np.concatenate((row, np.asarray([False]))):
            if bool(item):
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if runs:
            counts += np.histogram(runs, bins=np.asarray(edges))[0]
    return counts


def classify_states(values: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    if epsilon < 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie in [0,0.5)")
    sample = _as_float(values)
    states = np.full(sample.shape, INTERIOR_STATE, dtype=np.int8)
    states[sample <= epsilon] = ZERO_STATE
    states[sample >= 1.0 - epsilon] = ONE_STATE
    return states


def _transition_category(states: np.ndarray) -> np.ndarray:
    value = np.asarray(states)
    if value.shape[-1] < 2:
        raise ValueError("state sequence needs at least two hours")
    left = value[..., :-1]
    right = value[..., 1:]
    left_interior = left == INTERIOR_STATE
    right_interior = right == INTERIOR_STATE
    category = np.full(left.shape, 1, dtype=np.int8)
    category[left_interior & right_interior] = 0
    category[(~left_interior) & (~right_interior)] = 2
    return category


def _safe_mean(values: np.ndarray) -> float | None:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    return None if sample.size == 0 else float(sample.mean())


def per_day_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    zero_epsilon: float = 0.0,
) -> dict[str, np.ndarray]:
    values = _as_float(scenarios)
    truth = _as_float(observations)
    if values.ndim != 4 or truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("per-day metric inputs do not align")

    level_score = empirical_crps(values, truth, member_axis=1)
    sample_ramp = np.diff(values, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)
    ramp_score = empirical_crps(sample_ramp, truth_ramp, member_axis=1)
    prediction = values.mean(axis=1)
    ramp_prediction = sample_ramp.mean(axis=1)
    lower = np.quantile(values, 0.05, axis=1)
    upper = np.quantile(values, 0.95, axis=1)

    sample_max_ramp = np.max(np.abs(sample_ramp), axis=(2, 3))
    truth_max_ramp = np.max(np.abs(truth_ramp), axis=(1, 2))
    max_ramp_score = empirical_crps(sample_max_ramp, truth_max_ramp, member_axis=1)

    sample_zero = values <= zero_epsilon
    truth_zero = truth <= zero_epsilon
    sample_any_zero = sample_zero.any(axis=-1).mean(axis=1)
    truth_any_zero = truth_zero.any(axis=-1)
    sample_longest = _longest_true_run(sample_zero).mean(axis=1)
    truth_longest = _longest_true_run(truth_zero)
    sample_transition_rate = np.not_equal(sample_zero[..., 1:], sample_zero[..., :-1]).mean(axis=(1, 2, 3))
    truth_transition_rate = np.not_equal(truth_zero[..., 1:], truth_zero[..., :-1]).mean(axis=(1, 2))

    return {
        "level_CRPS": level_score.mean(axis=(1, 2)),
        "ramp_CRPS": ramp_score.mean(axis=(1, 2)),
        "level_MAE": np.abs(prediction - truth).mean(axis=(1, 2)),
        "ramp_MAE": np.abs(ramp_prediction - truth_ramp).mean(axis=(1, 2)),
        "mean_abs_ramp_ratio": np.mean(np.abs(sample_ramp), axis=(1, 2, 3))
        / np.maximum(np.mean(np.abs(truth_ramp), axis=(1, 2)), 1e-12),
        "max_abs_ramp_CRPS": max_ramp_score,
        "max_abs_ramp_bias": sample_max_ramp.mean(axis=1) - truth_max_ramp,
        "coverage_90": ((truth >= lower) & (truth <= upper)).mean(axis=(1, 2)),
        "width_90": (upper - lower).mean(axis=(1, 2)),
        "zero_rate_abs_error": np.abs(sample_zero.mean(axis=(1, 2, 3)) - truth_zero.mean(axis=(1, 2))),
        "any_zero_Brier": np.mean((sample_any_zero - truth_any_zero) ** 2, axis=1),
        "zero_longest_run_MAE": np.mean(np.abs(sample_longest - truth_longest), axis=1),
        "zero_transition_rate_abs_error": np.abs(sample_transition_rate - truth_transition_rate),
    }


def masked_per_day_scores(
    scenarios: np.ndarray,
    observations: np.ndarray,
    valid_level_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """CRPS sensitivity analysis excluding originally missing targets.

    A ramp is valid only when both endpoint observations were originally
    present.  The mask is applied after each cell-level proper score is
    computed, so forecast members are never treated as replicates.
    """

    values = _as_float(scenarios)
    truth = _as_float(observations)
    valid = np.asarray(valid_level_mask, dtype=bool)
    if values.ndim != 4:
        raise ValueError("masked score scenarios must be [day,member,zone,hour]")
    if truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("masked score observations do not align")
    if valid.shape != truth.shape:
        raise ValueError("valid level mask does not align with observations")

    level = empirical_crps(values, truth, member_axis=1)
    ramp = empirical_crps(np.diff(values, axis=-1), np.diff(truth, axis=-1), member_axis=1)
    valid_ramp = valid[..., :-1] & valid[..., 1:]

    def reduce(score: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat_score = score.reshape(score.shape[0], -1)
        flat_mask = mask.reshape(mask.shape[0], -1)
        counts = flat_mask.sum(axis=1)
        total = np.where(flat_mask, flat_score, 0.0).sum(axis=1)
        mean = np.divide(
            total,
            counts,
            out=np.full(score.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )
        return mean, counts.astype(np.int64)

    level_mean, level_count = reduce(level, valid)
    ramp_mean, ramp_count = reduce(ramp, valid_ramp)
    return {
        "level_CRPS_clean": level_mean,
        "ramp_CRPS_clean": ramp_mean,
        "valid_level_cells": level_count,
        "valid_ramp_cells": ramp_count,
    }


def atom_duration_summary(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, Any]:
    values = _as_float(scenarios)
    truth = _as_float(observations)
    sample_zero = values <= epsilon
    truth_zero = truth <= epsilon
    sample_runs = _run_histogram(sample_zero)
    truth_runs = _run_histogram(truth_zero)
    sample_distribution = sample_runs / max(int(sample_runs.sum()), 1)
    truth_distribution = truth_runs / max(int(truth_runs.sum()), 1)
    duration_tv = (
        None
        if int(sample_runs.sum()) == 0 or int(truth_runs.sum()) == 0
        else float(0.5 * np.abs(sample_distribution - truth_distribution).sum())
    )
    return {
        "epsilon": float(epsilon),
        "forecast_zero_rate": float(sample_zero.mean()),
        "observed_zero_rate": float(truth_zero.mean()),
        "forecast_mean_longest_run": float(_longest_true_run(sample_zero).mean()),
        "observed_mean_longest_run": float(_longest_true_run(truth_zero).mean()),
        "duration_bins": ["1", "2", "3-4", "5-8", "9-16", "17-24"],
        "forecast_duration_distribution": sample_distribution.tolist(),
        "observed_duration_distribution": truth_distribution.tolist(),
        "duration_total_variation": duration_tv,
        "forecast_run_count": int(sample_runs.sum()),
        "observed_run_count": int(truth_runs.sum()),
    }


def generated_state_attribution(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    scenario_states: np.ndarray | None = None,
    epsilon: float = 0.0,
) -> list[dict[str, Any]]:
    """Diagnose ramp first-term error conditional on truth and member states.

    This is not a decomposition of proper CRPS: the diversity term is not
    partitioned.  The oracle-matched score is explicitly diagnostic because
    it selects members using the realised truth state.
    """

    values = _as_float(scenarios)
    truth = _as_float(observations)
    if scenario_states is None:
        member_state = classify_states(values, epsilon)
        state_source = f"threshold_epsilon={epsilon:.8g}"
    else:
        member_state = np.asarray(scenario_states, dtype=np.int8)
        if member_state.shape != values.shape:
            raise ValueError("scenario states do not align with scenario values")
        if not np.isin(member_state, (ZERO_STATE, INTERIOR_STATE, ONE_STATE)).all():
            raise ValueError("scenario states contain an unknown code")
        state_source = "archive_explicit_state"
    truth_state = classify_states(truth, epsilon)
    member_transition = _transition_category(member_state)
    truth_transition = _transition_category(truth_state)
    sample_ramp = np.diff(values, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)
    absolute_error = np.abs(sample_ramp - truth_ramp[:, None])

    records: list[dict[str, Any]] = []
    for truth_code, truth_name in enumerate(TRANSITION_NAMES):
        position_mask = truth_transition == truth_code
        positions = int(position_mask.sum())
        for generated_code, generated_name in enumerate(TRANSITION_NAMES):
            mask = position_mask[:, None] & (member_transition == generated_code)
            records.append(
                {
                    "truth_transition": truth_name,
                    "generated_transition": generated_name,
                    "truth_positions": positions,
                    "member_cases": int(mask.sum()),
                    "member_share_within_truth": float(mask.sum() / max(positions * values.shape[1], 1)),
                    "ramp_first_term_MAE": _safe_mean(absolute_error[mask]),
                    "state_source": state_source,
                }
            )

        oracle_scores: list[float] = []
        oracle_member_counts: list[int] = []
        position_indices = np.argwhere(position_mask)
        for day_index, zone_index, lead_index in position_indices:
            matching = member_transition[
                day_index, :, zone_index, lead_index
            ] == truth_code
            count = int(matching.sum())
            if count < 2:
                continue
            sample = sample_ramp[day_index, matching, zone_index, lead_index]
            observed = truth_ramp[day_index, zone_index, lead_index]
            score = empirical_crps(sample[None], np.asarray([observed]), member_axis=1)[0]
            oracle_scores.append(float(score))
            oracle_member_counts.append(count)
        records.append(
            {
                "truth_transition": truth_name,
                "generated_transition": "oracle_same_category",
                "truth_positions": positions,
                "member_cases": int(sum(oracle_member_counts)),
                "member_share_within_truth": None,
                "ramp_first_term_MAE": None,
                "oracle_empirical_CRPS": _safe_mean(np.asarray(oracle_scores)),
                "oracle_position_coverage": float(len(oracle_scores) / max(positions, 1)),
                "oracle_mean_members": _safe_mean(np.asarray(oracle_member_counts)),
                "state_source": state_source,
                "warning": "truth-conditioned diagnostic; not a proper forecast score",
            }
        )
    return records


def _correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = _as_float(left)
    y = _as_float(right)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("correlation inputs must have equal [sample,zone] shape")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = x.T @ y
    denominator = np.sqrt(
        np.sum(x * x, axis=0)[:, None] * np.sum(y * y, axis=0)[None, :]
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def lagged_correlation(
    values: np.ndarray,
    *,
    lag: int,
    increments: bool,
    climatology: np.ndarray | None = None,
) -> np.ndarray:
    sample = _as_float(values)
    if sample.ndim not in (3, 4):
        raise ValueError("lagged correlation expects [day,zone,hour] or [day,member,zone,hour]")
    if climatology is not None:
        centre = _as_float(climatology)
        if centre.shape != sample.shape[-2:]:
            raise ValueError("climatology must have shape [zone,hour]")
        sample = sample - centre if sample.ndim == 3 else sample - centre[None, None]
    if increments:
        sample = np.diff(sample, axis=-1)
    periods = sample.shape[-1]
    if lag < 0 or lag >= periods:
        raise ValueError("lag must be non-negative and smaller than the time dimension")
    stop = periods if lag == 0 else periods - lag
    left = sample[..., :stop]
    right = sample[..., lag : lag + stop]
    left = np.moveaxis(left, -2, -1).reshape(-1, sample.shape[-2])
    right = np.moveaxis(right, -2, -1).reshape(-1, sample.shape[-2])
    return _correlation_matrix(left, right)


def cross_lag_summary(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    lags: Sequence[int] = (0, 1, 2, 3),
    climatology: np.ndarray | None = None,
) -> dict[str, Any]:
    values = _as_float(scenarios)
    truth = _as_float(observations)
    if values.ndim != 4 or truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("cross-lag inputs do not align")
    off_diagonal = ~np.eye(values.shape[2], dtype=bool)
    records: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for increments, domain in ((False, "level"), (True, "increment")):
        for lag in lags:
            forecast = lagged_correlation(
                values, lag=int(lag), increments=increments, climatology=climatology
            )
            observed = lagged_correlation(
                truth, lag=int(lag), increments=increments, climatology=climatology
            )
            difference = forecast - observed
            valid_off = off_diagonal & np.isfinite(difference)
            diagonal = np.diag(difference)
            records.append(
                {
                    "domain": domain,
                    "lag": int(lag),
                    "cross_zone_RMSE": float(np.sqrt(np.mean(difference[valid_off] ** 2))),
                    "same_zone_MAE": float(np.nanmean(np.abs(diagonal))),
                    "forecast_offdiag_mean": float(np.nanmean(forecast[off_diagonal])),
                    "observed_offdiag_mean": float(np.nanmean(observed[off_diagonal])),
                }
            )
            matrices[f"{domain}_lag{lag}_forecast"] = forecast
            matrices[f"{domain}_lag{lag}_observed"] = observed
    return {"records": records, "matrices": matrices}


def per_day_cross_lag_error(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    lag: int = 1,
    increments: bool = True,
) -> np.ndarray:
    values = _as_float(scenarios)
    truth = _as_float(observations)
    errors = np.empty(values.shape[0], dtype=np.float64)
    off_diagonal = ~np.eye(values.shape[2], dtype=bool)
    for day in range(values.shape[0]):
        forecast = lagged_correlation(values[day : day + 1], lag=lag, increments=increments)
        observed = lagged_correlation(truth[day : day + 1], lag=lag, increments=increments)
        difference = forecast - observed
        valid = off_diagonal & np.isfinite(difference)
        errors[day] = np.sqrt(np.mean(difference[valid] ** 2)) if valid.any() else np.nan
    return errors


def stratified_paired_bootstrap(
    differences: np.ndarray,
    strata: np.ndarray,
    *,
    repetitions: int = 5000,
    seed: int = 20260814,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    values = _as_float(differences).reshape(-1)
    groups = np.asarray(strata).reshape(-1)
    if values.shape != groups.shape:
        raise ValueError("differences and strata must have the same length")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == item) for item in unique]
    if any(len(item) == 0 for item in indices):
        raise ValueError("empty bootstrap stratum")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(item, size=len(item), replace=True) for item in indices]
        )
        estimates[iteration] = values[sampled].mean()
    return {
        "n_days": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "probability_below_zero": float(np.mean(estimates < 0.0)),
        "bootstrap_repetitions": int(repetitions),
    }


def summarize_per_day_by_regime(
    records: Sequence[Mapping[str, Any]],
    *,
    model_left: str,
    model_right: str,
    metric_names: Sequence[str],
    regime_names: Sequence[str],
    bootstrap_repetitions: int = 5000,
) -> list[dict[str, Any]]:
    """Pair model-day rows and summarize left-minus-right by NWP stratum."""

    lookup: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (int(record["outer"]), str(record["day"]), str(record["model"]))
        if key in lookup:
            raise ValueError(f"duplicate per-day model row: {key}")
        lookup[key] = record
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for key, left in lookup.items():
        if key[2] != model_left:
            continue
        right_key = (key[0], key[1], model_right)
        if right_key not in lookup:
            raise ValueError(f"missing paired model row: {right_key}")
        pairs.append((left, lookup[right_key]))

    output: list[dict[str, Any]] = []
    for regime in regime_names:
        labels = sorted({str(left[regime]) for left, _ in pairs})
        for label in labels:
            selected = [(left, right) for left, right in pairs if str(left[regime]) == label]
            if not selected:
                continue
            strata = np.asarray([int(left["outer"]) for left, _ in selected])
            for metric in metric_names:
                left_values = np.asarray([float(left[metric]) for left, _ in selected])
                right_values = np.asarray([float(right[metric]) for _, right in selected])
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                if not finite.any():
                    continue
                summary = stratified_paired_bootstrap(
                    left_values[finite] - right_values[finite],
                    strata[finite],
                    repetitions=bootstrap_repetitions,
                    seed=20260814 + len(output),
                )
                output.append(
                    {
                        "regime_dimension": regime,
                        "regime_label": label,
                        "metric": metric,
                        "left_model": model_left,
                        "right_model": model_right,
                        "left_mean": float(left_values[finite].mean()),
                        "right_mean": float(right_values[finite].mean()),
                        **summary,
                        "inferential_warning": None
                        if int(summary["n_days"]) >= 20
                        else "fewer than 20 paired days; descriptive only",
                    }
                )
    return output
