"""Advanced, finite-ensemble diagnostics for wind-power trajectories.

The functions in this module deliberately keep the calendar day as the
observational unit.  Ensemble members define a finite predictive
distribution; they are never treated as independent observations.

Notation used below follows the diagnostic report:

* ``local`` ramp scores average proper cell-wise scores over zone and lead;
* ``fleet`` first averages power over zones and then takes the hourly ramp;
* the lagged variogram score uses exponent p=1 and equal pair weights;
* an atom is a value no larger than ``epsilon`` (normally exact zero or a
  threshold fixed using training data only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .core import (
    INTERIOR_STATE,
    TRANSITION_NAMES,
    classify_states,
    empirical_crps,
)


Array = np.ndarray
Domain = Literal["level", "ramp"]
PairType = Literal["cross_zone", "same_zone_temporal"]
Reduction = Literal["per_day", "pooled", "both"]


LAG1_PATH_NAMES = ("interior_only", "any_atom_involving")


def _finite_float(values: Array, *, name: str) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_forecast_truth(scenarios: Array, observations: Array) -> tuple[Array, Array]:
    values = _finite_float(scenarios, name="scenarios")
    truth = _finite_float(observations, name="observations")
    if values.ndim != 4:
        raise ValueError("scenarios must have shape [day,member,zone,hour]")
    if truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("observations must align with scenario day/zone/hour axes")
    if values.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if values.shape[2] < 1 or values.shape[3] < 2:
        raise ValueError("at least one zone and two hourly leads are required")
    return values, truth


def ramp_crps_metrics(scenarios: Array, observations: Array) -> dict[str, Array]:
    """Return complementary daily ramp CRPS diagnostics.

    ``max_up`` and ``max_down`` are the member-wise daily extrema across all
    zone/lead ramp cells. ``total_variation`` is the member-wise mean, over
    zones, of the sum of absolute hourly ramps.  All returned arrays have one
    value per day and use the exact empirical-distribution (V-statistic) CRPS.
    """

    values, truth = _validate_forecast_truth(scenarios, observations)
    member_ramp = np.diff(values, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)

    member_up = np.maximum(member_ramp, 0.0)
    truth_up = np.maximum(truth_ramp, 0.0)
    member_down = np.maximum(-member_ramp, 0.0)
    truth_down = np.maximum(-truth_ramp, 0.0)

    fleet_member_ramp = np.diff(values.mean(axis=2), axis=-1)
    fleet_truth_ramp = np.diff(truth.mean(axis=1), axis=-1)
    fleet_member_up = np.maximum(fleet_member_ramp, 0.0)
    fleet_truth_up = np.maximum(fleet_truth_ramp, 0.0)
    fleet_member_down = np.maximum(-fleet_member_ramp, 0.0)
    fleet_truth_down = np.maximum(-fleet_truth_ramp, 0.0)

    member_max_up = member_up.max(axis=(2, 3))
    truth_max_up = truth_up.max(axis=(1, 2))
    member_max_down = member_down.max(axis=(2, 3))
    truth_max_down = truth_down.max(axis=(1, 2))
    member_tv = np.abs(member_ramp).sum(axis=-1).mean(axis=2)
    truth_tv = np.abs(truth_ramp).sum(axis=-1).mean(axis=1)

    return {
        "ramp_local_mean_CRPS": empirical_crps(
            member_ramp, truth_ramp, member_axis=1
        ).mean(axis=(1, 2)),
        "ramp_local_up_CRPS": empirical_crps(
            member_up, truth_up, member_axis=1
        ).mean(axis=(1, 2)),
        "ramp_local_down_CRPS": empirical_crps(
            member_down, truth_down, member_axis=1
        ).mean(axis=(1, 2)),
        "ramp_fleet_mean_CRPS": empirical_crps(
            fleet_member_ramp, fleet_truth_ramp, member_axis=1
        ).mean(axis=1),
        "ramp_fleet_up_CRPS": empirical_crps(
            fleet_member_up, fleet_truth_up, member_axis=1
        ).mean(axis=1),
        "ramp_fleet_down_CRPS": empirical_crps(
            fleet_member_down, fleet_truth_down, member_axis=1
        ).mean(axis=1),
        "ramp_max_up_CRPS": empirical_crps(
            member_max_up, truth_max_up, member_axis=1
        ),
        "ramp_max_down_CRPS": empirical_crps(
            member_max_down, truth_max_down, member_axis=1
        ),
        "ramp_total_variation_CRPS": empirical_crps(
            member_tv, truth_tv, member_axis=1
        ),
    }


def shuffle_member_trajectories(scenarios: Array, *, seed: int = 20260814) -> Array:
    """Destroy cross-zone member coupling without altering local paths.

    A separate member permutation is drawn for every day and zone, and that
    permutation is applied to the complete hourly trajectory.  Consequently
    the multiset of trajectories in every day-zone cell is exactly preserved:
    its point marginals, local ramps, and atom-run durations are unchanged.
    Cross-zone member alignment (and hence fleet-member trajectories) is not.
    """

    values = _finite_float(scenarios, name="scenarios")
    if values.ndim != 4:
        raise ValueError("scenarios must have shape [day,member,zone,hour]")
    days, members, zones, _ = values.shape
    if members < 2:
        raise ValueError("at least two ensemble members are required")
    result = np.empty_like(values)
    rng = np.random.default_rng(seed)
    for day in range(days):
        for zone in range(zones):
            permutation = rng.permutation(members)
            result[day, :, zone, :] = values[day, permutation, zone, :]
    return result


def _domain_values(values: Array, truth: Array, domain: Domain) -> tuple[Array, Array]:
    if domain == "level":
        return values, truth
    if domain == "ramp":
        return np.diff(values, axis=-1), np.diff(truth, axis=-1)
    raise ValueError("domain must be 'level' or 'ramp'")


def _pair_differences(
    values: Array,
    truth: Array,
    *,
    lag: int,
    pair_type: PairType,
) -> tuple[Array, Array]:
    """Return forecast [D,M,P,T] and truth [D,P,T] pair differences."""

    periods = values.shape[-1]
    if lag < 0 or lag >= periods:
        raise ValueError("lag must be non-negative and smaller than the domain length")

    if pair_type == "same_zone_temporal":
        if lag == 0:
            # Keeping this explicit makes a complete lag 0..6 table possible;
            # the score is necessarily zero and should not drive conclusions.
            return np.zeros_like(values), np.zeros_like(truth)
        return (
            values[..., :-lag] - values[..., lag:],
            truth[..., :-lag] - truth[..., lag:],
        )

    if pair_type != "cross_zone":
        raise ValueError("pair_type must be 'cross_zone' or 'same_zone_temporal'")
    zones = values.shape[2]
    if zones < 2:
        raise ValueError("cross-zone variogram scores require at least two zones")
    zone_left, zone_right = np.where(~np.eye(zones, dtype=bool))
    stop = periods - lag if lag else periods
    left = values[:, :, zone_left, :stop]
    right = values[:, :, zone_right, lag : lag + stop]
    truth_left = truth[:, zone_left, :stop]
    truth_right = truth[:, zone_right, lag : lag + stop]
    return left - right, truth_left - truth_right


def lagged_variogram_score(
    scenarios: Array,
    observations: Array,
    *,
    lag: int,
    domain: Domain,
    pair_type: PairType,
    p: float = 1.0,
    reduction: Reduction = "per_day",
) -> Array | float | dict[str, Array | float]:
    """Compute an equal-weight lagged variogram score.

    For each day and requested pair set, this evaluates

    ``mean_pairs((|y_i-y_j|^p - mean_m |x_mi-x_mj|^p)^2)``.

    ``pooled`` is the mean of the daily scores, preserving day as the unit of
    analysis.  The project comparison fixes ``p=1``; the argument is retained
    only so an accidental different exponent can be rejected explicitly.
    """

    values, truth = _validate_forecast_truth(scenarios, observations)
    if not np.isclose(p, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("the registered lagged variogram diagnostic fixes p=1")
    forecast_domain, truth_domain = _domain_values(values, truth, domain)
    member_difference, truth_difference = _pair_differences(
        forecast_domain,
        truth_domain,
        lag=int(lag),
        pair_type=pair_type,
    )
    forecast_variogram = np.mean(np.abs(member_difference), axis=1)
    observed_variogram = np.abs(truth_difference)
    score = np.mean((forecast_variogram - observed_variogram) ** 2, axis=(1, 2))
    pooled = float(score.mean())
    if reduction == "per_day":
        return score
    if reduction == "pooled":
        return pooled
    if reduction == "both":
        return {"per_day": score, "pooled": pooled}
    raise ValueError("reduction must be 'per_day', 'pooled', or 'both'")


def lagged_variogram_suite(
    scenarios: Array,
    observations: Array,
    *,
    lags: Sequence[int] = tuple(range(7)),
    include_shuffle_control: bool = False,
    shuffle_seed: int = 20260814,
) -> dict[str, Any]:
    """Evaluate level/ramp, cross-zone/same-zone lagged variograms.

    The result stores daily arrays under stable string keys and scalar means
    in records suitable for CSV/JSON output.  When requested, the same table
    is repeated after the whole-trajectory member-shuffle negative control.
    """

    values, truth = _validate_forecast_truth(scenarios, observations)
    unique_lags = tuple(dict.fromkeys(int(item) for item in lags))
    if not unique_lags:
        raise ValueError("at least one lag is required")

    variants: list[tuple[str, Array]] = [("original", values)]
    if include_shuffle_control:
        variants.append(
            ("member_trajectory_shuffle", shuffle_member_trajectories(values, seed=shuffle_seed))
        )

    per_day: dict[str, Array] = {}
    records: list[dict[str, Any]] = []
    for variant, sample in variants:
        for domain in ("level", "ramp"):
            domain_periods = sample.shape[-1] - (1 if domain == "ramp" else 0)
            for pair_type in ("cross_zone", "same_zone_temporal"):
                for lag in unique_lags:
                    if lag < 0 or lag >= domain_periods:
                        raise ValueError(
                            f"lag {lag} is invalid for {domain} domain length {domain_periods}"
                        )
                    key = f"{variant}__{domain}__{pair_type}__lag{lag}"
                    daily = lagged_variogram_score(
                        sample,
                        truth,
                        lag=lag,
                        domain=domain,
                        pair_type=pair_type,
                        reduction="per_day",
                    )
                    assert isinstance(daily, np.ndarray)
                    per_day[key] = daily
                    records.append(
                        {
                            "variant": variant,
                            "domain": domain,
                            "pair_type": pair_type,
                            "lag": lag,
                            "n_days": int(len(daily)),
                            "pooled_score": float(daily.mean()),
                        }
                    )
    return {"per_day": per_day, "records": records, "shuffle_seed": int(shuffle_seed)}


def zero_run_statistics(values: Array, *, epsilon: float = 0.0) -> dict[str, Array]:
    """Return daily scalar H/K/L statistics for zero-atom paths.

    Input ends in ``[zone,hour]`` and may have arbitrary leading axes.  H is
    total zero hours across zones, K is the total number of zero runs across
    zones, and L is the longest zero run among the zones.
    """

    sample = _finite_float(values, name="values")
    if sample.ndim < 2:
        raise ValueError("values must end in [zone,hour]")
    if epsilon < 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie in [0,0.5)")
    zero = sample <= epsilon
    total_hours = zero.sum(axis=(-2, -1), dtype=np.int64)
    run_starts = zero[..., 0].sum(axis=-1, dtype=np.int64)
    if zero.shape[-1] > 1:
        run_starts = run_starts + (
            (~zero[..., :-1]) & zero[..., 1:]
        ).sum(axis=(-2, -1), dtype=np.int64)

    current = np.zeros(zero.shape[:-1], dtype=np.int16)
    longest_by_zone = np.zeros_like(current)
    for hour in range(zero.shape[-1]):
        current = np.where(zero[..., hour], current + 1, 0)
        longest_by_zone = np.maximum(longest_by_zone, current)
    longest = longest_by_zone.max(axis=-1)
    return {
        "H_total_zero_hours": total_hours,
        "K_zero_run_count": run_starts,
        "L_longest_zero_run": longest,
    }


def atom_event_metrics(
    scenarios: Array,
    observations: Array,
    *,
    epsilon: float = 0.0,
) -> dict[str, Array]:
    """Daily zero-atom count/duration CRPS and transition Brier scores."""

    values, truth = _validate_forecast_truth(scenarios, observations)
    member_statistics = zero_run_statistics(values, epsilon=epsilon)
    truth_statistics = zero_run_statistics(truth, epsilon=epsilon)
    result: dict[str, Array] = {}
    for short, name in (
        ("H", "H_total_zero_hours"),
        ("K", "K_zero_run_count"),
        ("L", "L_longest_zero_run"),
    ):
        result[f"atom_{short}_CRPS"] = empirical_crps(
            member_statistics[name], truth_statistics[name], member_axis=1
        )

    member_zero = values <= epsilon
    truth_zero = truth <= epsilon
    probability_any = member_zero.any(axis=(2, 3)).mean(axis=1)
    observed_any = truth_zero.any(axis=(1, 2)).astype(np.float64)
    result["atom_any_zero_Brier"] = (probability_any - observed_any) ** 2

    member_entry = (~member_zero[..., :-1]) & member_zero[..., 1:]
    truth_entry = (~truth_zero[..., :-1]) & truth_zero[..., 1:]
    member_exit = member_zero[..., :-1] & (~member_zero[..., 1:])
    truth_exit = truth_zero[..., :-1] & (~truth_zero[..., 1:])
    entry_probability = member_entry.mean(axis=1)
    exit_probability = member_exit.mean(axis=1)
    result["atom_entry_transition_Brier"] = np.mean(
        (entry_probability - truth_entry) ** 2, axis=(1, 2)
    )
    result["atom_exit_transition_Brier"] = np.mean(
        (exit_probability - truth_exit) ** 2, axis=(1, 2)
    )
    return result


def _coarse_transition_category(states: Array) -> Array:
    """Match the repository's three generated-transition categories."""

    state = np.asarray(states, dtype=np.int8)
    if state.shape[-1] < 2:
        raise ValueError("state sequence needs at least two hours")
    left_interior = state[..., :-1] == INTERIOR_STATE
    right_interior = state[..., 1:] == INTERIOR_STATE
    result = np.full(left_interior.shape, 1, dtype=np.int8)
    result[left_interior & right_interior] = 0
    result[(~left_interior) & (~right_interior)] = 2
    return result


def generated_transition_crps_decomposition(
    scenarios: Array,
    observations: Array,
    *,
    scenario_states: Array | None = None,
    epsilon: float = 0.0,
) -> dict[str, Any]:
    """Exactly decompose local ramp CRPS by generated transition category.

    For every ramp cell the finite-ensemble CRPS is

    ``M^-1 sum_m |x_m-y| - (2 M^2)^-1 sum_m sum_n |x_m-x_n|``.

    The first term is assigned to the state category of member ``m``.  Half
    of each unordered diversity pair is assigned to each endpoint (equivalent
    to assigning the ordered-member sum shown above).  Thus category net
    contributions add exactly to the ordinary local ramp CRPS.  Pairwise
    distances are obtained from sorted values in O(M log M), without an MxM
    tensor.  A category contribution can be negative; only the total CRPS is
    constrained to be non-negative.
    """

    values, truth = _validate_forecast_truth(scenarios, observations)
    if scenario_states is None:
        member_states = classify_states(values, epsilon)
        state_source = f"threshold_epsilon={epsilon:.8g}"
    else:
        member_states = np.asarray(scenario_states, dtype=np.int8)
        if member_states.shape != values.shape:
            raise ValueError("scenario_states do not align with scenarios")
        if not np.isin(member_states, (0, 1, 2)).all():
            raise ValueError("scenario_states must contain only 0, 1, or 2")
        state_source = "archive_explicit_state"

    member_ramp = np.diff(values, axis=-1)
    truth_ramp = np.diff(truth, axis=-1)
    category = _coarse_transition_category(member_states)
    members = values.shape[1]

    # Put member last, sort once, and carry categories through the same order.
    ramp_last = np.moveaxis(member_ramp, 1, -1)
    category_last = np.moveaxis(category, 1, -1)
    order = np.argsort(ramp_last, axis=-1, kind="stable")
    ordered = np.take_along_axis(ramp_last, order, axis=-1)
    ordered_category = np.take_along_axis(category_last, order, axis=-1)

    prefix = np.cumsum(ordered, axis=-1)
    prefix_before = prefix - ordered
    total = prefix[..., -1, None]
    rank = np.arange(members, dtype=np.float64)
    distance_sum = (
        ordered * rank
        - prefix_before
        + (total - prefix)
        - ordered * (members - rank - 1.0)
    )
    if np.any(distance_sum < -1e-10):
        raise RuntimeError("negative member distance sum from sorted-distance identity")
    distance_sum = np.maximum(distance_sum, 0.0)
    absolute_error = np.abs(ordered - truth_ramp[..., None])

    first = np.empty((values.shape[0], len(TRANSITION_NAMES)), dtype=np.float64)
    pair = np.empty_like(first)
    counts = np.empty_like(first, dtype=np.int64)
    for code in range(len(TRANSITION_NAMES)):
        mask = ordered_category == code
        first_by_cell = np.sum(np.where(mask, absolute_error, 0.0), axis=-1) / members
        pair_by_cell = np.sum(np.where(mask, distance_sum, 0.0), axis=-1) / (
            2.0 * members * members
        )
        first[:, code] = first_by_cell.mean(axis=(1, 2))
        pair[:, code] = pair_by_cell.mean(axis=(1, 2))
        counts[:, code] = mask.sum(axis=(1, 2, 3))

    net = first - pair
    total_score = empirical_crps(member_ramp, truth_ramp, member_axis=1).mean(axis=(1, 2))
    reconstructed = net.sum(axis=1)
    tolerance = 2e-12 + 2e-10 * np.abs(total_score)
    if not np.all(np.abs(reconstructed - total_score) <= tolerance):
        raise RuntimeError("transition contributions do not reconstruct local ramp CRPS")

    records: list[dict[str, Any]] = []
    transition_definitions = {
        "interior_to_interior": "both ramp endpoints are interior",
        # This label is retained for compatibility with core.py; its category
        # is intentionally bidirectional and therefore means atom<->interior.
        "atom_to_interior": "exactly one endpoint is interior (atom<->interior)",
        "atom_to_atom": "neither endpoint is interior",
    }
    for code, name in enumerate(TRANSITION_NAMES):
        records.append(
            {
                "generated_transition": name,
                "transition_definition": transition_definitions[name],
                "state_source": state_source,
                "member_cases": int(counts[:, code].sum()),
                "member_share": float(counts[:, code].sum() / max(counts.sum(), 1)),
                "first_term": float(first[:, code].mean()),
                "pair_term": float(pair[:, code].mean()),
                "net_contribution": float(net[:, code].mean()),
            }
        )
    return {
        "transition_names": list(TRANSITION_NAMES),
        "transition_definitions": transition_definitions,
        "state_source": state_source,
        "first_term": first,
        "pair_term": pair,
        "net_contribution": net,
        "member_cases": counts,
        "total_local_ramp_CRPS": total_score,
        "reconstructed_local_ramp_CRPS": reconstructed,
        "records": records,
    }


def _lag1_path_category(states: Array) -> Array:
    """Classify three-point paths underlying adjacent increment pairs."""

    state = np.asarray(states, dtype=np.int8)
    if state.shape[-1] < 3:
        raise ValueError("lag-1 increment paths require at least three hourly levels")
    interior_only = (
        (state[..., :-2] == INTERIOR_STATE)
        & (state[..., 1:-1] == INTERIOR_STATE)
        & (state[..., 2:] == INTERIOR_STATE)
    )
    return np.where(interior_only, 0, 1).astype(np.int8)


def _standardized_product_by_category(
    left: Array,
    right: Array,
    category: Array,
) -> dict[str, Array]:
    """Decompose a daily Pearson numerator on its own global scale.

    The population means and standard deviations are computed across all
    cases within a day, before cases are divided into categories.  Therefore
    the category contributions sum exactly to that day's ordinary Pearson
    correlation.  This is an accounting identity, not a causal allocation.
    """

    x = _finite_float(left, name="left increments")
    y = _finite_float(right, name="right increments")
    labels = np.asarray(category, dtype=np.int8)
    if x.shape != y.shape or labels.shape != x.shape or x.ndim < 2:
        raise ValueError("increment pairs and path categories must align by day")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("path categories must contain only 0 or 1")

    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    count = x.shape[1]
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.mean(x_centered * x_centered, axis=1)
        * np.mean(y_centered * y_centered, axis=1)
    )
    valid_day = denominator > 1e-12
    standardized_product = np.divide(
        x_centered * y_centered,
        denominator[:, None],
        out=np.full_like(x_centered, np.nan),
        where=valid_day[:, None],
    )

    case_count = np.stack([(labels == code).sum(axis=1) for code in (0, 1)], axis=1)
    case_share = case_count.astype(np.float64) / float(count)
    contribution = np.empty_like(case_share)
    conditional_global_moment = np.full_like(case_share, np.nan)
    conditional_correlation = np.full_like(case_share, np.nan)
    for code in (0, 1):
        mask = labels == code
        contribution[:, code] = np.sum(
            np.where(mask, standardized_product, 0.0), axis=1
        ) / float(count)
        np.divide(
            contribution[:, code],
            case_share[:, code],
            out=conditional_global_moment[:, code],
            where=(case_share[:, code] > 0.0) & valid_day,
        )
        for day in range(x.shape[0]):
            selected = mask[day]
            if int(selected.sum()) < 2:
                continue
            selected_x = x[day, selected]
            selected_y = y[day, selected]
            selected_x = selected_x - selected_x.mean()
            selected_y = selected_y - selected_y.mean()
            selected_denominator = np.sqrt(
                np.mean(selected_x * selected_x) * np.mean(selected_y * selected_y)
            )
            if selected_denominator > 1e-12:
                conditional_correlation[day, code] = float(
                    np.mean(selected_x * selected_y) / selected_denominator
                )

    total_correlation = np.mean(standardized_product, axis=1)
    reconstructed = contribution.sum(axis=1)
    if not np.allclose(
        reconstructed[valid_day],
        total_correlation[valid_day],
        rtol=2e-12,
        atol=2e-12,
    ):
        raise RuntimeError("path contributions do not reconstruct daily correlation")
    return {
        "case_count": case_count,
        "case_share": case_share,
        "correlation_contribution": contribution,
        "conditional_global_standardized_moment": conditional_global_moment,
        "conditional_correlation": conditional_correlation,
        "total_correlation": total_correlation,
        "reconstructed_correlation": reconstructed,
        "valid_day": valid_day,
    }


def lag1_increment_state_decomposition(
    scenarios: Array,
    observations: Array,
    *,
    scenario_states: Array | None = None,
    epsilon: float = 0.0,
) -> dict[str, Any]:
    """Daily three-point attribution of same-zone lag-1 increment correlation.

    A lag-1 increment pair uses three consecutive levels,
    ``a=x[t+1]-x[t]`` and ``b=x[t+2]-x[t+1]``.  Each forecast member path is
    labelled either ``interior_only`` when all three states are interior, or
    ``any_atom_involving`` otherwise.  Truth paths receive the same two output
    labels.  For forecast and truth separately, the ordinary within-day
    Pearson correlation is split into category contributions using global
    within-day centring and scaling.  Consequently::

        signed_total_correlation_gap
          == sum(signed_correlation_gap_contribution, axis=category)

    exactly (up to floating-point roundoff).  Absolute error is reported only
    for the total because absolute values are not additively attributable.

    ``scenario_states`` should be supplied only when an archive contains an
    explicit model state.  If it is omitted (as for the zone-wise DDPM
    archives), states are *phenomenological output-boundary labels* obtained
    from ``x <= epsilon`` or ``x >= 1-epsilon``.  They are not evidence that
    the generator has a latent discrete state.  Calendar day remains the
    observational unit; ensemble members are integrated within each day's
    predictive distribution and are never returned as pseudo-replicates.

    Conditional correlations are included as descriptive diagnostics.  They
    use category-specific centring/scaling, are non-additive, and are ``NaN``
    when a day/category has insufficient variation.
    """

    values, truth = _validate_forecast_truth(scenarios, observations)
    if values.shape[-1] < 3:
        raise ValueError("lag-1 increment decomposition requires at least three hours")
    if epsilon < 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie in [0,0.5)")

    if scenario_states is None:
        member_states = classify_states(values, epsilon)
        forecast_state_source = f"threshold_derived_output_boundary_epsilon={epsilon:.8g}"
        forecast_state_semantics = (
            "output-value boundary label only; not a latent generative state"
        )
    else:
        member_states = np.asarray(scenario_states, dtype=np.int8)
        if member_states.shape != values.shape:
            raise ValueError("scenario_states do not align with scenarios")
        if not np.isin(member_states, (0, 1, 2)).all():
            raise ValueError("scenario_states must contain only 0, 1, or 2")
        forecast_state_source = "archive_explicit_state"
        forecast_state_semantics = "model-provided discrete state from the archive"

    truth_states = classify_states(truth, epsilon)
    forecast_category = _lag1_path_category(member_states)
    truth_category = _lag1_path_category(truth_states)
    forecast_increment = np.diff(values, axis=-1)
    truth_increment = np.diff(truth, axis=-1)
    forecast = _standardized_product_by_category(
        forecast_increment[..., :-1],
        forecast_increment[..., 1:],
        forecast_category,
    )
    observed = _standardized_product_by_category(
        truth_increment[..., :-1],
        truth_increment[..., 1:],
        truth_category,
    )

    signed_contribution_gap = (
        forecast["correlation_contribution"] - observed["correlation_contribution"]
    )
    signed_total_gap = forecast["total_correlation"] - observed["total_correlation"]
    reconstructed_gap = signed_contribution_gap.sum(axis=1)
    valid = forecast["valid_day"] & observed["valid_day"]
    if not np.allclose(
        reconstructed_gap[valid], signed_total_gap[valid], rtol=2e-12, atol=2e-12
    ):
        raise RuntimeError("path gap contributions do not reconstruct correlation gap")

    return {
        "path_names": list(LAG1_PATH_NAMES),
        "path_definitions": {
            "interior_only": "all three consecutive level states are interior",
            "any_atom_involving": "at least one of the three level states is a lower/upper atom",
        },
        "epsilon": float(epsilon),
        "forecast_state_source": forecast_state_source,
        "forecast_state_semantics": forecast_state_semantics,
        "truth_state_source": f"threshold_derived_output_boundary_epsilon={epsilon:.8g}",
        "truth_state_semantics": "observed output-boundary label, not a latent state",
        "forecast_case_count": forecast["case_count"],
        "forecast_case_share": forecast["case_share"],
        "truth_case_count": observed["case_count"],
        "truth_case_share": observed["case_share"],
        "forecast_correlation_contribution": forecast["correlation_contribution"],
        "truth_correlation_contribution": observed["correlation_contribution"],
        "signed_correlation_gap_contribution": signed_contribution_gap,
        "forecast_conditional_global_standardized_moment": forecast[
            "conditional_global_standardized_moment"
        ],
        "truth_conditional_global_standardized_moment": observed[
            "conditional_global_standardized_moment"
        ],
        "forecast_conditional_correlation": forecast["conditional_correlation"],
        "truth_conditional_correlation": observed["conditional_correlation"],
        "conditional_correlation_gap": (
            forecast["conditional_correlation"] - observed["conditional_correlation"]
        ),
        "forecast_total_correlation": forecast["total_correlation"],
        "truth_total_correlation": observed["total_correlation"],
        "signed_total_correlation_gap": signed_total_gap,
        "absolute_total_correlation_error": np.abs(signed_total_gap),
        "reconstructed_signed_total_correlation_gap": reconstructed_gap,
        "valid_day": valid,
        "unit_of_analysis": "calendar_day",
        "member_semantics": "finite predictive distribution; not independent observations",
        "attribution_caveat": (
            "descriptive accounting by generated/observed path label; not a causal decomposition"
        ),
    }


__all__ = [
    "atom_event_metrics",
    "generated_transition_crps_decomposition",
    "lag1_increment_state_decomposition",
    "lagged_variogram_score",
    "lagged_variogram_suite",
    "ramp_crps_metrics",
    "shuffle_member_trajectories",
    "zero_run_statistics",
]
