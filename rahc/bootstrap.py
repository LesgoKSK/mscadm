"""Paired calendar-day cluster bootstrap for the RAHC experiment.

The test archive contains several zones for each calendar date.  Treating the
zone-day rows as independent would make uncertainty intervals much too narrow,
so this module samples *unique calendar dates* with replacement and includes
all rows (and therefore all zones and hours) belonging to every sampled date.

The same bootstrap date multiplicities are used for all three model seeds and
for both methods.  A metric difference is formed within each seed first and
the three paired differences are averaged only afterwards.  The seeds are
never pooled as independent observations.

Conditional coverage statistics are also recomputed correctly inside every
replicate: successes and counts are accumulated for every date/group pair,
then resampled successes are divided by resampled counts.  In particular, the
implementation does not average precomputed daily coverage percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .evaluation import crps_cells, interval_cell_metrics


DEFAULT_REPLICATES = 10_000
DEFAULT_RANDOM_SEED = 20_260_718
DEFAULT_NOMINAL = 0.90
EXPECTED_MODEL_SEEDS = 3


@dataclass(frozen=True)
class _PreparedMethod:
    """Day-level sufficient statistics and cell-level interval hits."""

    crps_sum_by_day: np.ndarray
    covered_sum_by_day: np.ndarray
    winkler_sum_by_day: np.ndarray
    width_sum_by_day: np.ndarray
    covered_cells: tuple[np.ndarray, ...]


def _validate_inputs(
    baseline_scenarios_by_seed: Sequence[np.ndarray],
    method_scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    day: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    nominal: float,
    expected_unique_days: int | None,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    tuple[object, ...],
    dict[str, np.ndarray],
]:
    baseline = tuple(np.asarray(values, dtype=np.float64) for values in baseline_scenarios_by_seed)
    method = tuple(np.asarray(values, dtype=np.float64) for values in method_scenarios_by_seed)
    if len(baseline) != EXPECTED_MODEL_SEEDS or len(method) != EXPECTED_MODEL_SEEDS:
        raise ValueError(
            f"the registered protocol requires exactly {EXPECTED_MODEL_SEEDS} "
            "baseline and method scenario arrays"
        )

    truth = np.asarray(observations, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] < 1 or truth.shape[1] < 1:
        raise ValueError("observations must have shape [cases, hours]")
    if not np.isfinite(truth).all():
        raise ValueError("observations contain non-finite values")
    cases, hours = truth.shape

    for label, collection in (("baseline", baseline), ("method", method)):
        for seed, values in enumerate(collection):
            if values.ndim != 3 or values.shape[0] != cases or values.shape[2] != hours:
                raise ValueError(
                    f"{label} seed {seed} must have shape "
                    f"[{cases}, members, {hours}]"
                )
            if values.shape[1] < 2:
                raise ValueError(f"{label} seed {seed} has fewer than two members")
            if not np.isfinite(values).all():
                raise ValueError(f"{label} seed {seed} contains non-finite values")

    dates = np.asarray(day)
    if dates.shape != (cases,):
        raise ValueError(f"day must have shape ({cases},)")
    if dates.dtype.kind in {"f", "c"} and not np.isfinite(dates).all():
        raise ValueError("day contains non-finite values")
    unique_dates, date_codes = np.unique(dates, return_inverse=True)
    if expected_unique_days is not None and len(unique_dates) != expected_unique_days:
        raise ValueError(
            f"expected {expected_unique_days} unique calendar days, found {len(unique_dates)}"
        )
    if len(unique_dates) < 2:
        raise ValueError("at least two unique calendar days are required")

    if not assignments:
        raise ValueError("assignments must contain at least one conditional family")
    checked_assignments: dict[str, np.ndarray] = {}
    for family, raw_codes in assignments.items():
        if not isinstance(family, str) or not family:
            raise ValueError("assignment family names must be non-empty strings")
        codes = np.asarray(raw_codes)
        if codes.shape != (cases, hours):
            raise ValueError(
                f"assignment family {family!r} must have shape ({cases}, {hours})"
            )
        if codes.dtype.kind in {"f", "c"} and not np.isfinite(codes).all():
            raise ValueError(f"assignment family {family!r} contains non-finite values")
        try:
            if len(np.unique(codes)) == 0:
                raise ValueError(f"assignment family {family!r} is empty")
        except TypeError as error:
            raise ValueError(
                f"assignment family {family!r} must contain sortable scalar codes"
            ) from error
        checked_assignments[family] = codes

    if not 0.0 < float(nominal) < 1.0:
        raise ValueError("nominal must be strictly between zero and one")
    # Converting dates to Python scalars keeps the returned metadata JSON-safe
    # for ordinary integer/string archives while retaining readable datetimes.
    date_metadata = tuple(
        value.item() if isinstance(value, np.generic) else value for value in unique_dates
    )
    return (
        baseline,
        method,
        truth,
        date_codes.astype(np.int64, copy=False),
        date_metadata,
        checked_assignments,
    )


def _sum_cells_by_day(
    values: np.ndarray, date_codes: np.ndarray, number_of_days: int
) -> np.ndarray:
    """Sum an ``[N,H]`` cell statistic within each unique date."""

    return np.bincount(
        date_codes,
        weights=np.asarray(values, dtype=np.float64).sum(axis=1),
        minlength=number_of_days,
    ).astype(np.float64, copy=False)


def _prepare_method(
    scenarios_by_seed: tuple[np.ndarray, ...],
    observations: np.ndarray,
    date_codes: np.ndarray,
    number_of_days: int,
    nominal: float,
) -> _PreparedMethod:
    crps_days: list[np.ndarray] = []
    covered_days: list[np.ndarray] = []
    winkler_days: list[np.ndarray] = []
    width_days: list[np.ndarray] = []
    covered_cells: list[np.ndarray] = []
    for scenarios in scenarios_by_seed:
        crps = crps_cells(scenarios, observations)
        covered, width, winkler = interval_cell_metrics(scenarios, observations, nominal)
        crps_days.append(_sum_cells_by_day(crps, date_codes, number_of_days))
        covered_days.append(_sum_cells_by_day(covered, date_codes, number_of_days))
        winkler_days.append(_sum_cells_by_day(winkler, date_codes, number_of_days))
        width_days.append(_sum_cells_by_day(width, date_codes, number_of_days))
        covered_cells.append(np.asarray(covered, dtype=np.float64))
    return _PreparedMethod(
        crps_sum_by_day=np.stack(crps_days),
        covered_sum_by_day=np.stack(covered_days),
        winkler_sum_by_day=np.stack(winkler_days),
        width_sum_by_day=np.stack(width_days),
        covered_cells=tuple(covered_cells),
    )


def _group_count_sufficient_statistics(
    assignments: Mapping[str, np.ndarray],
    date_codes: np.ndarray,
    hours: int,
    number_of_days: int,
) -> tuple[np.ndarray, tuple[slice, ...], tuple[str, ...], tuple[int, ...], tuple[np.ndarray, ...]]:
    """Build date/group counts and reusable flattened joint indices."""

    cell_dates = np.repeat(date_codes, hours)
    count_parts: list[np.ndarray] = []
    family_slices: list[slice] = []
    family_names: list[str] = []
    group_counts: list[int] = []
    joint_indices: list[np.ndarray] = []
    offset = 0
    for family, raw_codes in assignments.items():
        _, inverse = np.unique(np.asarray(raw_codes).reshape(-1), return_inverse=True)
        groups = int(inverse.max()) + 1
        joint = cell_dates * groups + inverse
        counts = np.bincount(
            joint, minlength=number_of_days * groups
        ).reshape(number_of_days, groups)
        count_parts.append(counts.astype(np.float64, copy=False))
        family_slices.append(slice(offset, offset + groups))
        family_names.append(family)
        group_counts.append(groups)
        joint_indices.append(joint.astype(np.int64, copy=False))
        offset += groups
    return (
        np.concatenate(count_parts, axis=1),
        tuple(family_slices),
        tuple(family_names),
        tuple(group_counts),
        tuple(joint_indices),
    )


def _success_sufficient_statistics(
    covered_cells: tuple[np.ndarray, ...],
    joint_indices: tuple[np.ndarray, ...],
    group_counts: tuple[int, ...],
    number_of_days: int,
) -> tuple[np.ndarray, ...]:
    """Return one ``[S,D,G_total]`` success tensor per method."""

    per_seed: list[np.ndarray] = []
    for covered in covered_cells:
        flat_success = covered.reshape(-1)
        family_parts: list[np.ndarray] = []
        for joint, groups in zip(joint_indices, group_counts):
            values = np.bincount(
                joint,
                weights=flat_success,
                minlength=number_of_days * groups,
            ).reshape(number_of_days, groups)
            family_parts.append(values.astype(np.float64, copy=False))
        per_seed.append(np.concatenate(family_parts, axis=1))
    return tuple(per_seed)


def _global_replicate_metrics(
    weights: np.ndarray,
    cells_by_day: np.ndarray,
    baseline: _PreparedMethod,
    method: _PreparedMethod,
    nominal: float,
) -> dict[str, np.ndarray]:
    """Compute paired, mean-seed global differences for all replicates."""

    denominator = weights @ cells_by_day
    if np.any(denominator <= 0.0):
        raise RuntimeError("a bootstrap replicate contains no forecast cells")

    def means(day_sums: np.ndarray) -> np.ndarray:
        # day_sums is [S,D]; output is [R,S].
        return (weights @ day_sums.T) / denominator[:, None]

    baseline_crps = means(baseline.crps_sum_by_day)
    method_crps = means(method.crps_sum_by_day)
    baseline_winkler = means(baseline.winkler_sum_by_day)
    method_winkler = means(method.winkler_sum_by_day)
    baseline_width = means(baseline.width_sum_by_day)
    method_width = means(method.width_sum_by_day)
    baseline_coverage = means(baseline.covered_sum_by_day)
    method_coverage = means(method.covered_sum_by_day)

    return {
        "crps_improvement": (baseline_crps - method_crps).mean(axis=1),
        "winkler_90_improvement": (baseline_winkler - method_winkler).mean(axis=1),
        "coverage_absolute_error_90_improvement": (
            np.abs(baseline_coverage - nominal) - np.abs(method_coverage - nominal)
        ).mean(axis=1),
        # This is deliberately a change rather than an "improvement": a
        # positive value means the method produced wider central intervals.
        "interval_width_90_change": (method_width - baseline_width).mean(axis=1),
    }


def _conditional_summary_from_totals(
    successes: np.ndarray,
    counts: np.ndarray,
    family_slices: tuple[slice, ...],
    nominal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute family-equal ACE and WUCE from replicate/group totals.

    ``successes`` and ``counts`` both have shape ``[replicates, groups]``.
    Empty groups in a replicate are excluded from that family's mean and from
    the worst-group extremum.  Because every input cell belongs to one group
    in every family, at least one valid group is always present.
    """

    valid = counts > 0.0
    coverage = np.full(successes.shape, np.nan, dtype=np.float64)
    np.divide(successes, counts, out=coverage, where=valid)

    family_aces: list[np.ndarray] = []
    for family_slice in family_slices:
        family_valid = valid[:, family_slice]
        family_error = np.abs(coverage[:, family_slice] - nominal)
        numerator = np.nansum(family_error, axis=1)
        denominator = family_valid.sum(axis=1)
        value = np.full(len(successes), np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=value, where=denominator > 0)
        family_aces.append(value)
    stacked_aces = np.stack(family_aces, axis=1)
    valid_families = np.isfinite(stacked_aces)
    family_equal_ace = np.divide(
        np.nansum(stacked_aces, axis=1),
        valid_families.sum(axis=1),
        out=np.full(len(successes), np.nan, dtype=np.float64),
        where=valid_families.sum(axis=1) > 0,
    )

    undercoverage = np.where(valid, np.maximum(0.0, nominal - coverage), -np.inf)
    worst_undercoverage = np.max(undercoverage, axis=1)
    if not np.isfinite(family_equal_ace).all() or not np.isfinite(worst_undercoverage).all():
        raise RuntimeError("could not compute conditional metrics for a bootstrap replicate")
    return family_equal_ace, worst_undercoverage


def _conditional_replicate_metrics(
    weights: np.ndarray,
    count_by_day_group: np.ndarray,
    baseline_success_by_seed: tuple[np.ndarray, ...],
    method_success_by_seed: tuple[np.ndarray, ...],
    family_slices: tuple[slice, ...],
    nominal: float,
) -> dict[str, np.ndarray]:
    """Compute paired conditional differences after resampled aggregation."""

    counts = weights @ count_by_day_group
    ace_differences: list[np.ndarray] = []
    worst_under_differences: list[np.ndarray] = []
    for baseline_success, method_success in zip(
        baseline_success_by_seed, method_success_by_seed
    ):
        baseline_ace, baseline_worst_under = _conditional_summary_from_totals(
            weights @ baseline_success, counts, family_slices, nominal
        )
        method_ace, method_worst_under = _conditional_summary_from_totals(
            weights @ method_success, counts, family_slices, nominal
        )
        ace_differences.append(baseline_ace - method_ace)
        worst_under_differences.append(baseline_worst_under - method_worst_under)
    return {
        "conditional_family_equal_ACE_90_improvement": np.stack(
            ace_differences, axis=1
        ).mean(axis=1),
        "conditional_worst_undercoverage_90_improvement": np.stack(
            worst_under_differences, axis=1
        ).mean(axis=1),
    }


def _metric_record(
    point_estimate: float,
    bootstrap_values: np.ndarray,
    ci: tuple[float, float],
    *,
    direction: str,
    support_nonpositive: bool = False,
) -> dict[str, float | str]:
    lower, upper = np.quantile(bootstrap_values, ci)
    if support_nonpositive:
        support = float(np.mean(bootstrap_values <= 0.0))
        support_event = "metric <= 0 (no interval widening)"
    else:
        support = float(np.mean(bootstrap_values > 0.0))
        support_event = "metric > 0 (strict improvement)"
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_support_fraction": support,
        "bootstrap_support_event": support_event,
        "direction": direction,
    }


def paired_calendar_day_bootstrap(
    baseline_scenarios_by_seed: Sequence[np.ndarray],
    method_scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    day: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    *,
    replicates: int = DEFAULT_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    nominal: float = DEFAULT_NOMINAL,
    ci: tuple[float, float] = (0.025, 0.975),
    expected_unique_days: int | None = 50,
) -> dict[str, object]:
    """Run the registered paired calendar-day cluster bootstrap.

    Parameters
    ----------
    baseline_scenarios_by_seed, method_scenarios_by_seed:
        Exactly three aligned arrays per method, each ``[N,M,H]``.  Member
        counts may differ across methods because every score is computed
        within method before forming the paired difference.
    observations:
        Common observations with shape ``[N,H]``.
    day:
        Calendar-date identifier for every case, shape ``[N]``.  Repeated
        zone rows for the same date must carry the same identifier.
    assignments:
        Frozen, method-independent conditional group codes, one ``[N,H]``
        array per group family.
    replicates:
        Number of bootstrap samples.  The registered experiment uses 10,000.
    random_seed:
        Seed for sampling date multiplicities.  One shared matrix of
        multiplicities is applied to all model seeds and both methods.
    nominal:
        Central interval nominal coverage, fixed at 0.90 for the experiment.
    ci:
        Percentile confidence interval probabilities.
    expected_unique_days:
        Protocol guard; defaults to the 50 unique test dates.  Pass ``None``
        only for a deliberately different dataset or unit test.

    Returns
    -------
    dict
        Point estimates, percentile intervals, and bootstrap sign-support
        fractions.  All ``*_improvement`` metrics are baseline minus method,
        so positive is favorable.  ``interval_width_90_change`` is method
        minus baseline, so positive denotes widening; its support fraction is
        instead the fraction supporting non-widening (change <= 0).

    Notes
    -----
    ``bootstrap_support_fraction`` is a resampling sign fraction, not a
    Bayesian posterior probability and not a p-value.
    """

    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    if (
        len(ci) != 2
        or not 0.0 <= float(ci[0]) < float(ci[1]) <= 1.0
    ):
        raise ValueError("ci must contain two increasing probabilities in [0, 1]")
    ci = (float(ci[0]), float(ci[1]))

    (
        baseline,
        method,
        truth,
        date_codes,
        unique_dates,
        checked_assignments,
    ) = _validate_inputs(
        baseline_scenarios_by_seed,
        method_scenarios_by_seed,
        observations,
        day,
        assignments,
        nominal,
        expected_unique_days,
    )
    number_of_days = len(unique_dates)
    cases, hours = truth.shape
    cells_by_day = np.bincount(
        date_codes,
        weights=np.full(cases, hours, dtype=np.float64),
        minlength=number_of_days,
    ).astype(np.float64, copy=False)

    prepared_baseline = _prepare_method(
        baseline, truth, date_codes, number_of_days, float(nominal)
    )
    prepared_method = _prepare_method(
        method, truth, date_codes, number_of_days, float(nominal)
    )
    (
        count_by_day_group,
        family_slices,
        family_names,
        group_counts,
        joint_indices,
    ) = _group_count_sufficient_statistics(
        checked_assignments, date_codes, hours, number_of_days
    )
    baseline_success = _success_sufficient_statistics(
        prepared_baseline.covered_cells,
        joint_indices,
        group_counts,
        number_of_days,
    )
    method_success = _success_sufficient_statistics(
        prepared_method.covered_cells,
        joint_indices,
        group_counts,
        number_of_days,
    )

    # A multinomial row is exactly the multiplicity vector obtained by drawing
    # D dates with replacement.  Reusing this single matrix enforces pairing
    # across methods and all three model seeds.
    rng = np.random.default_rng(int(random_seed))
    probabilities = np.full(number_of_days, 1.0 / number_of_days)
    bootstrap_weights = rng.multinomial(
        number_of_days, probabilities, size=int(replicates)
    ).astype(np.float64, copy=False)
    full_sample_weights = np.ones((1, number_of_days), dtype=np.float64)

    bootstrap_metrics = _global_replicate_metrics(
        bootstrap_weights,
        cells_by_day,
        prepared_baseline,
        prepared_method,
        float(nominal),
    )
    bootstrap_metrics.update(
        _conditional_replicate_metrics(
            bootstrap_weights,
            count_by_day_group,
            baseline_success,
            method_success,
            family_slices,
            float(nominal),
        )
    )
    point_metrics = _global_replicate_metrics(
        full_sample_weights,
        cells_by_day,
        prepared_baseline,
        prepared_method,
        float(nominal),
    )
    point_metrics.update(
        _conditional_replicate_metrics(
            full_sample_weights,
            count_by_day_group,
            baseline_success,
            method_success,
            family_slices,
            float(nominal),
        )
    )

    directions = {
        "crps_improvement": "baseline minus method; positive is better",
        "winkler_90_improvement": "baseline minus method; positive is better",
        "coverage_absolute_error_90_improvement": (
            "absolute baseline coverage error minus absolute method coverage error; "
            "positive is better"
        ),
        "interval_width_90_change": (
            "method minus baseline; positive means wider method intervals"
        ),
        "conditional_family_equal_ACE_90_improvement": (
            "baseline family-equal ACE minus method family-equal ACE; positive is better"
        ),
        "conditional_worst_undercoverage_90_improvement": (
            "baseline WUCE minus method WUCE; positive is better"
        ),
    }
    metrics: dict[str, dict[str, float | str]] = {}
    for name, bootstrap_values in bootstrap_metrics.items():
        metrics[name] = _metric_record(
            float(point_metrics[name][0]),
            bootstrap_values,
            ci,
            direction=directions[name],
            support_nonpositive=name == "interval_width_90_change",
        )

    return {
        "protocol": {
            "cluster_unit": "unique calendar day",
            "unique_calendar_days": number_of_days,
            "cases": cases,
            "hours": hours,
            "model_seeds": EXPECTED_MODEL_SEEDS,
            "replicates": int(replicates),
            "random_seed": int(random_seed),
            "nominal_coverage": float(nominal),
            "percentile_interval": [ci[0], ci[1]],
            "shared_date_resamples_across_methods_and_seeds": True,
            "seed_aggregation": (
                "within-seed paired difference, then arithmetic mean across three seeds"
            ),
            "conditional_aggregation": (
                "resample date/group successes and counts, recompute group coverage, "
                "average groups within family, then average families equally"
            ),
            "conditional_families": list(family_names),
            "conditional_groups_per_family": {
                family: groups for family, groups in zip(family_names, group_counts)
            },
            "support_fraction_warning": (
                "bootstrap sign support only; not a posterior probability or p-value"
            ),
        },
        "metrics": metrics,
    }


def paired_day_cluster_bootstrap(
    baseline_scenarios_by_seed: Sequence[np.ndarray],
    method_scenarios_by_seed: Sequence[np.ndarray],
    observations: np.ndarray,
    day: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    **kwargs: object,
) -> dict[str, object]:
    """Concise alias for :func:`paired_calendar_day_bootstrap`."""

    return paired_calendar_day_bootstrap(
        baseline_scenarios_by_seed,
        method_scenarios_by_seed,
        observations,
        day,
        assignments,
        **kwargs,
    )


def _self_test() -> None:
    """Small deterministic smoke test; run with ``python -m rahc.bootstrap``."""

    days = np.repeat(np.arange(50), 2)
    cases = len(days)
    hours = 4
    members = 21
    truth = np.broadcast_to(
        0.55 + 0.02 * np.sin(np.arange(hours))[None, :], (cases, hours)
    ).copy()
    narrow_offsets = np.linspace(-0.01, 0.01, members)[None, :, None]
    useful_offsets = np.linspace(-0.10, 0.10, members)[None, :, None]
    baseline = []
    method = []
    for seed in range(3):
        baseline.append(
            np.broadcast_to(0.30 + 0.001 * seed + narrow_offsets, (cases, members, hours)).copy()
        )
        method.append(truth[:, None, :] + useful_offsets + 0.0005 * seed)
    zone = np.broadcast_to((np.arange(cases) % 10 + 1)[:, None], (cases, hours))
    hour = np.broadcast_to(np.arange(1, hours + 1)[None, :], (cases, hours))
    regime = ((zone + hour) % 5) + 1
    result = paired_calendar_day_bootstrap(
        baseline,
        method,
        truth,
        days,
        {"zone": zone, "hour": hour, "regime": regime},
        replicates=200,
        random_seed=7,
    )
    metrics = result["metrics"]
    for name in (
        "crps_improvement",
        "winkler_90_improvement",
        "coverage_absolute_error_90_improvement",
        "conditional_family_equal_ACE_90_improvement",
        "conditional_worst_undercoverage_90_improvement",
    ):
        assert metrics[name]["point_estimate"] > 0.0, name
        assert metrics[name]["bootstrap_support_fraction"] > 0.95, name
    assert metrics["interval_width_90_change"]["point_estimate"] > 0.0
    assert result["protocol"]["unique_calendar_days"] == 50
    print("rahc.bootstrap self-test passed")


__all__ = [
    "DEFAULT_NOMINAL",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_REPLICATES",
    "paired_calendar_day_bootstrap",
    "paired_day_cluster_bootstrap",
]


if __name__ == "__main__":
    _self_test()
