"""Frozen regime grouping and conditional interval metrics for RAHC.

The grouping protocol is fitted only from *raw validation forecasts*.  When
several model seeds are supplied, the ensemble mean and ensemble spread are
computed for each seed and then averaged across seeds.  This makes group
membership common to every calibration method evaluated on the same raw
forecasts and prevents method-dependent regrouping.

All public functions use NumPy arrays and return pandas objects or plain
dictionaries; observations are never accepted by :meth:`GroupingProtocol.fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


N_ZONES = 10
N_QUINTILES = 5
QUINTILE_PROBABILITIES = np.linspace(0.0, 1.0, N_QUINTILES + 1)
GROUP_FAMILIES = (
    "zone",
    "hour",
    "wind_quintile",
    "spread_quintile",
    "zone_x_spread",
)


def _validate_zones(zone: np.ndarray, cases: int) -> np.ndarray:
    values = np.asarray(zone)
    if values.shape != (cases,):
        raise ValueError(f"zone must have shape ({cases},)")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("zone must contain integer zone identifiers")
    values = values.astype(np.int64, copy=False)
    if np.any((values < 1) | (values > N_ZONES)):
        raise ValueError("zone identifiers must be in [1, 10]")
    return values


def _forecast_features(raw_scenarios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cross-seed mean ensemble location and spread, both ``[N, H]``.

    ``raw_scenarios`` may be ``[N, M, H]`` for one seed or
    ``[S, N, M, H]`` for aligned forecasts from multiple seeds.  Spread is
    computed within each seed before averaging, so between-seed shifts do not
    artificially inflate the uncertainty regime feature.
    """

    values = np.asarray(raw_scenarios)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4:
        raise ValueError(
            "raw_scenarios must have shape [cases, members, hours] or "
            "[seeds, cases, members, hours]"
        )
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 2 or values.shape[3] < 1:
        raise ValueError("raw_scenarios dimensions must be non-empty and include >=2 members")
    if not np.isfinite(values).all():
        raise ValueError("raw_scenarios contains non-finite values")
    values = values.astype(np.float64, copy=False)
    per_seed_mean = values.mean(axis=2)
    per_seed_spread = values.std(axis=2, ddof=0)
    return per_seed_mean.mean(axis=0), per_seed_spread.mean(axis=0)


def _strict_quantile_edges(values: np.ndarray) -> np.ndarray:
    """Compute six quintile edges and deterministically repair duplicate cuts.

    Clipped or discrete forecasts can yield repeated empirical quantiles.  The
    protocol retains the required six-edge representation and moves each
    non-increasing edge to the next representable float.  This preserves a
    deterministic five-bin assignment without using test-set information.
    """

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("quintile fitting values must be finite and non-empty")
    edges = np.quantile(sample, QUINTILE_PROBABILITIES).astype(np.float64, copy=False)
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("could not construct strictly increasing quintile edges")
    return edges


@dataclass(frozen=True)
class GroupingProtocol:
    """Validation-fitted, zone-conditional regime grouping protocol.

    Parameters
    ----------
    mean_edges:
        Frozen ensemble-mean quintile edges with shape ``[10, 6]``.
    spread_edges:
        Frozen ensemble-spread quintile edges with shape ``[10, 6]``.

    Notes
    -----
    Use the same aligned raw-forecast array in :meth:`assign` for every method
    being compared.  In particular, do not derive group membership from a
    method's calibrated interval widths.
    """

    mean_edges: np.ndarray
    spread_edges: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_edges, dtype=np.float64)
        spread = np.asarray(self.spread_edges, dtype=np.float64)
        expected = (N_ZONES, N_QUINTILES + 1)
        if mean.shape != expected or spread.shape != expected:
            raise ValueError(f"edge arrays must both have shape {expected}")
        if not np.isfinite(mean).all() or not np.isfinite(spread).all():
            raise ValueError("edge arrays must be finite")
        if np.any(np.diff(mean, axis=1) <= 0.0) or np.any(np.diff(spread, axis=1) <= 0.0):
            raise ValueError("edge arrays must be strictly increasing within each zone")
        object.__setattr__(self, "mean_edges", mean.copy())
        object.__setattr__(self, "spread_edges", spread.copy())

    @classmethod
    def fit(cls, validation_raw_scenarios: np.ndarray, zone: np.ndarray) -> "GroupingProtocol":
        """Fit zone-wise quintile cuts from raw validation forecasts only.

        With input shape ``[seeds, cases, members, hours]``, seed-specific
        ensemble mean/spread features are averaged before the zone-wise cuts
        are fitted.  Every one of the ten zones must occur in validation data.
        """

        mean_feature, spread_feature = _forecast_features(validation_raw_scenarios)
        zones = _validate_zones(zone, len(mean_feature))
        mean_edges = np.empty((N_ZONES, N_QUINTILES + 1), dtype=np.float64)
        spread_edges = np.empty_like(mean_edges)
        for zone_id in range(1, N_ZONES + 1):
            selected = zones == zone_id
            if not selected.any():
                raise ValueError(f"validation data does not contain zone {zone_id}")
            mean_edges[zone_id - 1] = _strict_quantile_edges(mean_feature[selected])
            spread_edges[zone_id - 1] = _strict_quantile_edges(spread_feature[selected])
        return cls(mean_edges=mean_edges, spread_edges=spread_edges)

    @staticmethod
    def _zone_bins(values: np.ndarray, zones: np.ndarray, edges: np.ndarray) -> np.ndarray:
        bins = np.empty(values.shape, dtype=np.int8)
        for zone_id in range(1, N_ZONES + 1):
            selected = zones == zone_id
            # Internal frozen cuts define integer quintiles 1,...,5.  Exact
            # boundary values enter the higher bin, matching np.digitize's
            # conventional left-closed intervals.
            bins[selected] = np.digitize(
                values[selected], edges[zone_id - 1, 1:-1], right=False
            ).astype(np.int8) + 1
        return bins

    def assign(self, raw_scenarios: np.ndarray, zone: np.ndarray) -> dict[str, np.ndarray]:
        """Assign all registered group families to forecast cells.

        Returns a dictionary of integer arrays with shape ``[cases, hours]``.
        The ``zone_x_spread`` code is ``(zone - 1) * 5 + spread_quintile`` and
        therefore ranges from 1 to 50.
        """

        mean_feature, spread_feature = _forecast_features(raw_scenarios)
        zones = _validate_zones(zone, len(mean_feature))
        hours = mean_feature.shape[1]
        zone_cells = np.broadcast_to(zones[:, None], (len(zones), hours)).copy()
        hour_cells = np.broadcast_to(
            np.arange(1, hours + 1, dtype=np.int16)[None, :], (len(zones), hours)
        ).copy()
        wind = self._zone_bins(mean_feature, zones, self.mean_edges)
        spread = self._zone_bins(spread_feature, zones, self.spread_edges)
        return {
            "zone": zone_cells,
            "hour": hour_cells,
            "wind_quintile": wind,
            "spread_quintile": spread,
            "zone_x_spread": ((zone_cells - 1) * N_QUINTILES + spread).astype(np.int16),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable leakage-auditable representation."""

        return {
            "mean_edges": self.mean_edges.tolist(),
            "spread_edges": self.spread_edges.tolist(),
            "edge_shape": [N_ZONES, N_QUINTILES + 1],
            "feature_definition": (
                "per-seed raw ensemble mean/std, then arithmetic mean across aligned seeds"
            ),
            "fit_split": "validation raw forecasts only",
            "zone_conditional": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GroupingProtocol":
        """Restore a protocol produced by :meth:`to_dict`."""

        return cls(
            mean_edges=np.asarray(payload["mean_edges"], dtype=np.float64),
            spread_edges=np.asarray(payload["spread_edges"], dtype=np.float64),
        )


def _family_labels(hours: int) -> dict[str, list[tuple[int, str]]]:
    return {
        "zone": [(index, f"zone_{index}") for index in range(1, N_ZONES + 1)],
        "hour": [(index, f"hour_{index:02d}") for index in range(1, hours + 1)],
        "wind_quintile": [
            (index, f"wind_q{index}") for index in range(1, N_QUINTILES + 1)
        ],
        "spread_quintile": [
            (index, f"spread_q{index}") for index in range(1, N_QUINTILES + 1)
        ],
        "zone_x_spread": [
            (
                (zone_id - 1) * N_QUINTILES + quintile,
                f"zone_{zone_id}_spread_q{quintile}",
            )
            for zone_id in range(1, N_ZONES + 1)
            for quintile in range(1, N_QUINTILES + 1)
        ],
    }


def group_interval_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    zone: np.ndarray,
    protocol: GroupingProtocol,
    *,
    grouping_raw_scenarios: np.ndarray,
    nominal: float = 0.90,
) -> pd.DataFrame:
    """Compute conditional central-interval diagnostics for registered groups.

    Parameters
    ----------
    scenarios:
        Scenarios for the method being scored, shape ``[N, M, H]``.
    observations:
        Realizations, shape ``[N, H]``.
    zone:
        Zone identifier for each case, shape ``[N]`` with values 1--10.
    protocol:
        Protocol fitted on validation raw forecasts.
    grouping_raw_scenarios:
        Common raw forecasts used only to assign regimes.  This can be one
        seed ``[N, M, H]`` or aligned seeds ``[S, N, M, H]``.  Pass the same
        value when comparing different calibrated methods.
    nominal:
        Central interval probability.  The registered experiment uses 0.90.

    Returns
    -------
    pandas.DataFrame
        One row for every expected group, including empty groups.  Empty
        groups have ``n=0`` and NaN-valued metrics so distribution shift is
        visible rather than silently dropping the group.
    """

    forecast = np.asarray(scenarios)
    truth = np.asarray(observations)
    if forecast.ndim != 3:
        raise ValueError("scenarios must have shape [cases, members, hours]")
    cases, members, hours = forecast.shape
    if members < 2:
        raise ValueError("scenarios must contain at least two ensemble members")
    if truth.shape != (cases, hours):
        raise ValueError(f"observations must have shape ({cases}, {hours})")
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal must be strictly between zero and one")
    if not np.isfinite(forecast).all() or not np.isfinite(truth).all():
        raise ValueError("scenarios and observations must be finite")
    zones = _validate_zones(zone, cases)
    assignments = protocol.assign(grouping_raw_scenarios, zones)
    if any(values.shape != (cases, hours) for values in assignments.values()):
        raise ValueError("grouping raw forecasts do not align with scored scenarios")

    alpha = 1.0 - float(nominal)
    lower = np.quantile(forecast, alpha / 2.0, axis=1)
    upper = np.quantile(forecast, 1.0 - alpha / 2.0, axis=1)
    covered = (truth >= lower) & (truth <= upper)
    width = upper - lower
    winkler = width.copy()
    below = truth < lower
    above = truth > upper
    winkler[below] += (2.0 / alpha) * (lower[below] - truth[below])
    winkler[above] += (2.0 / alpha) * (truth[above] - upper[above])

    rows: list[dict[str, float | int | str]] = []
    for family, labels in _family_labels(hours).items():
        codes = assignments[family]
        for code, label in labels:
            mask = codes == code
            count = int(mask.sum())
            if count:
                coverage = float(covered[mask].mean())
                ace = abs(coverage - nominal)
                undercoverage = max(0.0, nominal - coverage)
                overcoverage = max(0.0, coverage - nominal)
                mean_width = float(width[mask].mean())
                mean_winkler = float(winkler[mask].mean())
            else:
                coverage = ace = undercoverage = overcoverage = np.nan
                mean_width = mean_winkler = np.nan
            rows.append(
                {
                    "family": family,
                    "group": label,
                    "group_code": int(code),
                    "nominal": float(nominal),
                    "coverage_90": coverage,
                    "ACE_90": ace,
                    "undercoverage_90": undercoverage,
                    "overcoverage_90": overcoverage,
                    "interval_width_90": mean_width,
                    "winkler_score_90": mean_winkler,
                    "n": count,
                }
            )
    return pd.DataFrame(rows)


def summarize_group_metrics(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Summarize groups within families and then weight families equally.

    Macro quantities are first averaged over the non-empty groups in each
    family and then averaged over families, so the 50 ``zone_x_spread`` cells
    cannot dominate a five-group family.  Worst-case quantities are extrema
    over all non-empty registered groups.

    Returns
    -------
    family_summary, overall_summary
        The first object contains one row per registered family.  The second
        is a plain dictionary with equally weighted macro metrics plus WACE,
        WUCE, maximum overcoverage, and minimum group coverage.
    """

    required = {
        "family",
        "coverage_90",
        "ACE_90",
        "undercoverage_90",
        "overcoverage_90",
        "interval_width_90",
        "winkler_score_90",
        "n",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics frame is missing columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("metrics frame is empty")

    records: list[dict[str, float | int | str]] = []
    for family in GROUP_FAMILIES:
        group = metrics.loc[metrics["family"] == family]
        if group.empty:
            raise ValueError(f"metrics frame does not contain family {family!r}")
        nonempty = group.loc[group["n"] > 0]
        if nonempty.empty:
            raise ValueError(f"family {family!r} has no non-empty groups")
        records.append(
            {
                "family": family,
                "mean_coverage_90": float(nonempty["coverage_90"].mean()),
                "mean_ACE_90": float(nonempty["ACE_90"].mean()),
                "mean_undercoverage_90": float(nonempty["undercoverage_90"].mean()),
                "mean_overcoverage_90": float(nonempty["overcoverage_90"].mean()),
                "mean_interval_width_90": float(nonempty["interval_width_90"].mean()),
                "mean_winkler_score_90": float(nonempty["winkler_score_90"].mean()),
                "WACE_90": float(nonempty["ACE_90"].max()),
                "WUCE_90": float(nonempty["undercoverage_90"].max()),
                "max_overcoverage_90": float(nonempty["overcoverage_90"].max()),
                "min_group_coverage_90": float(nonempty["coverage_90"].min()),
                "groups": int(len(group)),
                "nonempty_groups": int(len(nonempty)),
                "empty_groups": int(len(group) - len(nonempty)),
                "samples_across_groups": int(nonempty["n"].sum()),
            }
        )
    family_summary = pd.DataFrame(records)
    nonempty_all = metrics.loc[metrics["n"] > 0]
    overall: dict[str, float | int] = {
        "family_equal_mean_coverage_90": float(family_summary["mean_coverage_90"].mean()),
        "family_equal_ACE_90": float(family_summary["mean_ACE_90"].mean()),
        "family_equal_undercoverage_90": float(
            family_summary["mean_undercoverage_90"].mean()
        ),
        "family_equal_overcoverage_90": float(
            family_summary["mean_overcoverage_90"].mean()
        ),
        "family_equal_interval_width_90": float(
            family_summary["mean_interval_width_90"].mean()
        ),
        "family_equal_winkler_score_90": float(
            family_summary["mean_winkler_score_90"].mean()
        ),
        "WACE_90": float(nonempty_all["ACE_90"].max()),
        "WUCE_90": float(nonempty_all["undercoverage_90"].max()),
        "max_overcoverage_90": float(nonempty_all["overcoverage_90"].max()),
        "min_group_coverage_90": float(nonempty_all["coverage_90"].min()),
        "families": int(len(family_summary)),
        "groups": int(len(metrics)),
        "nonempty_groups": int(len(nonempty_all)),
        "empty_groups": int(len(metrics) - len(nonempty_all)),
    }
    return family_summary, overall


__all__ = [
    "GROUP_FAMILIES",
    "GroupingProtocol",
    "group_interval_metrics",
    "summarize_group_metrics",
]
