"""Paired calendar-day bootstrap selection for CAA-RAHC.

The selector implements the safety contract of the CAA experiment rather
than a weighted objective.  A candidate first has to be non-inferior to A0
under paired date-cluster resampling.  Conditional calibration error is used
only inside that feasible set.  A separately labelled ``unconstrained``
candidate is audited and selected on a separate A5 path; it can never leak
into the constrained A4 decision.

Inputs are daily mean scores.  Arrays have shape ``[model_replicate, day]``
or ``[day]``.  The same bootstrap date multiplicities are applied to every
candidate, metric, and model replicate.  Model replicates (the three training
seeds in the registered experiment) are robustness replicates sharing the
same observations, not independent data clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


METRIC_KEYS = (
    "CRPS",
    "MAE",
    "VS",
    "ramp_CRPS",
    "winkler_90",
    "width_90",
)
RELATIVE_ONE_PERCENT_KEYS = ("MAE", "VS", "ramp_CRPS", "winkler_90")
SELECTION_POLICIES = ("baseline", "constrained", "unconstrained")


@dataclass(frozen=True)
class CandidateRecord:
    """Daily sufficient statistics and deterministic tie-break metadata.

    Parameters
    ----------
    name:
        Unique configuration identifier.  The registered primary baseline is
        named ``A0``.
    metrics:
        Exactly the six :data:`METRIC_KEYS`.  Every value is a finite array
        with shape ``[days]`` or ``[model_replicates, days]``.
    conditional_ace90:
        A scalar, a vector of per-replicate values, or an array whose first
        dimension indexes model replicates.  Remaining dimensions (for
        example conditional groups) are averaged within replicate before the
        cross-replicate mean is formed.
    gate_maximum, width_cap, shrinkage:
        Frozen hyperparameters used only for stable tie-breaking.  Lower
        Winkler, lower gate, lower width cap, and then stronger (larger)
        shrinkage are preferred.
    selection_policy:
        ``constrained`` for the A4 path, ``unconstrained`` for the A5
        ablation, or ``baseline`` for A0.  Unconstrained records are never
        eligible for the constrained selection.
    metadata:
        Optional JSON-like experiment metadata.  It is normalized to a JSON
        serializable representation in the returned audit.
    """

    name: str
    metrics: Mapping[str, Any]
    conditional_ace90: Any
    gate_maximum: float = 0.0
    width_cap: float = 0.0
    shrinkage: float = 0.0
    selection_policy: str = "constrained"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _json_value(value: Any) -> Any:
    """Recursively convert NumPy values and common containers to JSON data."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_record(value: CandidateRecord | Mapping[str, Any]) -> CandidateRecord:
    if isinstance(value, CandidateRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("candidates must contain CandidateRecord or mapping values")
    return CandidateRecord(**dict(value))


def _metric_array(value: Any, *, name: str, metric: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 2:
        raise ValueError(
            f"{name}.{metric} must have shape [days] or [replicates,days] "
            "with at least two days"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name}.{metric} contains non-finite values")
    if np.any(result < 0.0):
        raise ValueError(f"{name}.{metric} must be non-negative")
    return result


def _conditional_summary(value: Any, model_replicates: int) -> dict[str, Any]:
    """Normalize scalar/per-seed/per-seed-by-group conditional ACE values."""

    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("conditional_ace90 must be finite, non-negative, and non-empty")
    if array.ndim == 0:
        per_replicate = array.reshape(1)
        interpretation = "single aggregate scalar"
    elif array.shape[0] == model_replicates:
        per_replicate = array.reshape(model_replicates, -1).mean(axis=1)
        interpretation = "first dimension is model replicate; remainder averaged within replicate"
    else:
        per_replicate = np.asarray([array.mean()], dtype=np.float64)
        interpretation = "aggregate array; all entries averaged"
    return {
        "mean": float(per_replicate.mean()),
        "per_model_replicate_mean": [float(item) for item in per_replicate],
        "input_shape": list(array.shape),
        "interpretation": interpretation,
    }


def _quantile_summary(values: np.ndarray, upper_probability: float) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "upper": float(np.quantile(values, upper_probability)),
        "maximum": float(np.max(values)),
    }


def _constraint(value: Any, limit: float, passed: bool, definition: str) -> dict[str, Any]:
    return {
        "value": _json_value(value),
        "limit": float(limit),
        "passed": bool(passed),
        "definition": definition,
    }


def _selection_key(audit: Mapping[str, Any]) -> tuple[Any, ...]:
    """Registered deterministic ordering inside a selection path."""

    return (
        float(audit["conditional_ace90"]["mean"]),
        float(audit["point_metrics"]["winkler_90"]),
        float(audit["tie_break_metadata"]["gate_maximum"]),
        float(audit["tie_break_metadata"]["width_cap"]),
        -float(audit["tie_break_metadata"]["shrinkage"]),
        str(audit["name"]),
    )


def select_candidates(
    candidates: Sequence[CandidateRecord | Mapping[str, Any]],
    *,
    baseline_name: str = "A0",
    day_ids: Sequence[Any] | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260719,
    upper_probability: float = 0.95,
    crps_relative_margin: float = 0.005,
    secondary_relative_margin: float = 0.01,
    width_absolute_margin: float = 0.05,
) -> dict[str, Any]:
    """Select constrained A4 and unconstrained A5 configurations.

    The constrained candidate is feasible only if all of the following hold:

    * the paired date-bootstrap upper quantile of relative CRPS degradation is
      at most 0.5%;
    * every model-replicate full-sample CRPS degradation is at most 0.5%;
    * the bootstrap upper quantile of relative degradation in MAE, VS,
      ramp-CRPS, and Winkler-90 is at most 1%; and
    * both the point estimate and bootstrap upper quantile of absolute
      Width-90 increase are at most 0.05.

    A0 is assigned exact zero differences and is always feasible.  Therefore
    the constrained path always has a solution and returns A0 exactly with
    ``fallback=true`` when no constrained CAA candidate passes.  The A5 path
    ignores feasibility but considers only records explicitly labelled
    ``unconstrained`` (plus A0), keeping the ablation separate from A4.
    """

    if not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not 0.5 < float(upper_probability) < 1.0:
        raise ValueError("upper_probability must be in (0.5,1)")
    for name, margin in (
        ("crps_relative_margin", crps_relative_margin),
        ("secondary_relative_margin", secondary_relative_margin),
        ("width_absolute_margin", width_absolute_margin),
    ):
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    records = [_as_record(item) for item in candidates]
    if not records:
        raise ValueError("at least the A0 baseline candidate is required")
    names = [record.name for record in records]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    if baseline_name not in names:
        raise ValueError(f"baseline candidate {baseline_name!r} is missing")

    normalized: dict[str, dict[str, np.ndarray]] = {}
    conditional: dict[str, dict[str, Any]] = {}
    expected_shape: tuple[int, int] | None = None
    for record in records:
        if record.selection_policy not in SELECTION_POLICIES:
            raise ValueError(
                f"{record.name}.selection_policy must be one of {SELECTION_POLICIES}"
            )
        if set(record.metrics) != set(METRIC_KEYS):
            raise ValueError(
                f"{record.name}.metrics must contain exactly {list(METRIC_KEYS)}"
            )
        for field_name, value in (
            ("gate_maximum", record.gate_maximum),
            ("width_cap", record.width_cap),
            ("shrinkage", record.shrinkage),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{record.name}.{field_name} must be finite and non-negative")
        metric_values = {
            metric: _metric_array(record.metrics[metric], name=record.name, metric=metric)
            for metric in METRIC_KEYS
        }
        shape = metric_values["CRPS"].shape
        if any(value.shape != shape for value in metric_values.values()):
            raise ValueError(f"all metrics for {record.name} must share one shape")
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all candidates must have the same [replicate,day] shape")
        normalized[record.name] = metric_values
        conditional[record.name] = _conditional_summary(record.conditional_ace90, shape[0])

    assert expected_shape is not None
    model_replicates, number_of_days = expected_shape
    if day_ids is None:
        dates = [str(index) for index in range(number_of_days)]
    else:
        if len(day_ids) != number_of_days:
            raise ValueError("day_ids length does not match the metric day dimension")
        dates = [str(item) for item in day_ids]
        if len(set(dates)) != len(dates):
            raise ValueError("day_ids must identify unique calendar dates")

    baseline = normalized[baseline_name]
    for metric in METRIC_KEYS[:-1]:
        per_seed_baseline = baseline[metric].mean(axis=1)
        if np.any(per_seed_baseline <= 0.0):
            raise ValueError(f"baseline {metric} means must be strictly positive")

    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_weights = rng.multinomial(
        number_of_days,
        np.full(number_of_days, 1.0 / number_of_days),
        size=int(bootstrap_replicates),
    ).astype(np.float64, copy=False)
    bootstrap_denominator = float(number_of_days)

    # Compute the baseline bootstrap means once.  Shape is [B,S].
    baseline_bootstrap = {
        metric: (bootstrap_weights @ baseline[metric].T) / bootstrap_denominator
        for metric in METRIC_KEYS
    }

    audits: list[dict[str, Any]] = []
    record_by_name = {record.name: record for record in records}
    for candidate_name in sorted(names):
        record = record_by_name[candidate_name]
        values = normalized[candidate_name]
        is_baseline = candidate_name == baseline_name
        point_metrics = {
            metric: float(values[metric].mean(axis=1).mean()) for metric in METRIC_KEYS
        }

        bootstrap_summaries: dict[str, dict[str, Any]] = {}
        relative_samples: dict[str, np.ndarray] = {}
        if is_baseline:
            for metric in METRIC_KEYS[:-1]:
                relative_samples[metric] = np.zeros(int(bootstrap_replicates), dtype=np.float64)
                bootstrap_summaries[metric] = {
                    "scale": "relative candidate-minus-A0 degradation",
                    **_quantile_summary(relative_samples[metric], float(upper_probability)),
                }
            width_samples = np.zeros(int(bootstrap_replicates), dtype=np.float64)
            per_seed_crps = np.zeros(model_replicates, dtype=np.float64)
        else:
            candidate_bootstrap = {
                metric: (bootstrap_weights @ values[metric].T) / bootstrap_denominator
                for metric in METRIC_KEYS
            }
            for metric in METRIC_KEYS[:-1]:
                seed_relative = (
                    candidate_bootstrap[metric] - baseline_bootstrap[metric]
                ) / baseline_bootstrap[metric]
                relative_samples[metric] = seed_relative.mean(axis=1)
                bootstrap_summaries[metric] = {
                    "scale": "relative candidate-minus-A0 degradation",
                    **_quantile_summary(relative_samples[metric], float(upper_probability)),
                }
            width_samples = (
                candidate_bootstrap["width_90"] - baseline_bootstrap["width_90"]
            ).mean(axis=1)
            per_seed_crps = (
                values["CRPS"].mean(axis=1) - baseline["CRPS"].mean(axis=1)
            ) / baseline["CRPS"].mean(axis=1)
        width_point = float(
            (values["width_90"].mean(axis=1) - baseline["width_90"].mean(axis=1)).mean()
        ) if not is_baseline else 0.0
        bootstrap_summaries["width_90"] = {
            "scale": "absolute candidate-minus-A0 change",
            **_quantile_summary(width_samples, float(upper_probability)),
        }

        constraints: dict[str, dict[str, Any]] = {}
        crps_upper = bootstrap_summaries["CRPS"]["upper"]
        constraints["CRPS_bootstrap_upper_relative"] = _constraint(
            crps_upper,
            float(crps_relative_margin),
            is_baseline or crps_upper <= crps_relative_margin,
            "paired date-bootstrap upper relative degradation",
        )
        constraints["CRPS_each_model_replicate_point_relative"] = _constraint(
            [float(item) for item in per_seed_crps],
            float(crps_relative_margin),
            is_baseline or bool(np.all(per_seed_crps <= crps_relative_margin)),
            "every model-replicate full-sample relative degradation",
        )
        for metric in RELATIVE_ONE_PERCENT_KEYS:
            upper = bootstrap_summaries[metric]["upper"]
            constraints[f"{metric}_bootstrap_upper_relative"] = _constraint(
                upper,
                float(secondary_relative_margin),
                is_baseline or upper <= secondary_relative_margin,
                "paired date-bootstrap upper relative degradation",
            )
        constraints["width_90_point_absolute_change"] = _constraint(
            width_point,
            float(width_absolute_margin),
            is_baseline or width_point <= width_absolute_margin,
            "full-sample absolute candidate-minus-A0 width change",
        )
        width_upper = bootstrap_summaries["width_90"]["upper"]
        constraints["width_90_bootstrap_upper_absolute_change"] = _constraint(
            width_upper,
            float(width_absolute_margin),
            is_baseline or width_upper <= width_absolute_margin,
            "paired date-bootstrap upper absolute candidate-minus-A0 width change",
        )
        all_constraints_passed = is_baseline or all(
            item["passed"] for item in constraints.values()
        )
        effective_policy = "baseline" if is_baseline else record.selection_policy
        eligible_constrained = is_baseline or (
            effective_policy == "constrained" and all_constraints_passed
        )
        eligible_unconstrained = is_baseline or effective_policy == "unconstrained"
        audits.append(
            {
                "name": candidate_name,
                "selection_policy": effective_policy,
                "is_baseline": is_baseline,
                "all_constraints_passed": bool(all_constraints_passed),
                "eligible_for_constrained_A4": bool(eligible_constrained),
                "eligible_for_unconstrained_A5": bool(eligible_unconstrained),
                "point_metrics": point_metrics,
                "conditional_ace90": conditional[candidate_name],
                "bootstrap_differences_vs_A0": bootstrap_summaries,
                "constraints": constraints,
                "tie_break_metadata": {
                    "gate_maximum": float(record.gate_maximum),
                    "width_cap": float(record.width_cap),
                    "shrinkage": float(record.shrinkage),
                    "order": (
                        "conditional_ace90, winkler_90, gate_maximum, width_cap, "
                        "stronger_shrinkage, candidate_name"
                    ),
                },
                "metadata": _json_value(record.metadata),
            }
        )

    constrained_pool = [item for item in audits if item["eligible_for_constrained_A4"]]
    # A0 is guaranteed to be in this pool, making fallback exact and total.
    selected_constrained = min(constrained_pool, key=_selection_key)
    unconstrained_pool = [item for item in audits if item["eligible_for_unconstrained_A5"]]
    selected_unconstrained = min(unconstrained_pool, key=_selection_key)

    return {
        "protocol": {
            "name": "CAA paired-calendar-day constrained selector v1",
            "baseline": baseline_name,
            "metric_keys": list(METRIC_KEYS),
            "lower_is_better": True,
            "cluster_unit": "unique calendar day",
            "unique_days": int(number_of_days),
            "day_ids": dates,
            "model_replicates": int(model_replicates),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "paired_weights_across_candidates_metrics_and_model_replicates": True,
            "upper_probability": float(upper_probability),
            "margins": {
                "CRPS_relative": float(crps_relative_margin),
                "MAE_VS_ramp_CRPS_winkler_90_relative": float(
                    secondary_relative_margin
                ),
                "width_90_absolute_change": float(width_absolute_margin),
            },
            "model_replicate_rule": (
                "CRPS full-sample relative degradation must pass separately in every "
                "model replicate; bootstrap deltas are averaged across model replicates"
            ),
            "A4_A5_separation": (
                "constrained records can enter only A4; unconstrained records can enter "
                "only A5; A0 is present in both paths"
            ),
        },
        "constrained_A4": {
            "selected": selected_constrained["name"],
            "fallback": selected_constrained["name"] == baseline_name,
            "feasible_candidates": [
                item["name"] for item in constrained_pool
            ],
            "selection_key": _json_value(_selection_key(selected_constrained)),
        },
        "unconstrained_A5": {
            "selected": selected_unconstrained["name"],
            "fallback": selected_unconstrained["name"] == baseline_name,
            "eligible_candidates": [
                item["name"] for item in unconstrained_pool
            ],
            "constraints_ignored_for_selection": True,
            "selection_key": _json_value(_selection_key(selected_unconstrained)),
        },
        "candidates": audits,
    }


__all__ = [
    "METRIC_KEYS",
    "RELATIVE_ONE_PERCENT_KEYS",
    "SELECTION_POLICIES",
    "CandidateRecord",
    "select_candidates",
]
