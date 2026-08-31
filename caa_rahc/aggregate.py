"""Combine identically configured calibration records across outer replications."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .selection import CandidateRecord, METRIC_KEYS


def aggregate_outer_records(
    records_by_outer: Mapping[int, Sequence[CandidateRecord]],
) -> list[CandidateRecord]:
    """Stack model-replicate axes while retaining the common 50 date clusters.

    The registered three outer models use the same calibration dates.  Outer
    replications are therefore additional model replicates, not independent
    date clusters.  Metrics are concatenated on axis zero and retain the same
    date axis for paired bootstrap selection.
    """

    if not records_by_outer:
        raise ValueError("at least one outer record set is required")
    grouped: dict[str, list[tuple[int, CandidateRecord]]] = defaultdict(list)
    expected_names: set[str] | None = None
    for outer, records in sorted(records_by_outer.items()):
        names = [record.name for record in records]
        if len(names) != len(set(names)):
            raise ValueError(f"outer {outer} contains duplicate candidate names")
        current = set(names)
        if expected_names is None:
            expected_names = current
        elif current != expected_names:
            missing = sorted(expected_names - current)
            extra = sorted(current - expected_names)
            raise ValueError(
                f"candidate catalog differs in outer {outer}; missing={missing}, extra={extra}"
            )
        for record in records:
            grouped[record.name].append((int(outer), record))
    result: list[CandidateRecord] = []
    for name in sorted(grouped):
        entries = grouped[name]
        reference = entries[0][1]
        for outer, record in entries[1:]:
            if (
                record.selection_policy != reference.selection_policy
                or record.gate_maximum != reference.gate_maximum
                or record.width_cap != reference.width_cap
                or record.shrinkage != reference.shrinkage
            ):
                raise ValueError(f"candidate {name} metadata differs in outer {outer}")
        metrics: dict[str, np.ndarray] = {}
        expected_days: int | None = None
        for metric in METRIC_KEYS:
            parts = []
            for outer, record in entries:
                values = np.asarray(record.metrics[metric], dtype=np.float64)
                if values.ndim == 1:
                    values = values[None, :]
                if values.ndim != 2:
                    raise ValueError(f"outer {outer} candidate {name}.{metric} is not 2D")
                if expected_days is None:
                    expected_days = values.shape[1]
                elif values.shape[1] != expected_days:
                    raise ValueError("calibration date dimensions differ across outers")
                parts.append(values)
            metrics[metric] = np.concatenate(parts, axis=0)
        conditional_parts = []
        for outer, record in entries:
            value = np.asarray(record.conditional_ace90, dtype=np.float64)
            if value.ndim == 0:
                value = value[None]
            conditional_parts.append(value.reshape(-1))
        result.append(
            CandidateRecord(
                name=name,
                metrics=metrics,
                conditional_ace90=np.concatenate(conditional_parts),
                gate_maximum=reference.gate_maximum,
                width_cap=reference.width_cap,
                shrinkage=reference.shrinkage,
                selection_policy=reference.selection_policy,
                metadata={
                    "outer_replications": [outer for outer, _ in entries],
                    "model_replicates_per_outer": [
                        int(np.asarray(record.metrics["CRPS"]).reshape(-1, metrics["CRPS"].shape[1]).shape[0])
                        for _, record in entries
                    ],
                    "outer_metadata": {
                        f"outer{outer}": _jsonable(record.metadata) for outer, record in entries
                    },
                },
            )
        )
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = ["aggregate_outer_records"]
