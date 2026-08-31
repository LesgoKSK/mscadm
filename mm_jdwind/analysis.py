from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import JointSplitData
from .metrics import evaluate_joint, per_date_joint_metrics


def legacy_archive_to_joint(
    scenarios: np.ndarray,
    observations: np.ndarray,
    day: np.ndarray,
    zone: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize legacy zone-major archives to day-major joint tensors."""

    dates = np.asarray(day).astype("datetime64[D]")
    zones = np.asarray(zone, dtype=np.int64)
    order = np.lexsort((zones, dates.astype(np.int64)))
    sorted_dates = dates[order]
    sorted_zones = zones[order]
    unique_dates = np.unique(sorted_dates)
    expected_zones = np.tile(np.arange(1, 11, dtype=np.int64), len(unique_dates))
    if not np.array_equal(sorted_zones, expected_zones):
        raise ValueError("legacy archive does not contain zones 1..10 for every date")
    if not np.array_equal(
        sorted_dates, np.repeat(unique_dates, 10)
    ):
        raise ValueError("legacy archive day ordering is incomplete")
    values = np.asarray(scenarios)[order]
    truth = np.asarray(observations)[order]
    members = values.shape[1]
    joint = values.reshape(len(unique_dates), 10, members, 24).transpose(0, 2, 1, 3)
    joint_truth = truth.reshape(len(unique_dates), 10, 24)
    return joint, joint_truth, unique_dates


def load_caa_family(
    root: str | Path, *, outer: int, seed: int, family: str
) -> dict[str, np.ndarray]:
    base = Path(root) / f"outer{outer}" / "scenarios"
    with np.load(
        base / f"{family}_seed{seed}_test_final.npz", allow_pickle=False
    ) as stored:
        scenarios, observations, dates = legacy_archive_to_joint(
            stored["scenarios"],
            stored["observations"],
            stored["day"],
            stored["zone"],
        )
    with np.load(
        base / f"analytic_atoms_seed{seed}_test_final.npz", allow_pickle=False
    ) as atoms:
        p0, _, atom_dates = legacy_archive_to_joint(
            atoms[f"{family}_pi0"][:, None, :],
            atoms["observations"],
            atoms["day"],
            atoms["zone"],
        )
        p1, _, _ = legacy_archive_to_joint(
            atoms[f"{family}_pi1"][:, None, :],
            atoms["observations"],
            atoms["day"],
            atoms["zone"],
        )
    if not np.array_equal(dates, atom_dates):
        raise ValueError("CAA scenario and atom archives have different dates")
    return {
        "scenarios": scenarios,
        "observations": observations,
        "day": dates,
        "zero_probability": p0[:, 0],
        "one_probability": p1[:, 0],
    }


def split_for_archive(
    condition: np.ndarray, observations: np.ndarray, dates: np.ndarray
) -> JointSplitData:
    return JointSplitData(
        condition=condition,
        target=observations.astype(np.float32),
        day=dates.astype("datetime64[D]"),
        zones=np.arange(1, 11, dtype=np.int64),
    )


def metric_record(
    scenarios: np.ndarray,
    split: JointSplitData,
    zero_probability: np.ndarray,
    one_probability: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    return (
        evaluate_joint(
            scenarios,
            split,
            zero_probability=zero_probability,
            one_probability=one_probability,
        ),
        per_date_joint_metrics(scenarios, split.target),
    )


def paired_date_bootstrap(
    records: Iterable[dict[str, Any]],
    *,
    metric: str,
    higher_is_better: bool = False,
    replicates: int = 5_000,
    seed: int = 20260727,
) -> dict[str, float]:
    """Average paired seed differences within a date, then bootstrap dates."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        baseline = np.asarray(record["baseline"][metric], dtype=np.float64)
        method = np.asarray(record["method"][metric], dtype=np.float64)
        dates = np.asarray(record["dates"]).astype("datetime64[D]")
        if baseline.shape != method.shape or baseline.shape != dates.shape:
            raise ValueError("paired record shapes do not align")
        difference = method - baseline if higher_is_better else baseline - method
        for date, value in zip(dates.astype(str), difference):
            grouped[date].append(float(value))
    date_differences = np.asarray(
        [np.mean(grouped[date]) for date in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(date_differences), size=(replicates, len(date_differences))
    )
    sampled = date_differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "difference_definition": (
            "method-baseline" if higher_is_better else "baseline-method"
        ),
        "dates": int(len(date_differences)),
        "point": float(date_differences.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "probability_improvement": float(np.mean(sampled > 0.0)),
        "replicates": int(replicates),
    }


def summarize_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(records[0]["metrics"])
    return {
        name: {
            "mean": float(np.mean([item["metrics"][name] for item in records])),
            "seed_outer_sd": float(
                np.std([item["metrics"][name] for item in records], ddof=1)
            ),
            "minimum": float(min(item["metrics"][name] for item in records)),
            "maximum": float(max(item["metrics"][name] for item in records)),
        }
        for name in names
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )


__all__ = [
    "legacy_archive_to_joint",
    "load_caa_family",
    "metric_record",
    "paired_date_bootstrap",
    "split_for_archive",
    "summarize_metric_records",
    "write_json",
]
