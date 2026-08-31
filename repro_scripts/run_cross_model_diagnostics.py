#!/usr/bin/env python3
"""Run leakage-controlled trajectory diagnostics on the frozen scenario archives.

This script deliberately keeps the MM/DDPM and STGF/DDPM date panels separate.
Calendar days, rather than scenario members, are the observational units used by
the paired bootstrap.  DDPM archives were generated zone by zone; consequently
their cross-zone/cross-lag results are labelled descriptive and must not be read
as a controlled comparison of diffusion and flow model classes.

The command is intended to be run from the repository root::

    MPLCONFIGDIR=/tmp/mpl_crossdiag \
      python repro_scripts/run_cross_model_diagnostics.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cross_model_diagnostics.core import (  # noqa: E402
    atom_duration_summary,
    empirical_crps,
    per_day_cross_lag_error,
    per_day_metrics,
    stratified_paired_bootstrap,
    summarize_per_day_by_regime,
    cross_lag_summary,
    validate_archive_arrays,
)
from cross_model_diagnostics.advanced import (  # noqa: E402
    atom_event_metrics,
    generated_transition_crps_decomposition,
    lag1_increment_state_decomposition,
    lagged_variogram_suite,
    ramp_crps_metrics,
)
from cross_model_diagnostics.data import (  # noqa: E402
    assign_regimes,
    date_sha256,
    fit_regime_thresholds,
    legacy_day_split,
    load_raw_joint_days,
    nwp_day_features,
    train_zero_epsilon,
)


PRIMARY_PANEL = "mm_ddpm"
SECONDARY_PANEL = "stgf_ddpm"
LOWER_IS_BETTER_METRICS = (
    "level_CRPS",
    "ramp_CRPS",
    "level_MAE",
    "ramp_MAE",
    "max_abs_ramp_CRPS",
    "max_abs_ramp_abs_bias",
    "mean_abs_ramp_ratio_abs_error",
    "coverage_90_abs_error",
    "zero_rate_abs_error",
    "any_zero_Brier",
    "zero_longest_run_MAE",
    "zero_transition_rate_abs_error",
    "cross_lag_increment_lag1_RMSE",
    "level_CRPS_clean",
    "ramp_CRPS_clean",
    "level_MAE_clean",
    "ramp_MAE_clean",
    "ramp_local_mean_CRPS",
    "ramp_local_up_CRPS",
    "ramp_local_down_CRPS",
    "ramp_fleet_mean_CRPS",
    "ramp_fleet_up_CRPS",
    "ramp_fleet_down_CRPS",
    "ramp_max_up_CRPS",
    "ramp_max_down_CRPS",
    "ramp_total_variation_CRPS",
)
REGIME_METRICS = (
    "level_CRPS",
    "ramp_CRPS",
    "max_abs_ramp_CRPS",
    "cross_lag_increment_lag1_RMSE",
    "level_CRPS_clean",
    "ramp_CRPS_clean",
    "ramp_fleet_mean_CRPS",
    "ramp_max_up_CRPS",
    "ramp_max_down_CRPS",
    "ramp_total_variation_CRPS",
)
REGIME_NAMES = (
    "calm",
    "strong",
    "speed_volatile",
    "turning",
    "spatial_heterogeneous",
)
POSITIVE_REGIME_LABELS = {
    "calm": "calm",
    "strong": "strong",
    "speed_volatile": "speed_volatile",
    "turning": "turning",
    "spatial_heterogeneous": "spatial_heterogeneous",
}
FEATURE_NAMES = (
    "mean_ws100",
    "rms_dws100",
    "weighted_turn100",
    "nwp_ramp",
    "direction_shift",
    "spatial_dispersion",
    "mean_shear100_10",
)
ATOM_EPSILON_SCHEMES = ("exact", "epsilon_0p001", "epsilon_0p01", "train_derived")
ATOM_EVENT_SCHEMES = {"exact": 0.0, "epsilon_0p01": 0.01}
ATOM_EVENT_METRICS = (
    "atom_H_CRPS",
    "atom_K_CRPS",
    "atom_L_CRPS",
    "atom_any_zero_Brier",
    "atom_entry_transition_Brier",
    "atom_exit_transition_Brier",
)


@dataclass(frozen=True)
class ModelSpec:
    panel: str
    model: str
    seeds: tuple[int, ...]
    path_template: str
    joint_scope: str
    has_explicit_states: bool = False

    def path(self, root: Path, outer: int, seed: int) -> Path:
        return root / self.path_template.format(outer=outer, seed=seed)


@dataclass
class Archive:
    path: Path
    scenarios: np.ndarray
    observations: np.ndarray
    days: np.ndarray
    states: np.ndarray | None


MODEL_SPECS = (
    ModelSpec(
        PRIMARY_PANEL,
        "MM_mass",
        (0, 1, 2),
        "outputs/mm_jdwind_confirmation_v1/outer{outer}/scenarios/flow_mass_preserving_seed{seed}_test.npz",
        "joint_10_zone_archive",
        True,
    ),
    ModelSpec(
        PRIMARY_PANEL,
        "MM_none",
        (0, 1, 2),
        "outputs/mm_jdwind_confirmation_v1/outer{outer}/scenarios/flow_none_seed{seed}_test.npz",
        "joint_10_zone_archive",
        True,
    ),
    ModelSpec(
        PRIMARY_PANEL,
        "DDPM_product",
        (0,),
        "outputs/ddpm_confirmation_v1/outer{outer}/scenarios/ddpm_seed{seed}_test.npz",
        "descriptive_only_product_of_zonewise_marginals",
    ),
    ModelSpec(
        SECONDARY_PANEL,
        "STGF",
        (0, 1, 2),
        "outputs/stgf_confirmation_v2/outer{outer}/scenarios/stgf_seed{seed}_test.npz",
        "joint_10_zone_archive",
    ),
    ModelSpec(
        SECONDARY_PANEL,
        "Time_domain",
        (0,),
        "outputs/stgf_confirmation_v2/outer{outer}/scenarios/time_domain_seed{seed}_test.npz",
        "joint_10_zone_archive",
    ),
    ModelSpec(
        SECONDARY_PANEL,
        "Time_frequency",
        (0,),
        "outputs/stgf_confirmation_v2/outer{outer}/scenarios/time_frequency_seed{seed}_test.npz",
        "joint_10_zone_archive",
    ),
    ModelSpec(
        SECONDARY_PANEL,
        "Graph_only",
        (0,),
        "outputs/stgf_confirmation_v2/outer{outer}/scenarios/graph_only_seed{seed}_test.npz",
        "joint_10_zone_archive",
    ),
    ModelSpec(
        SECONDARY_PANEL,
        "DDPM_product",
        (0,),
        "outputs/ddpm_stgf_confirmation_v2/outer{outer}/scenarios/ddpm_seed{seed}_test.npz",
        "descriptive_only_product_of_zonewise_marginals",
    ),
)


REGISTRY_PATHS = {
    PRIMARY_PANEL: Path("repro_configs/mm_jdwind_confirmation_splits.json"),
    SECONDARY_PANEL: Path("repro_configs/stgf_confirmation_splits.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[D]"))
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError(f"refusing to create an empty CSV: {path}")
    frame = pd.DataFrame([dict(record) for record in records])
    frame.to_csv(path, index=False, float_format="%.10g", quoting=csv.QUOTE_MINIMAL)


def _safe_npz_key(*parts: Any) -> str:
    text = "__".join(str(part) for part in parts)
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def _finite_mean(values: Iterable[float]) -> float:
    sample = np.asarray(list(values), dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    return float(finite.mean()) if finite.size else float("nan")


def _registry_days(registry: Mapping[str, Any], outer: int) -> np.ndarray:
    days = np.asarray(registry["outer_test_dates"][str(outer)], dtype="datetime64[D]")
    if days.shape != (50,) or len(np.unique(days)) != 50:
        raise ValueError(f"outer {outer} registry does not contain 50 unique days")
    return days


def _raw_indices(raw_days: np.ndarray, selected_days: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(raw_days, selected_days)
    if np.any(indices >= len(raw_days)) or not np.array_equal(raw_days[indices], selected_days):
        raise ValueError("archive days are not a strict subset of the raw calendar")
    return indices


def _load_archives(
    root: Path,
    raw: Mapping[str, np.ndarray],
    registries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, int, int], Archive], list[dict[str, Any]]]:
    archives: dict[tuple[str, str, int, int], Archive] = {}
    inventory: list[dict[str, Any]] = []
    canonical: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for spec in MODEL_SPECS:
        for outer in (1, 2, 3):
            expected_days = _registry_days(registries[spec.panel], outer)
            for seed in spec.seeds:
                path = spec.path(root, outer, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as payload:
                    scenarios, observations, days = validate_archive_arrays(
                        payload["scenarios"], payload["observations"], payload["day"]
                    )
                    if not np.array_equal(days, expected_days):
                        raise ValueError(f"archive day order differs from frozen registry: {path}")
                    if "zones" in payload and not np.array_equal(
                        np.asarray(payload["zones"]), np.arange(1, 11)
                    ):
                        raise ValueError(f"archive zone order differs from 1..10: {path}")
                    states = None
                    if spec.has_explicit_states:
                        if "states" not in payload:
                            raise ValueError(f"explicit MM states missing from {path}")
                        states = np.asarray(payload["states"], dtype=np.int8)
                        if states.shape != scenarios.shape or not np.isin(states, (0, 1, 2)).all():
                            raise ValueError(f"invalid explicit state tensor in {path}")

                panel_outer = (spec.panel, outer)
                if panel_outer not in canonical:
                    canonical[panel_outer] = (days.copy(), observations.copy())
                else:
                    canonical_days, canonical_observations = canonical[panel_outer]
                    if not np.array_equal(days, canonical_days):
                        raise ValueError(f"paired archive dates drifted in {path}")
                    if not np.array_equal(observations, canonical_observations):
                        raise ValueError(f"paired observations drifted in {path}")

                raw_index = _raw_indices(np.asarray(raw["day"]), days)
                if not np.array_equal(observations, np.asarray(raw["target"])[raw_index]):
                    raise ValueError(f"archive observations differ from raw loader in {path}")
                key = (spec.panel, spec.model, seed, outer)
                archives[key] = Archive(path, scenarios, observations, days, states)
                inventory.append(
                    {
                        "panel": spec.panel,
                        "model": spec.model,
                        "seed": seed,
                        "outer": outer,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha256(path),
                        "n_days": scenarios.shape[0],
                        "n_members": scenarios.shape[1],
                        "n_zones": scenarios.shape[2],
                        "n_hours": scenarios.shape[3],
                        "date_sha256": date_sha256(days),
                        "joint_scope": spec.joint_scope,
                        "explicit_states": states is not None,
                    }
                )

    for panel in (PRIMARY_PANEL, SECONDARY_PANEL):
        panel_days = [_registry_days(registries[panel], outer) for outer in (1, 2, 3)]
        joined = np.concatenate(panel_days)
        if len(np.unique(joined)) != 150:
            raise ValueError(f"{panel} outer folds overlap; pairing would be invalid")
    primary_days = np.concatenate(
        [_registry_days(registries[PRIMARY_PANEL], outer) for outer in (1, 2, 3)]
    )
    secondary_days = np.concatenate(
        [_registry_days(registries[SECONDARY_PANEL], outer) for outer in (1, 2, 3)]
    )
    if np.intersect1d(primary_days, secondary_days).size:
        raise ValueError("primary and secondary panels overlap; they must remain separate")
    return archives, inventory


def _fit_panel_references(
    raw: Mapping[str, np.ndarray], registries: Mapping[str, Mapping[str, Any]]
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
    dict[str, float],
    dict[str, Any],
]:
    """Fit one frozen, outcome-independent NWP definition per date panel.

    Each panel uses the 481 legacy-training days left after excluding all 150
    of that panel's diagnostic dates.  The secondary panel is interpreted on
    its own dates and never borrowed as a replicate of the primary panel.
    """

    raw_days = np.asarray(raw["day"]).astype("datetime64[D]")
    legacy = legacy_day_split(raw_days)
    features = nwp_day_features(dict(raw))
    thresholds_by_panel: dict[str, dict[str, Any]] = {}
    regimes_by_panel: dict[str, dict[str, np.ndarray]] = {}
    epsilon_by_panel: dict[str, float] = {}
    reference_audit: dict[str, Any] = {
        "legacy_train_days": int(len(legacy["train"])),
        "panels": {},
        "zero_epsilon_rule": "1st percentile of positive interior target values, clipped to [1e-4,0.02]",
    }
    for panel in (PRIMARY_PANEL, SECONDARY_PANEL):
        diagnostic_days = np.unique(
            np.concatenate(
                [_registry_days(registries[panel], outer) for outer in (1, 2, 3)]
            )
        )
        reference_days = np.setdiff1d(legacy["train"], diagnostic_days, assume_unique=True)
        train_mask = np.isin(raw_days, reference_days)
        if len(reference_days) != 481 or int(train_mask.sum()) != 481:
            raise ValueError(f"{panel} leakage-free reference must contain exactly 481 days")

        thresholds = fit_regime_thresholds(features, train_mask)
        thresholds.update(
            {
                "fit_scope": f"legacy train excluding all 150 {panel} diagnostic dates",
                "fit_date_sha256": date_sha256(reference_days),
                "excluded_diagnostic_dates": int(len(diagnostic_days)),
                "panel": panel,
            }
        )
        thresholds_by_panel[panel] = thresholds
        regimes_by_panel[panel] = assign_regimes(features, thresholds)

        epsilon_with_forward_fill = train_zero_epsilon(np.asarray(raw["target"]), train_mask)
        valid = train_mask[:, None, None] & ~np.asarray(raw["target_was_missing"], dtype=bool)
        target = np.asarray(raw["target"], dtype=np.float64)
        positive = target[valid & (target > 0.0) & (target < 1.0)]
        clean_epsilon = float(np.clip(np.quantile(positive, 0.01), 1e-4, 0.02))
        epsilon_by_panel[panel] = clean_epsilon
        reference_audit["panels"][panel] = {
            "diagnostic_days": int(len(diagnostic_days)),
            "reference_days": int(train_mask.sum()),
            "reference_date_sha256": date_sha256(reference_days),
            "train_zero_epsilon_clean": clean_epsilon,
            "train_zero_epsilon_including_forward_filled_cells": epsilon_with_forward_fill,
        }
    return features, thresholds_by_panel, regimes_by_panel, epsilon_by_panel, reference_audit


def _masked_clean_metrics(
    scenarios: np.ndarray,
    observations: np.ndarray,
    missing: np.ndarray,
    base: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Re-score affected days, retaining base scores on fully observed days."""

    missing = np.asarray(missing, dtype=bool)
    affected = missing.any(axis=(1, 2))
    clean_level_count = (~missing).sum(axis=(1, 2)).astype(np.int64)
    ramp_missing = missing[..., :-1] | missing[..., 1:]
    clean_ramp_count = (~ramp_missing).sum(axis=(1, 2)).astype(np.int64)
    output = {
        "level_CRPS_clean": np.asarray(base["level_CRPS"]).copy(),
        "ramp_CRPS_clean": np.asarray(base["ramp_CRPS"]).copy(),
        "level_MAE_clean": np.asarray(base["level_MAE"]).copy(),
        "ramp_MAE_clean": np.asarray(base["ramp_MAE"]).copy(),
        "clean_level_cell_count": clean_level_count,
        "clean_ramp_cell_count": clean_ramp_count,
        "missing_target_cell_count": missing.sum(axis=(1, 2)).astype(np.int64),
        "missing_affected_ramp_count": ramp_missing.sum(axis=(1, 2)).astype(np.int64),
        "has_raw_missing_target": affected.astype(np.int8),
    }
    for day_index in np.flatnonzero(affected):
        values = scenarios[day_index : day_index + 1]
        truth = observations[day_index : day_index + 1]
        level_score = empirical_crps(values, truth, member_axis=1)[0]
        level_mae = np.abs(values.mean(axis=1)[0] - truth[0])
        level_valid = ~missing[day_index]
        ramp_values = np.diff(values, axis=-1)
        ramp_truth = np.diff(truth, axis=-1)
        ramp_score = empirical_crps(ramp_values, ramp_truth, member_axis=1)[0]
        ramp_mae = np.abs(ramp_values.mean(axis=1)[0] - ramp_truth[0])
        ramp_valid = ~ramp_missing[day_index]
        output["level_CRPS_clean"][day_index] = float(level_score[level_valid].mean())
        output["ramp_CRPS_clean"][day_index] = float(ramp_score[ramp_valid].mean())
        output["level_MAE_clean"][day_index] = float(level_mae[level_valid].mean())
        output["ramp_MAE_clean"][day_index] = float(ramp_mae[ramp_valid].mean())
    return output


def _per_day_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
    raw: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    regimes_by_panel: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    metric_names: list[str] | None = None
    raw_days = np.asarray(raw["day"])
    for spec in MODEL_SPECS:
        for outer in (1, 2, 3):
            for seed in spec.seeds:
                archive = archives[(spec.panel, spec.model, seed, outer)]
                raw_index = _raw_indices(raw_days, archive.days)
                missing = np.asarray(raw["target_was_missing"])[raw_index]
                metrics = per_day_metrics(archive.scenarios, archive.observations, zero_epsilon=0.0)
                metrics.update(ramp_crps_metrics(archive.scenarios, archive.observations))
                metrics.update(
                    _masked_clean_metrics(archive.scenarios, archive.observations, missing, metrics)
                )
                metrics["cross_lag_increment_lag1_RMSE"] = per_day_cross_lag_error(
                    archive.scenarios, archive.observations, lag=1, increments=True
                )
                metrics["cross_lag_increment_lag1_RMSE_clean_days"] = np.where(
                    missing.any(axis=(1, 2)),
                    np.nan,
                    metrics["cross_lag_increment_lag1_RMSE"],
                )
                metrics["max_abs_ramp_abs_bias"] = np.abs(metrics["max_abs_ramp_bias"])
                metrics["mean_abs_ramp_ratio_abs_error"] = np.abs(
                    metrics["mean_abs_ramp_ratio"] - 1.0
                )
                metrics["coverage_90_abs_error"] = np.abs(metrics["coverage_90"] - 0.9)
                if metric_names is None:
                    metric_names = list(metrics)
                elif set(metric_names) != set(metrics):
                    raise RuntimeError("per-day metric schema changed between archives")

                for day_offset, day in enumerate(archive.days):
                    row: dict[str, Any] = {
                        "panel": spec.panel,
                        "outer": outer,
                        "model": spec.model,
                        "seed": seed,
                        "day": str(day),
                        "member_count": archive.scenarios.shape[1],
                        "joint_scope": spec.joint_scope,
                    }
                    for name, values in metrics.items():
                        row[name] = float(values[day_offset])
                    for name, values in features.items():
                        row[name] = float(values[raw_index[day_offset]])
                    for name, values in regimes_by_panel[spec.panel].items():
                        row[name] = str(values[raw_index[day_offset]])
                    records.append(row)
    if metric_names is None:
        raise RuntimeError("no per-day metrics were produced")
    return records, metric_names


def _aggregate_seed_day_records(
    seed_records: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_records:
        key = (str(row["panel"]), int(row["outer"]), str(row["model"]), str(row["day"]))
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for (panel, outer, model, day), rows in sorted(grouped.items()):
        first = rows[0]
        row: dict[str, Any] = {
            "panel": panel,
            "outer": outer,
            "model": model,
            "seed": "mean",
            "seed_count": len(rows),
            "day": day,
            "member_count_per_seed": int(first["member_count"]),
            "joint_scope": first["joint_scope"],
        }
        for name in metric_names:
            values = np.asarray([float(item[name]) for item in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[name] = float(finite.mean()) if finite.size else float("nan")
            row[f"seed_sd__{name}"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else float("nan"))
            )
        for name in FEATURE_NAMES:
            row[name] = float(first[name])
        for name in REGIME_NAMES:
            row[name] = str(first[name])
        aggregated.append(row)

    stability: list[dict[str, Any]] = []
    by_model_seed: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in seed_records:
        by_model_seed.setdefault(
            (str(row["panel"]), str(row["model"]), int(row["seed"])), []
        ).append(row)
    models = sorted({(key[0], key[1]) for key in by_model_seed})
    for panel, model in models:
        seeds = sorted(key[2] for key in by_model_seed if key[:2] == (panel, model))
        for metric in metric_names:
            seed_means = np.asarray(
                [
                    _finite_mean(
                        float(row[metric]) for row in by_model_seed[(panel, model, seed)]
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            finite_seed_means = seed_means[np.isfinite(seed_means)]
            stability.append(
                {
                    "panel": panel,
                    "model": model,
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "mean_across_seed_means": float(finite_seed_means.mean())
                    if finite_seed_means.size
                    else None,
                    "sd_across_seed_means": float(finite_seed_means.std(ddof=1))
                    if finite_seed_means.size > 1
                    else (0.0 if finite_seed_means.size == 1 else None),
                    "minimum_seed_mean": float(finite_seed_means.min())
                    if finite_seed_means.size
                    else None,
                    "maximum_seed_mean": float(finite_seed_means.max())
                    if finite_seed_means.size
                    else None,
                }
            )
    return aggregated, stability


def _paired_bootstrap_records(
    day_records: Sequence[Mapping[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["panel"]), int(row["outer"]), str(row["day"]), str(row["model"])): row
        for row in day_records
    }
    output: list[dict[str, Any]] = []
    for left_model in ("MM_mass", "MM_none"):
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for key, left in lookup.items():
            panel, outer, day, model = key
            if panel != PRIMARY_PANEL or model != left_model:
                continue
            right_key = (panel, outer, day, "DDPM_product")
            if right_key not in lookup:
                raise ValueError(f"missing primary DDPM pair for {right_key}")
            pairs.append((left, lookup[right_key]))
        if len(pairs) != 150:
            raise ValueError(f"expected 150 paired days for {left_model}, found {len(pairs)}")
        for metric_index, metric in enumerate(LOWER_IS_BETTER_METRICS):
            left_values = np.asarray([float(left[metric]) for left, _ in pairs])
            right_values = np.asarray([float(right[metric]) for _, right in pairs])
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            strata = np.asarray([int(left["outer"]) for left, _ in pairs])[finite]
            result = stratified_paired_bootstrap(
                left_values[finite] - right_values[finite],
                strata,
                repetitions=repetitions,
                seed=20260814 + 1000 * (left_model == "MM_none") + metric_index,
            )
            right_mean = float(right_values[finite].mean())
            output.append(
                {
                    "panel": PRIMARY_PANEL,
                    "left_model": left_model,
                    "right_model": "DDPM_product",
                    "metric": metric,
                    "direction": "lower_is_better",
                    "left_mean": float(left_values[finite].mean()),
                    "right_mean": right_mean,
                    **result,
                    "relative_mean_difference_pct": 100.0
                    * float(result["mean_difference"])
                    / max(abs(right_mean), 1e-12),
                    "relative_ci_low_pct": 100.0
                    * float(result["ci_low"])
                    / max(abs(right_mean), 1e-12),
                    "relative_ci_high_pct": 100.0
                    * float(result["ci_high"])
                    / max(abs(right_mean), 1e-12),
                    "unit_of_inference": "paired calendar day, stratified by outer fold",
                }
            )
    return output


def _secondary_paired_bootstrap_records(
    day_records: Sequence[Mapping[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    """Paired calendar-day contrasts within the disjoint secondary panel."""

    lookup = {
        (str(row["panel"]), int(row["outer"]), str(row["day"]), str(row["model"])): row
        for row in day_records
    }
    output: list[dict[str, Any]] = []
    left_models = ("STGF", "Time_domain", "Time_frequency", "Graph_only")
    for model_index, left_model in enumerate(left_models):
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for key, left in lookup.items():
            panel, outer, day, model = key
            if panel != SECONDARY_PANEL or model != left_model:
                continue
            right_key = (panel, outer, day, "DDPM_product")
            if right_key not in lookup:
                raise ValueError(f"missing secondary DDPM pair for {right_key}")
            pairs.append((left, lookup[right_key]))
        if len(pairs) != 150:
            raise ValueError(
                f"expected 150 secondary paired days for {left_model}, found {len(pairs)}"
            )
        for metric_index, metric in enumerate(LOWER_IS_BETTER_METRICS):
            left_values = np.asarray([float(left[metric]) for left, _ in pairs])
            right_values = np.asarray([float(right[metric]) for _, right in pairs])
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            strata = np.asarray([int(left["outer"]) for left, _ in pairs])[finite]
            result = stratified_paired_bootstrap(
                left_values[finite] - right_values[finite],
                strata,
                repetitions=repetitions,
                seed=20262814 + model_index * 1000 + metric_index,
            )
            right_mean = float(right_values[finite].mean())
            output.append(
                {
                    "panel": SECONDARY_PANEL,
                    "left_model": left_model,
                    "right_model": "DDPM_product",
                    "metric": metric,
                    "direction": "lower_is_better",
                    "left_mean": float(left_values[finite].mean()),
                    "right_mean": right_mean,
                    **result,
                    "relative_mean_difference_pct": 100.0
                    * float(result["mean_difference"])
                    / max(abs(right_mean), 1e-12),
                    "left_seed_aggregation": "per-day mean across seeds 0-2"
                    if left_model == "STGF"
                    else "single frozen seed0",
                    "right_seed_aggregation": "single frozen seed0",
                    "unit_of_inference": "paired calendar day, stratified by outer fold",
                    "panel_warning": "secondary panel is analyzed independently and is not pooled with the primary panel",
                }
            )
    return output


def _stratified_regime_interaction_bootstrap(
    paired_differences: np.ndarray,
    outer: np.ndarray,
    positive: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a regime-minus-complement contrast of paired model gaps.

    Counts are fixed within the six outer-by-membership strata.  Vectorized
    index sampling keeps the registered 5,000-repetition analysis inexpensive.
    """

    values = np.asarray(paired_differences, dtype=np.float64).reshape(-1)
    folds = np.asarray(outer, dtype=np.int64).reshape(-1)
    membership = np.asarray(positive, dtype=bool).reshape(-1)
    if not (values.shape == folds.shape == membership.shape):
        raise ValueError("interaction bootstrap arrays must have equal shape")
    if not np.isfinite(values).all():
        raise ValueError("interaction bootstrap contains non-finite paired gaps")
    if repetitions < 100:
        raise ValueError("interaction bootstrap requires at least 100 repetitions")
    n_positive = int(membership.sum())
    n_complement = int((~membership).sum())
    if n_positive == 0 or n_complement == 0:
        raise ValueError("regime and complement must both contain days")

    rng = np.random.default_rng(seed)
    positive_sum = np.zeros(repetitions, dtype=np.float64)
    complement_sum = np.zeros(repetitions, dtype=np.float64)
    stratum_counts: dict[str, int] = {}
    for fold in sorted(np.unique(folds)):
        for is_positive, destination, label in (
            (True, positive_sum, "regime"),
            (False, complement_sum, "complement"),
        ):
            stratum = values[(folds == fold) & (membership == is_positive)]
            if stratum.size == 0:
                raise ValueError(f"empty outer{fold}/{label} interaction stratum")
            sampled_index = rng.integers(
                0, stratum.size, size=(repetitions, stratum.size), endpoint=False
            )
            destination += stratum[sampled_index].sum(axis=1)
            stratum_counts[f"outer{fold}_{label}_days"] = int(stratum.size)

    bootstrap = positive_sum / n_positive - complement_sum / n_complement
    regime_gap = float(values[membership].mean())
    complement_gap = float(values[~membership].mean())
    return {
        "n_days": int(len(values)),
        "n_regime_days": n_positive,
        "n_complement_days": n_complement,
        "regime_gap": regime_gap,
        "complement_gap": complement_gap,
        "interaction_difference": regime_gap - complement_gap,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "probability_below_zero": float(np.mean(bootstrap < 0.0)),
        "bootstrap_repetitions": int(repetitions),
        **stratum_counts,
    }


def _primary_regime_interaction_records(
    day_records: Sequence[Mapping[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    """Difference-in-differences for each registered binary NWP regime."""

    lookup = {
        (str(row["panel"]), int(row["outer"]), str(row["day"]), str(row["model"])): row
        for row in day_records
    }
    output: list[dict[str, Any]] = []
    for model_index, left_model in enumerate(("MM_mass", "MM_none")):
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for key, left in lookup.items():
            panel, outer, day, model = key
            if panel != PRIMARY_PANEL or model != left_model:
                continue
            right_key = (panel, outer, day, "DDPM_product")
            if right_key not in lookup:
                raise ValueError(f"missing primary interaction pair: {right_key}")
            pairs.append((left, lookup[right_key]))
        if len(pairs) != 150:
            raise ValueError(f"expected 150 primary interaction pairs for {left_model}")

        outer = np.asarray([int(left["outer"]) for left, _ in pairs])
        for regime_index, regime in enumerate(REGIME_NAMES):
            positive_label = POSITIVE_REGIME_LABELS[regime]
            labels = np.asarray([str(left[regime]) for left, _ in pairs])
            positive = labels == positive_label
            complement_labels = sorted(set(labels[~positive].tolist()))
            if len(complement_labels) != 1:
                raise ValueError(f"{regime} is not a registered binary regime")
            for metric_index, metric in enumerate(REGIME_METRICS):
                left_values = np.asarray([float(left[metric]) for left, _ in pairs])
                right_values = np.asarray([float(right[metric]) for _, right in pairs])
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                result = _stratified_regime_interaction_bootstrap(
                    left_values[finite] - right_values[finite],
                    outer[finite],
                    positive[finite],
                    repetitions=repetitions,
                    seed=20263814
                    + model_index * 10_000
                    + regime_index * 100
                    + metric_index,
                )
                output.append(
                    {
                        "panel": PRIMARY_PANEL,
                        "regime_dimension": regime,
                        "regime_positive_label": positive_label,
                        "complement_label": complement_labels[0],
                        "left_model": left_model,
                        "right_model": "DDPM_product",
                        "metric": metric,
                        "direction": "lower_is_better",
                        **result,
                        "interaction_definition": "mean(MM-DDPM | regime) - mean(MM-DDPM | complement)",
                        "negative_interaction_interpretation": "MM has a stronger relative advantage in the named regime",
                        "resampling_unit": "paired calendar day",
                        "bootstrap_strata": "outer fold x fixed regime membership",
                    }
                )
    return output


def _regime_records(
    day_records: Sequence[Mapping[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    primary = [row for row in day_records if row["panel"] == PRIMARY_PANEL]
    for left_model in ("MM_mass", "MM_none"):
        rows = summarize_per_day_by_regime(
            primary,
            model_left=left_model,
            model_right="DDPM_product",
            metric_names=REGIME_METRICS,
            regime_names=REGIME_NAMES,
            bootstrap_repetitions=repetitions,
        )
        for row in rows:
            row["panel"] = PRIMARY_PANEL
            row["direction"] = "lower_is_better"
        output.extend(rows)
    return output


def _stack_archives(
    archives: Mapping[tuple[str, str, int, int], Archive],
    spec: ModelSpec,
    seed: int,
    *,
    include_states: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    selected = [archives[(spec.panel, spec.model, seed, outer)] for outer in (1, 2, 3)]
    scenarios = np.concatenate([archive.scenarios for archive in selected], axis=0)
    observations = np.concatenate([archive.observations for archive in selected], axis=0)
    days = np.concatenate([archive.days for archive in selected])
    states = None
    if include_states:
        if any(archive.states is None for archive in selected):
            raise ValueError(f"requested absent explicit states for {spec.model}")
        states = np.concatenate([archive.states for archive in selected if archive.states is not None])
    if len(days) != 150 or len(np.unique(days)) != 150:
        raise ValueError(f"pooled {spec.panel}/{spec.model}/seed{seed} does not have 150 unique days")
    return scenarios, observations, days, states


def _cross_lag_records(
    archives: Mapping[tuple[str, str, int, int], Archive], raw: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    seed_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    raw_days = np.asarray(raw["day"])
    for spec in MODEL_SPECS:
        for seed in spec.seeds:
            scenarios, observations, days, _ = _stack_archives(archives, spec, seed)
            raw_index = _raw_indices(raw_days, days)
            clean_days = ~np.asarray(raw["target_was_missing"])[raw_index].any(axis=(1, 2))
            for data_scope, mask in (
                ("all_days", np.ones(len(days), dtype=bool)),
                ("exclude_raw_missing_affected_days", clean_days),
            ):
                result = cross_lag_summary(scenarios[mask], observations[mask])
                for row in result["records"]:
                    seed_rows.append(
                        {
                            "panel": spec.panel,
                            "outer_scope": "pooled_3_outer_folds",
                            "model": spec.model,
                            "seed": seed,
                            "seed_count": 1,
                            "data_scope": data_scope,
                            "n_days": int(mask.sum()),
                            "joint_scope": spec.joint_scope,
                            **row,
                        }
                    )
                for name, matrix in result["matrices"].items():
                    matrices[
                        _safe_npz_key(spec.panel, spec.model, f"seed{seed}", data_scope, name)
                    ] = matrix.astype(np.float32)

    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = (
            str(row["panel"]),
            str(row["model"]),
            str(row["data_scope"]),
            str(row["domain"]),
            int(row["lag"]),
        )
        grouped.setdefault(key, []).append(row)
    mean_rows: list[dict[str, Any]] = []
    numeric = (
        "cross_zone_RMSE",
        "same_zone_MAE",
        "forecast_offdiag_mean",
        "observed_offdiag_mean",
    )
    for key, rows in sorted(grouped.items()):
        panel, model, data_scope, domain, lag = key
        mean_rows.append(
            {
                "panel": panel,
                "outer_scope": "pooled_3_outer_folds",
                "model": model,
                "seed": "mean",
                "seed_count": len(rows),
                "data_scope": data_scope,
                "n_days": int(rows[0]["n_days"]),
                "joint_scope": rows[0]["joint_scope"],
                "domain": domain,
                "lag": lag,
                **{
                    name: float(np.mean([float(row[name]) for row in rows])) for name in numeric
                },
            }
        )
    return seed_rows + mean_rows, matrices


def _lagged_variogram_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate the registered lagged variogram suite and shuffle control."""

    seed_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(MODEL_SPECS):
        for outer in (1, 2, 3):
            for seed in spec.seeds:
                archive = archives[(spec.panel, spec.model, seed, outer)]
                shuffle_seed = 20260814 + spec_index * 10_000 + outer * 100 + seed
                suite = lagged_variogram_suite(
                    archive.scenarios,
                    archive.observations,
                    lags=tuple(range(7)),
                    include_shuffle_control=True,
                    shuffle_seed=shuffle_seed,
                )
                if len(suite["records"]) != 56 or len(suite["per_day"]) != 56:
                    raise RuntimeError("advanced lagged variogram suite schema changed")
                for descriptor in suite["records"]:
                    key = (
                        f"{descriptor['variant']}__{descriptor['domain']}__"
                        f"{descriptor['pair_type']}__lag{descriptor['lag']}"
                    )
                    daily = np.asarray(suite["per_day"][key], dtype=np.float64)
                    if daily.shape != (50,) or not np.isfinite(daily).all():
                        raise RuntimeError(f"invalid variogram daily array: {key}")
                    for day_index, day in enumerate(archive.days):
                        seed_rows.append(
                            {
                                "panel": spec.panel,
                                "outer": outer,
                                "model": spec.model,
                                "seed": seed,
                                "day": str(day),
                                "variant": descriptor["variant"],
                                "domain": descriptor["domain"],
                                "pair_type": descriptor["pair_type"],
                                "lag": int(descriptor["lag"]),
                                "variogram_score": float(daily[day_index]),
                                "shuffle_seed": shuffle_seed,
                                "joint_scope": spec.joint_scope,
                            }
                        )

                # A whole-trajectory permutation must leave every same-zone
                # variogram exactly invariant up to floating-point rounding.
                for domain in ("level", "ramp"):
                    for lag in range(7):
                        original = suite["per_day"][
                            f"original__{domain}__same_zone_temporal__lag{lag}"
                        ]
                        shuffled = suite["per_day"][
                            f"member_trajectory_shuffle__{domain}__same_zone_temporal__lag{lag}"
                        ]
                        if not np.allclose(original, shuffled, rtol=0.0, atol=2e-14):
                            raise RuntimeError(
                                "member shuffle changed a same-zone trajectory diagnostic"
                            )

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    group_names = (
        "panel",
        "outer",
        "model",
        "day",
        "variant",
        "domain",
        "pair_type",
        "lag",
    )
    for row in seed_rows:
        grouped.setdefault(tuple(row[name] for name in group_names), []).append(row)
    day_rows: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        first = selected[0]
        values = np.asarray([float(row["variogram_score"]) for row in selected])
        output = dict(zip(group_names, key))
        output.update(
            {
                "seed": "mean",
                "seed_count": len(selected),
                "variogram_score": float(values.mean()),
                "seed_sd__variogram_score": float(values.std(ddof=1))
                if len(values) > 1
                else 0.0,
                "joint_scope": first["joint_scope"],
            }
        )
        day_rows.append(output)

    summary_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    summary_names = ("panel", "model", "variant", "domain", "pair_type", "lag")
    for row in day_rows:
        summary_groups.setdefault(tuple(row[name] for name in summary_names), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, selected in sorted(
        summary_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        if len(selected) != 150:
            raise RuntimeError(f"variogram summary does not contain 150 days: {key}")
        first = selected[0]
        values = np.asarray([float(row["variogram_score"]) for row in selected])
        summary_rows.append(
            {
                **dict(zip(summary_names, key)),
                "seed": "mean",
                "seed_count": int(first["seed_count"]),
                "n_days": len(selected),
                "pooled_score": float(values.mean()),
                "day_sd": float(values.std(ddof=1)),
                "mean_daily_seed_sd": float(
                    np.mean([float(row["seed_sd__variogram_score"]) for row in selected])
                ),
                "joint_scope": first["joint_scope"],
                "unit_of_aggregation": "calendar day after within-day seed averaging",
            }
        )

    summary_lookup = {
        (
            str(row["panel"]),
            str(row["model"]),
            str(row["domain"]),
            str(row["pair_type"]),
            int(row["lag"]),
            str(row["variant"]),
        ): row
        for row in summary_rows
    }
    delta_rows: list[dict[str, Any]] = []
    for key, original in summary_lookup.items():
        panel, model, domain, pair_type, lag, variant = key
        if variant != "original":
            continue
        shuffled = summary_lookup[
            (panel, model, domain, pair_type, lag, "member_trajectory_shuffle")
        ]
        original_score = float(original["pooled_score"])
        shuffled_score = float(shuffled["pooled_score"])
        delta_rows.append(
            {
                "panel": panel,
                "model": model,
                "domain": domain,
                "pair_type": pair_type,
                "lag": lag,
                "original_score": original_score,
                "member_shuffle_score": shuffled_score,
                "shuffle_minus_original": shuffled_score - original_score,
                "relative_change_pct": 100.0
                * (shuffled_score - original_score)
                / max(abs(original_score), 1e-12),
                "negative_control_interpretation": (
                    "must_be_zero_by_construction"
                    if pair_type == "same_zone_temporal"
                    else "cross_zone_member_coupling_sensitivity"
                ),
                "joint_scope": original["joint_scope"],
            }
        )
    return seed_rows, day_rows, summary_rows, delta_rows


def _atom_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
    raw: Mapping[str, np.ndarray],
    train_epsilon_by_panel: Mapping[str, float],
) -> list[dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    raw_days = np.asarray(raw["day"])
    for spec in MODEL_SPECS:
        epsilons = {
            "exact": 0.0,
            "epsilon_0p001": 0.001,
            "epsilon_0p01": 0.01,
            "train_derived": float(train_epsilon_by_panel[spec.panel]),
        }
        for seed in spec.seeds:
            scenarios, observations, days, _ = _stack_archives(archives, spec, seed)
            raw_index = _raw_indices(raw_days, days)
            complete_zone_day = ~np.asarray(raw["target_was_missing"])[raw_index].any(axis=-1)
            selected_scenarios = np.transpose(scenarios, (0, 2, 1, 3))[complete_zone_day]
            selected_observations = observations[complete_zone_day]
            selected_scenarios = selected_scenarios[:, :, None, :]
            selected_observations = selected_observations[:, None, :]
            for epsilon_scheme in ATOM_EPSILON_SCHEMES:
                epsilon = epsilons[epsilon_scheme]
                for data_scope, values, truth, n_zone_days in (
                    (
                        "all_zone_days",
                        scenarios,
                        observations,
                        observations.shape[0] * observations.shape[1],
                    ),
                    (
                        "exclude_raw_missing_affected_zone_days",
                        selected_scenarios,
                        selected_observations,
                        int(complete_zone_day.sum()),
                    ),
                ):
                    result = atom_duration_summary(values, truth, epsilon=epsilon)
                    row: dict[str, Any] = {
                        "panel": spec.panel,
                        "outer_scope": "pooled_3_outer_folds",
                        "model": spec.model,
                        "seed": seed,
                        "seed_count": 1,
                        "data_scope": data_scope,
                        "n_days": 150,
                        "n_zone_days": n_zone_days,
                        "epsilon_scheme": epsilon_scheme,
                        "joint_scope": spec.joint_scope,
                    }
                    for name, value in result.items():
                        if name in (
                            "duration_bins",
                            "forecast_duration_distribution",
                            "observed_duration_distribution",
                        ):
                            continue
                        row[name] = value
                    for bin_name, forecast, observed in zip(
                        result["duration_bins"],
                        result["forecast_duration_distribution"],
                        result["observed_duration_distribution"],
                    ):
                        safe_bin = bin_name.replace("-", "_to_")
                        row[f"forecast_duration_p_{safe_bin}"] = forecast
                        row[f"observed_duration_p_{safe_bin}"] = observed
                    seed_rows.append(row)

    identifier_names = (
        "panel",
        "outer_scope",
        "model",
        "data_scope",
        "n_days",
        "n_zone_days",
        "epsilon_scheme",
        "epsilon",
        "joint_scope",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = tuple(row[name] for name in identifier_names)
        grouped.setdefault(key, []).append(row)
    mean_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output = dict(zip(identifier_names, key))
        output["seed"] = "mean"
        output["seed_count"] = len(rows)
        numeric_names = [
            name
            for name in rows[0]
            if name not in set(identifier_names) | {"seed", "seed_count"}
        ]
        for name in numeric_names:
            values = np.asarray(
                [float(row[name]) if row.get(name) is not None else np.nan for row in rows],
                dtype=np.float64,
            )
            output[name] = float(np.nanmean(values)) if np.isfinite(values).any() else None
        mean_rows.append(output)
    return seed_rows + mean_rows


def _atom_event_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
    bootstrap_repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return daily atom-event proper scores and primary paired contrasts."""

    seed_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for outer in (1, 2, 3):
            for seed in spec.seeds:
                archive = archives[(spec.panel, spec.model, seed, outer)]
                for epsilon_scheme, epsilon in ATOM_EVENT_SCHEMES.items():
                    metrics = atom_event_metrics(
                        archive.scenarios, archive.observations, epsilon=epsilon
                    )
                    if set(metrics) != set(ATOM_EVENT_METRICS):
                        raise RuntimeError("advanced atom-event metric schema changed")
                    for day_index, day in enumerate(archive.days):
                        seed_rows.append(
                            {
                                "panel": spec.panel,
                                "outer": outer,
                                "model": spec.model,
                                "seed": seed,
                                "day": str(day),
                                "epsilon_scheme": epsilon_scheme,
                                "epsilon": epsilon,
                                "joint_scope": spec.joint_scope,
                                **{
                                    name: float(values[day_index])
                                    for name, values in metrics.items()
                                },
                            }
                        )

    grouped: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = (
            str(row["panel"]),
            int(row["outer"]),
            str(row["model"]),
            str(row["day"]),
            str(row["epsilon_scheme"]),
        )
        grouped.setdefault(key, []).append(row)
    day_rows: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        panel, outer, model, day, epsilon_scheme = key
        first = selected[0]
        output: dict[str, Any] = {
            "panel": panel,
            "outer": outer,
            "model": model,
            "seed": "mean",
            "seed_count": len(selected),
            "day": day,
            "epsilon_scheme": epsilon_scheme,
            "epsilon": float(first["epsilon"]),
            "joint_scope": first["joint_scope"],
        }
        for name in ATOM_EVENT_METRICS:
            values = np.asarray([float(row[name]) for row in selected])
            output[name] = float(values.mean())
            output[f"seed_sd__{name}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        day_rows.append(output)

    lookup = {
        (
            str(row["panel"]),
            int(row["outer"]),
            str(row["day"]),
            str(row["model"]),
            str(row["epsilon_scheme"]),
        ): row
        for row in day_rows
    }
    bootstrap_rows: list[dict[str, Any]] = []
    for left_model in ("MM_mass", "MM_none"):
        for epsilon_index, epsilon_scheme in enumerate(ATOM_EVENT_SCHEMES):
            pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for key, left in lookup.items():
                panel, outer, day, model, scheme = key
                if panel != PRIMARY_PANEL or model != left_model or scheme != epsilon_scheme:
                    continue
                right_key = (panel, outer, day, "DDPM_product", scheme)
                if right_key not in lookup:
                    raise ValueError(f"missing atom-event DDPM pair: {right_key}")
                pairs.append((left, lookup[right_key]))
            if len(pairs) != 150:
                raise ValueError(
                    f"expected 150 atom-event pairs for {left_model}/{epsilon_scheme}"
                )
            strata = np.asarray([int(left["outer"]) for left, _ in pairs])
            for metric_index, metric in enumerate(ATOM_EVENT_METRICS):
                left_values = np.asarray([float(left[metric]) for left, _ in pairs])
                right_values = np.asarray([float(right[metric]) for _, right in pairs])
                result = stratified_paired_bootstrap(
                    left_values - right_values,
                    strata,
                    repetitions=bootstrap_repetitions,
                    seed=20261814
                    + 1000 * (left_model == "MM_none")
                    + 100 * epsilon_index
                    + metric_index,
                )
                bootstrap_rows.append(
                    {
                        "panel": PRIMARY_PANEL,
                        "left_model": left_model,
                        "right_model": "DDPM_product",
                        "epsilon_scheme": epsilon_scheme,
                        "epsilon": ATOM_EVENT_SCHEMES[epsilon_scheme],
                        "metric": metric,
                        "direction": "lower_is_better",
                        "left_mean": float(left_values.mean()),
                        "right_mean": float(right_values.mean()),
                        **result,
                        "unit_of_inference": "paired calendar day, stratified by outer fold",
                    }
                )
    return seed_rows, day_rows, bootstrap_rows


def _generated_transition_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact, fast generated-state decomposition for the registered subset."""

    requested: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("MM_mass", (0, 1, 2)),
        ("MM_none", (0,)),
        ("DDPM_product", (0,)),
    )
    definitions: dict[str, str] = {}
    seed_day_rows: list[dict[str, Any]] = []
    for model, seeds in requested:
        spec = next(
            item
            for item in MODEL_SPECS
            if item.panel == PRIMARY_PANEL and item.model == model
        )
        for outer in (1, 2, 3):
            for seed in seeds:
                archive = archives[(PRIMARY_PANEL, model, seed, outer)]
                explicit = archive.states if model.startswith("MM_") else None
                result = generated_transition_crps_decomposition(
                    archive.scenarios,
                    archive.observations,
                    scenario_states=explicit,
                    epsilon=0.0,
                )
                definitions.update(result["transition_definitions"])
                total = np.asarray(result["total_local_ramp_CRPS"])
                reconstructed = np.asarray(result["reconstructed_local_ramp_CRPS"])
                maximum_error = float(np.max(np.abs(total - reconstructed)))
                if maximum_error > 5e-11:
                    raise RuntimeError("generated-transition CRPS failed reconstruction audit")
                for day_index, day in enumerate(archive.days):
                    total_cases = int(np.asarray(result["member_cases"])[day_index].sum())
                    for code, transition in enumerate(result["transition_names"]):
                        cases = int(result["member_cases"][day_index, code])
                        seed_day_rows.append(
                            {
                                "panel": PRIMARY_PANEL,
                                "outer": outer,
                                "model": model,
                                "seed": seed,
                                "day": str(day),
                                "generated_transition": transition,
                                "transition_definition": definitions[transition],
                                "state_source": result["state_source"],
                                "epsilon": 0.0,
                                "member_cases": cases,
                                "member_share": float(cases / max(total_cases, 1)),
                                "first_term": float(result["first_term"][day_index, code]),
                                "pair_term": float(result["pair_term"][day_index, code]),
                                "net_contribution": float(
                                    result["net_contribution"][day_index, code]
                                ),
                                "total_local_ramp_CRPS": float(total[day_index]),
                                "reconstructed_local_ramp_CRPS": float(
                                    reconstructed[day_index]
                                ),
                                "absolute_reconstruction_error": float(
                                    abs(total[day_index] - reconstructed[day_index])
                                ),
                                "scope_note": (
                                    "all_three_frozen_seeds"
                                    if model == "MM_mass"
                                    else "registered_seed0_subset"
                                ),
                            }
                        )

    # Per-day seed means retain the day as the observational unit.  The member
    # case count is averaged for multi-seed models because seed replications
    # are not additional observed days.
    daily_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    daily_names = ("panel", "outer", "model", "day", "generated_transition")
    for row in seed_day_rows:
        daily_groups.setdefault(tuple(row[name] for name in daily_names), []).append(row)
    mean_day_rows: list[dict[str, Any]] = []
    for key, selected in sorted(
        daily_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        first = selected[0]
        output = dict(zip(daily_names, key))
        output.update(
            {
                "seed": "mean",
                "seed_count": len(selected),
                "transition_definition": first["transition_definition"],
                "state_source": first["state_source"],
                "epsilon": 0.0,
                "scope_note": first["scope_note"],
            }
        )
        for name in (
            "member_cases",
            "member_share",
            "first_term",
            "pair_term",
            "net_contribution",
            "total_local_ramp_CRPS",
            "reconstructed_local_ramp_CRPS",
            "absolute_reconstruction_error",
        ):
            output[name] = float(np.mean([float(row[name]) for row in selected]))
        mean_day_rows.append(output)

    all_day_rows = seed_day_rows + mean_day_rows
    summary_groups: dict[tuple[str, str, Any], list[Mapping[str, Any]]] = {}
    for row in all_day_rows:
        summary_groups.setdefault(
            (str(row["model"]), str(row["generated_transition"]), row["seed"]), []
        ).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (model, transition, seed), selected in sorted(
        summary_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        # Seed-mean and individual-seed groups each cover exactly 150 days.
        if len(selected) != 150:
            raise RuntimeError("generated-transition summary is not based on 150 days")
        first = selected[0]
        summary_rows.append(
            {
                "panel": PRIMARY_PANEL,
                "model": model,
                "seed": seed,
                "seed_count": int(first.get("seed_count", 1)),
                "n_days": len(selected),
                "generated_transition": transition,
                "transition_definition": first["transition_definition"],
                "state_source": first["state_source"],
                "epsilon": 0.0,
                "mean_member_cases_per_day": float(
                    np.mean([float(row["member_cases"]) for row in selected])
                ),
                "member_share": float(
                    np.mean([float(row["member_share"]) for row in selected])
                ),
                "first_term": float(
                    np.mean([float(row["first_term"]) for row in selected])
                ),
                "pair_term": float(
                    np.mean([float(row["pair_term"]) for row in selected])
                ),
                "net_contribution": float(
                    np.mean([float(row["net_contribution"]) for row in selected])
                ),
                "total_local_ramp_CRPS": float(
                    np.mean([float(row["total_local_ramp_CRPS"]) for row in selected])
                ),
                "maximum_absolute_reconstruction_error": float(
                    np.max(
                        [float(row["absolute_reconstruction_error"]) for row in selected]
                    )
                ),
                "scope_note": first["scope_note"],
                "decomposition_identity": "sum(net_contribution)=ordinary local ramp CRPS",
            }
        )
    return all_day_rows, summary_rows


def _lag1_state_attribution_records(
    archives: Mapping[tuple[str, str, int, int], Archive],
    bootstrap_repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Descriptive lag-1 correlation accounting for MM_mass and DDPM.

    MM uses its archived model states.  DDPM uses phenomenological output
    boundary labels; these must never be described as latent DDPM states.
    """

    epsilon_schemes = {"exact": 0.0, "epsilon_0p01": 0.01}
    requested = (("MM_mass", (0, 1, 2)), ("DDPM_product", (0,)))
    seed_rows: list[dict[str, Any]] = []
    path_names = ("interior_only", "any_atom_involving")
    for model, seeds in requested:
        for outer in (1, 2, 3):
            for seed in seeds:
                archive = archives[(PRIMARY_PANEL, model, seed, outer)]
                explicit_states = archive.states if model == "MM_mass" else None
                for epsilon_scheme, epsilon in epsilon_schemes.items():
                    result = lag1_increment_state_decomposition(
                        archive.scenarios,
                        archive.observations,
                        scenario_states=explicit_states,
                        epsilon=epsilon,
                    )
                    if tuple(result["path_names"]) != path_names:
                        raise RuntimeError("lag-1 state attribution path schema changed")
                    signed = np.asarray(result["signed_total_correlation_gap"])
                    reconstructed = np.asarray(
                        result["reconstructed_signed_total_correlation_gap"]
                    )
                    valid = np.asarray(result["valid_day"], dtype=bool)
                    if not np.allclose(
                        signed[valid], reconstructed[valid], rtol=2e-12, atol=2e-12
                    ):
                        raise RuntimeError("lag-1 state contributions failed reconstruction")
                    if not np.allclose(
                        np.asarray(result["absolute_total_correlation_error"])[valid],
                        np.abs(signed[valid]),
                        rtol=0.0,
                        atol=2e-14,
                    ):
                        raise RuntimeError("lag-1 absolute error identity failed")

                    for day_index, day in enumerate(archive.days):
                        row: dict[str, Any] = {
                            "panel": PRIMARY_PANEL,
                            "outer": outer,
                            "model": model,
                            "seed": seed,
                            "day": str(day),
                            "epsilon_scheme": epsilon_scheme,
                            "epsilon": epsilon,
                            "valid_day": bool(valid[day_index]),
                            "forecast_state_source": result["forecast_state_source"],
                            "forecast_state_semantics": result["forecast_state_semantics"],
                            "truth_state_source": result["truth_state_source"],
                            "truth_state_semantics": result["truth_state_semantics"],
                            "unit_of_analysis": result["unit_of_analysis"],
                            "member_semantics": result["member_semantics"],
                            "attribution_caveat": result["attribution_caveat"],
                            "forecast_total_correlation": float(
                                result["forecast_total_correlation"][day_index]
                            ),
                            "truth_total_correlation": float(
                                result["truth_total_correlation"][day_index]
                            ),
                            "signed_total_correlation_gap": float(signed[day_index]),
                            "absolute_total_correlation_error": float(
                                result["absolute_total_correlation_error"][day_index]
                            ),
                            "reconstructed_signed_total_correlation_gap": float(
                                reconstructed[day_index]
                            ),
                        }
                        for code, path_name in enumerate(path_names):
                            prefix = f"path__{path_name}"
                            row[f"{prefix}__definition"] = result["path_definitions"][
                                path_name
                            ]
                            for name in (
                                "forecast_case_count",
                                "forecast_case_share",
                                "truth_case_count",
                                "truth_case_share",
                                "forecast_correlation_contribution",
                                "truth_correlation_contribution",
                                "signed_correlation_gap_contribution",
                                "forecast_conditional_global_standardized_moment",
                                "truth_conditional_global_standardized_moment",
                                "forecast_conditional_correlation",
                                "truth_conditional_correlation",
                                "conditional_correlation_gap",
                            ):
                                row[f"{prefix}__{name}"] = float(
                                    result[name][day_index, code]
                                )
                        seed_rows.append(row)

    group_names = ("panel", "outer", "model", "day", "epsilon_scheme", "epsilon")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault(tuple(row[name] for name in group_names), []).append(row)
    day_rows: list[dict[str, Any]] = []
    numeric_names = (
        "forecast_total_correlation",
        "truth_total_correlation",
        "signed_total_correlation_gap",
        "absolute_total_correlation_error",
        "reconstructed_signed_total_correlation_gap",
        *tuple(
            f"path__{path_name}__{name}"
            for path_name in path_names
            for name in (
                "forecast_case_count",
                "forecast_case_share",
                "truth_case_count",
                "truth_case_share",
                "forecast_correlation_contribution",
                "truth_correlation_contribution",
                "signed_correlation_gap_contribution",
                "forecast_conditional_global_standardized_moment",
                "truth_conditional_global_standardized_moment",
                "forecast_conditional_correlation",
                "truth_conditional_correlation",
                "conditional_correlation_gap",
            )
        ),
    )
    for key, selected in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        first = selected[0]
        output: dict[str, Any] = {
            **dict(zip(group_names, key)),
            "seed": "mean",
            "seed_count": len(selected),
            "valid_day": bool(all(bool(row["valid_day"]) for row in selected)),
            "forecast_state_source": first["forecast_state_source"],
            "forecast_state_semantics": first["forecast_state_semantics"],
            "truth_state_source": first["truth_state_source"],
            "truth_state_semantics": first["truth_state_semantics"],
            "unit_of_analysis": "calendar_day",
            "member_semantics": first["member_semantics"],
            "attribution_caveat": first["attribution_caveat"],
            "absolute_error_seed_aggregation": "mean of per-seed absolute correlation errors",
        }
        for path_name in path_names:
            output[f"path__{path_name}__definition"] = first[
                f"path__{path_name}__definition"
            ]
        for name in numeric_names:
            values = np.asarray([float(row[name]) for row in selected])
            finite_values = values[np.isfinite(values)]
            output[name] = (
                float(finite_values.mean()) if finite_values.size else float("nan")
            )
            if name in (
                "signed_total_correlation_gap",
                "absolute_total_correlation_error",
                "path__interior_only__signed_correlation_gap_contribution",
                "path__any_atom_involving__signed_correlation_gap_contribution",
            ):
                output[f"seed_sd__{name}"] = (
                    float(finite_values.std(ddof=1))
                    if finite_values.size > 1
                    else (0.0 if finite_values.size == 1 else float("nan"))
                )
        output["absolute_of_seed_mean_signed_gap"] = abs(
            float(output["signed_total_correlation_gap"])
        )
        contribution_sum = sum(
            float(output[f"path__{path_name}__signed_correlation_gap_contribution"])
            for path_name in path_names
        )
        if not np.isclose(
            contribution_sum,
            float(output["signed_total_correlation_gap"]),
            rtol=2e-12,
            atol=2e-12,
        ):
            raise RuntimeError("seed-mean lag-1 contributions failed reconstruction")
        day_rows.append(output)

    lookup = {
        (
            int(row["outer"]),
            str(row["day"]),
            str(row["model"]),
            str(row["epsilon_scheme"]),
        ): row
        for row in day_rows
    }
    bootstrap_rows: list[dict[str, Any]] = []
    metric_specs = (
        (
            "absolute_total_correlation_error",
            "total absolute same-zone lag-1 increment correlation error",
            "lower_is_better",
        ),
        (
            "signed_total_correlation_gap",
            "forecast minus observed total correlation",
            "signed_no_universal_preference",
        ),
        (
            "path__interior_only__signed_correlation_gap_contribution",
            "signed contribution of three-point interior-only paths",
            "signed_no_universal_preference",
        ),
        (
            "path__any_atom_involving__signed_correlation_gap_contribution",
            "signed contribution of paths involving at least one output-boundary atom",
            "signed_no_universal_preference",
        ),
    )
    for epsilon_index, (epsilon_scheme, epsilon) in enumerate(epsilon_schemes.items()):
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for outer in (1, 2, 3):
            mm_days = sorted(
                str(row["day"])
                for row in day_rows
                if row["model"] == "MM_mass"
                and int(row["outer"]) == outer
                and row["epsilon_scheme"] == epsilon_scheme
            )
            for day in mm_days:
                pairs.append(
                    (
                        lookup[(outer, day, "MM_mass", epsilon_scheme)],
                        lookup[(outer, day, "DDPM_product", epsilon_scheme)],
                    )
                )
        if len(pairs) != 150:
            raise ValueError(f"expected 150 lag-1 paired days for {epsilon_scheme}")
        strata = np.asarray([int(left["outer"]) for left, _ in pairs])
        for metric_index, (metric, semantics, direction) in enumerate(metric_specs):
            left_values = np.asarray([float(left[metric]) for left, _ in pairs])
            right_values = np.asarray([float(right[metric]) for _, right in pairs])
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            # Signed metrics deliberately share a resampling stream so their
            # additive accounting relationship is preserved replicate-wise.
            bootstrap_seed = (
                20264814 + epsilon_index * 100 + (1 if metric_index == 0 else 2)
            )
            result = stratified_paired_bootstrap(
                left_values[finite] - right_values[finite],
                strata[finite],
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed,
            )
            bootstrap_rows.append(
                {
                    "panel": PRIMARY_PANEL,
                    "left_model": "MM_mass",
                    "right_model": "DDPM_product",
                    "epsilon_scheme": epsilon_scheme,
                    "epsilon": epsilon,
                    "metric": metric,
                    "metric_semantics": semantics,
                    "direction": direction,
                    "left_mean": float(left_values[finite].mean()),
                    "right_mean": float(right_values[finite].mean()),
                    **result,
                    "left_seed_aggregation": "per-day mean across MM_mass seeds 0-2",
                    "right_seed_aggregation": "single frozen DDPM seed0",
                    "unit_of_inference": "paired calendar day, stratified by outer fold",
                    "forecast_state_semantics": "MM uses archive explicit states; DDPM uses threshold-derived output-boundary labels only, not latent states",
                    "attribution_caveat": "descriptive correlation accounting by path label; not a causal decomposition",
                    "absolute_attribution_warning": "only signed gaps are additive; absolute total error has no additive category attribution",
                }
            )
    return seed_rows, day_rows, bootstrap_rows


def _regime_assignment_records(
    raw: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    regimes_by_panel: Mapping[str, Mapping[str, np.ndarray]],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_days = np.asarray(raw["day"])
    output: list[dict[str, Any]] = []
    for panel in (PRIMARY_PANEL, SECONDARY_PANEL):
        for outer in (1, 2, 3):
            days = _registry_days(registries[panel], outer)
            indices = _raw_indices(raw_days, days)
            for offset, day in enumerate(days):
                index = indices[offset]
                output.append(
                    {
                        "panel": panel,
                        "outer": outer,
                        "day": str(day),
                        **{name: float(values[index]) for name, values in features.items()},
                        **{
                            name: str(values[index])
                            for name, values in regimes_by_panel[panel].items()
                        },
                        "missing_target_cells": int(
                            np.asarray(raw["target_was_missing"])[index].sum()
                        ),
                        "raw_missing_affected_day": bool(
                            np.asarray(raw["target_was_missing"])[index].any()
                        ),
                    }
                )
    return output


def _make_plots(
    output_dir: Path,
    bootstrap: Sequence[Mapping[str, Any]],
    cross_lag: Sequence[Mapping[str, Any]],
    atom: Sequence[Mapping[str, Any]],
    regime: Sequence[Mapping[str, Any]],
    variogram: Sequence[Mapping[str, Any]],
    transition: Sequence[Mapping[str, Any]],
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_crossdiag")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    plot_metrics = ("level_CRPS", "ramp_CRPS", "max_abs_ramp_CRPS", "cross_lag_increment_lag1_RMSE")
    selected = [row for row in bootstrap if row["metric"] in plot_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, model in zip(axes, ("MM_mass", "MM_none")):
        rows = [row for row in selected if row["left_model"] == model]
        x = np.arange(len(rows))
        means = np.asarray([row["relative_mean_difference_pct"] for row in rows])
        lows = np.asarray([row["relative_ci_low_pct"] for row in rows])
        highs = np.asarray([row["relative_ci_high_pct"] for row in rows])
        axis.bar(x, means, color=np.where(means <= 0, "#238636", "#cf222e"), alpha=0.85)
        axis.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=4)
        axis.axhline(0, color="#57606a", linewidth=1)
        axis.set_xticks(x, [row["metric"].replace("_", "\n") for row in rows], fontsize=8)
        axis.set_title(f"{model} minus DDPM (paired days)")
        axis.set_ylabel("Relative score difference (%)\nnegative favors MM")
        axis.grid(axis="y", alpha=0.25)
    path = output_dir / "primary_paired_differences.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, panel in zip(axes, (PRIMARY_PANEL, SECONDARY_PANEL)):
        rows = [
            row
            for row in cross_lag
            if row["panel"] == panel
            and row["seed"] == "mean"
            and row["data_scope"] == "all_days"
            and row["domain"] == "increment"
        ]
        for model in dict.fromkeys(row["model"] for row in rows):
            model_rows = sorted((row for row in rows if row["model"] == model), key=lambda row: row["lag"])
            axis.plot(
                [row["lag"] for row in model_rows],
                [row["cross_zone_RMSE"] for row in model_rows],
                marker="o",
                label=model,
            )
        axis.set_title(f"{panel}: increment cross-zone correlation")
        axis.set_xlabel("lag (hours after differencing)")
        axis.set_ylabel("off-diagonal correlation RMSE")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path = output_dir / "pooled_cross_lag_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    primary_atom = [
        row
        for row in atom
        if row["panel"] == PRIMARY_PANEL
        and row["seed"] == "mean"
        and row["data_scope"] == "all_zone_days"
        and row["epsilon_scheme"] == "train_derived"
    ]
    bins = ("1", "2", "3_to_4", "5_to_8", "9_to_16", "17_to_24")
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(bins))
    observed = primary_atom[0] if primary_atom else None
    if observed is not None:
        axis.plot(
            x,
            [observed[f"observed_duration_p_{name}"] for name in bins],
            color="black",
            linewidth=2.5,
            marker="o",
            label="observed",
        )
    for row in primary_atom:
        axis.plot(
            x,
            [row[f"forecast_duration_p_{name}"] for name in bins],
            marker="o",
            label=row["model"],
        )
    axis.set_xticks(x, [name.replace("_to_", "-") for name in bins])
    axis.set_xlabel("zero-atom run duration (hours)")
    axis.set_ylabel("share of zero runs")
    axis.set_title("Primary panel: train-derived atom threshold")
    axis.grid(alpha=0.25)
    axis.legend()
    path = output_dir / "primary_atom_duration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    rows = [
        row
        for row in regime
        if row["left_model"] == "MM_mass"
        and row["metric"] == "ramp_CRPS"
        and row["regime_dimension"] in ("speed_volatile", "turning", "spatial_heterogeneous")
    ]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [f"{row['regime_dimension']}\n{row['regime_label']}" for row in rows]
    x = np.arange(len(rows))
    means = np.asarray([row["mean_difference"] for row in rows])
    lows = np.asarray([row["ci_low"] for row in rows])
    highs = np.asarray([row["ci_high"] for row in rows])
    axis.bar(x, means, color=np.where(means <= 0, "#238636", "#cf222e"), alpha=0.85)
    axis.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=4)
    axis.axhline(0, color="#57606a", linewidth=1)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylabel("Ramp CRPS difference (MM_mass - DDPM)")
    axis.set_title("Primary paired contrast within fixed NWP regimes")
    axis.grid(axis="y", alpha=0.25)
    path = output_dir / "primary_nwp_regime_ramp.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"MM_mass": "#0969da", "MM_none": "#8250df", "DDPM_product": "#cf222e"}
    for axis, domain in zip(axes, ("level", "ramp")):
        rows = [
            row
            for row in variogram
            if row["panel"] == PRIMARY_PANEL
            and row["domain"] == domain
            and row["pair_type"] == "cross_zone"
        ]
        for model in ("MM_mass", "MM_none", "DDPM_product"):
            for variant, style in (("original", "-"), ("member_trajectory_shuffle", "--")):
                selected_rows = sorted(
                    (
                        row
                        for row in rows
                        if row["model"] == model and row["variant"] == variant
                    ),
                    key=lambda row: int(row["lag"]),
                )
                axis.plot(
                    [int(row["lag"]) for row in selected_rows],
                    [float(row["pooled_score"]) for row in selected_rows],
                    linestyle=style,
                    marker="o" if variant == "original" else None,
                    color=colors[model],
                    label=f"{model} / {'original' if variant == 'original' else 'shuffle'}",
                )
        axis.set_title(f"Primary {domain} cross-zone variogram")
        axis.set_xlabel("lag")
        axis.set_ylabel("variogram score (lower is better)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    path = output_dir / "lagged_variogram_shuffle_control.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    transition_rows = [row for row in transition if row["seed"] == "mean"]
    models = ("MM_mass", "MM_none", "DDPM_product")
    transition_names = ("interior_to_interior", "atom_to_interior", "atom_to_atom")
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.24
    for index, name in enumerate(transition_names):
        values = [
            float(
                next(
                    row["net_contribution"]
                    for row in transition_rows
                    if row["model"] == model and row["generated_transition"] == name
                )
            )
            for model in models
        ]
        axis.bar(x + (index - 1) * width, values, width=width, label=name)
    axis.axhline(0, color="#57606a", linewidth=1)
    axis.set_xticks(x, models)
    axis.set_ylabel("net contribution to local ramp CRPS")
    axis.set_title("Exact generated-transition CRPS decomposition")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    path = output_dir / "generated_transition_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)
    return written


def _input_audit_records(root: Path, archive_inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in archive_inventory]
    extra_paths = [root / path for path in REGISTRY_PATHS.values()]
    extra_paths.extend(sorted((root / "Data").glob("*.csv")))
    extra_paths.extend(
        [
            root / "repro_scripts" / "run_cross_model_diagnostics.py",
            root / "cross_model_diagnostics" / "core.py",
            root / "cross_model_diagnostics" / "data.py",
            root / "cross_model_diagnostics" / "advanced.py",
        ]
    )
    for path in extra_paths:
        output.append(
            {
                "panel": "shared_input",
                "model": "raw_or_registry",
                "seed": "",
                "outer": "",
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "n_days": "",
                "n_members": "",
                "n_zones": "",
                "n_hours": "",
                "date_sha256": "",
                "joint_scope": "",
                "explicit_states": "",
            }
        )
    return output


def run(root: Path, output_dir: Path, bootstrap_repetitions: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registries = {
        panel: json.loads((root / path).read_text(encoding="utf-8"))
        for panel, path in REGISTRY_PATHS.items()
    }
    raw = load_raw_joint_days(root / "Data")
    features, thresholds_by_panel, regimes_by_panel, train_epsilon_by_panel, reference_audit = _fit_panel_references(
        raw, registries
    )
    archives, archive_inventory = _load_archives(root, raw, registries)
    seed_day_records, metric_names = _per_day_records(
        archives, raw, features, regimes_by_panel
    )
    day_records, seed_stability = _aggregate_seed_day_records(seed_day_records, metric_names)
    bootstrap = _paired_bootstrap_records(day_records, bootstrap_repetitions)
    secondary_bootstrap = _secondary_paired_bootstrap_records(
        day_records, bootstrap_repetitions
    )
    regime_summary = _regime_records(day_records, bootstrap_repetitions)
    regime_interaction = _primary_regime_interaction_records(
        day_records, bootstrap_repetitions
    )
    cross_lag, matrices = _cross_lag_records(archives, raw)
    variogram_seed, variogram_day, variogram_summary, variogram_delta = (
        _lagged_variogram_records(archives)
    )
    atom = _atom_records(archives, raw, train_epsilon_by_panel)
    atom_event_seed, atom_event_day, atom_event_bootstrap = _atom_event_records(
        archives, bootstrap_repetitions
    )
    transition_day, transition_summary = _generated_transition_records(archives)
    lag1_state_seed, lag1_state_day, lag1_state_bootstrap = (
        _lag1_state_attribution_records(archives, bootstrap_repetitions)
    )
    assignments = _regime_assignment_records(
        raw, features, regimes_by_panel, registries
    )

    artifacts: list[Path] = []
    outputs: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("input_inventory.csv", _input_audit_records(root, archive_inventory)),
        ("per_day_metrics_seed.csv", seed_day_records),
        ("per_day_metrics.csv", day_records),
        ("seed_stability.csv", seed_stability),
        ("primary_paired_bootstrap.csv", bootstrap),
        ("secondary_paired_bootstrap.csv", secondary_bootstrap),
        ("primary_regime_bootstrap.csv", regime_summary),
        ("primary_regime_interaction_bootstrap.csv", regime_interaction),
        ("regime_assignments.csv", assignments),
        ("cross_lag_summary.csv", cross_lag),
        ("lagged_variogram_per_day_seed.csv", variogram_seed),
        ("lagged_variogram_per_day.csv", variogram_day),
        ("lagged_variogram_summary.csv", variogram_summary),
        ("lagged_variogram_shuffle_delta.csv", variogram_delta),
        ("atom_duration_summary.csv", atom),
        ("atom_event_per_day_seed.csv", atom_event_seed),
        ("atom_event_per_day.csv", atom_event_day),
        ("atom_event_primary_bootstrap.csv", atom_event_bootstrap),
        ("generated_transition_decomposition_per_day.csv", transition_day),
        ("generated_transition_decomposition_summary.csv", transition_summary),
        ("lag1_state_attribution_per_day_seed.csv", lag1_state_seed),
        ("lag1_state_attribution_per_day.csv", lag1_state_day),
        ("lag1_state_attribution_bootstrap.csv", lag1_state_bootstrap),
    ]
    for filename, rows in outputs:
        path = output_dir / filename
        _write_csv(path, rows)
        artifacts.append(path)

    matrix_path = output_dir / "cross_lag_matrices.npz"
    np.savez_compressed(matrix_path, **matrices)
    artifacts.append(matrix_path)

    threshold_path = output_dir / "nwp_regime_thresholds.json"
    _write_json(
        threshold_path,
        {
            "thresholds_by_panel": thresholds_by_panel,
            "reference_audit": reference_audit,
            "regime_rule": {
                "calm": "mean WS100 at or below the panel-reference q25",
                "strong": "mean WS100 at or above the panel-reference q75",
                "speed_volatile": "RMS hourly WS100 change at or above the panel-reference q75",
                "turning": "speed-weighted adjacent-vector turning angle at or above the panel-reference q75",
                "spatial_heterogeneous": "mean hourly cross-zone WS100 dispersion at or above the panel-reference q75",
            },
        },
    )
    artifacts.append(threshold_path)

    artifacts.extend(
        _make_plots(
            output_dir,
            bootstrap,
            cross_lag,
            atom,
            regime_summary,
            variogram_summary,
            transition_summary,
        )
    )

    panel_days = {
        panel: np.concatenate([_registry_days(registries[panel], outer) for outer in (1, 2, 3)])
        for panel in (PRIMARY_PANEL, SECONDARY_PANEL)
    }
    missing = np.asarray(raw["target_was_missing"], dtype=bool)
    missing_audit: dict[str, Any] = {
        "raw_all_calendar": {
            "missing_cells": int(missing.sum()),
            "affected_days": int(missing.any(axis=(1, 2)).sum()),
            "affected_zone_days": int(missing.any(axis=2).sum()),
        }
    }
    for panel, days in panel_days.items():
        indices = _raw_indices(np.asarray(raw["day"]), days)
        panel_missing = missing[indices]
        missing_audit[panel] = {
            "missing_cells": int(panel_missing.sum()),
            "affected_ramps": int(
                (panel_missing[..., :-1] | panel_missing[..., 1:]).sum()
            ),
            "affected_days": int(panel_missing.any(axis=(1, 2)).sum()),
            "affected_zone_days": int(panel_missing.any(axis=2).sum()),
            "clean_days_for_cross_lag": int((~panel_missing.any(axis=(1, 2))).sum()),
            "clean_zone_days_for_atom": int((~panel_missing.any(axis=2)).sum()),
        }

    summary = {
        "schema": "cross_model_architecture_diagnostic_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": root.as_posix(),
        "output_directory": output_dir.as_posix(),
        "panels": {
            PRIMARY_PANEL: {
                "role": "primary paired panel",
                "models": ["MM_mass (seeds 0-2)", "MM_none (seeds 0-2)", "DDPM_product (seed 0)"],
                "n_unique_days": 150,
                "date_sha256": date_sha256(panel_days[PRIMARY_PANEL]),
            },
            SECONDARY_PANEL: {
                "role": "secondary architecture/ablation panel; never paired or pooled with primary",
                "models": [
                    "STGF (seeds 0-2)",
                    "Time_domain (seed 0)",
                    "Time_frequency (seed 0)",
                    "Graph_only (seed 0)",
                    "DDPM_product (seed 0)",
                ],
                "n_unique_days": 150,
                "date_sha256": date_sha256(panel_days[SECONDARY_PANEL]),
            },
        },
        "reference_audit": reference_audit,
        "missing_target_sensitivity": missing_audit,
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "unit": "paired calendar day",
            "strata": "outer fold",
            "scenario_members_are_not_resampling_units": True,
            "secondary_pairing": "STGF is averaged across seeds 0-2 within each day before pairing; other secondary models and DDPM use frozen seed0",
            "regime_interaction": "paired MM-DDPM day gaps; regime-minus-complement bootstrap stratified by outer fold and fixed regime membership",
        },
        "interpretation_limits": [
            "Both panels are frozen internal confirmation panels, not external validation.",
            "Primary and secondary panels use disjoint dates and are never combined or cross-paired.",
            "DDPM scenarios are products of independently generated zone marginals; cross-zone and cross-lag diagnostics are descriptive only and cannot identify a diffusion-versus-flow effect.",
            "Generated-transition attribution is an exact additive decomposition of ordinary local ramp CRPS; category contributions may be negative although their sum is non-negative.",
            "The member-trajectory shuffle preserves each day-zone local path ensemble and is a negative control for cross-zone member coupling, not a competing forecast.",
            "Lag-1 state attribution is a descriptive accounting identity for correlation gaps, not a causal decomposition; only signed category gaps are additive.",
            "DDPM lag-1 categories are phenomenological output-boundary threshold labels and are not evidence of a latent discrete DDPM state.",
            "Raw missing targets were repository-consistently forward-filled; clean-cell/day/zone-day sensitivity outputs quantify their influence.",
        ],
        "row_counts": {
            "per_day_seed": len(seed_day_records),
            "per_day_seed_averaged": len(day_records),
            "primary_bootstrap": len(bootstrap),
            "secondary_bootstrap": len(secondary_bootstrap),
            "primary_regime_bootstrap": len(regime_summary),
            "primary_regime_interaction_bootstrap": len(regime_interaction),
            "cross_lag": len(cross_lag),
            "lagged_variogram_per_day_seed": len(variogram_seed),
            "lagged_variogram_per_day_seed_averaged": len(variogram_day),
            "lagged_variogram_summary": len(variogram_summary),
            "lagged_variogram_shuffle_delta": len(variogram_delta),
            "atom_duration": len(atom),
            "atom_event_per_day_seed": len(atom_event_seed),
            "atom_event_per_day_seed_averaged": len(atom_event_day),
            "atom_event_primary_bootstrap": len(atom_event_bootstrap),
            "generated_transition_per_day": len(transition_day),
            "generated_transition_summary": len(transition_summary),
            "lag1_state_attribution_per_day_seed": len(lag1_state_seed),
            "lag1_state_attribution_per_day_seed_averaged": len(lag1_state_day),
            "lag1_state_attribution_bootstrap": len(lag1_state_bootstrap),
        },
        "artifacts": [path.name for path in artifacts] + ["summary.json", "manifest.json", "SHA256SUMS"],
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    artifacts.append(summary_path)

    manifest_entries = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(artifacts)
    ]
    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "cross_model_diagnostics_output_manifest_v1",
            "note": "Manifest covers every listed diagnostic artifact; SHA256SUMS also covers this manifest.",
            "files": manifest_entries,
        },
    )
    checksum_entries = manifest_entries + [
        {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    ]
    checksums_path = output_dir / "SHA256SUMS"
    checksums_path.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in checksum_entries),
        encoding="ascii",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <repo>/outputs/cross_model_diagnostics)",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5000,
        help="stratified paired bootstrap repetitions (minimum 100; default: 5000)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs" / "cross_model_diagnostics"
    )
    summary = run(root, output_dir, args.bootstrap_repetitions)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
