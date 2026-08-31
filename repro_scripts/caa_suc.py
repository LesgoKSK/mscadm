"""Descriptive, paired RTS-24 SUC sensitivity for locked CAA outputs.

This runner is intentionally downstream of ``selection.lock.json`` and the
locked outer-test application.  It compares A0 with the selected/fallback A4
output, optionally adding A2.  It is not a CAA success gate and performs no
method selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from repro.suc import reduce_scenarios, rts24, solve_suc
from repro_scripts.run_caa_calibration import load_context, validate_caa_lock


DEFAULT_CONFIG = Path("repro_configs/caa_rahc_frozen.json")
SELECTION_LOCK_NAME = "selection.lock.json"
OUTER_TEST_MANIFEST_NAME = "caa_test_final_manifest.json"
DATE_RANK_NUMERATORS = (1, 2, 3)
DATE_RANK_DENOMINATOR = 4
WIND_CAPACITY_MW = 1200.0
DEFAULT_REDUCTION_SEED = 20260722
DEFAULT_ENVIRONMENT_SEED = 20260723
SOLVER_METRICS = (
    "total_cost",
    "startup_cost",
    "energy_cost",
    "penalty_cost",
    "wind_curtailment",
    "load_shedding",
)


@dataclass(frozen=True)
class LockedArchive:
    path: Path
    scenarios: np.ndarray
    observations: np.ndarray
    zone: np.ndarray
    day: np.ndarray
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _archive_path(root: Path, outer: int, family: str, seed: int) -> Path:
    return (
        root
        / f"outer{outer}"
        / "scenarios"
        / f"{family}_seed{seed}_test_final.npz"
    )


def _load_lock_on_disk(context: Any, validated: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(context.root) / SELECTION_LOCK_NAME
    if not path.is_file():
        raise RuntimeError(f"SUC is locked; missing selection lock: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != dict(validated):
        raise RuntimeError("validated selection lock differs from selection.lock.json")
    if stored.get("schema") != "caa_rahc_selection_lock_v1":
        raise RuntimeError("selection lock has the wrong CAA schema")
    selected = stored.get("selected")
    if not isinstance(selected, dict) or "main_A4" not in selected:
        raise RuntimeError("selection lock lacks the main_A4 decision")
    return stored, _sha256_file(path)


def _registry_by_path(manifest: Mapping[str, Any], directory: Path) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError(f"outer-test manifest lacks a file registry: {directory}")
    registry: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RuntimeError("outer-test manifest contains a malformed file record")
        relative = str(record["path"])
        if relative in registry:
            raise RuntimeError(f"duplicate outer-test registry path: {relative}")
        resolved = (directory / relative).resolve()
        try:
            resolved.relative_to(directory.resolve())
        except ValueError as error:
            raise RuntimeError("outer-test registry path escapes its directory") from error
        registry[relative] = record
    return registry


def _load_outer_manifest(
    root: Path,
    *,
    outer: int,
    families: Sequence[str],
    seeds: Sequence[int],
    lock_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    directory = root / f"outer{outer}" / "scenarios"
    path = directory / OUTER_TEST_MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError(f"outer{outer} locked test-output manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "caa_rahc_locked_test_outputs_v1":
        raise RuntimeError(f"outer{outer} locked test-output manifest has the wrong schema")
    if int(manifest.get("outer", -1)) != int(outer):
        raise RuntimeError(f"outer{outer} test-output manifest outer mismatch")
    if manifest.get("selection_lock_sha256") != lock_sha256:
        raise RuntimeError(f"outer{outer} test-output manifest selection-lock hash mismatch")
    available = manifest.get("families")
    if not isinstance(available, list) or not set(families).issubset(set(available)):
        raise RuntimeError(f"outer{outer} test-output manifest lacks a requested family")
    registry = _registry_by_path(manifest, directory)
    for family in families:
        for seed in seeds:
            relative = f"{family}_seed{seed}_test_final.npz"
            if relative not in registry:
                raise RuntimeError(
                    f"outer{outer} test-output registry lacks {family} seed{seed}"
                )
    return registry


def _metadata_object(value: np.ndarray, path: Path) -> dict[str, Any]:
    array = np.asarray(value)
    if array.ndim != 0:
        raise RuntimeError(f"archive metadata must be a scalar JSON string: {path}")
    parsed = json.loads(str(array.item()))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"archive metadata JSON must contain an object: {path}")
    return parsed


def _load_archive(
    path: Path,
    *,
    outer: int,
    family: str,
    seed: int,
    lock_sha256: str,
    registry_record: Mapping[str, Any],
    expected_members: int | None,
) -> LockedArchive:
    if not path.is_file():
        raise RuntimeError(f"locked outer-test archive is missing: {path}")
    expected_bytes = registry_record.get("bytes")
    expected_hash = registry_record.get("sha256")
    if int(expected_bytes) != int(path.stat().st_size) or expected_hash != _sha256_file(path):
        raise RuntimeError(f"locked outer-test archive is size/hash drifted: {path}")
    with np.load(path, allow_pickle=False) as stored:
        required = {"scenarios", "observations", "zone", "day", "metadata"}
        if not required.issubset(stored.files):
            raise RuntimeError(f"locked archive lacks required arrays: {path}")
        scenarios = np.asarray(stored["scenarios"], dtype=np.float64).copy()
        observations = np.asarray(stored["observations"], dtype=np.float64).copy()
        zone = np.asarray(stored["zone"]).copy()
        day = np.asarray(stored["day"]).copy()
        metadata = _metadata_object(stored["metadata"], path)
    cases = scenarios.shape[0] if scenarios.ndim == 3 else -1
    if scenarios.ndim != 3 or scenarios.shape[2] != 24:
        raise RuntimeError(f"scenarios must have shape [case, member, 24]: {path}")
    if observations.shape != (cases, 24):
        raise RuntimeError(f"observations must have shape [case, 24]: {path}")
    if zone.shape != (cases,) or day.shape != (cases,):
        raise RuntimeError(f"zone/day must each have shape [case]: {path}")
    if expected_members is not None and scenarios.shape[1] != expected_members:
        raise RuntimeError(
            f"{path} has {scenarios.shape[1]} members, expected {expected_members}"
        )
    if not np.isfinite(scenarios).all() or np.any((scenarios < 0.0) | (scenarios > 1.0)):
        raise RuntimeError(f"archive scenarios are non-finite or outside [0,1]: {path}")
    if not np.isfinite(observations).all() or np.any(
        (observations < 0.0) | (observations > 1.0)
    ):
        raise RuntimeError(f"archive observations are non-finite or outside [0,1]: {path}")
    expected_metadata = {
        "schema": "caa_rahc_locked_test_scenarios_v1",
        "outer": int(outer),
        "model_seed": int(seed),
        "family": family,
        "selection_lock_sha256": lock_sha256,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"{path} metadata mismatch for {key}")
    return LockedArchive(path, scenarios, observations, zone, day, metadata)


def _load_outer_archives(
    root: Path,
    *,
    outer: int,
    families: Sequence[str],
    seeds: Sequence[int],
    lock_sha256: str,
    expected_members: int | None,
) -> dict[str, dict[int, LockedArchive]]:
    registry = _load_outer_manifest(
        root,
        outer=outer,
        families=families,
        seeds=seeds,
        lock_sha256=lock_sha256,
    )
    archives: dict[str, dict[int, LockedArchive]] = {}
    for family in families:
        archives[family] = {}
        for seed in seeds:
            path = _archive_path(root, outer, family, seed)
            archives[family][seed] = _load_archive(
                path,
                outer=outer,
                family=family,
                seed=seed,
                lock_sha256=lock_sha256,
                registry_record=registry[path.name],
                expected_members=expected_members,
            )
    reference = archives["A0"][seeds[0]]
    for family in families:
        for seed in seeds:
            archive = archives[family][seed]
            if archive.scenarios.shape != reference.scenarios.shape:
                raise RuntimeError(
                    f"outer{outer} scenario count differs across method/seed archives"
                )
            if not np.array_equal(archive.zone, reference.zone):
                raise RuntimeError(f"outer{outer} zone metadata is not paired")
            if not np.array_equal(archive.day, reference.day):
                raise RuntimeError(f"outer{outer} day metadata is not paired")
            if not np.array_equal(archive.observations, reference.observations):
                raise RuntimeError(f"outer{outer} realized observations are not paired")
    return archives


def _normalized_days(values: np.ndarray) -> np.ndarray:
    try:
        days = np.asarray(values).astype("datetime64[D]")
    except (TypeError, ValueError) as error:
        raise RuntimeError("outer-test day metadata is not calendar-date compatible") from error
    if np.any(np.isnat(days)):
        raise RuntimeError("outer-test day metadata contains NaT")
    return days


def _representative_cases(day: np.ndarray, zone: np.ndarray, *, outer: int) -> list[dict[str, Any]]:
    days = _normalized_days(day)
    unique_days = np.unique(days)
    if len(unique_days) < 5:
        raise RuntimeError(f"outer{outer} needs at least five dates for the frozen rank rule")
    date_ranks = [
        int((len(unique_days) - 1) * numerator // DATE_RANK_DENOMINATOR)
        for numerator in DATE_RANK_NUMERATORS
    ]
    if len(set(date_ranks)) != len(DATE_RANK_NUMERATORS):
        raise RuntimeError(f"outer{outer} frozen date ranks are not distinct")
    cases: list[dict[str, Any]] = []
    zones = np.asarray(zone)
    for slot, date_rank in enumerate(date_ranks):
        selected_day = unique_days[date_rank]
        matching = np.flatnonzero(days == selected_day)
        if len(matching) == 0:
            raise AssertionError("selected calendar date has no archive row")
        zone_order = np.argsort(zones[matching], kind="stable")
        ordered = matching[zone_order]
        archive_index = int(ordered[(len(ordered) - 1) // 2])
        cases.append(
            {
                "outer": int(outer),
                "representative_slot": int(slot),
                "date_rank": int(date_rank),
                "available_date_count": int(len(unique_days)),
                "archive_index": archive_index,
                "calendar_date": np.datetime_as_string(selected_day, unit="D"),
                "zone": _jsonable(zones[archive_index]),
            }
        )
    return cases


def _paired_seed(base: int, outer: int, slot: int) -> int:
    return int(base) + int(outer) * 1000 + int(slot)


def _pooled_scenarios(
    archives: Mapping[int, LockedArchive], seeds: Sequence[int], archive_index: int
) -> np.ndarray:
    members = {int(archives[seed].scenarios.shape[1]) for seed in seeds}
    if len(members) != 1:
        raise RuntimeError("model seeds cannot be equally pooled: member counts differ")
    return np.concatenate(
        [archives[seed].scenarios[archive_index] for seed in seeds], axis=0
    )


def _hit_time_limit(status: str) -> bool:
    normalized = status.casefold().replace("_", " ").replace("-", " ")
    return "time limit" in normalized


def _failed_stage(
    stage: str, error: BaseException, time_limit: float, elapsed: float
) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{stage}_success": False,
        f"{stage}_status": "ERROR",
        f"{stage}_solve_seconds": float(elapsed),
        f"{stage}_time_limit_seconds": float(time_limit),
        f"{stage}_hit_time_limit": False,
        f"{stage}_error_type": type(error).__name__,
        f"{stage}_error_message": str(error),
    }
    for metric in SOLVER_METRICS:
        result[f"{stage}_{metric}"] = np.nan
    return result


def _not_run_realized(time_limit: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "realized_success": False,
        "realized_status": "NOT_RUN_PLANNED_FAILED",
        "realized_solve_seconds": 0.0,
        "realized_time_limit_seconds": float(time_limit),
        "realized_hit_time_limit": False,
        "realized_error_type": "",
        "realized_error_message": "planned solve failed; realized solve not attempted",
    }
    for metric in SOLVER_METRICS:
        result[f"realized_{metric}"] = np.nan
    return result


def _solve_stage(
    stage: str,
    wind: np.ndarray,
    probabilities: np.ndarray,
    *,
    system: Any,
    time_limit: float,
    mip_gap: float,
    shedding_penalty: float,
    curtailment_penalty: float,
    fixed_commitment: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    started = time.perf_counter()
    try:
        result = solve_suc(
            wind,
            probabilities,
            system=system,
            fixed_commitment=fixed_commitment,
            shedding_penalty=shedding_penalty,
            curtailment_penalty=curtailment_penalty,
            mip_gap=mip_gap,
            time_limit=time_limit,
        )
        if not isinstance(result, Mapping):
            raise TypeError("solve_suc must return a mapping")
        status = str(result.get("status", "MISSING_STATUS"))
        fields: dict[str, Any] = {
            f"{stage}_success": True,
            f"{stage}_status": status,
            f"{stage}_solve_seconds": float(time.perf_counter() - started),
            f"{stage}_time_limit_seconds": float(time_limit),
            f"{stage}_hit_time_limit": _hit_time_limit(status),
            f"{stage}_error_type": "",
            f"{stage}_error_message": "",
        }
        for metric in SOLVER_METRICS:
            value = float(result[metric])
            if not np.isfinite(value):
                raise FloatingPointError(f"solve_suc returned non-finite {metric}")
            fields[f"{stage}_{metric}"] = value
        commitment: np.ndarray | None = None
        if stage == "planned":
            if "commitment" not in result:
                raise KeyError("planned solve result lacks commitment")
            commitment = np.asarray(result["commitment"], dtype=np.float64)
            if commitment.shape != (12, 24) or not np.isfinite(commitment).all():
                raise ValueError("planned commitment must be finite with shape [12,24]")
        return fields, commitment
    except Exception as error:  # recorded by design; the paired run continues
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": stage,
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return _failed_stage(stage, error, time_limit, elapsed), None


def _summarize(frame: pd.DataFrame, families: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in families:
        selected = frame.loc[frame["family"] == family]
        row: dict[str, Any] = {
            "family": family,
            "paired_cases": int(len(selected)),
            "successful_cases": int(
                np.sum(selected["planned_success"] & selected["realized_success"])
            ),
            "planned_failed_cases": int(np.sum(~selected["planned_success"])),
            "realized_failed_cases": int(np.sum(~selected["realized_success"])),
            "planned_time_limit_status_cases": int(np.sum(selected["planned_hit_time_limit"])),
            "realized_time_limit_status_cases": int(np.sum(selected["realized_hit_time_limit"])),
            "mean_planned_solve_seconds": float(selected["planned_solve_seconds"].mean()),
            "mean_realized_solve_seconds": float(selected["realized_solve_seconds"].mean()),
        }
        for stage in ("planned", "realized"):
            for metric in SOLVER_METRICS:
                row[f"mean_{stage}_{metric}"] = float(
                    selected[f"{stage}_{metric}"].mean(skipna=True)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _assert_complete_pairing(frame: pd.DataFrame, families: Sequence[str]) -> None:
    key = ["outer", "representative_slot"]
    invariant = [
        "archive_index",
        "calendar_date",
        "zone",
        "pooled_scenario_count",
        "reduced_scenario_count",
        "reduction_seed",
        "environment_seed",
        "clusters",
        "time_limit_seconds",
        "mip_gap",
        "shedding_penalty",
        "curtailment_penalty",
    ]
    expected = set(families)
    for pair, group in frame.groupby(key, sort=False):
        if set(group["family"]) != expected or len(group) != len(expected):
            raise AssertionError(f"incomplete method pairing for outer/slot {pair}")
        for column in invariant:
            if group[column].nunique(dropna=False) != 1:
                raise AssertionError(f"paired methods differ in {column} for {pair}")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output blocks write: {temporary}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output blocks write: {temporary}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_sensitivity(
    config: str | Path = DEFAULT_CONFIG,
    *,
    include_a2: bool = False,
    clusters: int = 10,
    time_limit: float = 120.0,
    mip_gap: float = 0.01,
    shedding_penalty: float = 1000.0,
    curtailment_penalty: float = 80.0,
    reduction_seed: int = DEFAULT_REDUCTION_SEED,
    environment_seed: int = DEFAULT_ENVIRONMENT_SEED,
) -> dict[str, Any]:
    """Run the frozen descriptive sensitivity; solver failures remain in outputs."""

    if clusters < 1:
        raise ValueError("clusters must be positive")
    if time_limit <= 0.0:
        raise ValueError("time_limit must be positive")
    if mip_gap < 0.0 or not np.isfinite(mip_gap):
        raise ValueError("mip_gap must be finite and non-negative")
    if shedding_penalty < 0.0 or curtailment_penalty < 0.0:
        raise ValueError("solver penalties must be non-negative")

    # The full frozen-experiment validator runs before any outer-test archive
    # path is constructed or opened.
    context = load_context(config)
    validated_lock = validate_caa_lock(context)
    lock, lock_sha256 = _load_lock_on_disk(context, validated_lock)
    root = Path(context.root)
    outers = tuple(sorted(int(value) for value in context.config["outer_splits"]))
    seeds = tuple(sorted(int(value) for value in context.config["model_seeds"]))
    if len(outers) != 3 or len(set(outers)) != 3:
        raise RuntimeError("the frozen SUC protocol requires exactly three outer splits")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("the frozen SUC protocol requires exactly three model seeds")
    families = ("A0", "A4", "A2") if include_a2 else ("A0", "A4")
    if include_a2 and "A2" not in lock["selected"]:
        raise RuntimeError("selection lock lacks the optional A2 decision")
    expected_members_value = context.config.get("sampling", {}).get("scenarios")
    expected_members = (
        int(expected_members_value) if expected_members_value is not None else None
    )

    records: list[dict[str, Any]] = []
    selected_cases: list[dict[str, Any]] = []
    archive_sources: dict[str, Any] = {}
    for outer in outers:
        archives = _load_outer_archives(
            root,
            outer=outer,
            families=families,
            seeds=seeds,
            lock_sha256=lock_sha256,
            expected_members=expected_members,
        )
        reference = archives["A0"][seeds[0]]
        cases = _representative_cases(reference.day, reference.zone, outer=outer)
        selected_cases.extend(cases)
        archive_sources[f"outer{outer}"] = {
            family: {
                f"seed{seed}": {
                    "path": str(archives[family][seed].path.resolve()),
                    "sha256": _sha256_file(archives[family][seed].path),
                }
                for seed in seeds
            }
            for family in families
        }
        for case in cases:
            archive_index = int(case["archive_index"])
            paired_reduction_seed = _paired_seed(
                reduction_seed, outer, int(case["representative_slot"])
            )
            paired_environment_seed = _paired_seed(
                environment_seed, outer, int(case["representative_slot"])
            )
            # rts24() is deterministic.  Reusing this exact object makes the
            # load curve and generator price coefficients identical by method.
            system = rts24()
            common_observation = reference.observations[archive_index]
            pooled_by_family: dict[str, np.ndarray] = {}
            reduced_by_family: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            pooled_counts: set[int] = set()
            for family in families:
                pooled = _pooled_scenarios(
                    archives[family], seeds, archive_index
                )
                pooled_by_family[family] = pooled
                pooled_counts.add(len(pooled))
                if clusters > len(pooled):
                    raise ValueError("clusters exceeds the equally pooled scenario count")
                reduced_by_family[family] = reduce_scenarios(
                    pooled,
                    clusters,
                    capacity=WIND_CAPACITY_MW,
                    seed=paired_reduction_seed,
                )
            if len(pooled_counts) != 1:
                raise RuntimeError("paired methods have different pooled scenario counts")
            pooled_count = pooled_counts.pop()
            for family in families:
                reduced, probabilities = reduced_by_family[family]
                base: dict[str, Any] = {
                    "family": family,
                    **case,
                    "model_seeds": ",".join(str(seed) for seed in seeds),
                    "seed_pooling": "equal members per seed, concatenated before reduction",
                    "source_members_per_seed": int(pooled_by_family[family].shape[0] // len(seeds)),
                    "pooled_scenario_count": int(pooled_count),
                    "reduced_scenario_count": int(len(reduced)),
                    "clusters": int(clusters),
                    "reduction_seed": int(paired_reduction_seed),
                    "environment_seed": int(paired_environment_seed),
                    "environment_seed_consumed": False,
                    "wind_capacity_mw": float(WIND_CAPACITY_MW),
                    "time_limit_seconds": float(time_limit),
                    "mip_gap": float(mip_gap),
                    "shedding_penalty": float(shedding_penalty),
                    "curtailment_penalty": float(curtailment_penalty),
                }
                planned, commitment = _solve_stage(
                    "planned",
                    reduced,
                    probabilities,
                    system=system,
                    time_limit=time_limit,
                    mip_gap=mip_gap,
                    shedding_penalty=shedding_penalty,
                    curtailment_penalty=curtailment_penalty,
                )
                if commitment is None:
                    realized = _not_run_realized(time_limit)
                else:
                    realized, _ = _solve_stage(
                        "realized",
                        common_observation[None, :] * WIND_CAPACITY_MW,
                        np.ones(1, dtype=np.float64),
                        system=system,
                        fixed_commitment=commitment,
                        time_limit=time_limit,
                        mip_gap=mip_gap,
                        shedding_penalty=shedding_penalty,
                        curtailment_penalty=curtailment_penalty,
                    )
                record = {**base, **planned, **realized}
                records.append(record)
                print(json.dumps(_jsonable(record), ensure_ascii=False), flush=True)

    daily = pd.DataFrame(records)
    _assert_complete_pairing(daily, families)
    summary = _summarize(daily, families)
    output = root / "suc"
    output.mkdir(parents=True, exist_ok=True)
    daily_path = output / "daily_results.csv"
    summary_path = output / "summary.csv"
    protocol_path = output / "protocol.json"
    _write_csv_atomic(daily_path, daily)
    _write_csv_atomic(summary_path, summary)
    failures = int(np.sum(~(daily["planned_success"] & daily["realized_success"])))
    protocol = {
        "schema": "caa_rahc_descriptive_suc_v1",
        "experiment": "locked CAA-RAHC outer-test RTS-24 SUC sensitivity",
        "statistical_scope": (
            "Descriptive sensitivity only; this is not an inferential comparison and "
            "is not a primary or secondary CAA success gate."
        ),
        "success_gate_role": "none",
        "selection_lock": {
            "path": str((root / SELECTION_LOCK_NAME).resolve()),
            "sha256": lock_sha256,
            "schema": lock["schema"],
        },
        "families": list(families),
        "locked_decisions": {
            "A0": {"selected": "A0", "fallback": False},
            "A4": lock["selected"]["main_A4"],
            **({"A2": lock["selected"]["A2"]} if include_a2 else {}),
        },
        "outer_splits": list(outers),
        "model_seeds": list(seeds),
        "representative_case_selection": {
            "cases_per_outer": 3,
            "total_cases": len(selected_cases),
            "unique_calendar_date_count": len(
                {str(case["calendar_date"]) for case in selected_cases}
            ),
            "date_sort": "ascending unique calendar date",
            "date_rank_rule": (
                "floor((number_of_unique_dates - 1) * numerator / 4), "
                "numerators fixed to [1,2,3]"
            ),
            "within_date_rule": (
                "sort rows by zone with stable archive-order ties; choose lower median row"
            ),
            "uses_method_performance": False,
            "cases": selected_cases,
        },
        "seed_pooling": {
            "rule": "equal-weight model-seed pooling by concatenation before K-means",
            "guard": "all three seeds and all methods must have identical member counts",
            "pooled_scenario_count": sorted(
                {int(value) for value in daily["pooled_scenario_count"]}
            ),
        },
        "paired_design": {
            "same_outer_date_zone_case": True,
            "same_realized_observation": True,
            "same_source_and_reduced_scenario_counts": True,
            "same_kmeans_seed_within_case": True,
            "same_rts24_system_object_within_case": True,
            "same_solver_parameters_within_case": True,
            "demand_and_prices": (
                "deterministic repro.suc.rts24 load and generator energy costs; "
                "no random perturbation is introduced"
            ),
            "environment_seed_base": int(environment_seed),
            "environment_seed_consumed": False,
        },
        "scenario_reduction": {
            "clusters": int(clusters),
            "capacity_mw": float(WIND_CAPACITY_MW),
            "seed_base": int(reduction_seed),
            "seed_rule": "base + outer*1000 + representative_slot; identical by method",
            "capacity_interpretation": (
                "one selected normalized GEFCom zone case is scaled to a 1200 MW "
                "single-zone system proxy; zones are not aggregated"
            ),
        },
        "solver": {
            "implementation": "repro.suc.solve_suc",
            "time_limit_seconds_per_solve": float(time_limit),
            "mip_gap": float(mip_gap),
            "shedding_penalty": float(shedding_penalty),
            "curtailment_penalty": float(curtailment_penalty),
            "failure_policy": (
                "solver exceptions are recorded with type/message and NaN metrics; "
                "a failed planned solve skips realized dispatch; the paired run continues"
            ),
            "failed_method_cases": failures,
        },
        "archives": archive_sources,
        "outputs": {
            "daily_results": str(daily_path.resolve()),
            "summary": str(summary_path.resolve()),
            "protocol": str(protocol_path.resolve()),
        },
    }
    _write_json_atomic(protocol_path, protocol)
    if failures:
        print(
            f"WARNING: {failures} method-case SUC records contain a failed solve; "
            "see daily_results.csv",
            file=sys.stderr,
            flush=True,
        )
    return {
        "daily": daily,
        "summary": summary,
        "protocol": protocol,
        "paths": {
            "daily": daily_path,
            "summary": summary_path,
            "protocol": protocol_path,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the descriptive paired SUC sensitivity on locked CAA outputs"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-a2", action="store_true")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--shedding-penalty", type=float, default=1000.0)
    parser.add_argument("--curtailment-penalty", type=float, default=80.0)
    parser.add_argument("--reduction-seed", type=int, default=DEFAULT_REDUCTION_SEED)
    parser.add_argument("--environment-seed", type=int, default=DEFAULT_ENVIRONMENT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_sensitivity(
        args.config,
        include_a2=args.include_a2,
        clusters=args.clusters,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        shedding_penalty=args.shedding_penalty,
        curtailment_penalty=args.curtailment_penalty,
        reduction_seed=args.reduction_seed,
        environment_seed=args.environment_seed,
    )
    print(result["summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DATE_RANK_DENOMINATOR",
    "DATE_RANK_NUMERATORS",
    "LockedArchive",
    "main",
    "run_sensitivity",
]
