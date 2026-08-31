"""Fail-closed evaluation of sealed, locked CAA outer-test artifacts.

The runner validates the complete locked-test registry before scoring A0--A6.
Conditional groups are loaded from the persisted test-raw assignments fitted
under the calibration-only grouping protocol; method scenarios never define
their own groups.  No training, calibration, selection, or scenario generation
is performed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from caa_rahc.atom_diagnostics import (
    ensemble_atom_quantization,
    pooled_atom_diagnostics,
)
from caa_rahc.metrics import analytic_atom_scores
from caa_rahc.outer_evaluation import evaluate_outer_tests
from rahc.group_metrics import (
    GROUP_FAMILIES,
    group_interval_metrics,
    summarize_group_metrics,
)
from repro_scripts.run_caa_calibration import load_context, validate_caa_lock


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "repro_configs" / "caa_rahc_frozen.json"
SELECTION_LOCK_NAME = "selection.lock.json"
TEST_MANIFEST_NAME = "caa_test_final_manifest.json"
FAMILIES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
ATOM_FAMILIES = ("A2", "A4", "A5", "A6")
OUTERS = (1, 2, 3)
SEEDS = (0, 1, 2)
EXPECTED_DATES_PER_OUTER = 50
EXPECTED_TOTAL_DATES = 150


class DigestCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def __call__(self, path: str | Path) -> str:
        value = Path(path)
        stat = value.stat()
        key = (str(value.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        if key not in self._cache:
            digest = hashlib.sha256()
            with value.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self._cache[key] = digest.hexdigest()
        return self._cache[key]


@dataclass(frozen=True)
class ScenarioArtifact:
    path: Path
    scenarios: np.ndarray
    observations: np.ndarray
    zone: np.ndarray
    day: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AtomArtifact:
    path: Path
    arrays: dict[str, np.ndarray]
    observations: np.ndarray
    zone: np.ndarray
    day: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OuterArtifacts:
    outer: int
    scenarios: dict[str, dict[int, ScenarioArtifact]]
    observations: np.ndarray
    zone: np.ndarray
    day: np.ndarray
    assignments: dict[str, np.ndarray]
    assignment_metadata: dict[str, Any]
    analytic_atoms: dict[int, AtomArtifact]
    diagnostics: dict[tuple[str, int], dict[str, Any]]
    manifest: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class FrozenAssignmentProtocol:
    """Adapter that makes persisted assignments usable by group_metrics."""

    assignments: Mapping[str, np.ndarray]
    expected_zone: np.ndarray

    def assign(self, _grouping_raw_scenarios: np.ndarray, zone: np.ndarray) -> dict[str, np.ndarray]:
        if not np.array_equal(np.asarray(zone), self.expected_zone):
            raise RuntimeError("frozen assignments were requested with different zone metadata")
        return {name: np.asarray(values).copy() for name, values in self.assignments.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _metadata(value: np.ndarray, *, source: Path) -> dict[str, Any]:
    array = np.asarray(value)
    if array.ndim != 0:
        raise RuntimeError(f"metadata must be a scalar JSON string: {source}")
    result = json.loads(str(array.item()))
    if not isinstance(result, dict):
        raise RuntimeError(f"metadata JSON must contain an object: {source}")
    return result


def _scenario_path(root: Path, outer: int, family: str, seed: int) -> Path:
    return root / f"outer{outer}" / "scenarios" / f"{family}_seed{seed}_test_final.npz"


def _atom_path(root: Path, outer: int, seed: int) -> Path:
    return root / f"outer{outer}" / "scenarios" / f"analytic_atoms_seed{seed}_test_final.npz"


def _audit_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".audit.json")


def _load_validated_lock(context: Any, digest: DigestCache) -> tuple[dict[str, Any], str]:
    validated = validate_caa_lock(context)
    path = Path(context.root) / SELECTION_LOCK_NAME
    if not path.is_file():
        raise RuntimeError(f"evaluation is locked; missing selection lock: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or stored != dict(validated):
        raise RuntimeError("validated lock differs from selection.lock.json")
    if stored.get("schema") != "caa_rahc_selection_lock_v1":
        raise RuntimeError("selection lock has the wrong schema")
    return stored, digest(path)


def _normalized_decision(family: str, value: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(value.get("selected", value.get("family", family)))
    fallback = bool(value.get("fallback", selected == "A0" and family != "A0"))
    if fallback != (selected == "A0" and family != "A0"):
        raise RuntimeError(f"selection lock has inconsistent fallback state for {family}")
    return {"selected": selected, "fallback": fallback, "payload": dict(value)}


def _resolve_decisions(lock: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    if isinstance(lock.get("selected"), Mapping):
        source = "selected"
        stored = lock["selected"]
        decisions: dict[str, dict[str, Any]] = {
            "A0": {"selected": "A0", "fallback": False, "payload": {"family": "A0"}}
        }
        for family in FAMILIES[1:]:
            key = "main_A4" if family == "A4" else family
            value = stored.get(key)
            if not isinstance(value, Mapping):
                raise RuntimeError(f"selection lock lacks decision {key}")
            decisions[family] = _normalized_decision(family, value)
        return decisions, source
    if isinstance(lock.get("selected_configs"), Mapping):
        source = "selected_configs"
        stored = lock["selected_configs"]
        if set(stored) != set(FAMILIES):
            raise RuntimeError("selected_configs must contain exactly A0--A6")
        return {
            family: _normalized_decision(family, stored[family])
            for family in FAMILIES
        }, source
    raise RuntimeError("selection lock has neither selected nor selected_configs")


def _validate_registry(
    directory: Path, records: Any, digest: DigestCache
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        raise RuntimeError("locked test manifest has no file registry")
    root = directory.resolve()
    registry: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RuntimeError("locked test registry contains a malformed record")
        relative = str(record["path"])
        if relative in registry:
            raise RuntimeError(f"locked test registry duplicates {relative}")
        path = (directory / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("locked test registry path escapes its directory") from error
        if not path.is_file():
            raise RuntimeError(f"registered locked test artifact is missing: {path}")
        if int(record.get("bytes", -1)) != int(path.stat().st_size):
            raise RuntimeError(f"registered locked test artifact size drift: {path}")
        if record.get("sha256") != digest(path):
            raise RuntimeError(f"registered locked test artifact hash drift: {path}")
        registry[relative] = record
    return registry


def _load_json_registered(path: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if path.name not in registry:
        raise RuntimeError(f"locked test registry omits {path.name}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"registered JSON is not an object: {path}")
    return result


def _validate_sidecar(
    path: Path,
    registry: Mapping[str, Any],
    *,
    digest: DigestCache,
    lock_sha256: str,
    schema: str,
) -> dict[str, Any]:
    sidecar_path = _audit_path(path)
    audit = _load_json_registered(sidecar_path, registry)
    if audit.get("schema") != schema:
        raise RuntimeError(f"audit sidecar schema mismatch: {sidecar_path}")
    if audit.get("archive_sha256") != digest(path):
        raise RuntimeError(f"audit sidecar archive hash mismatch: {path}")
    if audit.get("selection_lock_sha256") != lock_sha256:
        raise RuntimeError(f"audit sidecar selection-lock hash mismatch: {path}")
    return audit


def _load_scenario(
    path: Path,
    registry: Mapping[str, Any],
    *,
    outer: int,
    family: str,
    seed: int,
    lock_sha256: str,
    decision: Mapping[str, Any],
    digest: DigestCache,
) -> ScenarioArtifact:
    if path.name not in registry:
        raise RuntimeError(f"locked test registry omits {path.name}")
    _validate_sidecar(
        path,
        registry,
        digest=digest,
        lock_sha256=lock_sha256,
        schema="caa_rahc_locked_test_scenario_audit_v1",
    )
    with np.load(path, allow_pickle=False) as stored:
        required = {"scenarios", "observations", "zone", "day", "metadata"}
        if not required.issubset(stored.files):
            raise RuntimeError(f"locked scenario archive is incomplete: {path}")
        scenarios = np.asarray(stored["scenarios"], dtype=np.float64).copy()
        observations = np.asarray(stored["observations"], dtype=np.float64).copy()
        zone = np.asarray(stored["zone"]).copy()
        day = np.asarray(stored["day"]).copy()
        metadata = _metadata(stored["metadata"], source=path)
    if scenarios.ndim != 3 or scenarios.shape[2] != 24:
        raise RuntimeError(f"scenario archive must be [case,member,24]: {path}")
    cases = len(scenarios)
    if observations.shape != (cases, 24) or zone.shape != (cases,) or day.shape != (cases,):
        raise RuntimeError(f"scenario archive arrays do not align: {path}")
    if not np.isfinite(scenarios).all() or np.any((scenarios < 0.0) | (scenarios > 1.0)):
        raise RuntimeError(f"scenario archive is non-finite or outside [0,1]: {path}")
    if not np.isfinite(observations).all() or np.any(
        (observations < 0.0) | (observations > 1.0)
    ):
        raise RuntimeError(f"observation archive is non-finite or outside [0,1]: {path}")
    expected = {
        "schema": "caa_rahc_locked_test_scenarios_v1",
        "outer": int(outer),
        "model_seed": int(seed),
        "family": family,
        "selection_lock_sha256": lock_sha256,
        "selected": decision["selected"],
        "fallback": bool(decision["fallback"]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"{path} metadata mismatch for {key}")
    return ScenarioArtifact(path, scenarios, observations, zone, day, metadata)


def _load_assignments(
    path: Path,
    registry: Mapping[str, Any],
    *,
    outer: int,
    cases: int,
    hours: int,
    lock_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path.name not in registry:
        raise RuntimeError(f"locked test registry omits {path.name}")
    with np.load(path, allow_pickle=False) as stored:
        expected = {*GROUP_FAMILIES, "metadata"}
        if set(stored.files) != expected:
            raise RuntimeError("group assignment archive has the wrong array catalog")
        assignments = {
            family: np.asarray(stored[family]).copy() for family in GROUP_FAMILIES
        }
        metadata = _metadata(stored["metadata"], source=path)
    for family, values in assignments.items():
        if values.shape != (cases, hours):
            raise RuntimeError(f"group assignment {family} has the wrong shape")
        if not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError(f"group assignment {family} is not integer-valued")
    expected_metadata = {
        "outer": int(outer),
        "fit_split": "calibration raw only",
        "assignment_split": "test raw",
        "selection_lock_sha256": lock_sha256,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"group assignment metadata mismatch for {key}")
    if not isinstance(metadata.get("grouping_protocol_sha256"), str):
        raise RuntimeError("group assignment lacks frozen grouping protocol hash")
    return assignments, metadata


def _load_atoms(
    path: Path,
    registry: Mapping[str, Any],
    *,
    outer: int,
    seed: int,
    lock_sha256: str,
    digest: DigestCache,
) -> AtomArtifact:
    if path.name not in registry:
        raise RuntimeError(f"locked test registry omits {path.name}")
    _validate_sidecar(
        path,
        registry,
        digest=digest,
        lock_sha256=lock_sha256,
        schema="caa_rahc_test_analytic_atoms_audit_v1",
    )
    expected_arrays = {
        "structural_pi0",
        "structural_pi1",
        *{
            f"{family}_{suffix}"
            for family in ATOM_FAMILIES
            for suffix in ("pi0", "pi1")
        },
    }
    with np.load(path, allow_pickle=False) as stored:
        required = {*expected_arrays, "observations", "zone", "day", "metadata"}
        if set(stored.files) != required:
            raise RuntimeError(f"analytic atom archive has the wrong array catalog: {path}")
        arrays = {name: np.asarray(stored[name], dtype=np.float64).copy() for name in expected_arrays}
        observations = np.asarray(stored["observations"], dtype=np.float64).copy()
        zone = np.asarray(stored["zone"]).copy()
        day = np.asarray(stored["day"]).copy()
        metadata = _metadata(stored["metadata"], source=path)
    for family in ("structural", *ATOM_FAMILIES):
        p0 = arrays[f"{family}_pi0"]
        p1 = arrays[f"{family}_pi1"]
        if p0.shape != observations.shape or p1.shape != observations.shape:
            raise RuntimeError(f"analytic atom {family} arrays do not align: {path}")
        if (
            not np.isfinite(p0).all()
            or not np.isfinite(p1).all()
            or np.any(p0 < 0.0)
            or np.any(p1 < 0.0)
            or np.any(p0 + p1 > 1.0 + 1e-12)
        ):
            raise RuntimeError(f"analytic atom {family} probabilities are invalid: {path}")
    expected_metadata = {
        "schema": "caa_rahc_test_analytic_atoms_v1",
        "outer": int(outer),
        "model_seed": int(seed),
        "selection_lock_sha256": lock_sha256,
        "families": list(ATOM_FAMILIES),
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"analytic atom metadata mismatch for {key}: {path}")
    return AtomArtifact(path, arrays, observations, zone, day, metadata)


def _diagnostic_catalog(
    path: Path,
    registry: Mapping[str, Any],
    *,
    outer: int,
    lock_sha256: str,
    assignment_metadata: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    payload = _load_json_registered(path, registry)
    if payload.get("schema") != "caa_rahc_locked_test_application_audit_v1":
        raise RuntimeError(f"outer{outer} test diagnostics has the wrong schema")
    if payload.get("outer") != int(outer) or payload.get("selection_lock_sha256") != lock_sha256:
        raise RuntimeError(f"outer{outer} test diagnostics identity mismatch")
    if payload.get("grouping") != dict(assignment_metadata):
        raise RuntimeError(f"outer{outer} diagnostics grouping metadata drift")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"outer{outer} test diagnostics lacks records")
    catalog: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("test diagnostic record is not an object")
        family = str(record.get("family"))
        seed = int(record.get("model_seed", -1))
        key = (family, seed)
        if family not in FAMILIES or seed not in SEEDS or key in catalog:
            raise RuntimeError(f"outer{outer} test diagnostic catalog is malformed")
        decision = decisions[family]
        if record.get("selected") != decision["selected"] or bool(
            record.get("fallback", False)
        ) != bool(decision["fallback"]):
            raise RuntimeError(f"outer{outer} {family}/seed{seed} diagnostic lock drift")
        catalog[key] = record
    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if set(catalog) != expected:
        raise RuntimeError(f"outer{outer} test diagnostics is not A0--A6 x three seeds")
    return catalog


def _load_outer(
    root: Path,
    *,
    outer: int,
    lock_sha256: str,
    decisions: Mapping[str, Mapping[str, Any]],
    digest: DigestCache,
) -> OuterArtifacts:
    directory = root / f"outer{outer}" / "scenarios"
    manifest_path = directory / TEST_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"outer{outer} locked test manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"outer{outer} locked test manifest is not an object")
    expected_manifest = {
        "schema": "caa_rahc_locked_test_outputs_v1",
        "outer": int(outer),
        "selection_lock_sha256": lock_sha256,
        "families": list(FAMILIES),
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"outer{outer} locked test manifest mismatch for {key}")
    registry = _validate_registry(directory, manifest.get("files"), digest)
    required_names = {
        "group_assignments_test_final.npz",
        "caa_test_final_diagnostics.json",
        *{
            f"{family}_seed{seed}_test_final.npz"
            for family in FAMILIES
            for seed in SEEDS
        },
        *{
            f"{family}_seed{seed}_test_final.npz.audit.json"
            for family in FAMILIES
            for seed in SEEDS
        },
        *{f"analytic_atoms_seed{seed}_test_final.npz" for seed in SEEDS},
        *{f"analytic_atoms_seed{seed}_test_final.npz.audit.json" for seed in SEEDS},
    }
    if not required_names.issubset(registry):
        raise RuntimeError(
            f"outer{outer} registry omits required locked evaluation artifacts"
        )
    scenarios: dict[str, dict[int, ScenarioArtifact]] = {}
    for family in FAMILIES:
        scenarios[family] = {}
        for seed in SEEDS:
            scenarios[family][seed] = _load_scenario(
                _scenario_path(root, outer, family, seed),
                registry,
                outer=outer,
                family=family,
                seed=seed,
                lock_sha256=lock_sha256,
                decision=decisions[family],
                digest=digest,
            )
    reference = scenarios["A0"][0]
    for family in FAMILIES:
        for seed in SEEDS:
            archive = scenarios[family][seed]
            if archive.scenarios.shape != reference.scenarios.shape:
                raise RuntimeError(f"outer{outer} final scenario shapes are not paired")
            if not np.array_equal(archive.observations, reference.observations):
                raise RuntimeError(f"outer{outer} observations are not paired")
            if not np.array_equal(archive.zone, reference.zone) or not np.array_equal(
                archive.day, reference.day
            ):
                raise RuntimeError(f"outer{outer} zone/day identities are not paired")
            if (
                archive.metadata.get("grouping_protocol_sha256")
                != reference.metadata.get("grouping_protocol_sha256")
            ):
                raise RuntimeError(f"outer{outer} scenario grouping-protocol hash drift")
            if decisions[family]["fallback"] and not np.array_equal(
                archive.scenarios, scenarios["A0"][seed].scenarios
            ):
                raise RuntimeError(
                    f"outer{outer} {family}/seed{seed} fallback is not exact A0"
                )
    dates = np.asarray(reference.day).astype("datetime64[D]")
    if np.any(np.isnat(dates)) or len(np.unique(dates)) != EXPECTED_DATES_PER_OUTER:
        raise RuntimeError(f"outer{outer} does not contain exactly 50 unique test dates")
    assignments, assignment_metadata = _load_assignments(
        directory / "group_assignments_test_final.npz",
        registry,
        outer=outer,
        cases=len(reference.observations),
        hours=reference.observations.shape[1],
        lock_sha256=lock_sha256,
    )
    if (
        assignment_metadata["grouping_protocol_sha256"]
        != reference.metadata.get("grouping_protocol_sha256")
    ):
        raise RuntimeError(f"outer{outer} persisted assignment protocol hash drift")
    atoms: dict[int, AtomArtifact] = {}
    for seed in SEEDS:
        atom = _load_atoms(
            _atom_path(root, outer, seed),
            registry,
            outer=outer,
            seed=seed,
            lock_sha256=lock_sha256,
            digest=digest,
        )
        if not (
            np.array_equal(atom.observations, reference.observations)
            and np.array_equal(atom.zone, reference.zone)
            and np.array_equal(atom.day, reference.day)
        ):
            raise RuntimeError(f"outer{outer} analytic atoms are not case-aligned")
        sources = atom.metadata.get("sources")
        if not isinstance(sources, Mapping):
            raise RuntimeError(f"outer{outer} analytic atom archive lacks sources")
        for family in ATOM_FAMILIES:
            text = str(sources.get(family, ""))
            if decisions[family]["fallback"] != text.startswith("exact_A0"):
                raise RuntimeError(
                    f"outer{outer} {family}/seed{seed} analytic source/fallback mismatch"
                )
        atoms[seed] = atom
    diagnostics = _diagnostic_catalog(
        directory / "caa_test_final_diagnostics.json",
        registry,
        outer=outer,
        lock_sha256=lock_sha256,
        assignment_metadata=assignment_metadata,
        decisions=decisions,
    )
    return OuterArtifacts(
        outer=outer,
        scenarios=scenarios,
        observations=reference.observations,
        zone=reference.zone,
        day=reference.day,
        assignments=assignments,
        assignment_metadata=assignment_metadata,
        analytic_atoms=atoms,
        diagnostics=diagnostics,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def _validate_outer_dates(artifacts: Mapping[int, OuterArtifacts]) -> None:
    date_sets: dict[int, set[np.datetime64]] = {}
    for outer in OUTERS:
        dates = set(np.unique(np.asarray(artifacts[outer].day).astype("datetime64[D]")))
        if len(dates) != EXPECTED_DATES_PER_OUTER:
            raise RuntimeError(f"outer{outer} test date count is not 50")
        date_sets[outer] = dates
    for index, outer in enumerate(OUTERS):
        for other in OUTERS[index + 1 :]:
            if date_sets[outer].intersection(date_sets[other]):
                raise RuntimeError(f"outer{outer} and outer{other} test dates overlap")
    if len(set().union(*date_sets.values())) != EXPECTED_TOTAL_DATES:
        raise RuntimeError("sealed outer tests do not pool to exactly 150 dates")


def _methods(artifacts: Mapping[int, OuterArtifacts]) -> dict[str, dict[int, list[np.ndarray]]]:
    return {
        family: {
            outer: [artifacts[outer].scenarios[family][seed].scenarios for seed in SEEDS]
            for outer in OUTERS
        }
        for family in FAMILIES
    }


def _validate_overall_rows(rows: Any) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("outer evaluator returned no overall rows")
    frame = pd.DataFrame(rows)
    required = {"method", "outer", "seed", "CRPS", "coverage_90", "width_90", "winkler_90", "conditional_ACE90"}
    if not required.issubset(frame.columns):
        raise RuntimeError("outer evaluator overall rows omit registered metrics")
    expected = {
        (family, str(outer), seed)
        for family in FAMILIES
        for outer in (*OUTERS, "pooled")
        for seed in SEEDS
    }
    actual = {
        (str(row.method), str(row.outer), int(row.seed))
        for row in frame.itertuples(index=False)
    }
    if actual != expected or len(frame) != len(expected):
        raise RuntimeError("outer evaluator rows are not A0--A6 x outer/pooled x three seeds")
    return frame


def _conditional_metrics(
    methods: Mapping[str, Mapping[int, Sequence[np.ndarray]]],
    artifacts: Mapping[int, OuterArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    pooled_observations = np.concatenate([artifacts[outer].observations for outer in OUTERS])
    pooled_zone = np.concatenate([artifacts[outer].zone for outer in OUTERS])
    pooled_assignments = {
        family: np.concatenate([artifacts[outer].assignments[family] for outer in OUTERS])
        for family in GROUP_FAMILIES
    }
    protocol_by_outer = {
        outer: FrozenAssignmentProtocol(
            artifacts[outer].assignments, artifacts[outer].zone
        )
        for outer in OUTERS
    }
    pooled_protocol = FrozenAssignmentProtocol(pooled_assignments, pooled_zone)

    def score(
        method: str,
        outer: int | str,
        seed: int,
        scenarios: np.ndarray,
        observations: np.ndarray,
        zone: np.ndarray,
        protocol: FrozenAssignmentProtocol,
    ) -> None:
        # The token is deliberately method-independent and ignored by the
        # adapter.  It cannot encode calibrated means or spreads.
        common_assignment_token = np.empty((0,), dtype=np.float64)
        frame = group_interval_metrics(
            scenarios,
            observations,
            zone,
            protocol,  # type: ignore[arg-type]
            grouping_raw_scenarios=common_assignment_token,
            nominal=0.90,
        )
        frame.insert(0, "seed", int(seed))
        frame.insert(0, "outer", outer)
        frame.insert(0, "method", method)
        group_frames.append(frame)
        family_summary, overall = summarize_group_metrics(
            frame.drop(columns=["method", "outer", "seed"])
        )
        family_summary.insert(0, "seed", int(seed))
        family_summary.insert(0, "outer", outer)
        family_summary.insert(0, "method", method)
        for key, value in overall.items():
            family_summary[f"overall_{key}"] = value
        summary_frames.append(family_summary)

    for method in FAMILIES:
        for outer in OUTERS:
            for seed in SEEDS:
                score(
                    method,
                    outer,
                    seed,
                    np.asarray(methods[method][outer][seed]),
                    artifacts[outer].observations,
                    artifacts[outer].zone,
                    protocol_by_outer[outer],
                )
        for seed in SEEDS:
            score(
                method,
                "pooled",
                seed,
                np.concatenate([methods[method][outer][seed] for outer in OUTERS]),
                pooled_observations,
                pooled_zone,
                pooled_protocol,
            )
    return pd.concat(group_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def _atom_outputs(
    artifacts: Mapping[int, OuterArtifacts],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    by_family_seed: dict[str, dict[str, Any]] = {
        family: {} for family in ("structural", *ATOM_FAMILIES)
    }
    quantization: dict[str, Any] = {
        "schema": "caa_rahc_quantization_audit_v1",
        "protocol": {
            "analytic_probabilities": "pre-finite-ensemble quantization locked outputs",
            "pooled_dates": EXPECTED_TOTAL_DATES,
            "model_seeds": list(SEEDS),
        },
        "by_outer_family_seed": {},
        "pooled_by_family_seed": {family: {} for family in ATOM_FAMILIES},
    }
    for outer in OUTERS:
        quantization["by_outer_family_seed"][f"outer{outer}"] = {}
        truth = artifacts[outer].observations
        for seed in SEEDS:
            atom = artifacts[outer].analytic_atoms[seed]
            for family in ("structural", *ATOM_FAMILIES):
                p0 = atom.arrays[f"{family}_pi0"]
                p1 = atom.arrays[f"{family}_pi1"]
                scores = analytic_atom_scores(p0, p1, truth)
                score_rows.append(
                    {"family": family, "outer": outer, "seed": seed, **scores}
                )
                if family in ATOM_FAMILIES:
                    finite = ensemble_atom_quantization(
                        artifacts[outer].scenarios[family][seed].scenarios, p0, p1
                    )
                    quantization["by_outer_family_seed"][f"outer{outer}"][
                        f"{family}_seed{seed}"
                    ] = finite
    for family in ("structural", *ATOM_FAMILIES):
        for seed in SEEDS:
            zero_parts = [
                artifacts[outer].analytic_atoms[seed].arrays[f"{family}_pi0"]
                for outer in OUTERS
            ]
            one_parts = [
                artifacts[outer].analytic_atoms[seed].arrays[f"{family}_pi1"]
                for outer in OUTERS
            ]
            truth_parts = [artifacts[outer].observations for outer in OUTERS]
            scenario_parts = (
                None
                if family == "structural"
                else [
                    artifacts[outer].scenarios[family][seed].scenarios
                    for outer in OUTERS
                ]
            )
            result = pooled_atom_diagnostics(
                zero_parts, one_parts, truth_parts, scenario_parts
            )
            by_family_seed[family][f"seed{seed}"] = result
            score_rows.append(
                {
                    "family": family,
                    "outer": "pooled",
                    "seed": seed,
                    **result["analytic_scores"],
                }
            )
            if family in ATOM_FAMILIES:
                quantization["pooled_by_family_seed"][family][f"seed{seed}"] = result[
                    "finite_ensemble"
                ]
    # Canonical report/audit consumers use this key for the pooled 150-date
    # finite-ensemble result.  Keep the explicitly named detailed alias too.
    quantization["by_family_seed"] = quantization["pooled_by_family_seed"]
    diagnostics = {
        "schema": "caa_rahc_atom_diagnostics_v1",
        "protocol": {
            "scores": "analytic Brier/log loss independent of ensemble member count",
            "reliability_bins": "equal-count",
            "unique_calendar_days": EXPECTED_TOTAL_DATES,
            "families": ["structural", *ATOM_FAMILIES],
        },
        "by_family_seed": by_family_seed,
    }
    return pd.DataFrame(score_rows), diagnostics, quantization


def _strict_reversal_exists(before: np.ndarray, after: np.ndarray) -> bool:
    order = np.argsort(before, axis=1, kind="stable")
    first = np.take_along_axis(before, order, axis=1)
    second = np.take_along_axis(after, order, axis=1)
    prefix_max = np.maximum.accumulate(second, axis=1)
    suffix_min = np.minimum.accumulate(second[:, ::-1, :], axis=1)[:, ::-1, :]
    strict_boundary = first[:, :-1, :] < first[:, 1:, :]
    reversed_across_boundary = prefix_max[:, :-1, :] > suffix_min[:, 1:, :]
    return bool(np.any(strict_boundary & reversed_across_boundary))


def _rank_rows(
    artifacts: Mapping[int, OuterArtifacts],
    decisions: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank_keys = (
        "strict_reversals",
        "raw_ties_broken",
        "strict_pairs_collapsed",
        "raw_tied_pairs",
        "calibrated_tied_pairs",
        "stable_ordinal_rank_matches",
        "stable_ordinal_rank_total",
        "stable_ordinal_rank_fraction",
        "raw_boundary_fraction",
        "calibrated_boundary_fraction",
        "pair_comparisons",
    )
    for outer in OUTERS:
        for family in FAMILIES:
            for seed in SEEDS:
                diagnostic = artifacts[outer].diagnostics[(family, seed)]
                key = "rank_audit_raw_to_A0" if family == "A0" else "rank_audit_A0_to_family"
                audit = diagnostic.get(key)
                if not isinstance(audit, Mapping) or not set(rank_keys).issubset(audit):
                    raise RuntimeError(
                        f"outer{outer} {family}/seed{seed} lacks complete persisted rank audit"
                    )
                if family != "A0":
                    before = artifacts[outer].scenarios["A0"][seed].scenarios
                    after = artifacts[outer].scenarios[family][seed].scenarios
                    proven_reversal = _strict_reversal_exists(before, after)
                    if proven_reversal != (int(audit["strict_reversals"]) > 0):
                        raise RuntimeError(
                            f"outer{outer} {family}/seed{seed} rank diagnostic is not reproducible"
                        )
                transformation = diagnostic.get("transformation")
                transformation = transformation if isinstance(transformation, Mapping) else {}
                exact_a0 = bool(
                    family == "A0"
                    or decisions[family]["fallback"]
                    or diagnostic.get("exact_A0", False)
                )
                rows.append(
                    {
                        "method": family,
                        "outer": outer,
                        "seed": seed,
                        "selected": decisions[family]["selected"],
                        "fallback": bool(decisions[family]["fallback"]),
                        "exact_A0": exact_a0,
                        "comparison_source": "raw_to_A0" if family == "A0" else "A0_to_family",
                        **{name: audit[name] for name in rank_keys},
                        "central_values_changed_by_tail": transformation.get(
                            "central_values_changed_by_tail", 0 if exact_a0 else None
                        ),
                        "nonfinite_input_values": transformation.get(
                            "nonfinite_input_values", 0 if exact_a0 else None
                        ),
                        "nonfinite_intermediate_values": transformation.get(
                            "nonfinite_intermediate_values", 0 if exact_a0 else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _relative(new: float, baseline: float) -> float:
    if baseline <= 0.0:
        if new == baseline:
            return 0.0
        raise RuntimeError("relative outer-consistency metric has non-positive A0 denominator")
    return (new - baseline) / baseline


def _direction(delta: float) -> str:
    if delta < 0.0:
        return "improved"
    if delta > 0.0:
        return "worsened"
    return "tied"


def _outer_consistency(
    overall: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    fallback: bool,
) -> dict[str, Any]:
    constraints = config["constraints"]
    limits = {
        "CRPS_relative_degradation": float(constraints["crps_relative_degradation_u95"]),
        "secondary_relative_degradation": float(
            constraints["secondary_relative_degradation_u95"]
        ),
        "width_90_absolute_increase": float(constraints["width_absolute_increase_point"]),
    }
    secondary = ("MAE", "VS", "ramp_CRPS", "winkler_90")
    seed_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for outer in OUTERS:
        baseline_rows = overall.loc[
            (overall["method"] == "A0") & (overall["outer"].astype(str) == str(outer))
        ].sort_values("seed")
        primary_rows = overall.loc[
            (overall["method"] == "A4") & (overall["outer"].astype(str) == str(outer))
        ].sort_values("seed")
        if len(baseline_rows) != 3 or len(primary_rows) != 3:
            raise RuntimeError(f"outer{outer} consistency rows are incomplete")
        for seed in SEEDS:
            a0 = baseline_rows.loc[baseline_rows["seed"] == seed].iloc[0]
            a4 = primary_rows.loc[primary_rows["seed"] == seed].iloc[0]
            conditional_delta = float(a4["conditional_ACE90"] - a0["conditional_ACE90"])
            seed_rows.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "conditional_ACE90_delta": conditional_delta,
                    "conditional_ACE90_direction": _direction(conditional_delta),
                    "CRPS_relative_degradation": _relative(float(a4["CRPS"]), float(a0["CRPS"])),
                    **{
                        f"{metric}_relative_degradation": _relative(
                            float(a4[metric]), float(a0[metric])
                        )
                        for metric in secondary
                    },
                    "width_90_absolute_increase": float(a4["width_90"] - a0["width_90"]),
                }
            )
        baseline_mean = baseline_rows.mean(numeric_only=True)
        primary_mean = primary_rows.mean(numeric_only=True)
        conditional_delta = float(
            primary_mean["conditional_ACE90"] - baseline_mean["conditional_ACE90"]
        )
        crps_relative = _relative(float(primary_mean["CRPS"]), float(baseline_mean["CRPS"]))
        secondary_relative = {
            metric: _relative(float(primary_mean[metric]), float(baseline_mean[metric]))
            for metric in secondary
        }
        width_increase = float(primary_mean["width_90"] - baseline_mean["width_90"])
        violations = []
        if crps_relative > limits["CRPS_relative_degradation"]:
            violations.append("CRPS_relative_degradation")
        violations.extend(
            f"{metric}_relative_degradation"
            for metric, value in secondary_relative.items()
            if value > limits["secondary_relative_degradation"]
        )
        if width_increase > limits["width_90_absolute_increase"]:
            violations.append("width_90_absolute_increase")
        outer_rows.append(
            {
                "outer": outer,
                "conditional_ACE90_delta": conditional_delta,
                "conditional_ACE90_direction": _direction(conditional_delta),
                "CRPS_relative_degradation": crps_relative,
                **{
                    f"{metric}_relative_degradation": value
                    for metric, value in secondary_relative.items()
                },
                "width_90_absolute_increase": width_increase,
                "catastrophic": bool(violations),
                "catastrophic_violations": violations,
            }
        )
    improved = sum(row["conditional_ACE90_direction"] == "improved" for row in outer_rows)
    catastrophic = sum(bool(row["catastrophic"]) for row in outer_rows)
    passed = improved >= 2 and catastrophic == 0
    return {
        "schema": "caa_rahc_outer_consistency_v1",
        "primary": "A4",
        "baseline": "A0",
        "primary_fallback_exact_A0": bool(fallback),
        "catastrophic_rule": {
            "label": (
                "protocol-conservative label, not an ordinary-language severity claim: "
                "any outer three-seed mean crossing any frozen non-inferiority limit"
            ),
            "limits": limits,
        },
        "per_outer_seed": seed_rows,
        "per_outer": outer_rows,
        "same_primary_direction_outer_count": int(improved),
        "catastrophic_outer_count": int(catastrophic),
        "required_improved_outers": 2,
        "passed": bool(passed),
    }


def _success_gates(
    evaluator: Mapping[str, Any], consistency: Mapping[str, Any], *, fallback: bool
) -> dict[str, Any]:
    payload = dict(evaluator)
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise RuntimeError("outer evaluator returned no success-gate decisions")
    normalized = {str(key): bool(value) for key, value in gates.items()}
    if any(not isinstance(value, (bool, np.bool_)) for value in gates.values()):
        raise RuntimeError("outer evaluator success gates are not boolean")
    normalized["outer_consistency"] = bool(consistency["passed"])
    payload.update(
        {
            "schema": "caa_rahc_success_gates_v1",
            "gates": normalized,
            "passed": int(sum(normalized.values())),
            "total": int(len(normalized)),
            "all_passed": bool(all(normalized.values())),
            "primary_fallback_exact_A0": bool(fallback),
        }
    )
    return payload


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output blocks evaluation: {temporary}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output blocks evaluation: {temporary}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def refresh_experiment_manifest(
    root: str | Path,
    *,
    config_path: str | Path | None,
    lock_sha256: str,
    lock_decision_field: str,
    protocol: Mapping[str, Any],
    workspace: str | Path = WORKSPACE,
) -> dict[str, Any]:
    """Rebuild the root manifest; report generation may call this again."""

    output_root = Path(root).resolve()
    workspace_path = Path(workspace).resolve()
    destination = output_root / "experiment_manifest.json"
    paths = {
        path.resolve()
        for path in output_root.rglob("*")
        if path.is_file()
        and path.resolve() != destination.resolve()
        and not path.name.endswith(".tmp")
    }
    extras = [
        Path(__file__).resolve(),
        (workspace_path / "caa_rahc" / "outer_evaluation.py").resolve(),
        (workspace_path / "caa_rahc" / "atom_diagnostics.py").resolve(),
        (workspace_path / "rahc" / "group_metrics.py").resolve(),
    ]
    if config_path is not None and Path(config_path).is_file():
        extras.append(Path(config_path).resolve())
    report = workspace_path / "CAA_RAHC_EXPERIMENT_REPORT.md"
    if report.is_file():
        extras.append(report.resolve())
    # Completion treats every source in the frozen selection-code fingerprint
    # as critical, so register those sources instead of assuming a short list.
    for provenance_path in (
        output_root / "calibration" / "selection_analysis.json",
        output_root / "frozen_manifest.json",
    ):
        if not provenance_path.is_file():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"code provenance is not an object: {provenance_path}")
        catalogs = [
            provenance.get(key)
            for key in ("full_code", "full_code_fingerprint", "code")
            if isinstance(provenance.get(key), Mapping)
        ]
        for catalog in catalogs:
            records = catalog.get("files")
            if records is None:
                continue
            if not isinstance(records, list):
                raise RuntimeError(f"malformed code provenance catalog: {provenance_path}")
            for record in records:
                if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                    raise RuntimeError(f"malformed code provenance record: {provenance_path}")
                relative = Path(record["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"unsafe code provenance path: {relative}")
                source = (workspace_path / relative).resolve()
                try:
                    source.relative_to(workspace_path)
                except ValueError as error:
                    raise RuntimeError(f"code provenance escapes workspace: {relative}") from error
                if not source.is_file():
                    raise RuntimeError(f"registered code source is missing: {source}")
                extras.append(source)
    paths.update(path for path in extras if path.is_file())
    digest = DigestCache()
    files = [
        {
            "path": _display_path(path, workspace_path),
            "bytes": int(path.stat().st_size),
            "sha256": digest(path),
        }
        for path in sorted(paths, key=lambda value: str(value))
    ]
    payload = {
        "schema": "caa_rahc_experiment_manifest_v1",
        "generated_by": "repro_scripts.run_caa_evaluation",
        "selection_lock_sha256": lock_sha256,
        "lock_decision_field": lock_decision_field,
        "evaluation_protocol": _jsonable(protocol),
        "files": files,
    }
    _write_json_atomic(destination, payload)
    return payload


def run_evaluation(config: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate and evaluate all locked A0--A6 outer-test artifacts."""

    digest = DigestCache()
    context = load_context(config)
    root = Path(context.root)
    lock, lock_sha256 = _load_validated_lock(context, digest)
    decisions, decision_field = _resolve_decisions(lock)
    configured_outers = tuple(sorted(int(value) for value in context.config["outer_splits"]))
    configured_seeds = tuple(sorted(int(value) for value in context.config["model_seeds"]))
    if configured_outers != OUTERS or configured_seeds != SEEDS:
        raise RuntimeError("frozen evaluation requires outers/seeds [1,2,3]/[0,1,2]")
    # Every registry and every registered hash is validated before scoring.
    artifacts = {
        outer: _load_outer(
            root,
            outer=outer,
            lock_sha256=lock_sha256,
            decisions=decisions,
            digest=digest,
        )
        for outer in OUTERS
    }
    _validate_outer_dates(artifacts)
    methods = _methods(artifacts)
    observations = {outer: artifacts[outer].observations for outer in OUTERS}
    days = {outer: artifacts[outer].day for outer in OUTERS}
    assignments = {outer: artifacts[outer].assignments for outer in OUTERS}
    constraints = context.config["constraints"]
    evaluation = evaluate_outer_tests(
        methods,
        observations,
        days,
        assignments,
        baseline_name="A0",
        primary_name="A4",
        bootstrap_replicates=int(constraints["bootstrap_replicates"]),
        bootstrap_seed=int(constraints["bootstrap_seed"]),
        success_config=context.config["success_gates"],
        strict_reversals=int(
            sum(
                int(artifacts[outer].diagnostics[("A4", seed)]["rank_audit_A0_to_family"]["strict_reversals"])
                for outer in OUTERS
                for seed in SEEDS
            )
        ),
    )
    if evaluation.get("protocol", {}).get("unique_test_dates") != EXPECTED_TOTAL_DATES:
        raise RuntimeError("outer evaluator did not certify 150 unique test dates")
    overall = _validate_overall_rows(evaluation.get("rows"))
    conditional, conditional_summary = _conditional_metrics(methods, artifacts)
    atom_scores, atom_diagnostics, quantization = _atom_outputs(artifacts)
    ranks = _rank_rows(artifacts, decisions)
    consistency = _outer_consistency(
        overall, context.config, fallback=bool(decisions["A4"]["fallback"])
    )
    gates = _success_gates(
        evaluation.get("success_gates", {}),
        consistency,
        fallback=bool(decisions["A4"]["fallback"]),
    )

    metrics = root / "metrics"
    statistics = root / "statistics"
    _write_csv_atomic(metrics / "overall_metrics.csv", overall)
    _write_csv_atomic(metrics / "conditional_group_metrics.csv", conditional)
    _write_csv_atomic(metrics / "conditional_metrics.csv", conditional)
    _write_csv_atomic(metrics / "conditional_family_summary.csv", conditional_summary)
    _write_csv_atomic(metrics / "atom_scores.csv", atom_scores)
    _write_csv_atomic(metrics / "rank_audit.csv", ranks)
    _write_json_atomic(metrics / "atom_diagnostics.json", atom_diagnostics)
    _write_json_atomic(metrics / "quantization.json", quantization)
    _write_json_atomic(metrics / "quantization_audit.json", quantization)
    bootstrap_payload = {
        "schema": "caa_rahc_paired_calendar_day_bootstrap_v1",
        "protocol": {
            "replicates": int(constraints["bootstrap_replicates"]),
            "bootstrap_seed": int(constraints["bootstrap_seed"]),
            "unique_calendar_days": EXPECTED_TOTAL_DATES,
            "cluster": "calendar date",
            "outer_test_blocks_disjoint": True,
        },
        "comparisons": evaluation.get("paired_bootstrap", {}),
    }
    noninferiority_payload = {
        "schema": "caa_rahc_noninferiority_v1",
        "baseline": "A0",
        "comparisons": evaluation.get("noninferiority", {}),
        "primary": "A4",
        "primary_fallback_exact_A0": bool(decisions["A4"]["fallback"]),
    }
    _write_json_atomic(
        statistics / "paired_calendar_day_bootstrap.json", bootstrap_payload
    )
    _write_json_atomic(statistics / "noninferiority.json", noninferiority_payload)
    _write_json_atomic(statistics / "outer_consistency.json", consistency)
    _write_json_atomic(statistics / "success_gates.json", gates)
    protocol = {
        "schema": "caa_rahc_evaluation_protocol_v1",
        "outers": list(OUTERS),
        "model_seeds": list(SEEDS),
        "families": list(FAMILIES),
        "dates_per_outer": EXPECTED_DATES_PER_OUTER,
        "unique_calendar_days": EXPECTED_TOTAL_DATES,
        "paired_bootstrap_cluster": "calendar date",
        "conditional_assignment_source": (
            "calibration-fitted grouping protocol applied once to test raw forecasts; "
            "persisted group_assignments_test_final.npz reused unchanged for A0--A6"
        ),
        "conditional_groups_method_independent": True,
        "selection_lock_sha256": lock_sha256,
        "lock_decision_field": decision_field,
        "locked_decisions": decisions,
        "A4_fallback_exact_A0": bool(decisions["A4"]["fallback"]),
        "input_manifests": {
            f"outer{outer}": {
                "path": str(artifacts[outer].manifest_path.resolve()),
                "sha256": digest(artifacts[outer].manifest_path),
            }
            for outer in OUTERS
        },
    }
    config_path = getattr(context, "config_path", config)
    experiment_manifest = refresh_experiment_manifest(
        root,
        config_path=config_path,
        lock_sha256=lock_sha256,
        lock_decision_field=decision_field,
        protocol=protocol,
    )
    return {
        "protocol": protocol,
        "overall_metrics": overall,
        "conditional_metrics": conditional,
        "conditional_family_summary": conditional_summary,
        "atom_scores": atom_scores,
        "rank_audit": ranks,
        "atom_diagnostics": atom_diagnostics,
        "quantization": quantization,
        "paired_bootstrap": bootstrap_payload,
        "noninferiority": noninferiority_payload,
        "outer_consistency": consistency,
        "success_gates": gates,
        "experiment_manifest": experiment_manifest,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sealed locked CAA outer tests without fitting or selection"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_evaluation(args.config)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "success_gates": result["success_gates"],
                "outer_consistency": result["outer_consistency"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_TOTAL_DATES",
    "FAMILIES",
    "OUTERS",
    "SEEDS",
    "refresh_experiment_manifest",
    "run_evaluation",
]
