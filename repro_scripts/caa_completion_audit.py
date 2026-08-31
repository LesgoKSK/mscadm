"""Fail-closed completion audit for the frozen CAA-RAHC outer experiment.

The audit is deliberately independent of training and calibration code.  It
only reads sealed artifacts, recomputes their hashes, verifies their embedded
provenance, and writes two audit products:

``outputs/caa_rahc/completion_audit.json``
    Machine-readable checks, including explicit ``missing`` and ``failed``
    lists.  A scientifically unsuccessful result may still be complete; this
    audit checks that the registered analysis exists, not that every success
    gate passed.

``outputs/caa_rahc/manifest.csv``
    A fresh SHA-256 inventory of the completed tree and registered report.

No model code is imported and no formal experiment stage is run from here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / "outputs" / "caa_rahc"
DEFAULT_REPORT = WORKSPACE / "CAA_RAHC_EXPERIMENT_REPORT.md"

OUTERS = (1, 2, 3)
SEEDS = (0, 1, 2)
FAMILIES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
EXPECTED_DAYS_PER_OUTER = 50
AUTHORITATIVE_MODULE = "caa_rahc.candidates_nested"

FROZEN_MANIFEST = "frozen_manifest.json"
SELECTION_ANALYSIS = "calibration/selection_analysis.json"
SELECTION_LOCK = "selection.lock.json"
EXPERIMENT_MANIFEST = "experiment_manifest.json"

SUMMARY_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "overall_metrics": ("metrics/overall_metrics.csv", "metrics/overall.csv"),
    "conditional_metrics": (
        "metrics/conditional_metrics.csv",
        "metrics/conditional.csv",
    ),
    "atom_diagnostics": (
        "metrics/atom_diagnostics.json",
        "metrics/atom.json",
        "diagnostics/atom_diagnostics.json",
    ),
    "quantization_audit": (
        "metrics/quantization_audit.json",
        "metrics/quantization.json",
        "diagnostics/quantization_audit.json",
    ),
    "paired_calendar_day_bootstrap": (
        "statistics/paired_calendar_day_bootstrap.json",
        "statistics/bootstrap.json",
    ),
    "noninferiority": (
        "statistics/noninferiority.json",
        "statistics/non_inferiority.json",
    ),
    "success_gates": (
        "statistics/success_gates.json",
        "diagnostics/success_gates.json",
    ),
    "outer_consistency": (
        "statistics/outer_consistency.json",
        "diagnostics/outer_consistency.json",
    ),
    "suc_daily": ("suc/daily_results.csv",),
    "suc_summary": ("suc/summary.csv",),
    "suc_protocol": ("suc/protocol.json",),
    "figure_manifest": ("figures/figure_manifest.json",),
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_AUTHORITATIVE = (
    "caa_rahc.candidates",
    "caa_rahc/candidates.py",
    "caa_rahc\\candidates.py",
    "caa_rahc.zero_atom",
    "caa_rahc/zero_atom.py",
    "caa_rahc\\zero_atom.py",
    "caa_rahc.calibration",
    "caa_rahc/calibration.py",
    "caa_rahc\\calibration.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_path(root: Path, outer: int, seed: int) -> Path:
    return root / f"outer{outer}" / "runs" / "full" / f"seed{seed}" / "final.pt"


def raw_archive_path(root: Path, outer: int, seed: int, split: str) -> Path:
    return (
        root
        / f"outer{outer}"
        / "scenarios"
        / f"full_seed{seed}_{split}_raw.npz"
    )


def final_archive_path(root: Path, outer: int, seed: int, family: str) -> Path:
    return (
        root
        / f"outer{outer}"
        / "scenarios"
        / f"{family}_seed{seed}_test_final.npz"
    )


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".audit.json")


class DigestCache:
    def __init__(self) -> None:
        self._values: dict[tuple[str, int, int], str] = {}

    def __call__(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        if key not in self._values:
            self._values[key] = sha256_file(path)
        return self._values[key]


@dataclass
class AuditState:
    root: Path
    workspace: Path
    report: Path
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    critical_files: set[Path] = field(default_factory=set)
    digest: DigestCache = field(default_factory=DigestCache)

    def display(self, path: Path) -> str:
        resolved = path.resolve()
        for base in (self.workspace.resolve(), self.root.resolve()):
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                pass
        return str(resolved)

    def require(self, path: Path, *, label: str | None = None) -> bool:
        if path.is_file():
            self.critical_files.add(path.resolve())
            return True
        value = label or self.display(path)
        if value not in self.missing:
            self.missing.append(value)
        return False

    def record(self, name: str, passed: bool, detail: Any) -> None:
        self.checks[name] = {"passed": bool(passed), "detail": _jsonable(detail)}
        if not passed and name not in self.failed:
            self.failed.append(name)

    def guard(self, name: str, function: Any) -> Any:
        try:
            detail = function()
            self.record(name, True, detail)
            return detail
        except Exception as error:  # fail closed while preserving later findings
            self.record(
                name,
                False,
                {"error": type(error).__name__, "message": str(error)},
            )
            return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_hash(value: Any, *, name: str) -> str:
    text = str(value).lower()
    if not _HEX64.fullmatch(text):
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return text


def _metadata_from_archive(stored: Any) -> dict[str, Any]:
    if "metadata" not in stored.files:
        raise ValueError("archive is missing scalar JSON metadata")
    value = stored["metadata"]
    if value.shape != ():
        raise ValueError("archive metadata must be a scalar JSON string")
    result = json.loads(str(value.item()))
    if not isinstance(result, dict):
        raise ValueError("archive metadata JSON must be an object")
    return result


def _read_sidecar(
    state: AuditState,
    path: Path,
    *,
    hash_key: str,
) -> dict[str, Any]:
    sidecar = sidecar_path(path)
    if not state.require(sidecar):
        raise FileNotFoundError(sidecar)
    payload = _read_json(sidecar)
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar is not a JSON object: {sidecar}")
    recorded = _require_hash(payload.get(hash_key), name=f"{sidecar}:{hash_key}")
    actual = state.digest(path)
    if recorded != actual:
        raise ValueError(
            f"hash drift for {state.display(path)}: sidecar={recorded}, actual={actual}"
        )
    return payload


def _identity_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _validate_archive_arrays(
    path: Path,
    *,
    expected_dates: Sequence[str],
    state: AuditState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"scenarios", "observations", "zone", "day", "metadata"}
        missing = sorted(required.difference(stored.files))
        if missing:
            raise ValueError(f"archive lacks arrays {missing}: {state.display(path)}")
        scenarios = np.asarray(stored["scenarios"])
        observations = np.asarray(stored["observations"])
        zone = np.asarray(stored["zone"])
        day = np.asarray(stored["day"]).astype("datetime64[D]")
        metadata = _metadata_from_archive(stored)
        if scenarios.ndim != 3 or scenarios.shape[0] != len(day):
            raise ValueError(f"invalid scenario shape/alignment: {scenarios.shape}")
        if observations.shape[0] != len(day) or zone.shape != (len(day),):
            raise ValueError("observations, zone, and day are not case-aligned")
        if scenarios.shape[1] <= 0 or scenarios.shape[2] <= 0:
            raise ValueError("scenario member/hour axes must be non-empty")
        if not np.isfinite(scenarios).all():
            raise ValueError("scenario archive contains non-finite values")
        if float(scenarios.min()) < 0.0 or float(scenarios.max()) > 1.0:
            raise ValueError("scenario archive is outside [0,1]")
        unique_dates = np.unique(day).astype(str).tolist()
        if unique_dates != sorted(str(value) for value in expected_dates):
            raise ValueError(
                f"archive date identity differs from frozen manifest: {state.display(path)}"
            )
        signatures = {
            "observations_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
            "zone_sha256": hashlib.sha256(zone.tobytes()).hexdigest(),
            "day_sha256": hashlib.sha256(day.astype("datetime64[D]").tobytes()).hexdigest(),
            "cases": int(len(day)),
            "unique_dates": int(len(unique_dates)),
            "shape": list(scenarios.shape),
        }
    return metadata, signatures


def _audit_frozen_manifest(state: AuditState) -> dict[str, Any]:
    path = state.root / FROZEN_MANIFEST
    if not state.require(path):
        raise FileNotFoundError(path)
    manifest = _read_json(path)
    if manifest.get("schema") != "caa_rahc_frozen_outer_manifest_v1":
        raise ValueError("wrong frozen manifest schema")
    if manifest.get("outer_splits") != list(OUTERS):
        raise ValueError("frozen manifest must contain outer_splits [1,2,3]")
    if manifest.get("model_seeds") != list(SEEDS):
        raise ValueError("frozen manifest must contain model_seeds [0,1,2]")
    protocols = manifest.get("protocols")
    if not isinstance(protocols, dict):
        raise ValueError("frozen manifest protocols are missing")
    test_sets: dict[int, set[str]] = {}
    calibration_dates: dict[int, list[str]] = {}
    for outer in OUTERS:
        protocol = protocols.get(f"outer{outer}")
        if not isinstance(protocol, dict):
            raise ValueError(f"outer{outer} protocol is missing")
        dates = protocol.get("dates")
        if not isinstance(dates, dict):
            raise ValueError(f"outer{outer} date lists are missing")
        test = [str(value) for value in dates.get("test", [])]
        calibration = [str(value) for value in dates.get("calibration", [])]
        if len(test) != EXPECTED_DAYS_PER_OUTER or len(set(test)) != len(test):
            raise ValueError(f"outer{outer} test must have 50 unique dates")
        if len(calibration) != EXPECTED_DAYS_PER_OUTER or len(set(calibration)) != len(
            calibration
        ):
            raise ValueError(f"outer{outer} calibration must have 50 unique dates")
        count = protocol.get("calendar_day_counts", {})
        if int(count.get("test", -1)) != EXPECTED_DAYS_PER_OUTER:
            raise ValueError(f"outer{outer} calendar_day_counts.test != 50")
        if int(count.get("calibration", -1)) != EXPECTED_DAYS_PER_OUTER:
            raise ValueError(f"outer{outer} calendar_day_counts.calibration != 50")
        test_sets[outer] = set(test)
        calibration_dates[outer] = calibration
    for index, outer in enumerate(OUTERS):
        for other in OUTERS[index + 1 :]:
            overlap = test_sets[outer].intersection(test_sets[other])
            if overlap:
                raise ValueError(
                    f"sealed outer test dates overlap: outer{outer}/outer{other}: "
                    f"{sorted(overlap)[:5]}"
                )
    return {
        "payload": manifest,
        "path": path,
        "sha256": state.digest(path),
        "test_dates": {outer: sorted(values) for outer, values in test_sets.items()},
        "calibration_dates": calibration_dates,
    }


def _full_code_payload(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("full_code", "full_code_fingerprint", "code"):
        value = analysis.get(key)
        if isinstance(value, Mapping):
            return value
    raise ValueError("selection analysis lacks full-code fingerprint records")


def _audit_selection(
    state: AuditState, manifest_info: Mapping[str, Any]
) -> dict[str, Any]:
    analysis_path = state.root / SELECTION_ANALYSIS
    lock_path = state.root / SELECTION_LOCK
    if not state.require(analysis_path):
        raise FileNotFoundError(analysis_path)
    if not state.require(lock_path):
        raise FileNotFoundError(lock_path)
    analysis = _read_json(analysis_path)
    lock = _read_json(lock_path)
    if not isinstance(analysis, dict) or not isinstance(lock, dict):
        raise ValueError("selection analysis/lock must be JSON objects")
    analysis_hash = state.digest(analysis_path)
    lock_hash = state.digest(lock_path)
    if _require_hash(lock.get("analysis_sha256"), name="selection.lock analysis_sha256") != analysis_hash:
        raise ValueError("selection.lock analysis_sha256 does not match selection analysis")
    declared_analysis = lock.get("analysis_path", SELECTION_ANALYSIS)
    if Path(str(declared_analysis)).as_posix() != SELECTION_ANALYSIS:
        raise ValueError("selection.lock analysis_path is not the frozen analysis path")
    if _require_hash(lock.get("manifest_sha256"), name="selection.lock manifest_sha256") != manifest_info["sha256"]:
        raise ValueError("selection.lock manifest_sha256 does not match frozen manifest")

    fingerprint = _full_code_payload(analysis)
    records = fingerprint.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("full-code fingerprint must contain non-empty files records")
    normalized_records: list[dict[str, Any]] = []
    registered_paths: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("full-code file record is not an object")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe full-code path: {relative}")
        source = (state.workspace / relative).resolve()
        try:
            source.relative_to(state.workspace.resolve())
        except ValueError as error:
            raise ValueError(f"full-code path escapes workspace: {relative}") from error
        if not state.require(source, label=relative.as_posix()):
            raise FileNotFoundError(source)
        actual = state.digest(source)
        expected = _require_hash(item.get("sha256"), name=f"full-code {relative}")
        if actual != expected:
            raise ValueError(f"full-code hash drift: {relative.as_posix()}")
        if "bytes" in item and int(item["bytes"]) != source.stat().st_size:
            raise ValueError(f"full-code byte-size drift: {relative.as_posix()}")
        record = {
            "path": relative.as_posix(),
            "bytes": int(source.stat().st_size),
            "sha256": actual,
        }
        normalized_records.append(record)
        registered_paths.add(relative.as_posix().lower())
    combined = canonical_json_sha256(normalized_records)
    recorded_combined = _require_hash(
        fingerprint.get("combined_sha256"), name="selection analysis full-code combined_sha256"
    )
    if recorded_combined != combined:
        raise ValueError("selection analysis full-code combined hash is invalid")
    lock_full_hash = lock.get("full_code_sha256", lock.get("code_sha256"))
    if _require_hash(lock_full_hash, name="selection.lock full_code_sha256") != combined:
        raise ValueError("selection.lock full-code hash differs from selection analysis")
    if "caa_rahc/candidates_nested.py" not in registered_paths:
        raise ValueError("full-code fingerprint omits caa_rahc/candidates_nested.py")

    locked_at = _parse_timestamp(lock.get("locked_at_utc"), field_name="locked_at_utc")
    selected_configs = lock.get("selected_configs")
    if not isinstance(selected_configs, dict) or set(selected_configs) != set(FAMILIES):
        raise ValueError("selection.lock selected_configs must contain exactly A0--A6")
    if analysis.get("selected_configs") != selected_configs:
        raise ValueError("selection analysis and lock selected_configs differ")
    _audit_authority_payloads([analysis, lock])
    return {
        "analysis": analysis,
        "analysis_path": analysis_path,
        "analysis_sha256": analysis_hash,
        "lock": lock,
        "lock_path": lock_path,
        "lock_sha256": lock_hash,
        "locked_at": locked_at,
        "full_code_sha256": combined,
        "selected_configs": selected_configs,
        "full_code_files": len(records),
    }


def _audit_authority_payloads(payloads: Iterable[Mapping[str, Any]]) -> None:
    authoritative_values: list[str] = []

    def walk(value: Any, keys: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, keys + (str(key).lower(),))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, keys)
        elif isinstance(value, str) and any("authoritative" in key for key in keys):
            authoritative_values.append(value)

    for payload in payloads:
        walk(payload)
    normalized = [value.replace("\\", "/").lower() for value in authoritative_values]
    if not any(
        AUTHORITATIVE_MODULE.lower() in value
        or "caa_rahc/candidates_nested.py" in value
        for value in normalized
    ):
        raise ValueError("authoritative metadata does not name caa_rahc.candidates_nested")
    for value in normalized:
        # candidates_nested contains the prefix caa_rahc.candidates, so test
        # the exact prototype spellings after accepting the nested spelling.
        without_nested = value.replace("caa_rahc.candidates_nested", "")
        without_nested = without_nested.replace("caa_rahc/candidates_nested.py", "")
        for forbidden in _FORBIDDEN_AUTHORITATIVE:
            if forbidden.replace("\\", "/").lower() in without_nested:
                raise ValueError(
                    f"prototype module is labeled authoritative: {value}"
                )


def _audit_checkpoints(state: AuditState) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for outer in OUTERS:
        for seed in SEEDS:
            path = checkpoint_path(state.root, outer, seed)
            if not state.require(path):
                continue
            sidecar = _read_sidecar(state, path, hash_key="checkpoint_sha256")
            if int(sidecar.get("outer", -1)) != outer:
                raise ValueError(f"checkpoint sidecar outer mismatch: {state.display(path)}")
            if int(sidecar.get("model_seed", -1)) != seed:
                raise ValueError(f"checkpoint sidecar seed mismatch: {state.display(path)}")
            if sidecar.get("schema") != "caa_rahc_checkpoint_audit_v1":
                raise ValueError(f"checkpoint lacks registered audit schema: {path}")
            if sidecar.get("resume") is not False:
                raise ValueError(f"checkpoint was not certified fresh: {path}")
            records.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "path": state.display(path),
                    "sha256": state.digest(path),
                }
            )
    if len(records) != len(OUTERS) * len(SEEDS):
        raise FileNotFoundError(f"found {len(records)}/9 audited final checkpoints")
    return {"count": len(records), "records": records}


def _check_archive_identity(
    metadata: Mapping[str, Any], *, outer: int, seed: int, split: str
) -> None:
    if str(metadata.get("split")) != split:
        raise ValueError(f"archive metadata split is not {split}")
    if int(metadata.get("outer", -1)) != outer:
        raise ValueError("archive metadata outer mismatch")
    recorded_seed = _identity_value(metadata, "training_seed", "model_seed", "seed")
    if int(recorded_seed if recorded_seed is not None else -1) != seed:
        raise ValueError("archive metadata seed mismatch")


def _audit_raw_archives(
    state: AuditState,
    manifest_info: Mapping[str, Any],
    selection_info: Mapping[str, Any] | None,
    *,
    split: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    reference: dict[int, dict[str, Any]] = {}
    for outer in OUTERS:
        expected_dates = manifest_info[f"{split}_dates"][outer]
        for seed in SEEDS:
            path = raw_archive_path(state.root, outer, seed, split)
            if not state.require(path):
                continue
            sidecar = _read_sidecar(state, path, hash_key="archive_sha256")
            metadata, signatures = _validate_archive_arrays(
                path, expected_dates=expected_dates, state=state
            )
            _check_archive_identity(metadata, outer=outer, seed=seed, split=split)
            if split == "test":
                if selection_info is None:
                    raise ValueError("sealed test raw archives exist without valid selection lock")
                if metadata.get("generated_after_selection_lock") is not True:
                    raise ValueError("test metadata lacks generated_after_selection_lock=true")
                if _require_hash(
                    metadata.get("selection_lock_sha256"),
                    name="test metadata selection_lock_sha256",
                ) != selection_info["lock_sha256"]:
                    raise ValueError("test metadata selection-lock hash mismatch")
                analysis_value = metadata.get(
                    "selection_analysis_sha256", metadata.get("analysis_sha256")
                )
                if _require_hash(
                    analysis_value, name="test metadata selection_analysis_sha256"
                ) != selection_info["analysis_sha256"]:
                    raise ValueError("test metadata selection-analysis hash mismatch")
                generated_at = _parse_timestamp(
                    metadata.get("generated_at_utc"), field_name="generated_at_utc"
                )
                if generated_at <= selection_info["locked_at"]:
                    raise ValueError(
                        f"test raw archive was not generated after selection lock: {path}"
                    )
                if sidecar.get("selection_lock_sha256") not in (
                    None,
                    selection_info["lock_sha256"],
                ):
                    raise ValueError("test sidecar selection-lock hash mismatch")
            current_reference = {
                key: signatures[key]
                for key in ("observations_sha256", "zone_sha256", "day_sha256")
            }
            if outer not in reference:
                reference[outer] = current_reference
            elif reference[outer] != current_reference:
                raise ValueError(f"outer{outer} raw {split} archives are misaligned by seed")
            records.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "path": state.display(path),
                    "sha256": state.digest(path),
                    **signatures,
                }
            )
    if len(records) != 9:
        raise FileNotFoundError(f"found {len(records)}/9 audited {split} raw archives")
    return {"count": len(records), "records": records, "reference": reference}


def _collect_named_numbers(value: Any, aliases: set[str]) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in aliases and isinstance(item, (int, float, np.number)):
                result.append(float(item))
            result.extend(_collect_named_numbers(item, aliases))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_collect_named_numbers(item, aliases))
    return result


def _validate_transform_diagnostics(metadata: Mapping[str, Any]) -> dict[str, list[float]]:
    fields = {
        "strict_reversals": {"strict_reversals", "strict_rank_reversals"},
        "nonfinite": {
            "nonfinite",
            "nonfinite_values",
            "non_finite",
            "non_finite_values",
        },
        "central_changed": {
            "central_changed",
            "central_values_changed",
            "central_values_changed_by_tail",
        },
    }
    detail: dict[str, list[float]] = {}
    for name, aliases in fields.items():
        values = _collect_named_numbers(metadata, aliases)
        if not values:
            raise ValueError(f"transform metadata omits {name}")
        if any(value != 0.0 for value in values):
            raise ValueError(f"transform {name} is non-zero: {values}")
        detail[name] = values
    return detail


def _audit_final_archives(
    state: AuditState,
    manifest_info: Mapping[str, Any],
    selection_info: Mapping[str, Any],
    test_info: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    authority_payloads: list[Mapping[str, Any]] = []
    config_hashes: dict[str, set[str]] = {family: set() for family in FAMILIES}
    selected_configs = selection_info["selected_configs"]
    test_reference = test_info["reference"]
    for outer in OUTERS:
        expected_dates = manifest_info["test_dates"][outer]
        for seed in SEEDS:
            raw_path = raw_archive_path(state.root, outer, seed, "test")
            raw_hash = state.digest(raw_path) if raw_path.is_file() else None
            for family in FAMILIES:
                path = final_archive_path(state.root, outer, seed, family)
                if not state.require(path):
                    continue
                sidecar = _read_sidecar(state, path, hash_key="archive_sha256")
                metadata, signatures = _validate_archive_arrays(
                    path, expected_dates=expected_dates, state=state
                )
                _check_archive_identity(metadata, outer=outer, seed=seed, split="test")
                recorded_family = _identity_value(metadata, "family", "method")
                if str(recorded_family) != family:
                    raise ValueError(f"final archive family mismatch: {path}")
                if metadata.get("generated_after_selection_lock") is not True:
                    raise ValueError("final metadata lacks generated_after_selection_lock=true")
                if _require_hash(
                    metadata.get("selection_lock_sha256"),
                    name="final selection_lock_sha256",
                ) != selection_info["lock_sha256"]:
                    raise ValueError("final archive selection-lock hash mismatch")
                analysis_value = metadata.get(
                    "selection_analysis_sha256", metadata.get("analysis_sha256")
                )
                if _require_hash(
                    analysis_value, name="final selection_analysis_sha256"
                ) != selection_info["analysis_sha256"]:
                    raise ValueError("final archive selection-analysis hash mismatch")
                source_hash = metadata.get("source_test_raw_sha256")
                if _require_hash(source_hash, name="source_test_raw_sha256") != raw_hash:
                    raise ValueError("final archive is not tied to its sealed raw test archive")
                generated_at = _parse_timestamp(
                    metadata.get("generated_at_utc"), field_name="generated_at_utc"
                )
                if generated_at <= selection_info["locked_at"]:
                    raise ValueError("final archive timestamp precedes selection lock")
                config = metadata.get("selected_config")
                if config is None:
                    raise ValueError("final metadata omits selected_config")
                config_hash = canonical_json_sha256(config)
                declared_config_hash = metadata.get("selected_config_sha256", config_hash)
                if _require_hash(
                    declared_config_hash, name="selected_config_sha256"
                ) != config_hash:
                    raise ValueError("selected_config hash is invalid")
                if config != selected_configs[family]:
                    raise ValueError(
                        f"{family} selected_config differs from selection.lock"
                    )
                config_hashes[family].add(config_hash)
                diagnostics = _validate_transform_diagnostics(metadata)
                reference = {
                    key: signatures[key]
                    for key in ("observations_sha256", "zone_sha256", "day_sha256")
                }
                if reference != test_reference[outer]:
                    raise ValueError(f"final archive is misaligned with outer{outer} raw test")
                if sidecar.get("selected_config_sha256") not in (None, config_hash):
                    raise ValueError("final sidecar selected-config hash mismatch")
                authority_payloads.extend((metadata, sidecar))
                records.append(
                    {
                        "outer": outer,
                        "seed": seed,
                        "family": family,
                        "path": state.display(path),
                        "sha256": state.digest(path),
                        "selected_config_sha256": config_hash,
                        "diagnostics": diagnostics,
                        "shape": signatures["shape"],
                    }
                )
    expected_count = len(OUTERS) * len(SEEDS) * len(FAMILIES)
    if len(records) != expected_count:
        raise FileNotFoundError(
            f"found {len(records)}/{expected_count} hashed A0--A6 final archives"
        )
    inconsistent = {
        family: sorted(values) for family, values in config_hashes.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"selected config varies across outer/seed: {inconsistent}")
    _audit_authority_payloads(authority_payloads)
    return {
        "count": len(records),
        "records": records,
        "selected_config_sha256": {
            family: next(iter(config_hashes[family])) for family in FAMILIES
        },
        "strict_reversals": 0,
        "nonfinite": 0,
        "central_changed": 0,
    }


def _find_summary_path(state: AuditState, name: str) -> Path | None:
    for relative in SUMMARY_ARTIFACTS[name]:
        candidate = state.root / relative
        if candidate.is_file():
            state.critical_files.add(candidate.resolve())
            return candidate
    state.missing.append(SUMMARY_ARTIFACTS[name][0])
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _audit_summary_artifacts(state: AuditState) -> dict[str, Any]:
    paths = {name: _find_summary_path(state, name) for name in SUMMARY_ARTIFACTS}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(f"missing summary artifacts: {missing}")
    assert all(path is not None for path in paths.values())

    overall = _read_csv_rows(paths["overall_metrics"])  # type: ignore[arg-type]
    conditional = _read_csv_rows(paths["conditional_metrics"])  # type: ignore[arg-type]
    for label, rows in (("overall", overall), ("conditional", conditional)):
        method_key = "method" if "method" in rows[0] else "family"
        if method_key not in rows[0]:
            raise ValueError(f"{label} metrics omit method/family column")
        present_methods = {row[method_key] for row in rows}
        if not set(FAMILIES).issubset(present_methods):
            raise ValueError(f"{label} metrics omit A0--A6")
        if "outer" in rows[0]:
            present_outers = {
                int(row["outer"])
                for row in rows
                if str(row.get("outer", "")).isdigit()
            }
            if not set(OUTERS).issubset(present_outers):
                raise ValueError(f"{label} metrics omit one or more outer splits")

    json_payloads: dict[str, Any] = {}
    for name in (
        "atom_diagnostics",
        "quantization_audit",
        "paired_calendar_day_bootstrap",
        "noninferiority",
        "success_gates",
        "outer_consistency",
        "suc_protocol",
        "figure_manifest",
    ):
        payload = _read_json(paths[name])  # type: ignore[arg-type]
        if not isinstance(payload, (dict, list)) or len(payload) == 0:
            raise ValueError(f"{name} JSON is empty")
        json_payloads[name] = payload

    bootstrap = json_payloads["paired_calendar_day_bootstrap"]
    replicates = _collect_named_numbers(bootstrap, {"replicates", "bootstrap_replicates"})
    clusters = _collect_named_numbers(
        bootstrap, {"unique_calendar_days", "cluster_count", "clusters"}
    )
    if not replicates or any(int(value) != 5000 for value in replicates):
        raise ValueError("bootstrap does not certify 5000 replicates")
    if not clusters or any(int(value) != 150 for value in clusters):
        raise ValueError("bootstrap does not certify 150 disjoint outer test dates")

    gates = json_payloads["success_gates"]
    gate_map = gates.get("gates") if isinstance(gates, dict) else None
    if not isinstance(gate_map, dict) or not gate_map:
        raise ValueError("success_gates omits gate decisions")
    if int(gates.get("total", -1)) != len(gate_map):
        raise ValueError("success_gates total does not match gate decisions")
    if not all(isinstance(value, (bool, np.bool_)) for value in gate_map.values()):
        raise ValueError("success gate decisions must be boolean")

    suc_daily = _read_csv_rows(paths["suc_daily"])  # type: ignore[arg-type]
    suc_summary = _read_csv_rows(paths["suc_summary"])  # type: ignore[arg-type]

    figure_manifest = json_payloads["figure_manifest"]
    entries = (
        figure_manifest.get("files")
        if isinstance(figure_manifest, dict)
        else figure_manifest
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError("figure_manifest has no figure records")
    figure_records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("figure_manifest record is not an object")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe figure path: {relative}")
        path = state.root / relative
        if not state.require(path):
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".png", ".pdf", ".svg"} or path.stat().st_size == 0:
            raise ValueError(f"invalid figure artifact: {path}")
        expected = _require_hash(entry.get("sha256"), name=f"figure {relative}")
        if state.digest(path) != expected:
            raise ValueError(f"figure hash drift: {relative}")
        figure_records.append(
            {"path": relative.as_posix(), "sha256": expected, "bytes": path.stat().st_size}
        )

    return {
        "paths": {name: state.display(path) for name, path in paths.items()},
        "overall_rows": len(overall),
        "conditional_rows": len(conditional),
        "suc_daily_rows": len(suc_daily),
        "suc_summary_rows": len(suc_summary),
        "bootstrap_replicates": sorted(set(int(value) for value in replicates)),
        "bootstrap_clusters": sorted(set(int(value) for value in clusters)),
        "success_gates_passed": int(sum(bool(value) for value in gate_map.values())),
        "success_gates_total": len(gate_map),
        "figures": figure_records,
    }


def _manifest_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("files"), list):
        return list(payload["files"])
    if isinstance(payload, dict):
        result = []
        for path, detail in payload.items():
            if path == "schema" or not isinstance(detail, dict):
                continue
            result.append({"path": path, **detail})
        return result
    raise ValueError("experiment_manifest must be an object or contain files[]")


def _resolve_registered_path(state: AuditState, value: Any) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        workspace_candidate = (state.workspace / raw).resolve()
        root_candidate = (state.root / raw).resolve()
        if workspace_candidate.is_file():
            resolved = workspace_candidate
        elif root_candidate.is_file():
            resolved = root_candidate
        else:
            # Preserve the conventional workspace-relative interpretation in
            # the error message when neither candidate exists.
            resolved = workspace_candidate
    allowed = False
    for base in (state.workspace.resolve(), state.root.resolve()):
        try:
            resolved.relative_to(base)
            allowed = True
        except ValueError:
            pass
    if not allowed:
        raise ValueError(f"manifest path escapes registered roots: {value}")
    return resolved


def _audit_report_and_experiment_manifest(state: AuditState) -> dict[str, Any]:
    if not state.require(state.report, label=state.display(state.report)):
        raise FileNotFoundError(state.report)
    if state.report.stat().st_size < 100:
        raise ValueError("CAA report is unexpectedly short")
    manifest_path = state.root / EXPERIMENT_MANIFEST
    if not state.require(manifest_path):
        raise FileNotFoundError(manifest_path)
    records = _manifest_records(_read_json(manifest_path))
    if not records:
        raise ValueError("experiment_manifest contains no file records")
    covered: set[Path] = set()
    for record in records:
        path = _resolve_registered_path(state, record.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"experiment_manifest file is missing: {path}")
        actual = state.digest(path)
        expected = _require_hash(record.get("sha256"), name=f"manifest {path}")
        if actual != expected:
            raise ValueError(f"experiment_manifest hash drift: {state.display(path)}")
        if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
            raise ValueError(f"experiment_manifest byte-size drift: {state.display(path)}")
        covered.add(path.resolve())
    required_coverage = {
        path.resolve()
        for path in state.critical_files
        if path.name not in {EXPERIMENT_MANIFEST, "completion_audit.json", "manifest.csv"}
    }
    missing_coverage = sorted(
        state.display(path) for path in required_coverage.difference(covered)
    )
    if missing_coverage:
        raise ValueError(
            f"experiment_manifest omits {len(missing_coverage)} critical files: "
            f"{missing_coverage[:10]}"
        )
    return {
        "report": state.display(state.report),
        "report_bytes": state.report.stat().st_size,
        "manifest": state.display(manifest_path),
        "manifest_records": len(records),
        "critical_files_covered": len(required_coverage),
    }


def _write_manifest_csv(state: AuditState) -> tuple[Path, int]:
    excluded = {"completion_audit.json", "manifest.csv"}
    paths = [
        path.resolve()
        for path in state.root.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and not path.name.endswith(".tmp")
    ]
    if state.report.is_file():
        paths.append(state.report.resolve())
    rows = []
    for path in sorted(set(paths), key=lambda item: state.display(item)):
        rows.append(
            {
                "path": state.display(path),
                "bytes": int(path.stat().st_size),
                "sha256": state.digest(path),
            }
        )
    destination = state.root / "manifest.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination, len(rows)


def _write_completion(state: AuditState, payload: Mapping[str, Any]) -> Path:
    destination = state.root / "completion_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def audit_completion(
    root: str | Path = DEFAULT_ROOT,
    *,
    workspace: str | Path = WORKSPACE,
    report: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Audit a completed tree and always emit JSON/CSV audit products.

    The returned ``complete`` flag is the programmatic result.  Use
    :func:`main` for the registered non-zero process exit on failure.
    """

    state = AuditState(
        root=Path(root).resolve(),
        workspace=Path(workspace).resolve(),
        report=Path(report).resolve(),
    )
    state.root.mkdir(parents=True, exist_ok=True)

    manifest_info = state.guard("frozen_manifest_and_outer_dates", lambda: _audit_frozen_manifest(state))
    selection_info = None
    checkpoint_info = None
    calibration_info = None
    test_info = None
    final_info = None
    summary_info = None
    report_info = None

    if manifest_info is not None:
        selection_info = state.guard(
            "selection_lock_analysis_and_full_code",
            lambda: _audit_selection(state, manifest_info),
        )
        checkpoint_info = state.guard(
            "nine_audited_final_checkpoints", lambda: _audit_checkpoints(state)
        )
        calibration_info = state.guard(
            "nine_calibration_raw_archives",
            lambda: _audit_raw_archives(
                state, manifest_info, selection_info, split="calibration"
            ),
        )
        test_info = state.guard(
            "nine_sealed_test_raw_archives",
            lambda: _audit_raw_archives(
                state, manifest_info, selection_info, split="test"
            ),
        )
        if selection_info is not None and test_info is not None:
            final_info = state.guard(
                "sixty_three_hashed_final_archives",
                lambda: _audit_final_archives(
                    state, manifest_info, selection_info, test_info
                ),
            )
        else:
            state.record(
                "sixty_three_hashed_final_archives",
                False,
                "blocked by invalid selection lock or sealed test raw archives",
            )
    else:
        for name in (
            "selection_lock_analysis_and_full_code",
            "nine_audited_final_checkpoints",
            "nine_calibration_raw_archives",
            "nine_sealed_test_raw_archives",
            "sixty_three_hashed_final_archives",
        ):
            state.record(name, False, "blocked by invalid frozen manifest")

    summary_info = state.guard(
        "metrics_statistics_suc_and_figures", lambda: _audit_summary_artifacts(state)
    )
    # This check runs last because experiment_manifest must cover every
    # critical file discovered by all preceding checks.
    report_info = state.guard(
        "report_and_experiment_manifest",
        lambda: _audit_report_and_experiment_manifest(state),
    )

    missing_unique = sorted(set(state.missing))
    if missing_unique:
        state.record(
            "no_missing_artifacts",
            False,
            {"count": len(missing_unique), "paths": missing_unique},
        )
    else:
        state.record("no_missing_artifacts", True, {"count": 0, "paths": []})

    complete = all(item["passed"] for item in state.checks.values())
    manifest_csv, manifest_rows = _write_manifest_csv(state)
    payload: dict[str, Any] = {
        "schema": "caa_rahc_completion_audit_v1",
        "complete": bool(complete),
        "status": "complete" if complete else "incomplete",
        "root": str(state.root),
        "report": str(state.report),
        "contract": {
            "outers": list(OUTERS),
            "model_seeds": list(SEEDS),
            "families": list(FAMILIES),
            "days_per_outer": EXPECTED_DAYS_PER_OUTER,
            "authoritative_module": AUTHORITATIVE_MODULE,
            "prototype_policy": (
                "caa_rahc.candidates may be named only as prototype_role/streaming "
                "dependency; it and zero_atom/calibration may not be authoritative"
            ),
        },
        "counts": {
            "audited_checkpoints": (checkpoint_info or {}).get("count", 0),
            "calibration_raw_archives": (calibration_info or {}).get("count", 0),
            "sealed_test_raw_archives": (test_info or {}).get("count", 0),
            "final_scenario_archives": (final_info or {}).get("count", 0),
            "manifest_rows": manifest_rows,
        },
        "checks": state.checks,
        "missing": missing_unique,
        "failed": sorted(set(state.failed)),
        "manifest_csv": str(manifest_csv),
    }
    _write_completion(state, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed completion audit for frozen CAA-RAHC artifacts"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_completion(
        args.root, workspace=args.workspace, report=args.report
    )
    summary = {
        "complete": result["complete"],
        "missing": result["missing"],
        "failed": result["failed"],
        "completion_audit": str(Path(args.root).resolve() / "completion_audit.json"),
        "manifest_csv": result["manifest_csv"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_MODULE",
    "EXPECTED_DAYS_PER_OUTER",
    "FAMILIES",
    "OUTERS",
    "SEEDS",
    "audit_completion",
    "canonical_json_sha256",
    "checkpoint_path",
    "final_archive_path",
    "main",
    "raw_archive_path",
    "sha256_file",
    "sidecar_path",
]
