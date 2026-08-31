"""Frozen, auditable date protocol for the architecture-v1 experiments.

The two 150-day cross-model diagnostic panels have already influenced the
architecture decision.  They are therefore collected in the explicit
``r_seen`` role and quarantined from model fitting, checkpoint selection,
calibration, architecture selection, and final evaluation.

The local GEFCom2014 data contain no honest final test set after this reuse.
``final`` is consequently an empty, external-only reservation.  Code must not
silently relabel one of the local roles as a final test set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "architecture_v1_protocol_v1"
LOCAL_MODEL_ROLES = ("train", "validation", "calibration", "selection")
LOCAL_ROLES = (*LOCAL_MODEL_ROLES, "r_seen")
ALL_ROLES = (*LOCAL_ROLES, "final")
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "repro_configs" / "architecture_v1.json"
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA256 of a stable JSON representation."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def date_list_sha256(values: Iterable[object]) -> str:
    """Hash sorted ISO dates joined by newlines, without a trailing newline."""

    dates = _dates(values, allow_empty=True).astype(str).tolist()
    return hashlib.sha256("\n".join(dates).encode("ascii")).hexdigest()


def _dates(values: Iterable[object], *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(list(values), dtype="datetime64[D]")
    if result.ndim != 1:
        raise ValueError("date collection must be one-dimensional")
    if len(result) == 0:
        if allow_empty:
            return result
        raise ValueError("date collection must not be empty")
    if len(np.unique(result)) != len(result):
        raise ValueError("date collection contains duplicates")
    return np.sort(result)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_path(value: str | Path, *, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    project_candidate = config_path.resolve().parents[1] / candidate
    if project_candidate.exists():
        return project_candidate
    return (config_path.parent / candidate).resolve()


def _portable_path(path: Path, *, project_root: Path) -> str:
    """Prefer a repo-relative scientific identifier over a machine path."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        # External registries may legitimately live outside the repository;
        # their content hash, not this display label, remains authoritative.
        return path.name


def load_architecture_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    value = _read_json(source)
    if value.get("schema") != SCHEMA:
        raise ValueError(f"unexpected architecture-v1 schema in {source}")
    protocol = value.get("data_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("configuration is missing data_protocol")
    if protocol.get("r_seen_policy") != "diagnosis_only_quarantine":
        raise ValueError("r_seen_policy must be diagnosis_only_quarantine")
    if value.get("layout") != {
        "sample": "calendar_day",
        "zones": 10,
        "hours": 24,
        "condition_dim": 20,
    }:
        raise ValueError("architecture-v1 layout must remain [day,10,24,20]")
    return source, value


def _available_days_from_predictors(
    data_dir: Path, zones: Sequence[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read calendar keys from NWP files only; never open TestTar_W.csv."""

    per_zone: dict[int, np.ndarray] = {}
    files: list[dict[str, Any]] = []
    for zone in zones:
        parts: list[pd.DataFrame] = []
        for name in (f"Train_W_Zone{zone}.csv", f"TestPred_W_Zone{zone}.csv"):
            path = data_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, usecols=["ZONEID", "TIMESTAMP"])
            parts.append(frame)
            files.append(
                {
                    "path": name,
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
        joined = pd.concat(parts, ignore_index=True)
        if not (joined["ZONEID"] == zone).all():
            raise ValueError(f"zone identifier mismatch in predictor files for zone {zone}")
        timestamp = pd.to_datetime(joined["TIMESTAMP"], format="%Y%m%d %H:%M")
        if timestamp.duplicated().any():
            raise ValueError(f"duplicate predictor timestamp for zone {zone}")
        day = (timestamp - pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[D]")
        per_zone[zone] = np.sort(np.unique(day))
    reference = per_zone[int(zones[0])]
    for zone, days in per_zone.items():
        if not np.array_equal(days, reference):
            raise ValueError(f"predictor calendar for zone {zone} differs from zone {zones[0]}")
    return reference, {
        "scope": "predictor_calendar_only",
        "target_file_opened": False,
        "files": sorted(files, key=lambda item: item["path"]),
    }


def _load_r_seen(
    config: Mapping[str, Any], *, config_path: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    protocol = config["data_protocol"]
    specification = protocol.get("r_seen")
    if not isinstance(specification, dict):
        raise ValueError("data_protocol.r_seen is required")
    sources = specification.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("r_seen must contain at least two frozen quarantine registries")

    blocks: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for item in sources:
        registry_path = _resolve_path(item["path"], config_path=config_path)
        registry = _read_json(registry_path)
        if registry.get("schema") != item["schema"]:
            raise ValueError(f"unexpected registry schema: {registry_path}")
        field = item.get("date_blocks_field", "outer_test_dates")
        raw_blocks = registry.get(field)
        if not isinstance(raw_blocks, dict):
            raise ValueError(f"missing {field} in {registry_path}")
        dates = _dates(
            value for values in raw_blocks.values() for value in values
        )
        expected_count = int(item["expected_count"])
        expected_sha = str(item["expected_union_sha256"])
        if len(dates) != expected_count:
            raise ValueError(
                f"{registry_path} has {len(dates)} R-SEEN dates, expected {expected_count}"
            )
        actual_sha = date_list_sha256(dates)
        declared_sha_field = item.get(
            "declared_union_sha_field", "all_outer_test_sha256"
        )
        declared_sha = registry.get(declared_sha_field)
        if actual_sha != expected_sha or declared_sha != expected_sha:
            raise ValueError(f"R-SEEN registry union hash mismatch: {registry_path}")
        blocks.append(dates)
        audit.append(
            {
                "label": item["label"],
                "path": _portable_path(
                    registry_path, project_root=config_path.resolve().parents[1]
                ),
                "file_sha256": file_sha256(registry_path),
                "schema": item["schema"],
                "date_count": len(dates),
                "date_sha256": actual_sha,
            }
        )

    for left_index, left in enumerate(blocks):
        for right_index in range(left_index + 1, len(blocks)):
            if np.intersect1d(left, blocks[right_index]).size:
                raise ValueError(
                    "R-SEEN quarantine registries overlap: "
                    f"{sources[left_index]['label']} and "
                    f"{sources[right_index]['label']}"
                )
    union = np.sort(np.concatenate(blocks))
    expected_count = int(specification["expected_count"])
    expected_sha = str(specification["expected_union_sha256"])
    if len(union) != expected_count or date_list_sha256(union) != expected_sha:
        raise ValueError("combined R-SEEN catalog does not match the frozen config")
    return union, audit


def _allocate_local_roles(
    available: np.ndarray,
    r_seen: np.ndarray,
    specification: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not np.array_equal(np.intersect1d(r_seen, available), r_seen):
        missing = np.setdiff1d(r_seen, available).astype(str).tolist()
        raise ValueError(f"R-SEEN dates absent from predictor calendar: {missing}")
    remaining = np.setdiff1d(available, r_seen, assume_unique=True)
    expected_remaining = int(specification["expected_remaining_count"])
    if len(remaining) != expected_remaining:
        raise ValueError(
            f"remaining pool has {len(remaining)} dates, expected {expected_remaining}"
        )
    remaining_sha = date_list_sha256(remaining)
    if remaining_sha != specification["expected_remaining_sha256"]:
        raise ValueError("remaining date pool hash mismatch")

    order = tuple(specification["allocation_order"])
    if set(order) != set(LOCAL_MODEL_ROLES) or len(order) != len(LOCAL_MODEL_ROLES):
        raise ValueError("allocation_order must contain each local model role exactly once")
    counts = {key: int(value) for key, value in specification["role_counts"].items()}
    if set(counts) != set(LOCAL_MODEL_ROLES):
        raise ValueError("role_counts must specify all local model roles")
    if sum(counts.values()) != len(remaining):
        raise ValueError("local model role counts do not exhaust the non-R-SEEN pool")

    seed = int(specification["allocation_seed"])
    shuffled = remaining[np.random.default_rng(seed).permutation(len(remaining))]
    roles: dict[str, np.ndarray] = {}
    cursor = 0
    for role in order:
        roles[role] = np.sort(shuffled[cursor : cursor + counts[role]])
        cursor += counts[role]
    roles["r_seen"] = r_seen
    roles["final"] = np.asarray([], dtype="datetime64[D]")

    expected_hashes = specification["expected_role_sha256"]
    for role in (*LOCAL_MODEL_ROLES, "r_seen"):
        actual = date_list_sha256(roles[role])
        if actual != expected_hashes[role]:
            raise ValueError(f"frozen {role} date hash mismatch")
    return roles


def _assert_partition(roles: Mapping[str, np.ndarray], available: np.ndarray) -> None:
    for left_index, left in enumerate(LOCAL_ROLES):
        for right in LOCAL_ROLES[left_index + 1 :]:
            overlap = np.intersect1d(roles[left], roles[right])
            if overlap.size:
                raise ValueError(f"roles {left} and {right} overlap")
    union = np.sort(np.concatenate([roles[role] for role in LOCAL_ROLES]))
    if not np.array_equal(union, available):
        raise ValueError("local roles do not partition every predictor calendar date")
    if np.intersect1d(roles["r_seen"], roles["selection"]).size:
        raise ValueError("R-SEEN dates entered selection")
    if np.intersect1d(roles["r_seen"], roles["final"]).size:
        raise ValueError("R-SEEN dates entered final")
    if len(roles["final"]):
        raise ValueError("local GEFCom dates may not be assigned to final")


def _smoke_view(
    roles: Mapping[str, np.ndarray], smoke: Mapping[str, Any], enabled: bool
) -> dict[str, np.ndarray]:
    if not enabled:
        return {key: value.copy() for key, value in roles.items()}
    limits = smoke.get("max_days_per_role")
    if not isinstance(limits, dict):
        raise ValueError("smoke.max_days_per_role is required")
    result: dict[str, np.ndarray] = {}
    for role in LOCAL_ROLES:
        maximum = int(limits[role])
        if maximum < 1:
            raise ValueError(f"smoke role {role} must keep at least one date")
        result[role] = roles[role][:maximum].copy()
    result["final"] = roles["final"].copy()
    return result


def _month_blocks(dates: np.ndarray) -> dict[str, list[str]]:
    """Declare calendar-cluster units for dependent-date resampling."""

    result: dict[str, list[str]] = {}
    for value in dates.astype("datetime64[D]").astype(str).tolist():
        result.setdefault(value[:7], []).append(value)
    return result


@dataclass(frozen=True)
class ArchitectureProtocol:
    """Full frozen registry plus an optional, deterministic smoke view."""

    dates: Mapping[str, np.ndarray]
    full_dates: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    config_path: Path
    smoke: bool

    def role_dates(self, role: str, *, full: bool = False) -> np.ndarray:
        if role not in ALL_ROLES:
            raise KeyError(role)
        source = self.full_dates if full else self.dates
        return source[role].copy()

    def require_final_available(self) -> np.ndarray:
        """Fail closed until a separately frozen external registry is supplied."""

        raise RuntimeError(
            "architecture_v1 final test is external-only and is not registered; "
            "do not reuse a local GEFCom2014 role as final"
        )


def build_architecture_protocol(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    available_days: Iterable[object] | None = None,
    data_dir: str | Path | None = None,
    zones: Sequence[int] = tuple(range(1, 11)),
    smoke: bool | None = None,
) -> ArchitectureProtocol:
    """Build and verify the frozen architecture-v1 role registry.

    When ``available_days`` is omitted, calendar keys are read from NWP files
    only.  No target file is required to construct or audit the split.
    """

    source, config = load_architecture_config(config_path)
    protocol_spec = config["data_protocol"]
    predictor_calendar_audit: dict[str, Any]
    if available_days is None:
        root_value = data_dir if data_dir is not None else config["data_dir"]
        root = _resolve_path(root_value, config_path=source)
        available, predictor_calendar_audit = _available_days_from_predictors(root, zones)
    else:
        available = _dates(available_days)
        predictor_calendar_audit = {
            "scope": "caller_supplied_available_days",
            "target_file_opened": False,
        }
    expected_available = int(protocol_spec["expected_available_count"])
    if len(available) != expected_available:
        raise ValueError(
            f"predictor calendar has {len(available)} dates, expected {expected_available}"
        )
    if date_list_sha256(available) != protocol_spec["expected_available_sha256"]:
        raise ValueError("predictor calendar date hash mismatch")

    r_seen, source_audit = _load_r_seen(config, config_path=source)
    full_roles = _allocate_local_roles(available, r_seen, protocol_spec)
    _assert_partition(full_roles, available)
    smoke_enabled = bool(config["smoke"]["enabled"]) if smoke is None else bool(smoke)
    if smoke_enabled and config["smoke"].get("formal_config_policy") == "smoke_forbidden":
        raise RuntimeError("formal-v2 config forbids smoke/subset protocol views")
    active_roles = _smoke_view(full_roles, config["smoke"], smoke_enabled)

    full_dates_json = {
        role: full_roles[role].astype(str).tolist() for role in ALL_ROLES
    }
    active_dates_json = {
        role: active_roles[role].astype(str).tolist() for role in ALL_ROLES
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "name": "architecture_v1_leakage_safe_local_protocol",
        "config_path": _portable_path(source, project_root=source.parents[1]),
        "config_file_sha256": file_sha256(source),
        "allocation": {
            "seed": int(protocol_spec["allocation_seed"]),
            "algorithm": "numpy.default_rng(PCG64).permutation",
            "order": list(protocol_spec["allocation_order"]),
        },
        "roles": {
            "train": "model fitting only",
            "validation": "checkpoint/early-stop selection only",
            "calibration": "post-hoc calibration only; no architecture choice",
            "selection": "pre-registered architecture and ablation gates only",
            "r_seen": "architecture diagnosis only; quarantined from all fitting and scoring",
            "final": "external-only reservation; currently unavailable",
        },
        "full_dates": full_dates_json,
        "full_date_counts": {role: len(full_roles[role]) for role in ALL_ROLES},
        "full_date_sha256": {
            role: date_list_sha256(full_roles[role]) for role in ALL_ROLES
        },
        "calendar_month_blocks": {
            role: _month_blocks(full_roles[role]) for role in LOCAL_ROLES
        },
        "smoke": {
            "enabled": smoke_enabled,
            "dates": active_dates_json,
            "date_counts": {role: len(active_roles[role]) for role in ALL_ROLES},
            "date_sha256": {
                role: date_list_sha256(active_roles[role]) for role in ALL_ROLES
            },
        },
        "r_seen_sources": source_audit,
        "predictor_calendar": {
            **predictor_calendar_audit,
            "date_count": len(available),
            "date_sha256": date_list_sha256(available),
        },
        "missing_target_policy": config["preprocessing"]["target_missing"],
        "standardizers_fit_on": "raw-observed train cells only",
        "resampling_policy": {
            "primary_unit": "calendar day paired across models and training seeds",
            "dependence_cluster": "calendar month (YYYY-MM) from calendar_month_blocks",
            "warning": "random role allocation is not a contiguous moving-block design",
        },
        "external_final": protocol_spec["external_final"],
        "safety_invariants": {
            "split_before_target_imputation": True,
            "cross_role_target_fill": False,
            "r_seen_intersection_selection": 0,
            "r_seen_intersection_final": 0,
            "local_final_date_count": 0,
        },
    }
    manifest["protocol_sha256"] = canonical_sha256(manifest)
    return ArchitectureProtocol(
        dates=active_roles,
        full_dates=full_roles,
        manifest=manifest,
        config=config,
        config_path=source,
        smoke=smoke_enabled,
    )


def write_protocol_manifest(
    protocol: ArchitectureProtocol, path: str | Path
) -> dict[str, str]:
    """Write an audit manifest and SHA256 sidecar; return both hashes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        protocol.manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    destination.write_text(payload, encoding="utf-8")
    file_hash = file_sha256(destination)
    sidecar = destination.with_name(destination.name + ".sha256")
    sidecar.write_text(f"{file_hash}  {destination.name}\n", encoding="ascii")
    return {
        "protocol_sha256": str(protocol.manifest["protocol_sha256"]),
        "manifest_file_sha256": file_hash,
        "sidecar": str(sidecar),
    }


__all__ = [
    "ALL_ROLES",
    "ArchitectureProtocol",
    "DEFAULT_CONFIG_PATH",
    "LOCAL_MODEL_ROLES",
    "LOCAL_ROLES",
    "SCHEMA",
    "build_architecture_protocol",
    "canonical_sha256",
    "date_list_sha256",
    "file_sha256",
    "load_architecture_config",
    "write_protocol_manifest",
]
