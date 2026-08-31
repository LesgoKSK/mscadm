"""Generate the result-first Chinese report for the frozen CAA-RAHC study.

The report is a pure downstream renderer.  It reads only locked selection and
canonical evaluation artifacts, validates every reported number, and fails
closed before writing when a required file or field is absent.  It never runs
training, calibration, test application, evaluation, or SUC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / "outputs" / "caa_rahc"
DEFAULT_OUTPUT = WORKSPACE / "CAA_RAHC_EXPERIMENT_REPORT.md"
DEFAULT_PROTOCOL = WORKSPACE / "CAA_RAHC_FROZEN_PROTOCOL.md"

FAMILIES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
OUTERS = (1, 2, 3)
SEEDS = (0, 1, 2)
POINT_METRICS = (
    "CRPS",
    "coverage_90",
    "width_90",
    "winkler_90",
    "conditional_ACE90",
)
BOOTSTRAP_METRICS = {
    "CRPS": "crps_improvement",
    "coverage_90": "coverage_absolute_error_90_improvement",
    "width_90": "interval_width_90_change",
    "winkler_90": "winkler_90_improvement",
    "conditional_ACE90": "conditional_family_equal_ACE_90_improvement",
}

CANONICAL_PATHS = {
    "overall": "metrics/overall_metrics.csv",
    "conditional": "metrics/conditional_metrics.csv",
    "atom": "metrics/atom_diagnostics.json",
    "quantization": "metrics/quantization_audit.json",
    "rank": "metrics/rank_audit.csv",
    "bootstrap": "statistics/paired_calendar_day_bootstrap.json",
    "noninferiority": "statistics/noninferiority.json",
    "gates": "statistics/success_gates.json",
    "outer_consistency": "statistics/outer_consistency.json",
    "suc_daily": "suc/daily_results.csv",
    "suc_summary": "suc/summary.csv",
    "suc_protocol": "suc/protocol.json",
    "lock": "selection.lock.json",
    "selection_audit": "calibration/selection.audit.json",
    "experiment_manifest": "experiment_manifest.json",
}


class ReportInputError(RuntimeError):
    """A required, locked report input is absent or internally inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _need(mapping: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ReportInputError(f"{source} 缺少必需字段 {key}")
    return mapping[key]


def _finite(value: Any, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"{source} 不是数值: {value!r}") from error
    if not math.isfinite(result):
        raise ReportInputError(f"{source} 不是有限数值")
    return result


def _integer(value: Any, source: str) -> int:
    number = _finite(value, source)
    if not number.is_integer():
        raise ReportInputError(f"{source} 不是整数")
    return int(number)


def _boolean(value: Any, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ReportInputError(f"{source} 不是布尔值")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportInputError(f"无法读取 JSON: {path}: {error}") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ReportInputError(f"CSV 缺少表头: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReportInputError(f"无法读取 CSV: {path}: {error}") from error
    if not rows:
        raise ReportInputError(f"CSV 没有数据行: {path}")
    return rows


def _require_files(root: Path, protocol: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, relative in CANONICAL_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not protocol.is_file():
        missing.append(str(protocol))
    if missing:
        raise ReportInputError("缺少报告必需产物:\n- " + "\n- ".join(missing))
    return paths


def _normalize_selection_entry(family: str, value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportInputError(f"{source}.{family} 不是对象")
    entry = dict(value)
    selected = entry.get("selected", entry.get("candidate"))
    if selected is None:
        # selected_configs aliases sometimes store the configuration directly.
        selected = "A0" if bool(entry.get("fallback")) else family
    if not isinstance(selected, str) or not selected:
        raise ReportInputError(f"{source}.{family}.selected 无效")
    if "fallback" not in entry:
        raise ReportInputError(f"{source}.{family} 缺少必需字段 fallback")
    fallback = _boolean(entry["fallback"], f"{source}.{family}.fallback")
    if family != "A0" and fallback != (selected == "A0"):
        raise ReportInputError(f"{source}.{family} 的 selected/fallback 不一致")
    if family == "A0" and (selected != "A0" or fallback):
        raise ReportInputError("A0 必须是非回退基线")
    config = entry.get("config")
    if config is None and "selected" not in entry and "candidate" not in entry:
        config = {
            key: item
            for key, item in entry.items()
            if key not in {"fallback", "family", "decision_key"}
        }
    if not isinstance(config, Mapping):
        raise ReportInputError(f"{source}.{family}.config 缺失或不是对象")
    return {
        "family": family,
        "selected": selected,
        "fallback": fallback,
        "config": dict(config),
        "decision_key": entry.get("decision_key"),
    }


def _selection_catalog(lock: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    if lock.get("schema") != "caa_rahc_selection_lock_v1":
        raise ReportInputError("selection.lock.json schema 不正确")
    if "selected" in lock:
        raw = _need(lock, "selected", "selection.lock.json")
        if not isinstance(raw, Mapping):
            raise ReportInputError("selection.lock.json.selected 不是对象")
        expected_keys = {"main_A4", "A1", "A2", "A3", "A5", "A6"}
        if set(raw) != expected_keys:
            raise ReportInputError(
                "selection.lock.json.selected 必须恰含 main_A4,A1,A2,A3,A5,A6"
            )
        entries: dict[str, Any] = {
            "A0": {
                "family": "A0",
                "selected": "A0",
                "fallback": False,
                "config": {"definition": "locked A0 baseline"},
                "decision_key": "baseline",
            }
        }
        for family in FAMILIES[1:]:
            key = "main_A4" if family == "A4" else family
            entries[family] = _normalize_selection_entry(
                family, raw[key], f"selection.lock.json.selected.{key}"
            )
        audit_selected = _need(audit, "selected", "calibration/selection.audit.json")
        if audit_selected != raw:
            raise ReportInputError("selection.audit 与 selection.lock 的 selected 决策不同")
    elif "selected_configs" in lock:
        raw = lock["selected_configs"]
        if not isinstance(raw, Mapping) or set(raw) != set(FAMILIES):
            raise ReportInputError("selected_configs 别名必须恰含 A0--A6")
        entries = {
            family: _normalize_selection_entry(
                family, raw[family], f"selection.lock.json.selected_configs.{family}"
            )
            for family in FAMILIES
        }
        audit_alias = audit.get("selected_configs", audit.get("selected"))
        if audit_alias != raw:
            raise ReportInputError("selection.audit 与 lock 的 selected_configs 不同")
    else:
        raise ReportInputError("selection.lock 缺少 selected/selected_configs")

    protocol = _need(audit, "protocol", "calibration/selection.audit.json")
    if not isinstance(protocol, Mapping):
        raise ReportInputError("selection.audit.protocol 不是对象")
    access = str(_need(protocol, "test_access", "selection.audit.protocol")).lower()
    if "none" not in access or "calibration" not in access:
        raise ReportInputError("selection.audit 未证明选择阶段只读取 calibration")
    if lock.get("test_archives_accessed") is not False:
        raise ReportInputError("selection.lock 未证明选择阶段 test_archives_accessed=false")
    return entries


def _load_selection(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = _read_json(paths["lock"])
    if not isinstance(lock, dict):
        raise ReportInputError("selection.lock.json 不是对象")
    audit_relative = lock.get("selection_audit", CANONICAL_PATHS["selection_audit"])
    if Path(str(audit_relative)).as_posix() != CANONICAL_PATHS["selection_audit"]:
        raise ReportInputError("selection.lock 指向非 canonical selection.audit")
    actual_hash = sha256_file(paths["selection_audit"])
    recorded_hash = _need(lock, "selection_audit_sha256", "selection.lock.json")
    if recorded_hash != actual_hash:
        raise ReportInputError("selection.audit SHA-256 与 selection.lock 不符")
    audit_payload = _read_json(paths["selection_audit"])
    if not isinstance(audit_payload, dict):
        raise ReportInputError("selection.audit 不是对象")
    if audit_payload.get("schema") != "caa_rahc_calibration_selection_audit_v1":
        # selected_configs is a documented compatibility alias for synthetic
        # completion trees and early finalizers.
        if "selected_configs" not in audit_payload:
            raise ReportInputError("selection.audit schema 不正确")
    entries = _selection_catalog(lock, audit_payload)
    return lock, audit_payload, entries


def _outer_token(value: Any, source: str) -> str:
    text = str(value).strip()
    if text == "pooled":
        return text
    try:
        outer = int(text)
    except ValueError as error:
        raise ReportInputError(f"{source} outer 无效: {value!r}") from error
    if outer not in OUTERS:
        raise ReportInputError(f"{source} outer 不在 1,2,3/pooled")
    return str(outer)


@dataclass(frozen=True)
class MetricCatalog:
    rows: dict[tuple[str, str, int], dict[str, float]]
    pooled: dict[str, dict[str, float]]
    conditional_rows: list[dict[str, Any]]
    worst_a4_group: dict[str, Any]


def _load_metric_catalog(paths: Mapping[str, Path], *, a4_fallback: bool) -> MetricCatalog:
    raw_rows = _read_csv(paths["overall"])
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for index, row in enumerate(raw_rows, 2):
        source = f"overall_metrics.csv:{index}"
        method = str(_need(row, "method", source))
        if method not in FAMILIES:
            raise ReportInputError(f"{source} method 无效: {method}")
        outer = _outer_token(_need(row, "outer", source), source)
        seed = _integer(_need(row, "seed", source), f"{source}.seed")
        if seed not in SEEDS:
            raise ReportInputError(f"{source}.seed 不在 0,1,2")
        key = (method, outer, seed)
        if key in rows:
            raise ReportInputError(f"overall_metrics.csv 重复行 {key}")
        rows[key] = {
            metric: _finite(_need(row, metric, source), f"{source}.{metric}")
            for metric in POINT_METRICS
        }
    expected = {
        (family, outer, seed)
        for family in FAMILIES
        for outer in ("1", "2", "3", "pooled")
        for seed in SEEDS
    }
    if set(rows) != expected:
        missing = sorted(expected.difference(rows))
        extra = sorted(set(rows).difference(expected))
        raise ReportInputError(
            f"overall_metrics.csv catalog 不完整; missing={missing[:5]}, extra={extra[:5]}"
        )
    if a4_fallback:
        for outer in ("1", "2", "3", "pooled"):
            for seed in SEEDS:
                if rows[("A4", outer, seed)] != rows[("A0", outer, seed)]:
                    raise ReportInputError("A4 fallback=true，但 A4 test 指标不等于 A0")
    pooled = {
        family: {
            metric: sum(rows[(family, "pooled", seed)][metric] for seed in SEEDS)
            / len(SEEDS)
            for metric in POINT_METRICS
        }
        for family in FAMILIES
    }

    conditional_raw = _read_csv(paths["conditional"])
    required = {
        "method",
        "outer",
        "seed",
        "family",
        "group",
        "group_code",
        "coverage_90",
        "ACE_90",
        "undercoverage_90",
        "overcoverage_90",
        "interval_width_90",
        "winkler_score_90",
        "n",
    }
    conditional_rows: list[dict[str, Any]] = []
    for index, row in enumerate(conditional_raw, 2):
        source = f"conditional_metrics.csv:{index}"
        missing = required.difference(row)
        if missing:
            raise ReportInputError(f"{source} 缺少字段 {sorted(missing)}")
        method = row["method"]
        if method not in FAMILIES:
            raise ReportInputError(f"{source}.method 无效")
        outer = _outer_token(row["outer"], source)
        seed = _integer(row["seed"], f"{source}.seed")
        if seed not in SEEDS:
            raise ReportInputError(f"{source}.seed 无效")
        values = {
            key: _finite(row[key], f"{source}.{key}")
            for key in (
                "coverage_90",
                "ACE_90",
                "undercoverage_90",
                "overcoverage_90",
                "interval_width_90",
                "winkler_score_90",
                "n",
            )
        }
        if values["n"] <= 0:
            raise ReportInputError(f"{source}.n 必须为正")
        conditional_rows.append(
            {
                "method": method,
                "outer": outer,
                "seed": seed,
                "family": row["family"],
                "group": row["group"],
                "group_code": row["group_code"],
                **values,
            }
        )
    if not set(FAMILIES).issubset({row["method"] for row in conditional_rows}):
        raise ReportInputError("conditional_metrics.csv 未覆盖 A0--A6")
    pooled_a4 = [
        row
        for row in conditional_rows
        if row["method"] == "A4" and row["outer"] == "pooled"
    ]
    if not pooled_a4:
        raise ReportInputError("conditional_metrics.csv 缺少 pooled A4 行")
    worst = max(pooled_a4, key=lambda row: row["ACE_90"])
    return MetricCatalog(rows, pooled, conditional_rows, worst)


def _load_bootstrap(paths: Mapping[str, Path]) -> dict[str, dict[str, float | str]]:
    payload = _read_json(paths["bootstrap"])
    if not isinstance(payload, Mapping):
        raise ReportInputError("paired bootstrap JSON 不是对象")
    comparisons = _need(payload, "comparisons", "paired bootstrap")
    if not isinstance(comparisons, Mapping) or "A4" not in comparisons:
        raise ReportInputError("paired bootstrap.comparisons 缺少 A4")
    a4 = comparisons["A4"]
    if not isinstance(a4, Mapping):
        raise ReportInputError("paired bootstrap A4 不是对象")
    protocol = a4.get("protocol", payload.get("protocol"))
    if not isinstance(protocol, Mapping):
        raise ReportInputError("paired bootstrap 缺少 protocol")
    if _integer(_need(protocol, "replicates", "bootstrap.protocol"), "replicates") != 5000:
        raise ReportInputError("paired bootstrap replicates 必须为 5000")
    if _integer(
        _need(protocol, "unique_calendar_days", "bootstrap.protocol"),
        "unique_calendar_days",
    ) != 150:
        raise ReportInputError("paired bootstrap unique_calendar_days 必须为 150")
    metrics = _need(a4, "metrics", "bootstrap.comparisons.A4")
    if not isinstance(metrics, Mapping):
        raise ReportInputError("bootstrap A4.metrics 不是对象")
    result: dict[str, dict[str, float | str]] = {}
    for report_name, artifact_name in BOOTSTRAP_METRICS.items():
        record = _need(metrics, artifact_name, "bootstrap A4.metrics")
        if not isinstance(record, Mapping):
            raise ReportInputError(f"bootstrap {artifact_name} 不是对象")
        low = _finite(_need(record, "ci_low", artifact_name), f"{artifact_name}.ci_low")
        high = _finite(
            _need(record, "ci_high", artifact_name), f"{artifact_name}.ci_high"
        )
        point = _finite(
            _need(record, "point_estimate", artifact_name),
            f"{artifact_name}.point_estimate",
        )
        if low > high:
            raise ReportInputError(f"bootstrap {artifact_name} CI 上下界颠倒")
        direction = str(_need(record, "direction", artifact_name))
        result[report_name] = {
            "point": point,
            "low": low,
            "high": high,
            "direction": direction,
        }
    return result


def _load_gates_and_consistency(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = _read_json(paths["gates"])
    consistency = _read_json(paths["outer_consistency"])
    noninferiority = _read_json(paths["noninferiority"])
    if not isinstance(gates, Mapping) or not isinstance(consistency, Mapping):
        raise ReportInputError("success_gates/outer_consistency 必须是对象")
    gate_map = _need(gates, "gates", "success_gates")
    if not isinstance(gate_map, Mapping) or not gate_map:
        raise ReportInputError("success_gates.gates 缺失或为空")
    decisions = {
        str(name): _boolean(value, f"success_gates.gates.{name}")
        for name, value in gate_map.items()
    }
    passed = _integer(_need(gates, "passed", "success_gates"), "success_gates.passed")
    total = _integer(_need(gates, "total", "success_gates"), "success_gates.total")
    if total != len(decisions) or passed != sum(decisions.values()):
        raise ReportInputError("success_gates passed/total 与 gates 不一致")
    if _boolean(_need(gates, "all_passed", "success_gates"), "all_passed") != all(
        decisions.values()
    ):
        raise ReportInputError("success_gates all_passed 不一致")
    gates = {**dict(gates), "gates": decisions, "passed": passed, "total": total}

    if consistency.get("primary") != "A4" or consistency.get("baseline") != "A0":
        raise ReportInputError("outer_consistency primary/baseline 不是 A4/A0")
    consistency_passed = _boolean(
        _need(consistency, "passed", "outer_consistency"), "outer_consistency.passed"
    )
    per_outer_seed = _need(consistency, "per_outer_seed", "outer_consistency")
    per_outer = _need(consistency, "per_outer", "outer_consistency")
    if len(per_outer_seed) != 9 or len(per_outer) != 3:
        raise ReportInputError("outer_consistency 必须含 9 个 outer-seed 与 3 个 outer 汇总")
    same_count = _integer(
        _need(
            consistency,
            "same_primary_direction_outer_count",
            "outer_consistency",
        ),
        "same_primary_direction_outer_count",
    )
    if not 0 <= same_count <= 3:
        raise ReportInputError("same_primary_direction_outer_count 超出 0..3")
    if "outer_consistency" in decisions and decisions["outer_consistency"] != consistency_passed:
        raise ReportInputError("success gate 与 outer_consistency.json 决策不一致")
    if not isinstance(noninferiority, Mapping):
        raise ReportInputError("noninferiority.json 不是对象")
    if noninferiority.get("baseline") != "A0":
        raise ReportInputError("noninferiority baseline 不是 A0")
    comparisons = _need(noninferiority, "comparisons", "noninferiority")
    if not isinstance(comparisons, Mapping) or "A4" not in comparisons:
        raise ReportInputError("noninferiority.comparisons 缺少 A4")
    return gates, {**dict(consistency), "passed": consistency_passed, "same_count": same_count}


def _seed_entries(payload: Mapping[str, Any], family: str, source: str) -> list[Mapping[str, Any]]:
    by_family = _need(payload, "by_family_seed", source)
    if not isinstance(by_family, Mapping) or family not in by_family:
        raise ReportInputError(f"{source}.by_family_seed 缺少 {family}")
    raw = by_family[family]
    if not isinstance(raw, Mapping):
        raise ReportInputError(f"{source}.{family} 不是对象")
    entries = []
    for seed in SEEDS:
        value = raw.get(str(seed), raw.get(f"seed{seed}"))
        if not isinstance(value, Mapping):
            raise ReportInputError(f"{source}.{family} 缺少 seed{seed}")
        entries.append(value)
    return entries


def _mean_field(entries: Iterable[Mapping[str, Any]], key: str, source: str) -> float:
    values = [_finite(_need(entry, key, source), f"{source}.{key}") for entry in entries]
    return sum(values) / len(values)


def _max_field(entries: Iterable[Mapping[str, Any]], key: str, source: str) -> float:
    values = [_finite(_need(entry, key, source), f"{source}.{key}") for entry in entries]
    return max(values)


def _load_atom_summary(paths: Mapping[str, Path]) -> dict[str, Any]:
    atom = _read_json(paths["atom"])
    quant = _read_json(paths["quantization"])
    if not isinstance(atom, Mapping) or not isinstance(quant, Mapping):
        raise ReportInputError("atom/quantization JSON 必须是对象")
    atom_entries = _seed_entries(atom, "A4", "atom_diagnostics")
    quant_entries = _seed_entries(quant, "A4", "quantization_audit")
    scores = []
    reliability_bins = 0
    for entry in atom_entries:
        score = _need(entry, "analytic_scores", "atom_diagnostics.A4")
        if not isinstance(score, Mapping):
            raise ReportInputError("atom_diagnostics A4 analytic_scores 不是对象")
        for key in (
            "zero_Brier",
            "zero_log_loss",
            "zero_predicted_rate",
            "zero_observed_rate",
            "one_Brier",
            "one_log_loss",
            "one_predicted_rate",
            "one_observed_rate",
        ):
            _finite(_need(score, key, "A4 analytic_scores"), key)
        scores.append(score)
        zero_rel = _need(entry, "zero_reliability", "atom_diagnostics.A4")
        one_rel = _need(entry, "one_reliability", "atom_diagnostics.A4")
        if not isinstance(zero_rel, list) or not isinstance(one_rel, list):
            raise ReportInputError("atom reliability 必须是列表")
        reliability_bins += len(zero_rel) + len(one_rel)
    for entry in quant_entries:
        if _integer(_need(entry, "members", "quantization.A4"), "members") != 100:
            raise ReportInputError("finite-M quantization 必须对应 M=100")
        resolution = _finite(
            _need(entry, "probability_resolution", "quantization.A4"),
            "probability_resolution",
        )
        if not math.isclose(resolution, 0.01, rel_tol=0.0, abs_tol=1e-12):
            raise ReportInputError("M=100 probability_resolution 必须为 0.01")
    return {
        "zero_Brier": _mean_field(scores, "zero_Brier", "analytic_scores"),
        "zero_log_loss": _mean_field(scores, "zero_log_loss", "analytic_scores"),
        "zero_predicted_rate": _mean_field(
            scores, "zero_predicted_rate", "analytic_scores"
        ),
        "zero_observed_rate": _mean_field(
            scores, "zero_observed_rate", "analytic_scores"
        ),
        "one_Brier": _mean_field(scores, "one_Brier", "analytic_scores"),
        "one_log_loss": _mean_field(scores, "one_log_loss", "analytic_scores"),
        "one_predicted_rate": _mean_field(
            scores, "one_predicted_rate", "analytic_scores"
        ),
        "one_observed_rate": _mean_field(
            scores, "one_observed_rate", "analytic_scores"
        ),
        "members": 100,
        "probability_resolution": 0.01,
        "zero_quantization_MAE": _mean_field(
            quant_entries, "zero_quantization_MAE", "quantization.A4"
        ),
        "zero_quantization_max_abs": _max_field(
            quant_entries, "zero_quantization_max_abs", "quantization.A4"
        ),
        "one_quantization_MAE": _mean_field(
            quant_entries, "one_quantization_MAE", "quantization.A4"
        ),
        "one_quantization_max_abs": _max_field(
            quant_entries, "one_quantization_max_abs", "quantization.A4"
        ),
        "reliability_bins": reliability_bins,
    }


def _optional_number(row: Mapping[str, Any], key: str, source: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return _finite(value, f"{source}.{key}")


def _load_rank_summary(paths: Mapping[str, Path], *, a4_fallback: bool) -> dict[str, Any]:
    rows = _read_csv(paths["rank"])
    required = {
        "method",
        "outer",
        "seed",
        "selected",
        "fallback",
        "exact_A0",
        "comparison_source",
        "strict_reversals",
        "raw_ties_broken",
        "strict_pairs_collapsed",
        "raw_tied_pairs",
        "calibrated_tied_pairs",
        "stable_ordinal_rank_matches",
        "stable_ordinal_rank_total",
        "stable_ordinal_rank_fraction",
        "pair_comparisons",
        "central_values_changed_by_tail",
        "nonfinite_input_values",
        "nonfinite_intermediate_values",
    }
    catalog: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows, 2):
        source = f"rank_audit.csv:{index}"
        missing = required.difference(row)
        if missing:
            raise ReportInputError(f"{source} 缺少字段 {sorted(missing)}")
        method = row["method"]
        outer = _integer(row["outer"], f"{source}.outer")
        seed = _integer(row["seed"], f"{source}.seed")
        key = (method, outer, seed)
        if method not in FAMILIES or outer not in OUTERS or seed not in SEEDS or key in catalog:
            raise ReportInputError(f"{source} method/outer/seed 无效或重复")
        comparison = row["comparison_source"]
        expected_comparison = "raw_to_A0" if method == "A0" else "A0_to_family"
        if comparison != expected_comparison:
            raise ReportInputError(f"{source}.comparison_source 不正确")
        record = {
            "selected": row["selected"],
            "fallback": _boolean(row["fallback"], f"{source}.fallback"),
            "exact_A0": _boolean(row["exact_A0"], f"{source}.exact_A0"),
        }
        for field in (
            "strict_reversals",
            "raw_ties_broken",
            "strict_pairs_collapsed",
            "raw_tied_pairs",
            "calibrated_tied_pairs",
            "stable_ordinal_rank_matches",
            "stable_ordinal_rank_total",
            "stable_ordinal_rank_fraction",
            "pair_comparisons",
        ):
            record[field] = _finite(row[field], f"{source}.{field}")
        for field in (
            "central_values_changed_by_tail",
            "nonfinite_input_values",
            "nonfinite_intermediate_values",
        ):
            record[field] = _optional_number(row, field, source)
        catalog[key] = record
    expected = {
        (family, outer, seed)
        for family in FAMILIES
        for outer in OUTERS
        for seed in SEEDS
    }
    if set(catalog) != expected:
        raise ReportInputError("rank_audit.csv 必须恰含 A0--A6 × 3 outer × 3 seed")
    if a4_fallback and any(
        not catalog[("A4", outer, seed)]["exact_A0"]
        for outer in OUTERS
        for seed in SEEDS
    ):
        raise ReportInputError("A4 fallback=true，但 rank audit 未标 exact_A0")
    if not a4_fallback and any(
        catalog[("A4", outer, seed)]["fallback"]
        for outer in OUTERS
        for seed in SEEDS
    ):
        raise ReportInputError("A4 lock 非 fallback，但 rank audit 标为 fallback")

    by_family: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        selected = [
            catalog[(family, outer, seed)] for outer in OUTERS for seed in SEEDS
        ]
        matches = sum(item["stable_ordinal_rank_matches"] for item in selected)
        total = sum(item["stable_ordinal_rank_total"] for item in selected)
        by_family[family] = {
            "strict_reversals": int(sum(item["strict_reversals"] for item in selected)),
            "raw_ties_broken": int(sum(item["raw_ties_broken"] for item in selected)),
            "strict_pairs_collapsed": int(
                sum(item["strict_pairs_collapsed"] for item in selected)
            ),
            "raw_tied_pairs": int(sum(item["raw_tied_pairs"] for item in selected)),
            "calibrated_tied_pairs": int(
                sum(item["calibrated_tied_pairs"] for item in selected)
            ),
            "stable_rank_fraction": matches / total if total > 0 else 0.0,
            "central_changed": int(
                sum(
                    item["central_values_changed_by_tail"] or 0.0
                    for item in selected
                )
            ),
            "nonfinite": int(
                sum(
                    (item["nonfinite_input_values"] or 0.0)
                    + (item["nonfinite_intermediate_values"] or 0.0)
                    for item in selected
                )
            ),
        }
    if any(item["strict_reversals"] != 0 for item in by_family.values()):
        raise ReportInputError("rank audit 存在 strict reversals")
    if any(item["central_changed"] != 0 for item in by_family.values()):
        raise ReportInputError("tail transformation 改变了 central values")
    if any(item["nonfinite"] != 0 for item in by_family.values()):
        raise ReportInputError("rank/transform audit 存在 nonfinite values")
    return {"by_family": by_family, "rows": len(rows)}


def _load_suc(paths: Mapping[str, Path]) -> dict[str, Any]:
    daily = _read_csv(paths["suc_daily"])
    summary = _read_csv(paths["suc_summary"])
    protocol = _read_json(paths["suc_protocol"])
    if not isinstance(protocol, Mapping):
        raise ReportInputError("SUC protocol 不是对象")
    scope = str(_need(protocol, "statistical_scope", "suc.protocol")).lower()
    if "descriptive" not in scope or str(
        _need(protocol, "success_gate_role", "suc.protocol")
    ).lower() != "none":
        raise ReportInputError("SUC protocol 未声明 descriptive 且非 success gate")
    by_family: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(summary, 2):
        source = f"suc/summary.csv:{index}"
        family = str(_need(row, "family", source))
        if family in by_family:
            raise ReportInputError(f"SUC summary family 重复: {family}")
        paired = _integer(_need(row, "paired_cases", source), f"{source}.paired_cases")
        successful = _integer(
            _need(row, "successful_cases", source), f"{source}.successful_cases"
        )
        if not 0 <= successful <= paired:
            raise ReportInputError(f"{source} successful_cases 无效")
        cost = _optional_number(row, "mean_realized_total_cost", source)
        by_family[family] = {
            "paired_cases": paired,
            "successful_cases": successful,
            "mean_realized_total_cost": cost,
        }
    if not {"A0", "A4"}.issubset(by_family):
        raise ReportInputError("SUC summary 必须至少包含 A0 与 A4")
    if len(daily) == 0:
        raise ReportInputError("SUC daily_results 为空")
    return {"by_family": by_family, "daily_rows": len(daily), "protocol": protocol}


def _fmt(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt_ci(record: Mapping[str, Any]) -> str:
    return f"[{_fmt(float(record['low']))}, {_fmt(float(record['high']))}]"


def _config_text(config: Mapping[str, Any]) -> str:
    preferred = (
        "atom_strength",
        "tail_strength",
        "dmax",
        "width_delta_cap",
        "width_cap_reference",
        "strength",
        "tail_rule",
        "definition",
    )
    selected = {key: config[key] for key in preferred if key in config}
    if not selected:
        selected = dict(config)
    return "<br>".join(f"{key}={value}" for key, value in selected.items())


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _render_report(
    *,
    selection: Mapping[str, Any],
    metrics: MetricCatalog,
    bootstrap: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
    consistency: Mapping[str, Any],
    atom: Mapping[str, Any],
    rank: Mapping[str, Any],
    suc: Mapping[str, Any],
) -> str:
    a4 = selection["A4"]
    fallback = bool(a4["fallback"])
    fallback_sentence = (
        "A4 已精确回退到 A0；因此本轮不能据此宣称 A4 相对 A0 有预测改进。"
        if fallback
        else f"A4 未回退，锁定候选为 `{a4['selected']}`。"
    )
    p0 = metrics.pooled["A0"]
    p4 = metrics.pooled["A4"]
    result_rows = []
    labels = {
        "CRPS": "CRPS（A0−A4 为正表示改善）",
        "coverage_90": "90% coverage（CI 对应绝对误差改善）",
        "width_90": "W90（差值为 A4−A0）",
        "winkler_90": "Winkler-90（A0−A4 为正表示改善）",
        "conditional_ACE90": "conditional ACE90（A0−A4 为正表示改善）",
    }
    for metric in POINT_METRICS:
        record = bootstrap[metric]
        result_rows.append(
            (
                labels[metric],
                _fmt(p0[metric]),
                _fmt(p4[metric]),
                _fmt(float(record["point"])),
                _fmt_ci(record),
            )
        )

    selection_rows = [
        (
            family,
            selection[family]["selected"],
            "是" if selection[family]["fallback"] else "否",
            _config_text(selection[family]["config"]),
        )
        for family in FAMILIES
    ]
    ablation_rows = [
        (
            family,
            *(_fmt(metrics.pooled[family][metric]) for metric in POINT_METRICS),
        )
        for family in FAMILIES
    ]
    consistency_rows = []
    for outer in OUTERS:
        for seed in SEEDS:
            baseline = metrics.rows[("A0", str(outer), seed)]
            primary = metrics.rows[("A4", str(outer), seed)]
            crps_relative = (
                (primary["CRPS"] - baseline["CRPS"]) / baseline["CRPS"] * 100.0
            )
            consistency_rows.append(
                (
                    outer,
                    seed,
                    _fmt(crps_relative, 4) + "%",
                    _fmt((primary["coverage_90"] - baseline["coverage_90"]) * 100, 4)
                    + " pp",
                    _fmt(primary["width_90"] - baseline["width_90"]),
                    _fmt(
                        primary["conditional_ACE90"]
                        - baseline["conditional_ACE90"]
                    ),
                )
            )
    rank_rows = [
        (
            family,
            rank["by_family"][family]["strict_reversals"],
            rank["by_family"][family]["raw_ties_broken"],
            rank["by_family"][family]["strict_pairs_collapsed"],
            _fmt(rank["by_family"][family]["stable_rank_fraction"]),
            rank["by_family"][family]["central_changed"],
            rank["by_family"][family]["nonfinite"],
        )
        for family in FAMILIES
    ]
    suc_rows = []
    for family, values in suc["by_family"].items():
        cost = values["mean_realized_total_cost"]
        suc_rows.append(
            (
                family,
                f"{values['successful_cases']}/{values['paired_cases']}",
                "NA（无有限汇总）" if cost is None else _fmt(cost, 3),
            )
        )
    gate_rows = [
        (name, "通过" if passed else "未通过") for name, passed in gates["gates"].items()
    ]
    worst = metrics.worst_a4_group

    return f"""# CAA-RAHC 冻结实验报告

## 结论摘要（结果先行）

本结果属于 **post-freeze internal nested outer-split confirmation**（冻结后的内部 nested outer-split 确认），**不是外部确认**。{fallback_sentence}预注册成功门槛通过 **{gates['passed']}/{gates['total']}**；全部门槛是否通过：**{'是' if gates['all_passed'] else '否'}**。

下表的 A0/A4 点值来自 150 个互斥 sealed outer-test 日期上的三 seed pooled 结果。95% CI 来自 5,000 次按日历日成簇的 paired bootstrap，CI 对应表中明确写出的 A4−A0 或 A0−A4 差值定义，不能误读为单个方法点值的置信区间。

{_table(('指标', 'A0 点值', 'A4 点值', '配对差值点估计', '95% CI'), result_rows)}

## 1. 证据等级与数据使用边界

实验设计及阈值以 [CAA_RAHC_FROZEN_PROTOCOL.md](CAA_RAHC_FROZEN_PROTOCOL.md) 为准。三个 outer 各含 50 个 sealed test 日期，合计 150 日；本轮每个 outer 都重新训练三个底模 seed。旧研究中的 test 划分在本协议中**仅作为 calibration/候选选择数据**，没有充当本轮 sealed outer test。selection audit 明确记录选择阶段 test access 为 none，lock 记录 `test_archives_accessed=false`。

尽管方法、网格和成功门槛已冻结，本实验仍只是在同一 GEFCom2014 数据域内做内部确认。历史研究已接触过该数据集的全部年份，因此这里不能称为独立外部确认，也不能替代新年份、新场站或新公开数据集上的前瞻验证。

## 2. 方法与锁定流程

- **A0**：按小时 empirical PIT calibration 与 finite-ensemble linear tails，是所有约束和差值的基线。
- **A1**：旧 full RAHC 的 linear-tail 消融。
- **A2**：atom-only，用于分离边界原子修正。
- **A3**：regularized gate-only，不改变 atom。
- **A4**：atom + regularized local gate，预注册主方法；若约束集为空则精确回退 A0。
- **A5**：与 A4 候选相同，但选择时忽略非劣约束，仅作 unconstrained 消融。
- **A6**：atom + no-shrink gate，用于检验 shrinkage。

所有 calibration 决策先写入 `calibration/selection.audit.json`，其 SHA-256 再写入 `selection.lock.json`。同一锁定配置跨三个 outer 和三个 model seeds 使用；每个 outer 只重新拟合其 calibration-only 参数模型。

## 3. Calibration 选择结果

{_table(('族', '锁定候选', '回退 A0', '关键配置'), selection_rows)}

主路径结论：{fallback_sentence}

## 4. Sealed test 总体结果与 A0–A6 消融

以下均为三个 pooled seed 的算术平均；conditional ACE90 采用 family-equal 聚合。

{_table(('方法', 'CRPS', 'coverage90', 'W90', 'Winkler90', 'conditional ACE90'), ablation_rows)}

### A4 相对 A0 的 paired day-bootstrap

{_table(('指标', '差值方向', '点估计', '95% CI'), [(metric, bootstrap[metric]['direction'], _fmt(float(bootstrap[metric]['point'])), _fmt_ci(bootstrap[metric])) for metric in POINT_METRICS])}

## 5. 每个 outer/seed 的一致性

{_table(('outer', 'seed', 'CRPS 相对变化', 'coverage90 变化', 'W90 变化', 'conditional ACE90 变化'), consistency_rows)}

冻结 outer-consistency 审计结论为 **{'通过' if consistency['passed'] else '未通过'}**；三个 outer 中与主方向一致的数量为 **{consistency['same_count']}/3**。该判断还应用了 artifact 中冻结的 catastrophic rule，不能用 pooled 均值替代。

## 6. 条件校准

`conditional_metrics.csv` 共含 {len(metrics.conditional_rows)} 个 group-level 记录。pooled A4 中 ACE90 最大的记录属于 `{worst['family']}` / `{worst['group_code']}`，coverage90={_fmt(worst['coverage_90'])}，ACE90={_fmt(worst['ACE_90'])}，样本 cell 数 n={_fmt(worst['n'], 0)}。总体 family-equal conditional ACE90 的 A0/A4 点值及 paired CI 已在摘要表中报告；不能以单个最差 group 代替 family-equal 主统计量。

## 7. Boundary atoms 与 finite-M 量化

A4 的解析 atom 诊断按三个 seed 汇总：零事件 Brier={_fmt(atom['zero_Brier'])}、log loss={_fmt(atom['zero_log_loss'])}、平均预测率={_fmt(atom['zero_predicted_rate'])}、观察率={_fmt(atom['zero_observed_rate'])}；一事件对应 Brier={_fmt(atom['one_Brier'])}、log loss={_fmt(atom['one_log_loss'])}、平均预测率={_fmt(atom['one_predicted_rate'])}、观察率={_fmt(atom['one_observed_rate'])}。可靠性诊断共读取 {atom['reliability_bins']} 个 seed-bin 记录。

场景数固定为 M={atom['members']}，所以概率分辨率为 {atom['probability_resolution']:.2f}。解析 atom 到 finite ensemble 的零事件量化 MAE={_fmt(atom['zero_quantization_MAE'])}、跨 seed 最大绝对误差={_fmt(atom['zero_quantization_max_abs'])}；一事件分别为 {_fmt(atom['one_quantization_MAE'])} 与 {_fmt(atom['one_quantization_max_abs'])}。解析概率与场景中实际 0/1 频率不是同一个对象，尤其不能把稀有上边界事件的 1/M 离散化误差解释成结构概率估计误差。

## 8. Rank、ties 与数值安全

{_table(('方法', 'strict reversals', 'raw ties broken', 'strict pairs collapsed', 'stable ordinal rank fraction', 'central changed', 'nonfinite'), rank_rows)}

全部方法的 strict reversals、tail 引起的 central-value changes 与 nonfinite 计数均为零。`strict pairs collapsed` 可以因 atom/ties 而非零，它表示严格次序被压成 tie，不等同于反序；因此本报告同时给出 ties 与 stable ordinal rank，而不只报一个“无反序”结论。

## 9. SUC 描述性敏感性

{_table(('方法', '成功求解/paired cases', '平均 realized total cost'), suc_rows)}

SUC 共读取 {suc['daily_rows']} 条 paired method-case 记录。其 protocol 明确标为 **descriptive sensitivity only**，`success_gate_role=none`：它不是主/次成功门槛，也没有用于方法选择；这里不据此给出显著性或优越性结论。

## 10. 预注册成功门槛

{_table(('门槛', '结果'), gate_rows)}

合计通过 **{gates['passed']}/{gates['total']}**。科学结果是否成功与实验是否完整是两件事；即使门槛未全部通过，也必须完整保留 fallback、消融、CI 与失败门槛。

## 11. 限制与下一步

1. 这是同一数据域内的内部 outer-split confirmation，不是外部确认；下一步应使用未被任何旧研究接触的新年份、新场站或另一公开数据集。
2. 三 outer × 三 seeds 改善了内部稳定性审计，但不能把九个模型重复当成九份独立数据；不确定性以日历日为 cluster。
3. 上边界事件极少，解析 one-atom 结果应保守解释；需要更多真实边界事件才能验证可迁移性。
4. M=100 带来 0.01 的离散概率分辨率；下一步应预注册更大 ensemble 的量化敏感性，而不是事后选择 M。
5. SUC 仅是固定 RTS-24 proxy 下的描述性下游敏感性；需要独立系统、需求和成本设定做外部运营验证。
6. 下一轮确认必须重新训练底模、重新执行 selection lock，并保持 outer test 在锁定前不可访问。

## 12. 可追溯性

- 冻结协议：[CAA_RAHC_FROZEN_PROTOCOL.md](CAA_RAHC_FROZEN_PROTOCOL.md)
- 选择审计：`outputs/caa_rahc/calibration/selection.audit.json`
- 选择锁：`outputs/caa_rahc/selection.lock.json`
- 总体与条件指标：`outputs/caa_rahc/metrics/`
- paired bootstrap、非劣与成功门槛：`outputs/caa_rahc/statistics/`
- 描述性 SUC：`outputs/caa_rahc/suc/`
- 完整文件哈希：`outputs/caa_rahc/experiment_manifest.json`
"""


def _prepare_manifest_update(
    manifest_path: Path,
    *,
    workspace: Path,
    output: Path,
    report_bytes: bytes,
) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ReportInputError("experiment_manifest.json 缺少 files[]")
    try:
        relative = output.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise ReportInputError("报告输出必须位于 workspace 内") from error
    record = {
        "path": relative,
        "bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    files = []
    replaced = False
    seen: set[str] = set()
    for raw in payload["files"]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ReportInputError("experiment_manifest files[] 含畸形记录")
        path = str(raw["path"]).replace("\\", "/")
        if path in seen:
            raise ReportInputError(f"experiment_manifest path 重复: {path}")
        seen.add(path)
        if path == relative or Path(path).name == output.name:
            files.append(record)
            replaced = True
        else:
            files.append(dict(raw))
    if not replaced:
        files.append(record)
    files.sort(key=lambda item: str(item["path"]))
    return {**payload, "files": files}


def _write_report_and_manifest(
    output: Path,
    manifest_path: Path,
    content: str,
    manifest_payload: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = output.with_suffix(output.suffix + ".tmp")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if report_temporary.exists() or manifest_temporary.exists():
        raise ReportInputError("stale .tmp 文件阻止原子写入报告/manifest")
    report_temporary.write_bytes(content.encode("utf-8"))
    manifest_temporary.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_temporary.replace(output)
    manifest_temporary.replace(manifest_path)


def generate_report(
    root: str | Path = DEFAULT_ROOT,
    *,
    workspace: str | Path = WORKSPACE,
    output: str | Path = DEFAULT_OUTPUT,
    protocol: str | Path = DEFAULT_PROTOCOL,
) -> Path:
    """Validate canonical artifacts, render the report, and refresh its hash."""

    root_path = Path(root).resolve()
    workspace_path = Path(workspace).resolve()
    output_path = Path(output).resolve()
    protocol_path = Path(protocol).resolve()
    paths = _require_files(root_path, protocol_path)
    lock, selection_audit, selection = _load_selection(paths)
    del lock, selection_audit
    fallback = bool(selection["A4"]["fallback"])
    metrics = _load_metric_catalog(paths, a4_fallback=fallback)
    bootstrap = _load_bootstrap(paths)
    gates, consistency = _load_gates_and_consistency(paths)
    atom = _load_atom_summary(paths)
    rank = _load_rank_summary(paths, a4_fallback=fallback)
    suc = _load_suc(paths)
    content = _render_report(
        selection=selection,
        metrics=metrics,
        bootstrap=bootstrap,
        gates=gates,
        consistency=consistency,
        atom=atom,
        rank=rank,
        suc=suc,
    )
    encoded = content.encode("utf-8")
    manifest_payload = _prepare_manifest_update(
        paths["experiment_manifest"],
        workspace=workspace_path,
        output=output_path,
        report_bytes=encoded,
    )
    _write_report_and_manifest(
        output_path, paths["experiment_manifest"], content, manifest_payload
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the fail-closed Chinese CAA-RAHC experiment report"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = generate_report(
        args.root,
        workspace=args.workspace,
        output=args.output,
        protocol=args.protocol,
    )
    print(json.dumps({"report": str(path), "sha256": sha256_file(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReportInputError",
    "generate_report",
    "main",
    "sha256_file",
]
