from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from repro_scripts import caa_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection_entries(fallback: bool) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for family in caa_report.FAMILIES[1:]:
        selected = "A0" if family == "A4" and fallback else f"{family}_candidate"
        result["main_A4" if family == "A4" else family] = {
            "family": family,
            "selected": selected,
            "fallback": family == "A4" and fallback,
            "config": (
                {"definition": "exact A0"}
                if family == "A4" and fallback
                else {
                    "atom_strength": 0.5,
                    "tail_strength": 0.25,
                    "dmax": 0.25,
                    "width_delta_cap": 0.05,
                    "width_cap_reference": "atom_only",
                }
            ),
            "decision_key": "constrained_A4" if family == "A4" else family,
        }
    return result


def _point_values(family: str, outer: str, seed: int, fallback: bool) -> dict[str, float]:
    base = {
        "CRPS": 0.100 + seed * 0.001,
        "coverage_90": 0.860 + seed * 0.001,
        "width_90": 0.300 + seed * 0.001,
        "winkler_90": 0.500 + seed * 0.001,
        "conditional_ACE90": 0.080 + seed * 0.001,
    }
    if family == "A0" or (family == "A4" and fallback):
        return base
    offsets = {
        "A1": (0.002, 0.005, 0.005, 0.010, -0.005),
        "A2": (0.001, 0.010, 0.010, 0.005, -0.010),
        "A3": (0.000, 0.015, 0.015, -0.005, -0.015),
        "A4": (-0.005, 0.040, 0.020, -0.050, -0.030),
        "A5": (-0.006, 0.045, 0.030, -0.055, -0.032),
        "A6": (-0.003, 0.035, 0.025, -0.040, -0.025),
    }
    delta = offsets[family]
    return {
        metric: base[metric] + delta[index]
        for index, metric in enumerate(caa_report.POINT_METRICS)
    }


def _bootstrap_payload(fallback: bool) -> dict[str, object]:
    points = {
        "crps_improvement": 0.0 if fallback else 0.005,
        "coverage_absolute_error_90_improvement": 0.0 if fallback else 0.04,
        "interval_width_90_change": 0.0 if fallback else 0.02,
        "winkler_90_improvement": 0.0 if fallback else 0.05,
        "conditional_family_equal_ACE_90_improvement": 0.0 if fallback else 0.03,
    }
    metrics = {}
    for name, point in points.items():
        metrics[name] = {
            "point_estimate": point,
            "ci_low": point - 0.001,
            "ci_high": point + 0.001,
            "direction": (
                "method minus baseline; positive means wider method intervals"
                if name == "interval_width_90_change"
                else "baseline minus method; positive is better"
            ),
        }
    return {
        "schema": "caa_rahc_paired_calendar_day_bootstrap_v1",
        "protocol": {
            "replicates": 5000,
            "unique_calendar_days": 150,
            "cluster": "calendar date",
        },
        "comparisons": {"A4": {"metrics": metrics}},
    }


def _build_tree(
    tmp_path: Path, *, fallback: bool = False, selection_alias: bool = False
) -> tuple[Path, Path, Path, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "outputs" / "caa_rahc"
    root.mkdir(parents=True)
    protocol = workspace / "CAA_RAHC_FROZEN_PROTOCOL.md"
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    output = workspace / "CAA_RAHC_EXPERIMENT_REPORT.md"

    selected = _selection_entries(fallback)
    selection_audit_path = root / "calibration/selection.audit.json"
    if selection_alias:
        selected_configs = {
            "A0": {
                "selected": "A0",
                "fallback": False,
                "config": {"definition": "locked A0 baseline"},
            },
            **{
                family: selected["main_A4" if family == "A4" else family]
                for family in caa_report.FAMILIES[1:]
            },
        }
        audit_payload = {
            "protocol": {
                "test_access": "none; only calibration archives were loaded"
            },
            "selected_configs": selected_configs,
        }
        lock_selection = {"selected_configs": selected_configs}
    else:
        audit_payload = {
            "schema": "caa_rahc_calibration_selection_audit_v1",
            "protocol": {
                "test_access": "none; only calibration archives were loaded"
            },
            "selected": selected,
        }
        lock_selection = {"selected": selected}
    _write_json(selection_audit_path, audit_payload)
    _write_json(
        root / "selection.lock.json",
        {
            "schema": "caa_rahc_selection_lock_v1",
            "selection_audit": "calibration/selection.audit.json",
            "selection_audit_sha256": caa_report.sha256_file(selection_audit_path),
            "test_archives_accessed": False,
            **lock_selection,
        },
    )

    overall_rows = []
    for family in caa_report.FAMILIES:
        for outer in ("1", "2", "3", "pooled"):
            for seed in caa_report.SEEDS:
                overall_rows.append(
                    {
                        "method": family,
                        "outer": outer,
                        "seed": seed,
                        **_point_values(family, outer, seed, fallback),
                    }
                )
    _write_csv(root / "metrics/overall_metrics.csv", overall_rows)

    conditional_rows = []
    for family in caa_report.FAMILIES:
        for outer in ("1", "2", "3", "pooled"):
            for seed in caa_report.SEEDS:
                point = _point_values(family, outer, seed, fallback)
                conditional_rows.append(
                    {
                        "method": family,
                        "outer": outer,
                        "seed": seed,
                        "family": "zone",
                        "group": "zone-1",
                        "group_code": "z1",
                        "nominal": 0.9,
                        "coverage_90": point["coverage_90"],
                        "ACE_90": abs(point["coverage_90"] - 0.9),
                        "undercoverage_90": max(0.9 - point["coverage_90"], 0),
                        "overcoverage_90": max(point["coverage_90"] - 0.9, 0),
                        "interval_width_90": point["width_90"],
                        "winkler_score_90": point["winkler_90"],
                        "n": 100,
                    }
                )
    _write_csv(root / "metrics/conditional_metrics.csv", conditional_rows)

    analytic = {
        "zero_Brier": 0.01,
        "zero_log_loss": 0.05,
        "zero_predicted_rate": 0.02,
        "zero_observed_rate": 0.018,
        "one_Brier": 0.001,
        "one_log_loss": 0.005,
        "one_predicted_rate": 0.0002,
        "one_observed_rate": 0.0001,
    }
    _write_json(
        root / "metrics/atom_diagnostics.json",
        {
            "schema": "caa_rahc_atom_diagnostics_v1",
            "protocol": {"complete": True},
            "by_family_seed": {
                "A4": {
                    f"seed{seed}": {
                        "analytic_scores": analytic,
                        "zero_reliability": [{"bin": 1, "count": 100}],
                        "one_reliability": [{"bin": 1, "count": 100}],
                        "finite_ensemble": {"members": 100},
                    }
                    for seed in caa_report.SEEDS
                }
            },
        },
    )
    finite = {
        "members": 100,
        "probability_resolution": 0.01,
        "zero_quantization_MAE": 0.002,
        "zero_quantization_max_abs": 0.005,
        "one_quantization_MAE": 0.001,
        "one_quantization_max_abs": 0.004,
    }
    _write_json(
        root / "metrics/quantization_audit.json",
        {
            "schema": "caa_rahc_quantization_audit_v1",
            "by_family_seed": {
                "A4": {f"seed{seed}": finite for seed in caa_report.SEEDS}
            },
        },
    )

    rank_rows = []
    for family in caa_report.FAMILIES:
        for outer in caa_report.OUTERS:
            for seed in caa_report.SEEDS:
                exact = family == "A0" or (family == "A4" and fallback)
                rank_rows.append(
                    {
                        "method": family,
                        "outer": outer,
                        "seed": seed,
                        "selected": (
                            "A0"
                            if family == "A0" or (family == "A4" and fallback)
                            else f"{family}_candidate"
                        ),
                        "fallback": family == "A4" and fallback,
                        "exact_A0": exact,
                        "comparison_source": (
                            "raw_to_A0" if family == "A0" else "A0_to_family"
                        ),
                        "strict_reversals": 0,
                        "raw_ties_broken": 0,
                        "strict_pairs_collapsed": 2 if family in {"A2", "A4"} else 0,
                        "raw_tied_pairs": 3,
                        "calibrated_tied_pairs": 5,
                        "stable_ordinal_rank_matches": 100,
                        "stable_ordinal_rank_total": 100,
                        "stable_ordinal_rank_fraction": 1.0,
                        "raw_boundary_fraction": 0.01,
                        "calibrated_boundary_fraction": 0.02,
                        "pair_comparisons": 1000,
                        "central_values_changed_by_tail": 0,
                        "nonfinite_input_values": 0,
                        "nonfinite_intermediate_values": 0,
                    }
                )
    _write_csv(root / "metrics/rank_audit.csv", rank_rows)
    _write_json(root / "statistics/paired_calendar_day_bootstrap.json", _bootstrap_payload(fallback))
    _write_json(
        root / "statistics/noninferiority.json",
        {"schema": "caa_noninferiority_v1", "baseline": "A0", "comparisons": {"A4": {"complete": True}}},
    )
    gate_values = {
        "coverage_90": True,
        "conditional_ACE90_reduction": not fallback,
        "CRPS_noninferiority": True,
        "MAE_noninferiority": True,
        "VS_noninferiority": False,
        "ramp_CRPS_noninferiority": False,
        "winkler_90_noninferiority": True,
        "width_90_increase": True,
        "outer_consistency": False,
    }
    _write_json(
        root / "statistics/success_gates.json",
        {
            "schema": "caa_success_gates_v1",
            "passed": sum(gate_values.values()),
            "total": len(gate_values),
            "all_passed": all(gate_values.values()),
            "gates": gate_values,
        },
    )
    _write_json(
        root / "statistics/outer_consistency.json",
        {
            "schema": "caa_outer_consistency_v1",
            "primary": "A4",
            "baseline": "A0",
            "catastrophic_rule": "registered",
            "per_outer_seed": [
                {"outer": outer, "seed": seed, "direction": "primary"}
                for outer in caa_report.OUTERS
                for seed in caa_report.SEEDS
            ],
            "per_outer": [
                {"outer": outer, "direction": "primary", "catastrophic": False}
                for outer in caa_report.OUTERS
            ],
            "same_primary_direction_outer_count": 2,
            "passed": False,
        },
    )

    _write_csv(
        root / "suc/daily_results.csv",
        [
            {"family": family, "outer": 1, "planned_success": True, "realized_success": True}
            for family in ("A0", "A4")
        ],
    )
    _write_csv(
        root / "suc/summary.csv",
        [
            {
                "family": "A0",
                "paired_cases": 9,
                "successful_cases": 9,
                "mean_realized_total_cost": 1000.0,
            },
            {
                "family": "A4",
                "paired_cases": 9,
                "successful_cases": 9,
                "mean_realized_total_cost": 995.0 if not fallback else 1000.0,
            },
        ],
    )
    _write_json(
        root / "suc/protocol.json",
        {
            "schema": "caa_rahc_descriptive_suc_v1",
            "statistical_scope": "Descriptive sensitivity only",
            "success_gate_role": "none",
        },
    )
    _write_json(
        root / "experiment_manifest.json",
        {"schema": "caa_rahc_experiment_manifest_v1", "files": []},
    )
    return workspace, root, protocol, output


def test_success_report_is_result_first_traceable_and_refreshes_manifest(tmp_path: Path) -> None:
    workspace, root, protocol, output = _build_tree(tmp_path, fallback=False)

    result = caa_report.generate_report(
        root, workspace=workspace, output=output, protocol=protocol
    )

    assert result == output.resolve()
    text = output.read_text(encoding="utf-8")
    assert text.index("## 结论摘要（结果先行）") < text.index("## 1. 证据等级")
    assert "A4 未回退" in text
    assert "6/9" in text
    assert "post-freeze internal nested outer-split confirmation" in text
    assert "不是外部确认" in text
    assert "旧研究中的 test 划分" in text and "仅作为 calibration" in text
    assert "CRPS" in text and "coverage" in text and "Winkler" in text
    assert "finite-M" in text and "strict pairs collapsed" in text
    assert "descriptive sensitivity only" in text
    assert "[CAA_RAHC_FROZEN_PROTOCOL.md](CAA_RAHC_FROZEN_PROTOCOL.md)" in text

    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    report_record = next(
        item for item in manifest["files"] if item["path"] == "CAA_RAHC_EXPERIMENT_REPORT.md"
    )
    assert report_record["sha256"] == caa_report.sha256_file(output)
    assert report_record["bytes"] == output.stat().st_size


def test_fallback_report_uses_selected_configs_alias_and_forbids_improvement_claim(
    tmp_path: Path,
) -> None:
    workspace, root, protocol, output = _build_tree(
        tmp_path, fallback=True, selection_alias=True
    )

    caa_report.generate_report(root, workspace=workspace, output=output, protocol=protocol)

    text = output.read_text(encoding="utf-8")
    assert "A4 已精确回退到 A0" in text
    assert "不能据此宣称 A4 相对 A0 有预测改进" in text
    assert "| A4 | A0 | 是 |" in text


def test_missing_ci_field_fails_closed_without_writing_report(tmp_path: Path) -> None:
    workspace, root, protocol, output = _build_tree(tmp_path, fallback=False)
    path = root / "statistics/paired_calendar_day_bootstrap.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["comparisons"]["A4"]["metrics"]["crps_improvement"]["ci_high"]
    _write_json(path, payload)

    with pytest.raises(caa_report.ReportInputError, match="ci_high"):
        caa_report.generate_report(
            root, workspace=workspace, output=output, protocol=protocol
        )

    assert not output.exists()
