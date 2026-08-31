from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path("outputs/cr_mscadm")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def record(checks: dict, name: str, passed: bool, detail) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def main() -> None:
    checks: dict = {}

    expected_full = [ROOT / f"runs/full/seed{seed}/final.pt" for seed in range(3)]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in expected_full]
    record(
        checks,
        "three_independent_full_models",
        all(path.exists() for path in expected_full)
        and [value["seed"] for value in payloads] == [0, 1, 2]
        and all(value["step"] == 18_000 and value["variant"] == "full" for value in payloads),
        [{"path": str(path), "seed": value["seed"], "step": value["step"]} for path, value in zip(expected_full, payloads)],
    )

    other_runs = {
        "fixed": ROOT / "runs/fixed/seed0/final.pt",
        "hetero_no_crps": ROOT / "runs/hetero_no_crps/seed0/final.pt",
        "no_mask_control": ROOT / "runs/full_nomask/seed0/final.pt",
        "sample_mask_control": ROOT / "runs/full_samplemask/seed0/final.pt",
    }
    other_detail = {}
    other_ok = True
    for name, path in other_runs.items():
        value = torch.load(path, map_location="cpu", weights_only=False)
        other_detail[name] = {"path": str(path), "step": value["step"], "variant": value["variant"]}
        other_ok &= value["step"] == 18_000
    record(checks, "ablations_and_mask_controls", other_ok, other_detail)

    required_scenarios = [
        "baseline_mscadm_validation_raw.npz",
        "baseline_mscadm_test_raw.npz",
        "baseline_mscadm_test_calibrated.npz",
        "fixed_seed0_validation_raw.npz",
        "fixed_seed0_test_raw.npz",
        "hetero_no_crps_seed0_validation_raw.npz",
        "hetero_no_crps_seed0_test_raw.npz",
        "full_nomask_seed0_validation_raw.npz",
        "full_nomask_seed0_test_raw.npz",
        "full_nomask_seed0_test_calibrated.npz",
        "full_samplemask_seed0_validation_raw.npz",
        "full_samplemask_seed0_test_raw.npz",
        "full_samplemask_seed0_test_calibrated.npz",
    ]
    for seed in range(3):
        required_scenarios.extend(
            [
                f"full_seed{seed}_validation_raw.npz",
                f"full_seed{seed}_test_raw.npz",
                f"full_seed{seed}_test_calibrated.npz",
            ]
        )
    scenario_detail = {}
    test_reference = None
    scenario_ok = True
    for name in required_scenarios:
        path = ROOT / "scenarios" / name
        archive = np.load(path, allow_pickle=False)
        scenarios = archive["scenarios"]
        expected_shape = (500, 100, 24)
        valid = (
            scenarios.shape == expected_shape
            and np.isfinite(scenarios).all()
            and float(scenarios.min()) >= 0.0
            and float(scenarios.max()) <= 1.0
        )
        if "_test_" in name:
            signature = (archive["observations"].tobytes(), archive["zone"].tobytes(), archive["day"].tobytes())
            test_reference = signature if test_reference is None else test_reference
            valid &= signature == test_reference
        scenario_ok &= valid
        scenario_detail[name] = {"shape": list(scenarios.shape), "min": float(scenarios.min()), "max": float(scenarios.max())}
    record(checks, "scenario_archives_and_alignment", scenario_ok, scenario_detail)

    calibrations = ["baseline_mscadm", "full_seed0", "full_seed1", "full_seed2", "full_nomask_seed0", "full_samplemask_seed0"]
    calibration_detail = {}
    calibration_ok = True
    for name in calibrations:
        path = ROOT / "calibration" / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        strengths = [row["strength"] for row in payload["selection"]]
        objectives = [row["objective"] for row in payload["selection"]]
        selected = strengths[int(np.argmin(objectives))]
        calibration_detail[name] = {"selected_strength": selected, "candidates": strengths}
        calibration_ok &= strengths == [0.0, 0.25, 0.5, 0.75, 1.0]
    for name in ["baseline_mscadm_test_calibrated.npz"] + [f"full_seed{seed}_test_calibrated.npz" for seed in range(3)]:
        archive = np.load(ROOT / "scenarios" / name, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        calibration_ok &= metadata["rank_inversions"] == 0
        calibration_detail[name] = {"rank_inversions": metadata["rank_inversions"], "strength": metadata["strength"]}
    record(checks, "validation_only_calibration_and_zero_inversions", calibration_ok, calibration_detail)

    metrics = pd.read_csv(ROOT / "tables" / "all_metrics.csv")
    success = json.loads((ROOT / "tables" / "success_criteria.json").read_text(encoding="utf-8"))
    attribution = json.loads((ROOT / "tables" / "attribution_audit.json").read_text(encoding="utf-8"))
    record(
        checks,
        "metrics_success_and_attribution",
        len(metrics) == 11 and success["passed"] and not attribution["core_residual_plus_calibration_under_baseline_matched_sample_mask"]["passes_80pct_coverage_target"],
        {"metric_rows": len(metrics), "success": success, "attribution": attribution},
    )

    bootstrap = json.loads((ROOT / "tables" / "paired_bootstrap.json").read_text(encoding="utf-8"))
    bootstrap_ok = len(bootstrap) == 3 and all(
        run["CRPS"]["ci_low"] > 0 and run["VS"]["ci_low"] > 0 for run in bootstrap.values()
    )
    record(checks, "paired_day_bootstrap", bootstrap_ok, bootstrap)

    suc = pd.read_csv(ROOT / "suc" / "daily_results.csv")
    protocol = json.loads((ROOT / "suc" / "protocol.json").read_text(encoding="utf-8"))
    grouped_days = {name: sorted(part["day_index"].tolist()) for name, part in suc.groupby("model")}
    target_days = sorted(protocol["common_day_indices"])
    suc_ok = len(suc) == 28 and all(days == target_days for days in grouped_days.values())
    record(checks, "fair_seven_day_suc", suc_ok, {"rows": len(suc), "days": grouped_days})

    figure_detail = {}
    figures = sorted((ROOT / "figures").glob("*.png"))
    figures_ok = len(figures) >= 10
    for path in figures:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                figure_detail[path.name] = {"size": list(image.size), "bytes": path.stat().st_size}
                figures_ok &= image.width >= 800 and image.height >= 600
        except Exception as error:
            figure_detail[path.name] = {"error": str(error)}
            figures_ok = False
    record(checks, "figures_decode_and_resolution", figures_ok, figure_detail)

    xml = ET.parse(ROOT / "pytest.xml").getroot()
    test_ok = int(xml.attrib.get("failures", 0)) == 0 and int(xml.attrib.get("errors", 0)) == 0 and int(xml.attrib.get("tests", 0)) == 26
    record(checks, "full_test_suite", test_ok, xml.attrib)

    report = Path("CR_MSCADM_EXPERIMENT_REPORT.md")
    report_text = report.read_text(encoding="utf-8")
    report_ok = report.exists() and len(report_text) > 10_000 and "Feature-wise RCM" in report_text and "局限" in report_text
    record(checks, "complete_chinese_report", report_ok, {"path": str(report.resolve()), "characters": len(report_text)})

    source_paths = [
        Path("CR_MSCADM_EXPERIMENT_REPORT.md"),
        Path("repro_configs/cr_mscadm.json"),
        Path("cr_mscadm/model.py"),
        Path("cr_mscadm/calibration/__init__.py"),
        Path("cr_mscadm/training.py"),
        Path("cr_mscadm/experiment.py"),
    ]
    artifact_paths = [path for path in ROOT.rglob("*") if path.is_file() and path.name not in {"experiment_manifest.json", "completion_audit.json"}]
    manifest = {
        str(path.resolve()): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(set(source_paths + artifact_paths))
    }
    (ROOT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    record(checks, "sha256_manifest", len(manifest) >= 100, {"files": len(manifest), "path": str((ROOT / "experiment_manifest.json").resolve())})

    audit = {"passed": all(item["passed"] for item in checks.values()), "checks": checks}
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "checks": {name: value["passed"] for name, value in checks.items()}}, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
