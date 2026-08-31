from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every required formal-reproduction artifact")
    parser.add_argument("--root", default="outputs/full_reproduction")
    args = parser.parse_args()
    root = Path(args.root)
    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    scenario_root = root / "scenarios"
    all_zone = {
        "RAND": "rand_test.npz",
        "QRGBM": "qrgbm_test.npz",
        "WGAN": "wgan_reference_test.npz",
        "VAE": "vae_reference_test.npz",
        "NF": "nf_test.npz",
        "DDPM": "ddpm_test_250steps.npz",
        "MS-CADM": "mscadm_test_250steps.npz",
    }
    shapes = {}
    for model, filename in all_zone.items():
        path = scenario_root / filename
        if not path.exists():
            shapes[model] = None
            continue
        with np.load(path, allow_pickle=False) as archive:
            shapes[model] = list(archive["scenarios"].shape)
    record(
        "table1_all_models_and_shapes",
        set(shapes) == set(all_zone) and all(value == [500, 100, 24] for value in shapes.values()),
        shapes,
    )

    single_zone_expected = {
        "zone1_rand.npz",
        "zone1_qrgbm.npz",
        "zone1_wgan_reference.npz",
        "zone1_vae_reference.npz",
        "zone1_nf.npz",
        "zone1_ddpm.npz",
        "zone1_mscadm.npz",
    }
    single_shapes = {}
    for filename in sorted(single_zone_expected):
        path = scenario_root / "single_zone" / filename
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                single_shapes[filename] = list(archive["scenarios"].shape)
        else:
            single_shapes[filename] = None
    record(
        "table2_independently_trained_single_zone",
        all(value == [50, 100, 24] for value in single_shapes.values()),
        single_shapes,
    )

    step_paths = sorted((scenario_root / "sampling_steps").glob("mscadm_*steps.npz"))
    found_steps = sorted(int(path.stem.split("_")[1].replace("steps", "")) for path in step_paths)
    record("table3_sampling_steps", found_steps == [10, 20, 50, 100, 250], found_steps)

    ablation_paths = sorted((scenario_root / "ablations").glob("zone1_*.npz"))
    found_ablations = sorted(path.stem.replace("zone1_", "") for path in ablation_paths)
    ablation_shapes = {}
    for path in ablation_paths:
        with np.load(path, allow_pickle=False) as archive:
            ablation_shapes[path.stem] = list(archive["scenarios"].shape)
    record(
        "table4_independent_ablations",
        found_ablations == ["full", "no_adaln", "no_ce", "no_lv", "no_rcm"]
        and all(value == [50, 100, 24] for value in ablation_shapes.values()),
        {"variants": found_ablations, "shapes": ablation_shapes},
    )

    suc_path = root / "suc" / "daily_results.csv"
    suc = pd.read_csv(suc_path) if suc_path.exists() else pd.DataFrame()
    suc_models = ["ddpm", "mscadm", "nf", "qrgbm", "vae_reference", "wgan_reference"]
    day_sets = {
        model: sorted(suc.loc[suc["model"] == model, "day_index"].astype(int).tolist())
        for model in suc_models
    } if len(suc) else {}
    shared = len({tuple(values) for values in day_sets.values()}) == 1 if day_sets else False
    record(
        "table5_fair_suc",
        len(suc) == 42 and set(suc["model"]) == set(suc_models) and shared,
        {"rows": len(suc), "day_indices": day_sets},
    )

    all_zone_shards = sorted((root / "qrgbm_sharded").glob("hour_*.ubj"))
    zone_shards = sorted((root / "single_zone" / "zone1" / "qrgbm_sharded").glob("hour_*.ubj"))
    record(
        "qrgbm_hourly_checkpoints",
        len(all_zone_shards) == 24 and len(zone_shards) == 24,
        {"all_zone": len(all_zone_shards), "single_zone": len(zone_shards)},
    )

    required_tables = [
        "table1_all_zones.csv",
        "table2_zone1.csv",
        "table3_sampling_steps.csv",
        "table4_ablations.csv",
    ]
    table_state = {name: (root / "tables" / name).exists() for name in required_tables}
    table_state["table5_suc.csv"] = (root / "suc" / "table5_suc.csv").exists()
    record("tables_exist", all(table_state.values()), table_state)

    figure_paths = sorted((root / "figures").glob("*.png"))
    figures = {}
    for path in figure_paths:
        try:
            with Image.open(path) as image:
                image.verify()
                figures[path.name] = {"valid": True, "bytes": path.stat().st_size}
        except Exception as error:  # audit should report corrupt images, not crash early
            figures[path.name] = {"valid": False, "error": repr(error)}
    required_figure_names = {
        "figure5_intervals.png",
        "figure6_zone1_day1.png",
        "figure6_zone1_day2.png",
        "figure6_zone1_day3.png",
        "figure7_distribution.png",
        "figure4_qrgbm_test.png",
        "figure4_mscadm_test_250steps.png",
    }
    record(
        "figures_decode",
        required_figure_names.issubset(figures) and all(value["valid"] for value in figures.values()),
        {"count": len(figures), "figures": figures},
    )

    final_checkpoints = sorted(root.rglob("final.pt"))
    record(
        "trained_checkpoints",
        len(final_checkpoints) >= 16 and all(path.stat().st_size > 0 for path in final_checkpoints),
        [path.relative_to(root).as_posix() for path in final_checkpoints],
    )

    tests_log = root / "pipeline_logs" / "tests.log"
    test_text = tests_log.read_text(encoding="utf-8", errors="replace") if tests_log.exists() else ""
    record("test_suite", "22 passed" in test_text, test_text.splitlines()[-1:] or ["missing"])

    manifest = root / "experiment_manifest.json"
    record("hash_manifest", manifest.exists() and manifest.stat().st_size > 0, str(manifest.resolve()))

    passed = all(value["passed"] for value in checks.values())
    report = {"passed": passed, "checks": checks}
    destination = root / "completion_audit.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": {key: value["passed"] for key, value in checks.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
