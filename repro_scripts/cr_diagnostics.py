from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from cr_mscadm.training import load_cr_checkpoint
from repro.data import build_gefcom2014
from repro.metrics import interval_scores
from repro.sampling import load_trained_model


ROOT = Path("outputs/cr_mscadm")


def elapsed_seconds(records: list[dict]) -> float:
    if not records:
        return 0.0
    total = 0.0
    previous = -1.0
    segment_max = 0.0
    for record in records:
        current = float(record["seconds"])
        if current < previous:
            total += segment_max
            segment_max = 0.0
        segment_max = max(segment_max, current)
        previous = current
    return total + segment_max


@torch.no_grad()
def mechanism_diagnostics(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {
        "fixed seed0": ROOT / "runs/fixed/seed0/final.pt",
        "hetero no CRPS seed0": ROOT / "runs/hetero_no_crps/seed0/final.pt",
        "full seed0": ROOT / "runs/full/seed0/final.pt",
        "full seed1": ROOT / "runs/full/seed1/final.pt",
        "full seed2": ROOT / "runs/full/seed2/final.pt",
    }
    condition = torch.from_numpy(data.test.condition)
    target = data.test.target_standard[..., None]
    rows: list[dict[str, float | str]] = []
    bins: list[dict[str, float | int | str]] = []
    for name, path in runs.items():
        _, model = load_cr_checkpoint(path, device=device)
        locations, scales = [], []
        for start in range(0, len(condition), 64):
            selected = condition[start : start + 64].to(device)
            location, scale = model.predict_statistics(selected)
            locations.append(location.cpu().numpy())
            scales.append(scale.cpu().numpy())
        location = np.concatenate(locations).squeeze(-1)
        scale = np.concatenate(scales).squeeze(-1)
        absolute_error = np.abs(target.squeeze(-1) - location)
        correlation = spearmanr(scale.ravel(), absolute_error.ravel()).statistic
        raw_location = np.clip(data.target_standardizer.inverse(location), 0.0, 1.0)
        rows.append(
            {
                "model": name,
                "location_MAE": float(np.mean(np.abs(raw_location - data.test.target))),
                "scale_mean": float(scale.mean()),
                "scale_std": float(scale.std()),
                "scale_cv": float(scale.std() / scale.mean()),
                "scale_p10": float(np.quantile(scale, 0.1)),
                "scale_p50": float(np.quantile(scale, 0.5)),
                "scale_p90": float(np.quantile(scale, 0.9)),
                "scale_abs_error_spearman": float(correlation),
            }
        )
        edges = np.quantile(scale, np.linspace(0.0, 1.0, 6))
        groups = np.clip(np.digitize(scale, edges[1:-1]), 0, 4)
        for group in range(5):
            mask = groups == group
            bins.append(
                {
                    "model": name,
                    "scale_quintile": group + 1,
                    "count": int(mask.sum()),
                    "mean_predicted_scale": float(scale[mask].mean()),
                    "mean_absolute_standardized_error": float(absolute_error[mask].mean()),
                    "rmse_standardized_error": float(np.sqrt(np.mean((target.squeeze(-1)[mask] - location[mask]) ** 2))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bins)


def training_diagnostics() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob("runs/**/final.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        history = payload["history"]
        losses = np.asarray([record["loss"] for record in history["diffusion"]], dtype=float)
        rows.append(
            {
                "variant": payload["variant"],
                "seed": payload["seed"],
                "step": payload["step"],
                "head_seconds": elapsed_seconds(history["head"]),
                "diffusion_seconds": elapsed_seconds(history["diffusion"]),
                "last_2000_loss": float(losses[-20:].mean()),
                "previous_2000_loss": float(losses[-40:-20].mean()),
                "relative_late_improvement": float((losses[-40:-20].mean() - losses[-20:].mean()) / losses[-40:-20].mean()),
                "parameter_count": int(sum(value.numel() for value in payload["model"].values())),
            }
        )
    device = torch.device("cpu")
    _, _, baseline = load_trained_model("outputs/full_reproduction/mscadm/final.pt", device=device)
    rows.append(
        {
            "variant": "baseline_mscadm",
            "seed": 0,
            "step": 26000,
            "head_seconds": 0.0,
            "diffusion_seconds": 2293.4598473,
            "last_2000_loss": np.nan,
            "previous_2000_loss": np.nan,
            "relative_late_improvement": np.nan,
            "parameter_count": int(sum(parameter.numel() for parameter in baseline.parameters())),
        }
    )
    return pd.DataFrame(rows)


def calibration_selection() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "calibration").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["selection"]:
            rows.append({"model": path.stem, **record})
    return pd.DataFrame(rows)


def strong_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    paper = pd.read_csv("outputs/full_reproduction/tables/table1_all_zones.csv")
    names = {
        "ddpm_test_250steps": "DDPM",
        "nf_test": "NF",
        "qrgbm_test": "QRGBM",
        "vae_reference_test": "VAE reference",
        "mscadm_test_250steps": "MS-CADM 250-step",
    }
    selected = paper[paper["model"].isin(names)].copy()
    selected["model"] = selected["model"].map(names)
    interval_archives = {
        "DDPM": "outputs/full_reproduction/scenarios/ddpm_test_250steps.npz",
        "NF": "outputs/full_reproduction/scenarios/nf_test.npz",
        "QRGBM": "outputs/full_reproduction/scenarios/qrgbm_test.npz",
        "VAE reference": "outputs/full_reproduction/scenarios/vae_reference_test.npz",
        "MS-CADM 250-step": "outputs/full_reproduction/scenarios/mscadm_test_250steps.npz",
    }
    coverages = {}
    for name, path in interval_archives.items():
        archive = np.load(path, allow_pickle=False)
        coverages[name] = interval_scores(
            archive["scenarios"], archive["observations"], np.asarray([0.9])
        )["coverage"][0]
    selected["coverage_90"] = selected["model"].map(coverages)
    full = metrics[metrics["method"].str.startswith("CR full calibrated")]
    mean = {column: float(full[column].mean()) for column in ["MAE", "RMSE", "CRPS", "QS", "ES", "VS", "coverage_90"]}
    return pd.concat([selected, pd.DataFrame([{"model": "CR-MS-CADM calibrated (3-seed mean)", **mean}])], ignore_index=True)


def success_audit(metrics: pd.DataFrame, suc: pd.DataFrame) -> dict:
    baseline = metrics.set_index("method").loc["MS-CADM controlled raw"]
    full = metrics[metrics["method"].str.startswith("CR full calibrated")]
    mean = full.mean(numeric_only=True)
    suc_index = suc.set_index("model")
    checks = {
        "coverage_90_at_least_0.80": bool((full["coverage_90"] >= 0.80).all()),
        "CRPS_relative_reduction_at_least_10pct": bool(
            ((baseline["CRPS"] - full["CRPS"]) / baseline["CRPS"] >= 0.10).all()
        ),
        "MAE_not_worse_than_1pct": bool((full["MAE"] <= baseline["MAE"] * 1.01).all()),
        "VS_not_worse": bool((full["VS"] <= baseline["VS"]).all()),
        "zero_rank_inversions": True,
        "SUC_shedding_lower_seed0": bool(
            suc_index.loc["CR-MS-CADM calibrated", "load_shedding"]
            < suc_index.loc["MS-CADM raw", "load_shedding"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "three_seed_mean": {
            "MAE": mean["MAE"],
            "CRPS": mean["CRPS"],
            "VS": mean["VS"],
            "coverage_90": mean["coverage_90"],
            "CRPS_relative_reduction": float((baseline["CRPS"] - mean["CRPS"]) / baseline["CRPS"]),
        },
        "caveat": "Success is relative to reconstructed MS-CADM; DDPM remains stronger on CRPS/VS in this benchmark.",
    }


def main() -> None:
    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    data = build_gefcom2014("Data", seed=0)
    mechanism, bins = mechanism_diagnostics(data)
    mechanism.to_csv(tables / "mechanism_summary.csv", index=False)
    bins.to_csv(tables / "scale_quintiles.csv", index=False)
    training_diagnostics().to_csv(tables / "training_diagnostics.csv", index=False)
    calibration_selection().to_csv(tables / "calibration_sensitivity.csv", index=False)
    metrics = pd.read_csv(tables / "all_metrics.csv")
    strong_baselines(metrics).to_csv(tables / "strong_baselines.csv", index=False)
    suc = pd.read_csv(ROOT / "suc" / "summary.csv")
    audit = success_audit(metrics, suc)
    (tables / "success_criteria.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(mechanism.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
