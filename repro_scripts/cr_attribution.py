from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("outputs/cr_mscadm")


def main() -> None:
    all_metrics = pd.read_csv(ROOT / "tables" / "all_metrics.csv").set_index("method")
    nomask = pd.read_csv(ROOT / "tables" / "mask_control.csv").set_index("method")
    samplemask = pd.read_csv(ROOT / "tables" / "samplemask_control.csv").set_index("method")
    selected = pd.DataFrame(
        [
            {"method": "MS-CADM raw", **all_metrics.loc["MS-CADM controlled raw"].to_dict()},
            {"method": "No mask raw", **nomask.loc["full no-mask raw"].to_dict()},
            {"method": "No mask calibrated", **nomask.loc["full no-mask calibrated"].to_dict()},
            {"method": "Sample mask raw", **samplemask.loc["full sample-mask raw"].to_dict()},
            {"method": "Sample mask calibrated", **samplemask.loc["full sample-mask calibrated"].to_dict()},
            {"method": "Feature mask raw", **all_metrics.loc["CR full raw seed0"].to_dict()},
            {"method": "Feature mask calibrated", **all_metrics.loc["CR full calibrated seed0"].to_dict()},
        ]
    )
    selected.to_csv(ROOT / "tables" / "mask_attribution.csv", index=False)
    indexed = selected.set_index("method")
    baseline = indexed.loc["MS-CADM raw"]
    sample = indexed.loc["Sample mask calibrated"]
    feature = indexed.loc["Feature mask calibrated"]
    audit = {
        "core_residual_plus_calibration_under_baseline_matched_sample_mask": {
            "CRPS_relative_reduction": float((baseline.CRPS - sample.CRPS) / baseline.CRPS),
            "MAE_relative_reduction": float((baseline.MAE - sample.MAE) / baseline.MAE),
            "VS_relative_reduction": float((baseline.VS - sample.VS) / baseline.VS),
            "coverage_90_change": float(sample.coverage_90 - baseline.coverage_90),
            "passes_80pct_coverage_target": bool(sample.coverage_90 >= 0.8),
        },
        "additional_featurewise_RCM_effect_over_sample_mask": {
            "CRPS_relative_reduction": float((sample.CRPS - feature.CRPS) / sample.CRPS),
            "MAE_relative_reduction": float((sample.MAE - feature.MAE) / sample.MAE),
            "VS_relative_reduction": float((sample.VS - feature.VS) / sample.VS),
            "coverage_90_change": float(feature.coverage_90 - sample.coverage_90),
        },
        "interpretation": (
            "Conditional residual diffusion plus validation-only copula calibration is independently useful, "
            "but the registered coverage target is achieved only when feature-wise RCM is retained."
        ),
    }
    (ROOT / "tables" / "attribution_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    plot = selected[selected["method"].isin(
        ["MS-CADM raw", "No mask calibrated", "Sample mask calibrated", "Feature mask calibrated"]
    )]
    labels = ["MS raw", "No mask", "Sample mask", "Feature mask"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    colors = ["#777777", "#59A14F", "#F28E2B", "#E15759"]
    for ax, metric, title in zip(
        axes,
        ["CRPS", "coverage_90", "VS"],
        ["CRPS", "90% coverage", "Variogram score"],
    ):
        ax.bar(range(4), plot[metric], color=colors)
        ax.set_xticks(range(4), labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        if metric == "coverage_90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "figure10_mask_attribution.png", dpi=220)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
